from __future__ import annotations

from collections import deque
import multiprocessing as mp
import queue
import threading
import time
from typing import Callable


def _jpeg_to_tk_payload(jpeg: bytes) -> bytes:
    """Decode JPEG into the PPM/PGM bytes consumed by the optional Tk renderer."""
    import cv2
    import numpy as np

    encoded = np.frombuffer(jpeg, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("JPEG decode failed")
    height, width = int(image.shape[0]), int(image.shape[1])
    if image.ndim == 2:
        return f"P5\n{width} {height}\n255\n".encode("ascii") + image.tobytes()
    if int(image.shape[2]) == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return f"P6\n{width} {height}\n255\n".encode("ascii") + image.tobytes()


def _video_window_main(frame_queue, stop_event) -> None:
    """Run the optional video renderer in its own Python and Tcl/Tk process."""
    # Tk is only needed by this explicitly retained renderer.  Keep it out of
    # the transport/controller import path so the Qt application works on
    # Python installations without the optional Tk bindings.
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("工业相机实时画面（独立进程）")
    root.geometry("900x700")
    root.minsize(660, 520)

    fps_var = tk.StringVar(value="实际显示: 0.0 FPS")
    header = ttk.Frame(root)
    header.pack(fill="x", padx=8, pady=(8, 2))
    ttk.Label(header, text="独立进程视频：不等待状态栏和控制周期").pack(side="left")
    ttk.Label(header, textvariable=fps_var).pack(side="right")
    video_label = ttk.Label(root, text="等待视频帧", anchor="center")
    video_label.pack(fill="both", expand=True, padx=8, pady=(2, 8))

    photo = None
    last_frame_id = None
    display_times: deque[float] = deque(maxlen=120)

    def close() -> None:
        stop_event.set()
        root.destroy()

    def poll() -> None:
        nonlocal photo, last_frame_id
        if stop_event.is_set():
            root.destroy()
            return
        latest = None
        while True:
            try:
                item = frame_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                close()
                return
            latest = item
        if latest is not None:
            frame_id, encoding, payload = latest
            if frame_id != last_frame_id and payload:
                try:
                    if encoding == "jpeg":
                        photo = tk.PhotoImage(data=_jpeg_to_tk_payload(payload))
                    else:
                        photo = tk.PhotoImage(data=payload)
                    video_label.configure(image=photo, text="")
                    last_frame_id = frame_id
                    now = time.monotonic()
                    display_times.append(now)
                    while display_times and now - display_times[0] > 1.0:
                        display_times.popleft()
                    fps_var.set(f"实际显示: {len(display_times):.1f} FPS")
                except Exception:
                    video_label.configure(text="视频帧解码失败")
        root.after(15, poll)

    root.protocol("WM_DELETE_WINDOW", close)
    root.after(0, poll)
    try:
        root.mainloop()
    finally:
        stop_event.set()


class VideoProcessController:
    """Forward only the newest orchestrator preview frame to a video process."""

    def __init__(self, get_frame: Callable[[], object]):
        self._get_frame = get_frame
        self._context = mp.get_context("spawn")
        self._frame_queue = self._context.Queue(maxsize=1)
        self._process_stop = self._context.Event()
        self._thread_stop = threading.Event()
        self._process = self._context.Process(
            target=_video_window_main,
            args=(self._frame_queue, self._process_stop),
            name="monitor-video-process",
            daemon=True,
        )
        self._producer: threading.Thread | None = None

    def start(self) -> None:
        self._process.start()
        self._producer = threading.Thread(
            target=self._forward_latest_frames,
            name="monitor-video-forwarder",
            daemon=True,
        )
        self._producer.start()

    def is_alive(self) -> bool:
        return self._process.is_alive() and not self._process_stop.is_set()

    def _forward_latest_frames(self) -> None:
        last_frame_id = None
        last_send = 0.0
        send_interval = 1.0 / 30.0
        while not self._thread_stop.is_set() and self._process.is_alive() and not self._process_stop.is_set():
            now = time.perf_counter()
            remaining = send_interval - (now - last_send)
            if remaining > 0.0:
                self._thread_stop.wait(min(0.003, remaining))
                continue
            try:
                frame = self._get_frame()
                if frame is None or not frame.valid:
                    self._thread_stop.wait(0.003)
                    continue
                frame_id = int(frame.frame_id)
                if frame_id == last_frame_id:
                    # Do not consume a full 33 ms slot when the encoder is a
                    # fraction late. Poll briefly and send it as soon as ready.
                    self._thread_stop.wait(0.002)
                    continue
                if frame.frame_jpeg:
                    item = (frame_id, "jpeg", frame.frame_jpeg)
                elif frame.frame_png_base64:
                    item = (frame_id, "png", frame.frame_png_base64)
                else:
                    continue
                try:
                    self._frame_queue.put_nowait(item)
                except queue.Full:
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._frame_queue.put_nowait(item)
                    except queue.Full:
                        pass
                last_frame_id = frame_id
                last_send = time.perf_counter()
            except Exception:
                self._thread_stop.wait(0.01)

    def stop(self) -> None:
        self._thread_stop.set()
        self._process_stop.set()
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            pass

        process = self._process
        frame_queue = self._frame_queue

        def reap() -> None:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            frame_queue.close()
            frame_queue.cancel_join_thread()

        threading.Thread(target=reap, name="monitor-video-reaper", daemon=True).start()
