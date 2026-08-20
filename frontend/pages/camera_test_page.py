from __future__ import annotations

import base64
from collections import deque
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2

from backend.vision.service import VisionCameraService


class CameraTestPage(ttk.Frame):
    """Raw camera preview with no recognition, PID, or pump activity."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._camera = VisionCameraService(logger=app.runtime_logger)
        self._devices: list[dict[str, object]] = []
        self._preview_stop = threading.Event()
        self._preview_worker: threading.Thread | None = None
        self._frame_lock = threading.Lock()
        self._latest_frame: tuple[int, str] | None = None
        self._last_frame_id = 0
        self._photo = None
        self._poll_job = None
        self._display_times: deque[float] = deque(maxlen=120)
        self._visible = False
        self._start_generation = 0

        self.device_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="等待发现相机")
        self.fps_var = tk.StringVar(value="显示帧率: 0.0 FPS")
        self._build()

    def _build(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="x", padx=18, pady=(14, 8))
        ttk.Button(top, text="返回参数页", command=lambda: self.app.show_page("parameter")).pack(side="left")
        ttk.Label(top, text="纯相机画面测试", font=("Microsoft YaHei UI", 15, "bold")).pack(side="left", padx=16)

        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=18, pady=6)
        self.device_combo = ttk.Combobox(controls, textvariable=self.device_var, state="readonly", width=68)
        self.device_combo.pack(side="left", padx=(0, 6))
        self.discover_button = ttk.Button(controls, text="发现相机", command=self._discover)
        self.discover_button.pack(side="left", padx=3)
        self.start_button = ttk.Button(controls, text="启动纯净预览", command=self._start, state="disabled")
        self.start_button.pack(side="left", padx=3)
        self.stop_button = ttk.Button(controls, text="停止", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=3)

        info = ttk.Frame(self)
        info.pack(fill="x", padx=18, pady=4)
        ttk.Label(info, textvariable=self.status_var).pack(side="left")
        ttk.Label(info, textvariable=self.fps_var).pack(side="right")

        frame = ttk.LabelFrame(self, text="相机原始画面（无识别、无辅助线、无PID）")
        frame.pack(fill="both", expand=True, padx=18, pady=(4, 18))
        self.video_label = ttk.Label(frame, text="点击“发现相机”开始", anchor="center")
        self.video_label.pack(fill="both", expand=True, padx=6, pady=6)

    def on_show(self) -> None:
        self._visible = True
        if self._poll_job is None:
            self._poll_once()

    def on_hide(self) -> None:
        self._visible = False
        self._start_generation += 1
        self._stop()
        if self._poll_job is not None:
            self.after_cancel(self._poll_job)
            self._poll_job = None

    def _discover(self) -> None:
        self.discover_button.configure(state="disabled")
        self.status_var.set("正在发现相机…")
        found: list[dict[str, object]] = []

        def task() -> None:
            found.extend(self._camera.discover_cameras())

        def done() -> None:
            self.discover_button.configure(state="normal")
            self._devices = found
            labels = [
                f"{item.get('manufacturer', '')} {item.get('model', '')} | "
                f"{item.get('selected_backend') or item.get('backend_name', '')} | {item.get('unique_id', '')}"
                for item in found
            ]
            self.device_combo.configure(values=labels)
            if labels:
                self.device_combo.current(0)
                self.start_button.configure(state="normal")
                self.status_var.set(f"发现 {len(labels)} 台相机")
            else:
                self.status_var.set("未发现相机")

        def failed(exc: Exception) -> None:
            self.discover_button.configure(state="normal")
            self.status_var.set("相机发现失败")
            messagebox.showerror("相机发现失败", str(exc))

        self.app.run_backend_task(task, on_success=done, on_error=failed)

    def _selected_device(self) -> dict[str, object]:
        index = self.device_combo.current()
        if index < 0 or index >= len(self._devices):
            raise ValueError("请先选择相机")
        return self._devices[index]

    def _start(self) -> None:
        try:
            device = self._selected_device()
        except Exception as exc:
            messagebox.showerror("无法启动", str(exc))
            return
        self.start_button.configure(state="disabled")
        self.discover_button.configure(state="disabled")
        self.status_var.set("正在打开相机…")
        self._start_generation += 1
        generation = self._start_generation

        def task() -> None:
            unique_id = str(device.get("unique_id", ""))
            backend = str(device.get("selected_backend") or device.get("backend_name") or "")
            self._camera.select_camera(unique_id, backend or None)
            self._camera.open_camera()
            self._camera.configure_camera({"frame_rate": 100.0, "width": 720, "height": 540})
            self._camera.start_camera_stream()

        def done() -> None:
            if not self._visible or generation != self._start_generation:
                self._close_camera_async()
                return
            self.stop_button.configure(state="normal")
            self.status_var.set("纯净预览运行中；相机请求100 FPS，页面目标30 FPS")
            self._preview_stop = threading.Event()
            self._preview_worker = threading.Thread(
                target=self._preview_loop,
                args=(self._preview_stop,),
                name="camera-test-preview-loop",
                daemon=True,
            )
            self._preview_worker.start()

        def failed(exc: Exception) -> None:
            self.start_button.configure(state="normal")
            self.discover_button.configure(state="normal")
            self.status_var.set("相机启动失败")
            messagebox.showerror("相机启动失败", str(exc))

        self.app.run_backend_task(task, on_success=done, on_error=failed)

    def _preview_loop(self, stop_event: threading.Event) -> None:
        last_id = 0
        next_frame = time.perf_counter()
        while not stop_event.is_set():
            packet = self._camera.get_latest_frame()
            if packet.valid and packet.image is not None and int(packet.frame_id) != last_id:
                now = time.perf_counter()
                if now >= next_frame:
                    image = packet.image
                    height, width = image.shape[:2]
                    scale = min(1.0, 640.0 / width, 480.0 / height)
                    if scale < 0.999:
                        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
                    ok, encoded = cv2.imencode(".png", image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                    if ok:
                        payload = base64.b64encode(encoded.tobytes()).decode("ascii")
                        with self._frame_lock:
                            self._latest_frame = (int(packet.frame_id), payload)
                    next_frame = now + 1.0 / 30.0
                last_id = int(packet.frame_id)
            stop_event.wait(0.001)

    def _poll_once(self) -> None:
        with self._frame_lock:
            latest = self._latest_frame
            self._latest_frame = None
        if latest is not None and latest[0] != self._last_frame_id:
            try:
                self._photo = tk.PhotoImage(data=latest[1])
                self.video_label.configure(image=self._photo, text="")
                self._last_frame_id = latest[0]
                now = time.monotonic()
                self._display_times.append(now)
                while self._display_times and now - self._display_times[0] > 1.0:
                    self._display_times.popleft()
                self.fps_var.set(f"显示帧率: {len(self._display_times):.1f} FPS")
            except Exception as exc:
                self.status_var.set(f"画面显示失败: {exc}")
        self._poll_job = self.after(15, self._poll_once)

    def _stop(self) -> None:
        self._preview_stop.set()
        self._preview_worker = None
        self._latest_frame = None
        self._display_times.clear()
        self.fps_var.set("显示帧率: 0.0 FPS")
        self.stop_button.configure(state="disabled")
        self.start_button.configure(state="normal" if self._devices else "disabled")
        self.discover_button.configure(state="normal")

        self._close_camera_async()
        self.status_var.set("纯净预览已停止")

    def _close_camera_async(self) -> None:
        def close_camera() -> None:
            try:
                self._camera.stop_camera_stream()
                self._camera.close_camera()
            except Exception:
                pass

        threading.Thread(target=close_camera, name="camera-test-close", daemon=True).start()
