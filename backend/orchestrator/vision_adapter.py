from __future__ import annotations

import base64
import queue
import threading
import time
from dataclasses import replace
from typing import Any, Callable, Protocol, runtime_checkable

try:
    import cv2
except Exception:  # pragma: no cover - handled at runtime for local video only
    cv2 = None

from .models import RecognitionSnapshot

INDUSTRIAL_CAMERA_BACKENDS = {"hikrobot", "basler", "daheng", "flir", "allied_vision", "gentl"}
PREVIEW_MAX_WIDTH = 760
PREVIEW_MAX_HEIGHT = 560
PREVIEW_TARGET_INTERVAL_S = 1.0 / 30.0
PROCESS_TARGET_INTERVAL_S = 0.08


@runtime_checkable
class VisionAdapterProtocol(Protocol):
    def prepare_video(self, video_source_type: str, video_source: str, pixel_to_micron: float) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_snapshot(self) -> RecognitionSnapshot | dict[str, Any]: ...


class GenericVisionAdapter:
    def __init__(self, vision_service: Any) -> None:
        self.vision_service = vision_service

    def _call(self, names: list[str], *args, **kwargs):
        if self.vision_service is None:
            raise RuntimeError("未注入 vision_service")
        for name in names:
            fn = getattr(self.vision_service, name, None)
            if callable(fn):
                return fn(*args, **kwargs)
        raise AttributeError(f"vision_service 缺少可用接口: {names}")

    def prepare_video(self, video_source_type: str, video_source: str, pixel_to_micron: float) -> None:
        try:
            self._call(
                ["prepare_video", "prepare", "setup"],
                video_source_type=video_source_type,
                video_source=video_source,
                pixel_to_micron=pixel_to_micron,
            )
        except TypeError:
            self._call(["prepare_video", "prepare", "setup"], video_source_type, video_source, pixel_to_micron)

    def start(self) -> None:
        self._call(["start", "start_loop", "run"])

    def stop(self) -> None:
        self._call(["stop", "stop_loop", "shutdown"])

    def get_snapshot(self) -> RecognitionSnapshot | dict[str, Any]:
        return self._call(["get_snapshot", "get_latest_snapshot", "read_snapshot", "pull_result", "run_once"])


class PipelineVisionService:
    def __init__(self, logger: Callable[[str], None] | None = None) -> None:
        from ..vision.service import VisionCameraService

        self._log = logger or (lambda _msg: None)
        self._pipeline = None
        self._camera_service = VisionCameraService(logger=self._log)
        self._video_source_type = "camera"
        self._video_source = "0"
        self._selected_backend = ""
        self._pixel_to_micron = 1.0
        self._lock = threading.RLock()
        self._cap = None
        self._worker: threading.Thread | None = None
        self._process_worker: threading.Thread | None = None
        self._frame_queue: queue.Queue[tuple[int, float, Any]] = queue.Queue(maxsize=3)
        self._stop_event = threading.Event()
        self._last_processed_frame_id = 0
        self._last_processed_frame_timestamp = 0.0
        self._capture_frame_id = 0
        self._last_preview_publish_time = 0.0
        self._last_processing_submit_time = 0.0
        self._latest = self._empty_snapshot("当前无有效液滴通过")

    def _ensure_pipeline(self):
        if self._pipeline is None:
            from ..vision.config import default_config
            from ..vision.pipeline import VisionPipeline

            self._pipeline = VisionPipeline(default_config(), logger=self._log)
        return self._pipeline

    def _is_realtime_mode(self) -> bool:
        mode = self._video_source_type.strip().lower()
        return mode in {
            "camera",
            "realtime",
            "real_time",
            "live",
            "usb",
            "opencv",
            "hikrobot",
            "hikrobot_industrial_camera",
            "industrial_camera",
            "usb_camera",
        }

    def _empty_snapshot(self, reason: str) -> RecognitionSnapshot:
        return RecognitionSnapshot(
            frame_droplet_count=0,
            total_droplet_count=0,
            new_crossing_count=0,
            avg_diameter=None,
            single_cell_rate=0.0,
            valid_for_control=False,
            timestamp=time.time(),
            reason=reason,
            droplet_count=0,
            active_droplet_count=0,
            has_droplet=False,
            control_reason=reason,
            frame_png_base64=None,
            frame_width=0,
            frame_height=0,
            video_source_type=self._video_source_type,
            video_source=self._video_source,
            frame_id=0,
            frame_single_cell_count=0,
            frame_diameters=[],
            frame_diameter_sum=0.0,
            frame_avg_diameter=None,
            frame_single_cell_rate=None,
            frame_diameter_std=None,
            frame_diameter_cv=None,
            uniformity_valid=False,
            uniformity_status="当前无液滴",
            uniformity_reason=reason,
        )

    def _snapshot_with_error(self, reason: str) -> RecognitionSnapshot:
        with self._lock:
            current = self._latest
        if current.frame_png_base64:
            return replace(
                current,
                valid_for_control=False,
                timestamp=time.time(),
                reason=reason,
                control_reason=reason,
                uniformity_valid=False,
                uniformity_reason=reason,
            )
        return self._empty_snapshot(reason)

    def set_mvs_sdk_path(self, sdk_path: str) -> None:
        self._camera_service.set_mvs_sdk_path(sdk_path)

    def set_selected_backend(self, backend_name: str) -> None:
        self._selected_backend = str(backend_name or "").strip()

    def discover_cameras_result(self) -> dict[str, Any]:
        return self._camera_service.discover_cameras_result()

    def refresh_cameras_result(self) -> dict[str, Any]:
        return self._camera_service.refresh_cameras_result()

    def get_camera_devices(self) -> list[dict[str, Any]]:
        return self._camera_service.get_camera_devices()

    def select_camera(self, unique_id: str, backend_name: str | None = None) -> dict[str, Any]:
        self._video_source = str(unique_id or "")
        self._selected_backend = str(backend_name or "")
        return self._camera_service.select_camera(unique_id, backend_name)

    def test_camera(self) -> dict[str, Any]:
        return self._camera_service.test_camera()

    def get_camera_status(self) -> dict[str, Any]:
        return self._camera_service.get_camera_status()

    def prepare_video(self, video_source_type: str, video_source: str, pixel_to_micron: float) -> None:
        self.stop()
        with self._lock:
            self._video_source_type = str(video_source_type or "camera")
            self._video_source = str(video_source or "0")
            self._pixel_to_micron = float(pixel_to_micron) if float(pixel_to_micron) > 0 else 1.0
            self._last_processed_frame_id = 0
            self._last_processed_frame_timestamp = 0.0
            self._capture_frame_id = 0
            self._last_preview_publish_time = 0.0
            self._last_processing_submit_time = 0.0
            self._ensure_pipeline().reset()
            self._latest = self._empty_snapshot("视频输入已准备，等待识别")

        if self._is_realtime_mode():
            backend = self._selected_backend or _backend_from_mode(self._video_source_type)
            selected = self._camera_service.select_camera(self._video_source, backend or None)
            _require_industrial_camera(selected)
            self._log(
                "[VISION][CAMERA][SELECTED] "
                f"source=industrial_camera backend={selected.get('selected_backend') or selected.get('backend_name')} "
                f"vendor={selected.get('manufacturer')} model={selected.get('model')} "
                f"serial={selected.get('serial_number')} unique_id={selected.get('unique_id')}"
            )
            self._camera_service.open_camera()
            self._camera_service.start_camera_stream()
            deadline = time.time() + 3.0
            packet = self._camera_service.get_latest_frame()
            while time.time() < deadline and (not packet.valid or packet.image is None):
                time.sleep(0.03)
                packet = self._camera_service.get_latest_frame()
            if not packet.valid or packet.image is None:
                raise RuntimeError(packet.error or "实时相机未产生有效帧")
            self._log(
                "[VISION][CAMERA][FRAME][OK] "
                f"backend={packet.source_backend} frame_id={packet.frame_id} "
                f"width={packet.width} height={packet.height} pixel_format={packet.pixel_format}"
            )
            try:
                snapshot = self._snapshot_from_frame(packet.image)
                with self._lock:
                    self._latest = snapshot
                    self._last_processed_frame_id = int(packet.frame_id or 0)
                    self._last_processed_frame_timestamp = float(packet.timestamp or time.time())
            except Exception as exc:
                self._log(f"[VISION][PREVIEW][WARN] initial frame process failed: {exc}")
            self.start()
            self._log("[VISION][PREVIEW][START] realtime preview loop started")

    def _open_capture(self):
        if cv2 is None:
            raise RuntimeError("OpenCV/cv2 未安装，无法读取本地视频")
        cap = cv2.VideoCapture(self._video_source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {self._video_source}")
        return cap

    def start(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            if not self._is_realtime_mode():
                self._cap = self._open_capture()
            self._frame_queue = queue.Queue(maxsize=3)
            self._stop_event.clear()
            self._worker = threading.Thread(target=self._capture_loop, name="vision-capture-loop", daemon=True)
            self._process_worker = threading.Thread(target=self._process_loop, name="vision-processing-loop", daemon=True)
            self._worker.start()
            self._process_worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        worker = self._worker
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=1.0)
        process_worker = self._process_worker
        if process_worker is not None and process_worker.is_alive() and process_worker is not threading.current_thread():
            process_worker.join(timeout=1.0)
        with self._lock:
            cap = self._cap
            self._cap = None
            self._worker = None
            self._process_worker = None
        if cap is not None:
            cap.release()
        try:
            self._camera_service.stop_camera_stream()
            self._camera_service.close_camera()
        except Exception:
            pass

    def _encode_png_base64(self, frame) -> tuple[str | None, int, int]:
        if cv2 is None:
            return None, 0, 0
        try:
            preview = self._resize_preview_frame(frame)
            ok, buf = cv2.imencode(".png", preview, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
            if not ok:
                return None, int(preview.shape[1]), int(preview.shape[0])
            return base64.b64encode(buf.tobytes()).decode("ascii"), int(preview.shape[1]), int(preview.shape[0])
        except Exception:
            return None, 0, 0

    def _resize_preview_frame(self, frame):
        try:
            height = int(frame.shape[0])
            width = int(frame.shape[1])
        except Exception:
            return frame
        if width <= 0 or height <= 0:
            return frame
        scale = min(1.0, PREVIEW_MAX_WIDTH / float(width), PREVIEW_MAX_HEIGHT / float(height))
        if scale >= 0.999:
            return frame
        target = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(frame, target, interpolation=cv2.INTER_AREA)

    def _read_next_frame(self) -> tuple[bool, Any, str]:
        if self._is_realtime_mode():
            packet = self._camera_service.get_latest_frame()
            if not packet.valid or packet.image is None:
                return False, None, packet.error or "相机取帧异常"
            frame_id = int(packet.frame_id or 0)
            timestamp = float(packet.timestamp or 0.0)
            if (
                frame_id > 0
                and timestamp > 0.0
                and frame_id == self._last_processed_frame_id
                and timestamp <= self._last_processed_frame_timestamp
            ):
                return False, None, ""
            self._last_processed_frame_id = frame_id
            self._last_processed_frame_timestamp = timestamp or time.time()
            return True, packet.image, ""

        with self._lock:
            cap = self._cap
        if cap is None:
            return False, None, "视频源未打开"
        ok, frame = cap.read()
        if not ok:
            return False, None, "本地视频读取结束"
        return True, frame, ""

    def _snapshot_from_frame(
        self,
        frame,
        *,
        frame_png_base64: str | None = None,
        frame_width: int = 0,
        frame_height: int = 0,
        frame_id: int | None = None,
        timestamp: float | None = None,
    ) -> RecognitionSnapshot:
        result = self._ensure_pipeline().process_frame(frame)
        control = result.metrics.control
        avg_px = control.frame_avg_diameter
        active_count = int(result.metrics.control.frame_droplet_count)
        total_count = int(result.metrics.control.total_droplet_count)
        new_cross = int(result.metrics.control.new_crossing_count)
        has_droplet = active_count > 0
        control_reason = str(result.metrics.control.reason or "")
        frame_b64 = frame_png_base64
        width = int(frame_width or 0)
        height = int(frame_height or 0)
        if frame_b64 is None:
            frame_b64, width, height = self._encode_png_base64(result.annotated_frame)
        if frame_b64 is None:
            frame_b64, width, height = self._encode_png_base64(frame)
        scale = float(self._pixel_to_micron)
        frame_diameters = [float(value) * scale for value in control.frame_diameters]
        frame_avg_diameter = (float(avg_px) * scale) if avg_px is not None else None
        frame_diameter_sum = float(control.frame_diameter_sum) * scale
        frame_diameter_std = (
            float(control.frame_diameter_std) * scale
            if control.frame_diameter_std is not None
            else None
        )
        return RecognitionSnapshot(
            frame_droplet_count=active_count,
            total_droplet_count=total_count,
            new_crossing_count=new_cross,
            avg_diameter=frame_avg_diameter,
            single_cell_rate=float(control.frame_single_cell_rate or 0.0),
            valid_for_control=bool(result.metrics.control.valid_for_control and has_droplet),
            timestamp=float(timestamp or time.time()),
            reason=control_reason,
            droplet_count=total_count,
            active_droplet_count=active_count,
            has_droplet=has_droplet,
            control_reason=control_reason,
            frame_png_base64=frame_b64,
            frame_width=width,
            frame_height=height,
            video_source_type=self._video_source_type,
            video_source=self._video_source,
            frame_id=int(frame_id if frame_id is not None else result.frame_index),
            frame_single_cell_count=int(control.frame_single_cell_count),
            frame_diameters=frame_diameters,
            frame_diameter_sum=frame_diameter_sum,
            frame_avg_diameter=frame_avg_diameter,
            frame_single_cell_rate=control.frame_single_cell_rate,
            frame_diameter_std=frame_diameter_std,
            frame_diameter_cv=control.frame_diameter_cv,
            uniformity_valid=bool(control.uniformity_valid),
            uniformity_status=str(control.uniformity_status or ""),
            uniformity_reason=str(control.uniformity_reason or ""),
        )

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            ok, frame, error = self._read_next_frame()
            if not ok:
                if error and self._is_realtime_mode():
                    with self._lock:
                        self._latest = self._snapshot_with_error(error)
                elif error:
                    break
                time.sleep(0.03)
                continue
            try:
                with self._lock:
                    self._capture_frame_id += 1
                    frame_id = self._capture_frame_id
                timestamp = time.time()
                if self._should_publish_preview(timestamp):
                    self._publish_video_frame(frame, frame_id, timestamp)
                if self._should_submit_processing(timestamp):
                    self._submit_processing_frame(frame_id, timestamp, frame)
            except Exception as exc:
                self._log(f"[VISION][WARN] capture frame failed: {exc}")
                time.sleep(0.02)
        with self._lock:
            cap = self._cap
            self._cap = None
            self._worker = None
        if cap is not None:
            cap.release()

    def _process_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame_id, timestamp, frame = self._frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                with self._lock:
                    current = self._latest
                    frame_b64 = current.frame_png_base64
                    width = int(current.frame_width)
                    height = int(current.frame_height)
                snapshot = self._snapshot_from_frame(
                    frame,
                    frame_png_base64=frame_b64,
                    frame_width=width,
                    frame_height=height,
                    frame_id=frame_id,
                    timestamp=timestamp,
                )
                with self._lock:
                    current = self._latest
                    self._latest = replace(
                        snapshot,
                        frame_png_base64=current.frame_png_base64,
                        frame_width=current.frame_width,
                        frame_height=current.frame_height,
                        frame_id=max(int(current.frame_id), int(snapshot.frame_id)),
                        timestamp=max(float(current.timestamp), float(snapshot.timestamp)),
                    )
            except Exception as exc:
                self._log(f"[VISION][WARN] processing frame failed: {exc}")

    def _publish_video_frame(self, frame, frame_id: int, timestamp: float) -> None:
        frame_b64, width, height = self._encode_png_base64(frame)
        if frame_b64 is None:
            return
        with self._lock:
            self._latest = replace(
                self._latest,
                frame_png_base64=frame_b64,
                frame_width=width,
                frame_height=height,
                frame_id=int(frame_id),
                timestamp=float(timestamp),
                video_source_type=self._video_source_type,
                video_source=self._video_source,
            )

    def _submit_processing_frame(self, frame_id: int, timestamp: float, frame) -> None:
        item = (int(frame_id), float(timestamp), frame)
        try:
            self._frame_queue.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            self._frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._frame_queue.put_nowait(item)
        except queue.Full:
            pass

    def _should_publish_preview(self, timestamp: float) -> bool:
        with self._lock:
            last = float(self._last_preview_publish_time or 0.0)
            if float(timestamp) - last < PREVIEW_TARGET_INTERVAL_S:
                return False
            self._last_preview_publish_time = float(timestamp)
            return True

    def _should_submit_processing(self, timestamp: float) -> bool:
        with self._lock:
            last = float(self._last_processing_submit_time or 0.0)
            if float(timestamp) - last < PROCESS_TARGET_INTERVAL_S:
                return False
            self._last_processing_submit_time = float(timestamp)
            return True

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            ok, frame, error = self._read_next_frame()
            if not ok:
                if error and self._is_realtime_mode():
                    with self._lock:
                        self._latest = self._snapshot_with_error(error)
                elif error:
                    break
                time.sleep(0.03)
                continue
            try:
                snapshot = self._snapshot_from_frame(frame)
                with self._lock:
                    self._latest = snapshot
            except Exception as exc:
                self._log(f"[VISION][WARN] 帧处理失败: {exc}")
                time.sleep(0.02)
        with self._lock:
            cap = self._cap
            self._cap = None
            self._worker = None
        if cap is not None:
            cap.release()

    def get_snapshot(self) -> RecognitionSnapshot:
        with self._lock:
            return RecognitionSnapshot(
                frame_droplet_count=self._latest.frame_droplet_count,
                total_droplet_count=self._latest.total_droplet_count,
                new_crossing_count=self._latest.new_crossing_count,
                avg_diameter=self._latest.avg_diameter,
                single_cell_rate=self._latest.single_cell_rate,
                valid_for_control=self._latest.valid_for_control,
                timestamp=self._latest.timestamp,
                reason=self._latest.reason,
                droplet_count=self._latest.droplet_count,
                active_droplet_count=self._latest.active_droplet_count,
                has_droplet=self._latest.has_droplet,
                control_reason=self._latest.control_reason,
                frame_png_base64=self._latest.frame_png_base64,
                frame_width=self._latest.frame_width,
                frame_height=self._latest.frame_height,
                video_source_type=self._latest.video_source_type,
                video_source=self._latest.video_source,
                frame_id=self._latest.frame_id,
                frame_single_cell_count=self._latest.frame_single_cell_count,
                frame_diameters=list(self._latest.frame_diameters),
                frame_diameter_sum=self._latest.frame_diameter_sum,
                frame_avg_diameter=self._latest.frame_avg_diameter,
                frame_single_cell_rate=self._latest.frame_single_cell_rate,
                frame_diameter_std=self._latest.frame_diameter_std,
                frame_diameter_cv=self._latest.frame_diameter_cv,
                uniformity_valid=self._latest.uniformity_valid,
                uniformity_status=self._latest.uniformity_status,
                uniformity_reason=self._latest.uniformity_reason,
            )

    def run_once(self) -> RecognitionSnapshot:
        return self.get_snapshot()


def _backend_from_mode(mode: str) -> str:
    value = str(mode or "").strip().lower()
    aliases = {"alliedvision": "allied_vision"}
    value = aliases.get(value, value)
    return value if value in {"hikrobot", "basler", "daheng", "flir", "allied_vision", "gentl", "opencv"} else ""


def _require_industrial_camera(device: dict[str, Any]) -> None:
    backend = str(device.get("selected_backend", "") or device.get("backend_name", "") or "").strip().lower()
    device_type = str(device.get("device_type", "") or "").strip().lower()
    if backend in INDUSTRIAL_CAMERA_BACKENDS and device_type == "industrial_camera":
        return
    raise RuntimeError(
        "Realtime video must use an industrial camera. "
        f"Selected backend={backend or '--'}, device_type={device_type or '--'}."
    )
