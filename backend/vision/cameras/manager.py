from __future__ import annotations

import base64
import logging
import threading
import time
from dataclasses import asdict
from typing import Any

try:
    import cv2
except Exception:
    cv2 = None

from .base import BaseCameraAdapter, CameraBackendError
from .models import CameraDeviceInfo, CameraDiscoveryResult, CameraStatus, CameraTestResult, FrameData
from .registry import backend_sort_key, default_registry, device_dedupe_key


class CameraManager:
    def __init__(self, config: Any | None = None, logger=None) -> None:
        self.config = config
        self._log = logger or (lambda _msg: None)
        self.registry = default_registry(config=config, logger=self._log)
        self._lock = threading.RLock()
        self._devices: dict[str, CameraDeviceInfo] = {}
        self._selected_device: CameraDeviceInfo | None = None
        self._adapter: BaseCameraAdapter | None = None
        self._latest_frame = FrameData(None, 0, 0.0)
        self._camera_connected = False
        self._camera_streaming = False
        self._consecutive_frame_failures = 0
        self._dropped_frame_count = 0
        self._last_error = ""
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_discovery_result = CameraDiscoveryResult()

    def discover_all_result(self) -> CameraDiscoveryResult:
        self.close_selected()
        result = self.registry.discover_from_all_adapters_with_status()
        raw_devices = list(result.raw_devices)
        preferred = list(getattr(self.config, "preferred_backend_order", []) or [])
        grouped: dict[str, list[CameraDeviceInfo]] = {}
        for device in raw_devices:
            grouped.setdefault(device_dedupe_key(device), []).append(device)

        merged: dict[str, CameraDeviceInfo] = {}
        for key, candidates in grouped.items():
            candidates.sort(key=lambda item: backend_sort_key(item, preferred))
            selected = candidates[0]
            selected.available_backends = [item.backend_name for item in candidates]
            selected.selected_backend = selected.backend_name
            if len(candidates) > 1:
                self._log(
                    "[CAMERA][DEVICE][DEDUPLICATED] "
                    f"key={key} backends={','.join(selected.available_backends)} selected={selected.selected_backend}"
                )
            merged[selected.unique_id] = selected
        with self._lock:
            self._devices = merged
            result.raw_devices = raw_devices
            result.deduplicated_devices = list(merged.values())
            result.devices = list(merged.values())
            result.raw_device_count = len(raw_devices)
            result.final_device_count = len(merged)
            self._last_discovery_result = result
        self._log(f"[CAMERA][DISCOVERY] raw_count={len(raw_devices)}")
        self._log(f"[CAMERA][DISCOVERY] final_count={len(merged)}")
        return result

    def discover_all(self) -> list[CameraDeviceInfo]:
        return self.discover_all_result().devices

    def refresh_devices(self) -> list[CameraDeviceInfo]:
        return self.discover_all()

    def refresh_devices_result(self) -> CameraDiscoveryResult:
        return self.discover_all_result()

    def get_last_discovery_result(self) -> CameraDiscoveryResult:
        with self._lock:
            return self._last_discovery_result

    def get_devices(self) -> list[CameraDeviceInfo]:
        with self._lock:
            return list(self._devices.values())

    def get_device(self, unique_id: str) -> CameraDeviceInfo | None:
        with self._lock:
            return self._devices.get(unique_id)

    def select_device(self, unique_id: str, backend_name: str | None = None) -> CameraDeviceInfo:
        device = self.get_device(unique_id)
        if device is None:
            self.refresh_devices()
            device = self.get_device(unique_id)
        if device is None:
            raise CameraBackendError(f"未找到相机设备: {unique_id}")
        if backend_name and backend_name not in device.available_backends:
            raise CameraBackendError(f"设备不支持所选后端: {backend_name}")
        if backend_name:
            device.selected_backend = backend_name
            device.backend_name = backend_name
        with self._lock:
            self._selected_device = device
        self._log(
            "[CAMERA][SELECT] "
            f"vendor={device.manufacturer} model={device.model} sn={device.serial_number} "
            f"transport={device.transport_type} backend={device.selected_backend}"
        )
        return device

    def _build_adapter(self, device: CameraDeviceInfo) -> BaseCameraAdapter:
        adapter_cls = self.registry.get_adapter_for_device(device, device.selected_backend)
        return adapter_cls(config=self.config, logger=self._log)

    def test_device(self, camera_config: dict[str, Any] | None = None) -> CameraTestResult:
        device = self._require_selected()
        adapter = self._build_adapter(device)
        frames: list[FrameData] = []
        try:
            self._log("[CAMERA][TEST] 请关闭厂商官方相机软件及其预览窗口，避免设备被独占。")
            adapter.open(device)
            caps = adapter.get_capabilities()
            parameter_result = self._configure_adapter(adapter, camera_config or {})
            _apply_continuous_mode(adapter, caps)
            adapter.start_stream()
            deadline = time.time() + 6.0
            required = int(getattr(self.config, "test_frame_count", 3) or 3)
            seen: set[int] = set()
            while time.time() < deadline and len(frames) < required:
                frame = adapter.read_frame(int(getattr(self.config, "frame_timeout_ms", 1000) or 1000))
                if frame.valid and frame.image is not None and frame.frame_id not in seen:
                    if frame.width > 0 and frame.height > 0 and int(frame.image.size) > 0:
                        frames.append(frame)
                        seen.add(frame.frame_id)
                time.sleep(0.02)
            if len(frames) < required:
                raise CameraBackendError(f"测试取帧失败：需要 {required} 个有效帧，实际 {len(frames)} 个")
            preview = _preview_png(frames[-1])
            self._log(
                "[CAMERA][TEST][OK] "
                f"vendor={device.manufacturer} model={device.model} sn={device.serial_number} backend={device.selected_backend}"
            )
            return CameraTestResult(
                ok=True,
                message="测试取帧成功",
                unique_id=device.unique_id,
                backend_name=device.selected_backend,
                frames_read=len(frames),
                width=frames[-1].width,
                height=frames[-1].height,
                pixel_format=frames[-1].pixel_format,
                preview_png_base64=preview,
                device_info=device.to_dict(),
                capabilities=caps.to_dict(),
                applied_parameters=parameter_result["applied"],
                parameter_readback=parameter_result["readback"],
            )
        except Exception as exc:
            logging.exception("camera test failed")
            self._log(
                "[CAMERA][TEST][FAIL] "
                f"vendor={device.manufacturer} model={device.model} sn={device.serial_number} "
                f"backend={device.selected_backend} error={exc}"
            )
            return CameraTestResult(False, error=str(exc), unique_id=device.unique_id, backend_name=device.selected_backend)
        finally:
            try:
                adapter.stop_stream()
            except Exception:
                pass
            try:
                adapter.close()
            except Exception:
                pass

    def open_selected(self) -> None:
        device = self._require_selected()
        adapter = self._build_adapter(device)
        try:
            adapter.open(device)
            with self._lock:
                old = self._adapter
                self._adapter = adapter
                self._camera_connected = True
                self._last_error = ""
            if old is not None and old is not adapter:
                old.close()
            self._log(
                "[CAMERA][OPEN][OK] "
                f"vendor={device.manufacturer} model={device.model} sn={device.serial_number} backend={device.selected_backend}"
            )
        except CameraBackendError as exc:
            self._last_error = str(exc)
            self._log(f"[CAMERA][OPEN][FAIL] backend={device.selected_backend} error={exc}")
            raise
        except Exception as exc:
            logging.exception("camera open failed")
            self._last_error = str(exc)
            self._log(f"[CAMERA][OPEN][FAIL] backend={device.selected_backend} error={exc}")
            raise

    def configure_selected(self, camera_config: dict[str, Any] | None = None) -> dict[str, Any]:
        adapter = self._require_adapter()
        return self._configure_adapter(adapter, camera_config or {})

    def _configure_adapter(
        self,
        adapter: BaseCameraAdapter,
        camera_config: dict[str, Any],
    ) -> dict[str, Any]:
        caps = adapter.get_capabilities()
        applied: dict[str, Any] = {}
        readback: dict[str, Any] = {}
        skipped: dict[str, str] = {}
        for name, value in camera_config.items():
            cap = getattr(caps, name, None)
            if cap is None or not cap.supported:
                skipped[name] = "设备不支持"
                continue
            if cap.writable is False:
                skipped[name] = "参数只读"
                continue
            if value in (None, ""):
                continue
            if cap.min_value is not None and float(value) < float(cap.min_value):
                raise CameraBackendError(f"{name} 超出允许范围: {cap.min_value} - {cap.max_value}")
            if cap.max_value is not None and float(value) > float(cap.max_value):
                raise CameraBackendError(f"{name} 超出允许范围: {cap.min_value} - {cap.max_value}")
            adapter.set_feature(name, value)
            applied[name] = value
            if cap.readable:
                actual = adapter.get_feature(name)
                readback[name] = actual
            self._log(
                f"[CAMERA][PARAMETER][SET] backend={adapter.backend_name} "
                f"name={name} requested={value} readback={readback.get(name, 'unavailable')}"
            )
        return {"applied": applied, "readback": readback, "skipped": skipped}

    def start_stream(self) -> None:
        adapter = self._require_adapter()
        _apply_continuous_mode(adapter, adapter.get_capabilities())
        adapter.start_stream()
        self._stop_event.clear()
        with self._lock:
            self._camera_streaming = True
            self._camera_connected = True
            self._consecutive_frame_failures = 0
            self._dropped_frame_count = 0
            self._worker = threading.Thread(target=self._grab_loop, name="camera-manager-grab", daemon=True)
            self._worker.start()

    def stop_stream(self) -> None:
        self._stop_event.set()
        worker = self._worker
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=1.5)
        with self._lock:
            adapter = self._adapter
            self._worker = None
            self._camera_streaming = False
        if adapter is not None:
            adapter.stop_stream()

    def get_latest_frame(self) -> FrameData:
        with self._lock:
            if not self._camera_connected:
                return FrameData(None, self._latest_frame.frame_id, time.time(), valid=False, error=self._last_error or "相机连接中断")
            return self._latest_frame

    def get_status(self) -> CameraStatus:
        with self._lock:
            frame = self._latest_frame
            device = self._selected_device
            adapter = self._adapter
            return CameraStatus(
                selected_unique_id=device.unique_id if device else "",
                selected_backend=device.selected_backend if device else "",
                camera_connected=self._camera_connected,
                camera_streaming=self._camera_streaming,
                is_open=bool(adapter and adapter.is_open()),
                latest_frame_timestamp=frame.timestamp,
                frame_id=frame.frame_id,
                consecutive_frame_failures=self._consecutive_frame_failures,
                dropped_frame_count=self._dropped_frame_count,
                last_error=self._last_error,
                width=frame.width,
                height=frame.height,
                pixel_format=frame.pixel_format,
            )

    def close_selected(self) -> None:
        self.stop_stream()
        with self._lock:
            adapter = self._adapter
            self._adapter = None
            self._camera_connected = False
            self._camera_streaming = False
            self._latest_frame = FrameData(None, 0, 0.0)
        if adapter is not None:
            adapter.close()

    def _grab_loop(self) -> None:
        adapter = self._require_adapter()
        timeout = int(getattr(self.config, "frame_timeout_ms", 1000) or 1000)
        threshold = int(getattr(self.config, "frame_failure_threshold", 10) or 10)
        while not self._stop_event.is_set():
            frame = adapter.read_frame(timeout)
            with self._lock:
                if frame.valid and frame.image is not None:
                    if frame.frame_id == self._latest_frame.frame_id:
                        self._dropped_frame_count += 1
                    self._latest_frame = frame
                    self._camera_connected = True
                    self._camera_streaming = True
                    self._consecutive_frame_failures = 0
                    self._last_error = ""
                else:
                    self._consecutive_frame_failures += 1
                    self._last_error = frame.error or adapter.get_last_error() or "相机取帧失败"
                    self._log(f"[CAMERA][FRAME][FAIL] backend={adapter.backend_name} error={self._last_error}")
                    if self._consecutive_frame_failures >= threshold:
                        self._camera_connected = False
                        self._camera_streaming = False
                        self._latest_frame = FrameData(None, self._latest_frame.frame_id, time.time(), valid=False, error=self._last_error)
                        self._stop_event.set()
                        self._log(f"[CAMERA][DISCONNECTED] backend={adapter.backend_name} error={self._last_error}")
            time.sleep(0.001)

    def _require_selected(self) -> CameraDeviceInfo:
        with self._lock:
            if self._selected_device is None:
                raise CameraBackendError("尚未选择相机设备")
            return self._selected_device

    def _require_adapter(self) -> BaseCameraAdapter:
        with self._lock:
            if self._adapter is None:
                raise CameraBackendError("相机尚未打开")
            return self._adapter


def _apply_continuous_mode(adapter: BaseCameraAdapter, caps) -> None:
    if caps.acquisition_mode.supported and caps.acquisition_mode.writable:
        adapter.set_feature("acquisition_mode", "Continuous")
    if caps.trigger_mode.supported and caps.trigger_mode.writable:
        adapter.set_feature("trigger_mode", "Off")


def _preview_png(frame: FrameData) -> str | None:
    if frame.image is None or cv2 is None:
        return None
    ok, buf = cv2.imencode(".png", frame.image)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")
