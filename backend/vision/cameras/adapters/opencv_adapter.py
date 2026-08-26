from __future__ import annotations

import time
from typing import Any

from ..base import BaseCameraAdapter, CameraBackendError
from ..models import (
    CameraBackendStatus,
    CameraCapabilities,
    CameraDeviceInfo,
    CameraFeatureCapability,
    DEVICE_TYPE_USB,
    FrameData,
    TRANSPORT_UVC,
)


class OpenCVCameraAdapter(BaseCameraAdapter):
    backend_name = "opencv"
    supported_manufacturers = ("OpenCV", "UVC")
    backend_priority = 100

    def __init__(self, config: Any | None = None, logger=None) -> None:
        super().__init__(config=config, logger=logger)
        self._cv2 = _import_cv2()
        self._cap = None
        self._device: CameraDeviceInfo | None = None
        self._latest = FrameData(image=None, frame_id=0, timestamp=0.0, source_backend=self.backend_name)
        self._frame_id = 0
        self._streaming = False

    @classmethod
    def is_backend_available(cls, config: Any | None = None) -> tuple[bool, str]:
        try:
            _import_cv2()
            return True, "OpenCV available"
        except Exception as exc:
            return False, f"OpenCV/cv2 not available: {exc}"

    @classmethod
    def check_backend(cls, config: Any | None = None) -> CameraBackendStatus:
        ok, reason = cls.is_backend_available(config)
        return CameraBackendStatus(
            backend_name=cls.backend_name,
            display_name="OpenCV",
            backend_available=ok,
            sdk_loaded=ok,
            python_module_loaded=ok,
            error="" if ok else reason,
        )

    @classmethod
    def discover_devices(cls, config: Any | None = None, logger=None) -> list[CameraDeviceInfo]:
        cv2 = _import_cv2()
        devices: list[CameraDeviceInfo] = []
        indices = tuple(getattr(config, "opencv_scan_indices", tuple(range(4))) or tuple(range(4)))
        indices = tuple(idx for idx in indices if 0 <= int(idx) <= 3)
        backend_order = (
            ("CAP_DSHOW", getattr(cv2, "CAP_DSHOW", 700)),
            ("CAP_MSMF", getattr(cv2, "CAP_MSMF", 1400)),
            ("CAP_ANY", getattr(cv2, "CAP_ANY", 0)),
        )

        for index in indices:
            found = False
            for backend_name, api in backend_order:
                cap = cv2.VideoCapture(int(index), api)
                try:
                    if not cap.isOpened():
                        if logger:
                            logger(f"[CAMERA][OPENCV] index={index} backend={backend_name} unavailable")
                        continue
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        if logger:
                            logger(f"[CAMERA][OPENCV] index={index} backend={backend_name} no_valid_frame")
                        continue
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or frame.shape[1])
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame.shape[0])
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    unique_id = f"opencv:uvc:index:{index}"
                    caps = CameraCapabilities(
                        width=CameraFeatureCapability(True, True, True, width),
                        height=CameraFeatureCapability(True, True, True, height),
                        frame_rate=CameraFeatureCapability(True, True, True, fps),
                        pixel_format=CameraFeatureCapability(True, True, False, "BGR8", available_values=["BGR8"]),
                    )
                    devices.append(
                        CameraDeviceInfo(
                            device_id=f"opencv:{index}",
                            unique_id=unique_id,
                            backend_name=cls.backend_name,
                            manufacturer="OpenCV",
                            model=f"Camera Index {index}",
                            user_defined_name=f"Camera Index {index}",
                            device_type=DEVICE_TYPE_USB,
                            transport_type=TRANSPORT_UVC,
                            usb_path=f"index:{index}",
                            available=True,
                            status="available",
                            capabilities=caps,
                            available_backends=[cls.backend_name],
                            selected_backend=cls.backend_name,
                            backend_priority=cls.backend_priority,
                        )
                    )
                    found = True
                    if logger:
                        logger(f"[CAMERA][DISCOVERY] backend=opencv index={index} api={backend_name} valid_frame=True")
                    break
                finally:
                    cap.release()
            if not found and logger:
                logger(f"[CAMERA][OPENCV] index={index} unavailable")
        return devices

    def _index_from_device(self, device_info: CameraDeviceInfo) -> int:
        text = device_info.device_id.split(":", 1)[1] if ":" in device_info.device_id else device_info.device_id
        return int(float(text))

    def open(self, device_info: CameraDeviceInfo) -> None:
        self.close()
        index = self._index_from_device(device_info)
        cap = self._cv2.VideoCapture(index, getattr(self._cv2, "CAP_DSHOW", 700))
        if not cap.isOpened():
            cap.release()
            cap = self._cv2.VideoCapture(index, getattr(self._cv2, "CAP_ANY", 0))
        if not cap.isOpened():
            self._last_error = f"Cannot open UVC camera index {index}"
            raise CameraBackendError(self._last_error)
        self._cap = cap
        self._device = device_info
        self._last_error = ""

    def close(self) -> None:
        self.stop_stream()
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._device = None

    def start_stream(self) -> None:
        if self._cap is None:
            raise CameraBackendError("OpenCV camera is not open")
        self._streaming = True

    def stop_stream(self) -> None:
        self._streaming = False

    def read_frame(self, timeout_ms: int = 1000) -> FrameData:
        if self._cap is None or not self._streaming:
            return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error="OpenCV camera is not streaming")
        ok, frame = self._cap.read()
        host_monotonic = time.monotonic()
        now = time.time()
        if not ok or frame is None:
            self._last_error = "OpenCV camera read failed"
            return FrameData(None, self._frame_id, now, source_backend=self.backend_name, valid=False, error=self._last_error)
        self._frame_id += 1
        self._latest = FrameData(
            image=frame,
            frame_id=self._frame_id,
            timestamp=now,
            width=int(frame.shape[1]),
            height=int(frame.shape[0]),
            pixel_format="BGR8",
            source_backend=self.backend_name,
            device_unique_id=self._device.unique_id if self._device else "",
            valid=True,
            host_monotonic_timestamp=host_monotonic,
        )
        return self._latest

    def get_latest_frame(self) -> FrameData:
        return self._latest

    def get_device_info(self) -> CameraDeviceInfo | None:
        return self._device

    def get_capabilities(self) -> CameraCapabilities:
        if self._cap is None:
            return CameraCapabilities()
        return CameraCapabilities(
            width=CameraFeatureCapability(True, True, True, int(self._cap.get(self._cv2.CAP_PROP_FRAME_WIDTH) or 0)),
            height=CameraFeatureCapability(True, True, True, int(self._cap.get(self._cv2.CAP_PROP_FRAME_HEIGHT) or 0)),
            frame_rate=CameraFeatureCapability(True, True, True, float(self._cap.get(self._cv2.CAP_PROP_FPS) or 0.0)),
            pixel_format=CameraFeatureCapability(True, True, False, "BGR8", available_values=["BGR8"]),
        )

    def get_feature(self, name: str) -> Any:
        if self._cap is None:
            return None
        prop = {
            "width": self._cv2.CAP_PROP_FRAME_WIDTH,
            "height": self._cv2.CAP_PROP_FRAME_HEIGHT,
            "frame_rate": self._cv2.CAP_PROP_FPS,
            "exposure": self._cv2.CAP_PROP_EXPOSURE,
            "gain": self._cv2.CAP_PROP_GAIN,
        }.get(name)
        return self._cap.get(prop) if prop is not None else None

    def set_feature(self, name: str, value: Any) -> None:
        if self._cap is None:
            return
        prop = {
            "width": self._cv2.CAP_PROP_FRAME_WIDTH,
            "height": self._cv2.CAP_PROP_FRAME_HEIGHT,
            "frame_rate": self._cv2.CAP_PROP_FPS,
            "exposure": self._cv2.CAP_PROP_EXPOSURE,
            "gain": self._cv2.CAP_PROP_GAIN,
        }.get(name)
        if prop is not None:
            self._cap.set(prop, float(value))

    def is_open(self) -> bool:
        return bool(self._cap is not None and self._cap.isOpened())

    def is_streaming(self) -> bool:
        return bool(self._streaming)


def _import_cv2():
    import cv2

    return cv2

