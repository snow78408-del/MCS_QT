from __future__ import annotations

import time
from typing import Any

from ..base import BaseCameraAdapter, CameraBackendError
from ..models import CameraCapabilities, CameraDeviceInfo, CameraFeatureCapability, DEVICE_TYPE_INDUSTRIAL, FrameData


class DahengCameraAdapter(BaseCameraAdapter):
    backend_name = "daheng"
    supported_manufacturers = ("Daheng Imaging", "Daheng", "大恒图像")
    backend_priority = 12

    def __init__(self, config: Any | None = None, logger=None) -> None:
        super().__init__(config=config, logger=logger)
        self._manager = None
        self._camera = None
        self._device: CameraDeviceInfo | None = None
        self._latest = FrameData(None, 0, 0.0, source_backend=self.backend_name)
        self._frame_id = 0
        self._streaming = False

    @classmethod
    def is_backend_available(cls, config: Any | None = None) -> tuple[bool, str]:
        try:
            import gxipy  # noqa: F401
        except Exception as exc:
            return False, f"Galaxy SDK/gxipy 不可用: {exc}"
        return True, "Galaxy SDK 可用"

    @classmethod
    def discover_devices(cls, config: Any | None = None, logger=None) -> list[CameraDeviceInfo]:
        import gxipy as gx

        manager = gx.DeviceManager()
        _, infos = manager.update_device_list()
        devices: list[CameraDeviceInfo] = []
        for info in infos:
            manufacturer = str(info.get("vendor_name", "") or "Daheng Imaging")
            model = str(info.get("model_name", "") or "")
            serial = str(info.get("sn", "") or "")
            transport = str(info.get("device_class", "") or "Unknown")
            unique = f"{manufacturer}:{model}:{serial}" if serial else f"daheng:{info.get('device_id', model)}"
            devices.append(
                CameraDeviceInfo(
                    device_id=f"daheng:{serial or info.get('device_id', '')}",
                    unique_id=unique,
                    backend_name=cls.backend_name,
                    manufacturer=manufacturer,
                    model=model,
                    serial_number=serial,
                    user_defined_name=str(info.get("user_id", "") or ""),
                    device_type=DEVICE_TYPE_INDUSTRIAL,
                    transport_type=_transport(transport),
                    ip_address=str(info.get("ip", "") or ""),
                    available=True,
                    capabilities=_empty_caps(),
                    available_backends=[cls.backend_name],
                    selected_backend=cls.backend_name,
                    backend_priority=cls.backend_priority,
                    raw_info=info,
                )
            )
        return devices

    def open(self, device_info: CameraDeviceInfo) -> None:
        import gxipy as gx

        self.close()
        self._manager = gx.DeviceManager()
        self._manager.update_device_list()
        if device_info.serial_number:
            self._camera = self._manager.open_device_by_sn(device_info.serial_number)
        else:
            self._camera = self._manager.open_device_by_index(1)
        self._device = device_info
        self._device.capabilities = self.get_capabilities()

    def close(self) -> None:
        self.stop_stream()
        if self._camera is not None:
            try:
                self._camera.close_device()
            except Exception:
                pass
        self._camera = None
        self._device = None
        self._log("[CAMERA][CLOSE] backend=daheng")

    def start_stream(self) -> None:
        if self._camera is None:
            raise CameraBackendError("大恒相机未打开")
        self._camera.stream_on()
        self._streaming = True
        self._log("[CAMERA][STREAM][START] backend=daheng")

    def stop_stream(self) -> None:
        if self._camera is not None and self._streaming:
            try:
                self._camera.stream_off()
            except Exception:
                pass
        self._streaming = False

    def read_frame(self, timeout_ms: int = 1000) -> FrameData:
        if self._camera is None or not self._streaming:
            return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error="大恒相机未开始采集")
        try:
            raw = self._camera.data_stream[0].get_image(timeout_ms)
            if raw is None:
                return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error="大恒相机取帧超时")
            image = raw.convert("RGB").get_numpy_array()
            if image is None:
                return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error="大恒图像转换失败")
            image = image[:, :, ::-1].copy()
            self._frame_id += 1
            self._latest = FrameData(
                image=image,
                frame_id=self._frame_id,
                timestamp=time.time(),
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                pixel_format="BGR8",
                source_backend=self.backend_name,
                device_unique_id=self._device.unique_id if self._device else "",
                valid=True,
            )
            return self._latest
        except Exception as exc:
            self._last_error = str(exc)
            return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error=str(exc))

    def get_latest_frame(self) -> FrameData:
        return self._latest

    def get_device_info(self) -> CameraDeviceInfo | None:
        return self._device

    def get_capabilities(self) -> CameraCapabilities:
        return _caps_for_camera(self._camera) if self._camera is not None else _empty_caps()

    def get_feature(self, name: str) -> Any:
        feature = _feature_node(self._camera, name)
        if feature is None or not _feature_readable(feature):
            return None
        return feature.get()

    def set_feature(self, name: str, value: Any) -> None:
        feature = _feature_node(self._camera, name)
        if feature is None or not _feature_writable(feature) or not _feature_readable(feature):
            raise CameraBackendError(f"大恒参数 {name} 不支持可验证写入")
        try:
            result = feature.set(value)
            if result is False:
                raise RuntimeError("SDK rejected the value")
        except Exception as exc:
            self._last_error = str(exc)
            raise CameraBackendError(f"大恒参数 {name} 写入失败: {exc}") from exc

    def is_open(self) -> bool:
        return self._camera is not None

    def is_streaming(self) -> bool:
        return self._streaming


def _transport(value: str) -> str:
    v = value.upper()
    if "USB" in v:
        return "USB3"
    if "GEV" in v or "GIGE" in v:
        return "GigE"
    return value or "Unknown"


_FEATURE_NODES = {
    "exposure": "ExposureTime",
    "exposure_auto": "ExposureAuto",
    "gain": "Gain",
    "gain_auto": "GainAuto",
    "frame_rate": "AcquisitionFrameRate",
    "width": "Width",
    "height": "Height",
    "offset_x": "OffsetX",
    "offset_y": "OffsetY",
    "pixel_format": "PixelFormat",
    "trigger_mode": "TriggerMode",
    "trigger_source": "TriggerSource",
    "packet_size": "GevSCPSPacketSize",
    "acquisition_mode": "AcquisitionMode",
}


def _feature_node(camera: Any, name: str) -> Any:
    if camera is None:
        return None
    return getattr(camera, _FEATURE_NODES.get(name, name), None)


def _feature_readable(feature: Any) -> bool:
    for name in ("is_readable", "IsReadable"):
        status = getattr(feature, name, None)
        if callable(status):
            try:
                return bool(status())
            except Exception:
                return False
        if status is not None:
            return bool(status)
    return bool(getattr(feature, "readable", callable(getattr(feature, "get", None))))


def _feature_writable(feature: Any) -> bool:
    for name in ("is_writable", "is_writeable", "IsWritable"):
        status = getattr(feature, name, None)
        if callable(status):
            try:
                return bool(status())
            except Exception:
                return False
        if status is not None:
            return bool(status)
    return bool(getattr(feature, "writable", callable(getattr(feature, "set", None))))


def _caps_for_camera(camera: Any) -> CameraCapabilities:
    values = {}
    for name in CameraCapabilities.__dataclass_fields__:
        feature = _feature_node(camera, name)
        readable = feature is not None and _feature_readable(feature)
        writable = readable and _feature_writable(feature)
        current = None
        if readable:
            try:
                current = feature.get()
            except Exception:
                readable = False
                writable = False
        values[name] = CameraFeatureCapability(
            supported=bool(feature is not None and (readable or writable)),
            readable=readable,
            writable=writable,
            current_value=current,
            error="大恒参数不可验证写入" if feature is not None and not writable else ("大恒参数不可用" if feature is None else ""),
        )
    return CameraCapabilities(**values)


def _empty_caps() -> CameraCapabilities:
    return CameraCapabilities(**{
        name: CameraFeatureCapability(False, False, False, error="大恒参数能力需在相机打开后验证")
        for name in CameraCapabilities.__dataclass_fields__
    })

