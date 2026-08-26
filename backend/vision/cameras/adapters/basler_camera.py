from __future__ import annotations

import time
from typing import Any

from ..base import BaseCameraAdapter, CameraBackendError
from ..models import CameraCapabilities, CameraDeviceInfo, CameraFeatureCapability, DEVICE_TYPE_INDUSTRIAL, FrameData


class BaslerCameraAdapter(BaseCameraAdapter):
    backend_name = "basler"
    supported_manufacturers = ("Basler",)
    backend_priority = 11

    def __init__(self, config: Any | None = None, logger=None) -> None:
        super().__init__(config=config, logger=logger)
        self._camera = None
        self._converter = None
        self._device: CameraDeviceInfo | None = None
        self._latest = FrameData(None, 0, 0.0, source_backend=self.backend_name)
        self._frame_id = 0

    @classmethod
    def is_backend_available(cls, config: Any | None = None) -> tuple[bool, str]:
        try:
            from pypylon import pylon  # noqa: F401
        except Exception as exc:
            return False, f"pypylon/pylon Runtime 不可用: {exc}"
        return True, "pypylon 可用"

    @classmethod
    def discover_devices(cls, config: Any | None = None, logger=None) -> list[CameraDeviceInfo]:
        from pypylon import pylon

        devices: list[CameraDeviceInfo] = []
        for dev in pylon.TlFactory.GetInstance().EnumerateDevices():
            manufacturer = dev.GetVendorName() or "Basler"
            model = dev.GetModelName() or ""
            serial = dev.GetSerialNumber() or ""
            transport = dev.GetDeviceClass() or "Unknown"
            unique = f"{manufacturer}:{model}:{serial}" if serial else f"basler:{dev.GetFullName()}"
            devices.append(
                CameraDeviceInfo(
                    device_id=f"basler:{serial or dev.GetFullName()}",
                    unique_id=unique,
                    backend_name=cls.backend_name,
                    manufacturer=manufacturer,
                    model=model,
                    serial_number=serial,
                    user_defined_name=dev.GetUserDefinedName() or "",
                    device_type=DEVICE_TYPE_INDUSTRIAL,
                    transport_type=_transport(transport),
                    available=True,
                    capabilities=_empty_caps(),
                    available_backends=[cls.backend_name],
                    selected_backend=cls.backend_name,
                    backend_priority=cls.backend_priority,
                    raw_info=dev,
                )
            )
        return devices

    def open(self, device_info: CameraDeviceInfo) -> None:
        from pypylon import pylon

        self.close()
        factory = pylon.TlFactory.GetInstance()
        target = None
        for dev in factory.EnumerateDevices():
            if device_info.serial_number and dev.GetSerialNumber() == device_info.serial_number:
                target = dev
                break
            if dev.GetFullName() in device_info.device_id:
                target = dev
                break
        if target is None:
            raise CameraBackendError("未找到 Basler 相机")
        self._camera = pylon.InstantCamera(factory.CreateDevice(target))
        self._camera.Open()
        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self._device = device_info
        self._device.capabilities = self.get_capabilities()

    def close(self) -> None:
        self.stop_stream()
        if self._camera is not None:
            try:
                self._camera.Close()
            except Exception:
                pass
        self._camera = None
        self._device = None
        self._log("[CAMERA][CLOSE] backend=basler")

    def start_stream(self) -> None:
        from pypylon import pylon

        if self._camera is None:
            raise CameraBackendError("Basler 相机未打开")
        self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self._log("[CAMERA][STREAM][START] backend=basler")

    def stop_stream(self) -> None:
        if self._camera is not None and self._camera.IsGrabbing():
            self._camera.StopGrabbing()

    def read_frame(self, timeout_ms: int = 1000) -> FrameData:
        from pypylon import pylon

        if self._camera is None or not self._camera.IsGrabbing():
            return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error="Basler 未开始采集")
        result = self._camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_Return)
        try:
            if not result or not result.GrabSucceeded():
                err = result.GetErrorDescription() if result else "Basler 取帧超时"
                self._last_error = err
                return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error=err)
            image = self._converter.Convert(result).GetArray()
            host_monotonic = time.monotonic()
            hardware_id = int(getattr(result, "GetBlockID", lambda: 0)() or 0)
            hardware_timestamp = int(getattr(result, "GetTimeStamp", lambda: 0)() or 0)
            self._frame_id = hardware_id or (self._frame_id + 1)
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
                host_monotonic_timestamp=host_monotonic,
                hardware_frame_id=hardware_id,
                hardware_timestamp_ticks=hardware_timestamp,
            )
            return self._latest
        finally:
            if result:
                result.Release()

    def get_latest_frame(self) -> FrameData:
        return self._latest

    def get_device_info(self) -> CameraDeviceInfo | None:
        return self._device

    def get_capabilities(self) -> CameraCapabilities:
        return _caps_for_camera(self._camera) if self._camera is not None else _empty_caps()

    def get_feature(self, name: str) -> Any:
        node = _feature_node(self._camera, name)
        if node is None or not _node_readable(node):
            return None
        return node.GetValue()

    def set_feature(self, name: str, value: Any) -> None:
        node = _feature_node(self._camera, name)
        if node is None or not _node_writable(node) or not _node_readable(node):
            raise CameraBackendError(f"Basler 参数 {name} 不支持可验证写入")
        try:
            result = node.SetValue(value)
            if result is False:
                raise RuntimeError("SDK rejected the value")
        except Exception as exc:
            self._last_error = str(exc)
            raise CameraBackendError(f"Basler 参数 {name} 写入失败: {exc}") from exc

    def is_open(self) -> bool:
        return bool(self._camera is not None and self._camera.IsOpen())

    def is_streaming(self) -> bool:
        return bool(self._camera is not None and self._camera.IsGrabbing())


def _transport(value: str) -> str:
    v = value.upper()
    if "USB" in v:
        return "USB3"
    if "GIGE" in v or "GEV" in v:
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


def _node_readable(node: Any) -> bool:
    for name in ("IsReadable", "is_readable"):
        status = getattr(node, name, None)
        if callable(status):
            try:
                return bool(status())
            except Exception:
                return False
        if status is not None:
            return bool(status)
    return bool(getattr(node, "readable", callable(getattr(node, "GetValue", None))))


def _node_writable(node: Any) -> bool:
    for name in ("IsWritable", "is_writable", "is_writeable"):
        status = getattr(node, name, None)
        if callable(status):
            try:
                return bool(status())
            except Exception:
                return False
        if status is not None:
            return bool(status)
    return bool(getattr(node, "writable", callable(getattr(node, "SetValue", None))))


def _caps_for_camera(camera: Any) -> CameraCapabilities:
    values = {}
    for name in CameraCapabilities.__dataclass_fields__:
        node = _feature_node(camera, name)
        readable = node is not None and _node_readable(node)
        writable = readable and _node_writable(node)
        current = None
        if readable:
            try:
                current = node.GetValue()
            except Exception:
                readable = False
                writable = False
        values[name] = CameraFeatureCapability(
            supported=bool(node is not None and (readable or writable)),
            readable=readable,
            writable=writable,
            current_value=current,
            error="Basler 参数不可验证写入" if node is not None and not writable else ("Basler 参数不可用" if node is None else ""),
        )
    return CameraCapabilities(**values)


def _empty_caps() -> CameraCapabilities:
    return CameraCapabilities(**{
        name: CameraFeatureCapability(False, False, False, error="Basler 参数能力需在相机打开后验证")
        for name in CameraCapabilities.__dataclass_fields__
    })

