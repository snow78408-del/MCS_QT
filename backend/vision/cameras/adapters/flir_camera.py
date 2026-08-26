from __future__ import annotations

import math
import time
from typing import Any

from ..base import BaseCameraAdapter, CameraBackendError
from ..models import CameraCapabilities, CameraDeviceInfo, CameraFeatureCapability, DEVICE_TYPE_INDUSTRIAL, FrameData


_FEATURE_NODES: dict[str, tuple[str, str]] = {
    "exposure": ("ExposureTime", "float"),
    "gain": ("Gain", "float"),
    "frame_rate": ("AcquisitionFrameRate", "float"),
    "width": ("Width", "int"),
    "height": ("Height", "int"),
    "offset_x": ("OffsetX", "int"),
    "offset_y": ("OffsetY", "int"),
    "pixel_format": ("PixelFormat", "enum"),
    "trigger_mode": ("TriggerMode", "enum"),
    "acquisition_mode": ("AcquisitionMode", "enum"),
}


class FlirCameraAdapter(BaseCameraAdapter):
    backend_name = "flir"
    supported_manufacturers = ("Teledyne FLIR", "FLIR")
    backend_priority = 13

    def __init__(self, config: Any | None = None, logger=None) -> None:
        super().__init__(config=config, logger=logger)
        self._system = None
        self._camera_list = None
        self._camera = None
        self._device: CameraDeviceInfo | None = None
        self._latest = FrameData(None, 0, 0.0, source_backend=self.backend_name)
        self._frame_id = 0
        self._streaming = False

    @classmethod
    def is_backend_available(cls, config: Any | None = None) -> tuple[bool, str]:
        try:
            import PySpin  # noqa: F401
        except Exception as exc:
            return False, f"Spinnaker SDK/PySpin 不可用: {exc}"
        return True, "Spinnaker SDK 可用"

    @classmethod
    def discover_devices(cls, config: Any | None = None, logger=None) -> list[CameraDeviceInfo]:
        import PySpin

        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        devices: list[CameraDeviceInfo] = []
        try:
            for idx, cam in enumerate(cam_list):
                nodemap = cam.GetTLDeviceNodeMap()
                manufacturer = _node_string(PySpin, nodemap, "DeviceVendorName") or "Teledyne FLIR"
                model = _node_string(PySpin, nodemap, "DeviceModelName")
                serial = _node_string(PySpin, nodemap, "DeviceSerialNumber")
                transport = _node_string(PySpin, nodemap, "DeviceType") or "Unknown"
                unique = f"{manufacturer}:{model}:{serial}" if serial else f"flir:{idx}"
                # Feature writability is established only after Init(), so the
                # discovery record must not advertise guessed writable nodes.
                devices.append(
                    CameraDeviceInfo(
                        device_id=f"flir:{serial or idx}",
                        unique_id=unique,
                        backend_name=cls.backend_name,
                        manufacturer=manufacturer,
                        model=model,
                        serial_number=serial,
                        device_type=DEVICE_TYPE_INDUSTRIAL,
                        transport_type=_transport(transport),
                        available=True,
                        capabilities=_caps(),
                        available_backends=[cls.backend_name],
                        selected_backend=cls.backend_name,
                        backend_priority=cls.backend_priority,
                    )
                )
        finally:
            cam_list.Clear()
            system.ReleaseInstance()
        return devices

    def open(self, device_info: CameraDeviceInfo) -> None:
        import PySpin

        self.close()
        self._system = PySpin.System.GetInstance()
        self._camera_list = self._system.GetCameras()
        target = None
        for idx, cam in enumerate(self._camera_list):
            nodemap = cam.GetTLDeviceNodeMap()
            serial = _node_string(PySpin, nodemap, "DeviceSerialNumber")
            if serial == device_info.serial_number or device_info.device_id.endswith(str(idx)):
                target = cam
                break
        if target is None:
            raise CameraBackendError("未找到 FLIR 相机")
        self._camera = target
        self._camera.Init()
        self._device = device_info

    def close(self) -> None:
        self.stop_stream()
        if self._camera is not None:
            try:
                self._camera.DeInit()
            except Exception:
                pass
        if self._camera_list is not None:
            try:
                self._camera_list.Clear()
            except Exception:
                pass
        if self._system is not None:
            try:
                self._system.ReleaseInstance()
            except Exception:
                pass
        self._system = None
        self._camera_list = None
        self._camera = None
        self._device = None
        self._log("[CAMERA][CLOSE] backend=flir")

    def start_stream(self) -> None:
        if self._camera is None:
            raise CameraBackendError("FLIR 相机未打开")
        self._camera.BeginAcquisition()
        self._streaming = True
        self._log("[CAMERA][STREAM][START] backend=flir")

    def stop_stream(self) -> None:
        if self._camera is not None and self._streaming:
            try:
                self._camera.EndAcquisition()
            except Exception:
                pass
        self._streaming = False

    def read_frame(self, timeout_ms: int = 1000) -> FrameData:
        if self._camera is None or not self._streaming:
            return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error="FLIR 未开始采集")
        image = None
        try:
            image = self._camera.GetNextImage(timeout_ms)
            if image.IsIncomplete():
                return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error="FLIR 图像不完整")
            arr = image.GetNDArray()
            if arr.ndim == 2:
                out = arr
                fmt = "Mono8"
            else:
                out = arr[:, :, ::-1].copy()
                fmt = "BGR8"
            host_monotonic = time.monotonic()
            hardware_id = int(getattr(image, "GetFrameID", lambda: 0)() or 0)
            hardware_timestamp = int(getattr(image, "GetTimeStamp", lambda: 0)() or 0)
            self._frame_id = hardware_id or (self._frame_id + 1)
            self._latest = FrameData(out, self._frame_id, time.time(), int(out.shape[1]), int(out.shape[0]), fmt, self.backend_name, self._device.unique_id if self._device else "", True, host_monotonic_timestamp=host_monotonic, hardware_frame_id=hardware_id, hardware_timestamp_ticks=hardware_timestamp)
            return self._latest
        except Exception as exc:
            self._last_error = str(exc)
            return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error=str(exc))
        finally:
            if image is not None:
                image.Release()

    def get_latest_frame(self) -> FrameData:
        return self._latest

    def get_device_info(self) -> CameraDeviceInfo | None:
        return self._device

    def get_capabilities(self) -> CameraCapabilities:
        if self._camera is None:
            return _caps()
        import PySpin

        return _caps_for_camera(PySpin, self._camera)

    def get_feature(self, name: str) -> Any:
        node, kind = self._feature_node(name)
        if node is None or not _is_readable(_pyspin(), node):
            return None
        return _read_node(_pyspin(), node, kind)

    def set_feature(self, name: str, value: Any) -> None:
        pyspin = _pyspin()
        node, kind = self._feature_node(name)
        if node is None:
            raise CameraBackendError(f"FLIR 参数 {name} 不可用")
        if not _is_writable(pyspin, node) or not _is_readable(pyspin, node):
            raise CameraBackendError(f"FLIR 参数 {name} 不支持可验证写入")
        try:
            _write_node(pyspin, node, kind, value)
            actual = _read_node(pyspin, node, kind)
        except Exception as exc:
            self._last_error = str(exc)
            raise CameraBackendError(f"FLIR 参数 {name} 写入失败: {exc}") from exc
        if not _feature_values_equal(actual, value, kind):
            raise CameraBackendError(f"FLIR 参数 {name} 写入校验失败: requested={value}, readback={actual}")

    def is_open(self) -> bool:
        return self._camera is not None

    def is_streaming(self) -> bool:
        return self._streaming

    def _feature_node(self, name: str):
        canonical = "exposure" if name == "exposure_time" else name
        spec = _FEATURE_NODES.get(canonical)
        if self._camera is None or spec is None:
            return None, spec[1] if spec else "float"
        try:
            import PySpin

            raw = self._camera.GetNodeMap().GetNode(spec[0])
            if raw is None:
                return None, spec[1]
            converter = {
                "float": "CFloatPtr",
                "int": "CIntegerPtr",
                "enum": "CEnumerationPtr",
            }[spec[1]]
            return getattr(PySpin, converter)(raw), spec[1]
        except Exception:
            return None, spec[1]


def _pyspin():
    import PySpin

    return PySpin


def _is_readable(PySpin, node) -> bool:
    try:
        return bool(PySpin.IsReadable(node))
    except Exception:
        return bool(getattr(node, "readable", False))


def _is_writable(PySpin, node) -> bool:
    try:
        return bool(PySpin.IsWritable(node))
    except Exception:
        return bool(getattr(node, "writable", False))


def _read_node(PySpin, node, kind: str) -> Any:
    if kind == "enum":
        try:
            entry = node.GetCurrentEntry()
            if _is_readable(PySpin, entry):
                return str(entry.GetSymbolic())
        except Exception:
            pass
        return node.GetValue()
    return node.GetValue()


def _write_node(PySpin, node, kind: str, value: Any) -> None:
    if kind == "enum":
        text = str(value)
        try:
            entry = node.GetEntryByName(text)
            if entry is not None and _is_readable(PySpin, entry):
                node.SetIntValue(entry.GetValue())
                return
        except Exception:
            pass
        node.SetValue(text)
    elif kind == "int":
        node.SetValue(int(value))
    else:
        node.SetValue(float(value))


def _feature_values_equal(actual: Any, requested: Any, kind: str) -> bool:
    if kind == "enum":
        return str(actual).lower() == str(requested).lower()
    try:
        return math.isclose(float(actual), float(requested), rel_tol=1e-5, abs_tol=1e-5)
    except (TypeError, ValueError):
        return actual == requested


def _caps_for_camera(PySpin, camera) -> CameraCapabilities:
    values: dict[str, CameraFeatureCapability] = {}
    for name, (node_name, kind) in _FEATURE_NODES.items():
        try:
            raw = camera.GetNodeMap().GetNode(node_name)
            if raw is None:
                raise ValueError("node missing")
            converter = {"float": "CFloatPtr", "int": "CIntegerPtr", "enum": "CEnumerationPtr"}[kind]
            node = getattr(PySpin, converter)(raw)
            readable = _is_readable(PySpin, node)
            writable = _is_writable(PySpin, node) and readable
            current = _read_node(PySpin, node, kind) if readable else None
            minimum = _node_limit(node, "GetMin") if kind != "enum" and readable else None
            maximum = _node_limit(node, "GetMax") if kind != "enum" and readable else None
            increment = _node_limit(node, "GetInc") if kind == "int" and readable else None
            error = "" if writable else "FLIR 参数不可验证写入"
            values[name] = CameraFeatureCapability(True, readable, writable, current, minimum, maximum, increment, error=error)
        except Exception as exc:
            values[name] = CameraFeatureCapability(False, False, False, error=f"FLIR 参数不可用: {exc}")
    return CameraCapabilities(**values)


def _node_limit(node, method_name: str):
    method = getattr(node, method_name, None)
    if not callable(method):
        return None
    try:
        return method()
    except Exception:
        return None


def _caps() -> CameraCapabilities:
    unsupported = {
        name: CameraFeatureCapability(False, False, False, error="FLIR 参数能力需在相机打开后验证")
        for name in CameraCapabilities.__dataclass_fields__
    }
    return CameraCapabilities(**unsupported)


def _node_string(PySpin, nodemap, name: str) -> str:
    try:
        node = PySpin.CStringPtr(nodemap.GetNode(name))
        if PySpin.IsReadable(node):
            return str(node.GetValue())
    except Exception:
        return ""
    return ""


def _transport(value: str) -> str:
    v = value.upper()
    if "USB" in v:
        return "USB3"
    if "GIGE" in v or "GEV" in v:
        return "GigE"
    return value or "Unknown"
