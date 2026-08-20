from __future__ import annotations

from dataclasses import asdict, dataclass, fields, field
from typing import Any


DEVICE_TYPE_INDUSTRIAL = "industrial_camera"
DEVICE_TYPE_USB = "usb_camera"
DEVICE_TYPE_UNKNOWN = "unknown_camera"

TRANSPORT_GIGE = "GigE"
TRANSPORT_USB3 = "USB3"
TRANSPORT_USB2 = "USB2"
TRANSPORT_COAXPRESS = "CoaXPress"
TRANSPORT_CAMERALINK = "CameraLink"
TRANSPORT_GENTL = "GenTL"
TRANSPORT_UVC = "UVC"
TRANSPORT_UNKNOWN = "Unknown"


@dataclass(slots=True)
class CameraFeatureCapability:
    supported: bool = False
    readable: bool = False
    writable: bool = False
    current_value: Any | None = None
    min_value: float | int | None = None
    max_value: float | int | None = None
    increment: float | int | None = None
    available_values: list[Any] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CameraCapabilities:
    exposure: CameraFeatureCapability = field(default_factory=CameraFeatureCapability)
    exposure_auto: CameraFeatureCapability = field(default_factory=CameraFeatureCapability)
    gain: CameraFeatureCapability = field(default_factory=CameraFeatureCapability)
    gain_auto: CameraFeatureCapability = field(default_factory=CameraFeatureCapability)
    frame_rate: CameraFeatureCapability = field(default_factory=CameraFeatureCapability)
    width: CameraFeatureCapability = field(default_factory=CameraFeatureCapability)
    height: CameraFeatureCapability = field(default_factory=CameraFeatureCapability)
    offset_x: CameraFeatureCapability = field(default_factory=CameraFeatureCapability)
    offset_y: CameraFeatureCapability = field(default_factory=CameraFeatureCapability)
    pixel_format: CameraFeatureCapability = field(default_factory=CameraFeatureCapability)
    trigger_mode: CameraFeatureCapability = field(default_factory=CameraFeatureCapability)
    trigger_source: CameraFeatureCapability = field(default_factory=CameraFeatureCapability)
    packet_size: CameraFeatureCapability = field(default_factory=CameraFeatureCapability)
    acquisition_mode: CameraFeatureCapability = field(default_factory=CameraFeatureCapability)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name).to_dict() for name in self.__dataclass_fields__}


@dataclass(slots=True)
class CameraDeviceInfo:
    device_id: str
    unique_id: str
    backend_name: str
    manufacturer: str = ""
    model: str = ""
    serial_number: str = ""
    user_defined_name: str = ""
    device_type: str = DEVICE_TYPE_UNKNOWN
    transport_type: str = TRANSPORT_UNKNOWN
    ip_address: str = ""
    usb_path: str = ""
    gentl_producer: str = ""
    available: bool = True
    status: str = "discovered"
    capabilities: CameraCapabilities = field(default_factory=CameraCapabilities)
    error: str = ""
    available_backends: list[str] = field(default_factory=list)
    selected_backend: str = ""
    backend_priority: int = 100
    raw_info: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        # Do not call dataclasses.asdict(self) here: vendor SDK objects stored
        # in raw_info may be SWIG/ctypes handles and cannot be deep-copied or
        # pickled. Build the public payload field-by-field and intentionally
        # keep raw_info inside the backend process only.
        data: dict[str, Any] = {}
        for item in fields(self):
            if item.name == "raw_info":
                continue
            value = getattr(self, item.name)
            if item.name == "capabilities":
                data[item.name] = self.capabilities.to_dict()
            elif isinstance(value, list):
                data[item.name] = list(value)
            else:
                data[item.name] = value
        return data

    @classmethod
    def unavailable(cls, backend_name: str, error: str, manufacturer: str = "") -> "CameraDeviceInfo":
        return cls(
            device_id=f"{backend_name}:unavailable",
            unique_id=f"{backend_name}:unavailable",
            backend_name=backend_name,
            manufacturer=manufacturer,
            available=False,
            error=error,
            selected_backend=backend_name,
            available_backends=[backend_name],
        )


@dataclass(slots=True)
class FrameData:
    image: Any | None
    frame_id: int
    timestamp: float
    width: int = 0
    height: int = 0
    pixel_format: str = ""
    source_backend: str = ""
    device_unique_id: str = ""
    valid: bool = False
    error: str = ""


@dataclass(slots=True)
class CameraStatus:
    selected_unique_id: str = ""
    selected_backend: str = ""
    camera_connected: bool = False
    camera_streaming: bool = False
    is_open: bool = False
    latest_frame_timestamp: float = 0.0
    frame_id: int = 0
    consecutive_frame_failures: int = 0
    dropped_frame_count: int = 0
    last_error: str = ""
    width: int = 0
    height: int = 0
    pixel_format: str = ""
    frame_rate: float = 0.0


@dataclass(slots=True)
class CameraTestResult:
    ok: bool
    message: str = ""
    error: str = ""
    unique_id: str = ""
    backend_name: str = ""
    frames_read: int = 0
    width: int = 0
    height: int = 0
    pixel_format: str = ""
    frame_rate: float = 0.0
    preview_png_base64: str | None = None
    device_info: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    applied_parameters: dict[str, Any] = field(default_factory=dict)
    parameter_readback: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CameraBackendStatus:
    backend_name: str
    display_name: str = ""
    sdk_loaded: bool = False
    sdk_path: str = ""
    sdk_version: str = ""
    python_module_loaded: bool = False
    python_module_path: str = ""
    dll_loaded: bool = False
    dll_path: str = ""
    cti_loaded: bool = False
    cti_paths: list[str] = field(default_factory=list)
    backend_available: bool = False
    enum_return_code: str = ""
    raw_device_count: int = 0
    final_device_count: int = 0
    error: str = ""
    traceback: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CameraDiscoveryResult:
    devices: list[CameraDeviceInfo] = field(default_factory=list)
    raw_devices: list[CameraDeviceInfo] = field(default_factory=list)
    deduplicated_devices: list[CameraDeviceInfo] = field(default_factory=list)
    backend_statuses: list[CameraBackendStatus] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    raw_device_count: int = 0
    final_device_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "devices": [device.to_dict() for device in self.devices],
            "raw_devices": [device.to_dict() for device in self.raw_devices],
            "deduplicated_devices": [device.to_dict() for device in self.deduplicated_devices],
            "backend_statuses": [status.to_dict() for status in self.backend_statuses],
            "errors": list(self.errors),
            "raw_device_count": self.raw_device_count,
            "final_device_count": self.final_device_count,
        }
