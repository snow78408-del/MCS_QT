from __future__ import annotations

import ctypes
import os
import time
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_bool,
    c_float,
    c_int,
    c_ubyte,
    c_uint,
    c_ushort,
    c_void_p,
)
from pathlib import Path
from typing import Any

try:
    import cv2
except Exception:  # pragma: no cover - runtime dependency
    cv2 = None
try:
    import numpy as np
except Exception:  # pragma: no cover - runtime dependency
    np = None

from ..base import CameraBackendError
from ..models import (
    CameraCapabilities,
    CameraDeviceInfo,
    CameraFeatureCapability,
    DEVICE_TYPE_INDUSTRIAL,
    FrameData,
)


# ``WinDLL`` is only exposed by ctypes on Windows.  Keep this module
# importable on other platforms so the optional HIKROBOT backend can report
# itself as unavailable instead of preventing the camera registry from loading.
WinDLL = getattr(ctypes, "WinDLL", None)


DEFAULT_MVS_DLL_PATHS = (
    r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64\MvCameraControl.dll",
    r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64\MvCameraControl.dll",
    r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win32_i86\MvCameraControl.dll",
)

MV_MAX_DEVICE_NUM = 256
MV_GIGE_DEVICE = 0x00000001
MV_USB_DEVICE = 0x00000004
MV_ACCESS_EXCLUSIVE = 1

PIXEL_MONO8 = 0x01080001
PIXEL_BAYER_GR8 = 0x01080008
PIXEL_BAYER_RG8 = 0x01080009
PIXEL_BAYER_GB8 = 0x0108000A
PIXEL_BAYER_BG8 = 0x0108000B
PIXEL_RGB8 = 0x02180014
PIXEL_BGR8 = 0x02180015


class DirectDeviceInfoList(Structure):
    _fields_ = [
        ("nDeviceNum", c_uint),
        ("pDeviceInfo", c_void_p * MV_MAX_DEVICE_NUM),
    ]


class DirectFrameInfo(Structure):
    _fields_ = [
        ("nWidth", c_ushort),
        ("nHeight", c_ushort),
        ("enPixelType", c_uint),
        ("nFrameNum", c_uint),
        ("nDevTimeStampHigh", c_uint),
        ("nDevTimeStampLow", c_uint),
        ("nReserved0", c_uint),
        ("nHostTimeStamp", c_uint * 2),
        ("nFrameLen", c_uint),
        ("nSecondCount", c_uint),
        ("nCycleCount", c_uint),
        ("nCycleOffset", c_uint),
        ("fGain", c_float),
        ("fExposureTime", c_float),
        ("nAverageBrightness", c_uint),
        ("nRed", c_uint),
        ("nGreen", c_uint),
        ("nBlue", c_uint),
        ("nFrameCounter", c_uint),
        ("nTriggerIndex", c_uint),
        ("nInput", c_uint),
        ("nOutput", c_uint),
        ("nOffsetX", c_ushort),
        ("nOffsetY", c_ushort),
        ("nChunkWidth", c_ushort),
        ("nChunkHeight", c_ushort),
        ("nLostPacket", c_uint),
        ("nUnparsedChunkNum", c_uint),
        ("nReserved", c_uint * 36),
    ]


class DirectPixelConvertParam(Structure):
    _fields_ = [
        ("nWidth", c_ushort),
        ("nHeight", c_ushort),
        ("enSrcPixelType", c_uint),
        ("pSrcData", POINTER(c_ubyte)),
        ("nSrcDataLen", c_uint),
        ("enDstPixelType", c_uint),
        ("pDstBuffer", POINTER(c_ubyte)),
        ("nDstBufferSize", c_uint),
        ("nDstLen", c_uint),
        ("nReserved", c_uint * 4),
    ]


class DirectHikrobotDllCamera:
    backend_name = "hikrobot"

    def __init__(self, config: Any | None = None, logger=None) -> None:
        self.config = config
        self._log = logger or (lambda _msg: None)
        self._dll = None
        self._dll_path: Path | None = None
        self._dll_dir_handle = None
        self._handle = c_void_p()
        self._device: CameraDeviceInfo | None = None
        self._opened = False
        self._streaming = False
        self._frame_id = 0
        self._last_error = ""
        self._last_width = 0
        self._last_height = 0
        self._last_pixel_format = ""
        self._frame_buffer = None
        self._frame_buffer_size = 0

    @classmethod
    def find_dll(cls, config: Any | None = None) -> Path | None:
        candidates: list[Path] = []
        for value in _sdk_search_values(config):
            root = Path(value).expanduser()
            if root.is_file() and root.name.lower() == "mvcameracontrol.dll":
                candidates.append(root)
            elif root.exists():
                direct = root / "MvCameraControl.dll"
                if direct.exists():
                    candidates.append(direct)
                try:
                    candidates.extend(root.rglob("MvCameraControl.dll"))
                except Exception:
                    pass
        candidates.extend(Path(path) for path in DEFAULT_MVS_DLL_PATHS)
        for path in dict.fromkeys(candidates):
            if path.exists():
                return path
        return None

    @classmethod
    def is_available(cls, config: Any | None = None) -> tuple[bool, str, Path | None]:
        if WinDLL is None:
            return False, "HIKROBOT direct DLL backend requires Windows", None
        if cv2 is None or np is None:
            return False, "OpenCV or NumPy is not installed", None
        dll_path = cls.find_dll(config)
        if dll_path is None:
            return False, "MvCameraControl.dll was not found", None
        return True, f"HIKROBOT MVS DLL available: {dll_path}", dll_path

    @classmethod
    def discover_devices(cls, config: Any | None = None, logger=None) -> list[CameraDeviceInfo]:
        log = logger or (lambda _msg: None)
        direct = cls(config=config, logger=log)
        try:
            direct._load_dll()
            device_list = direct._enum_device_list()
        except Exception as exc:
            return [
                CameraDeviceInfo.unavailable(
                    backend_name=cls.backend_name,
                    manufacturer="HIKROBOT",
                    error=f"HIKROBOT MVS DLL unavailable: {exc}",
                )
            ]

        devices: list[CameraDeviceInfo] = []
        count = int(device_list.nDeviceNum)
        log(f"[HIKROBOT][DIRECT][ENUM] device_count={count}")
        for index in range(count):
            pointer = int(device_list.pDeviceInfo[index] or 0)
            devices.append(
                CameraDeviceInfo(
                    device_id=f"hikrobot:direct:{index}",
                    unique_id=f"HIKROBOT:DIRECT:{index}",
                    backend_name=cls.backend_name,
                    manufacturer="HIKROBOT",
                    model=f"MVS Device {index}",
                    serial_number=f"direct-{index}",
                    device_type=DEVICE_TYPE_INDUSTRIAL,
                    transport_type="MVS",
                    available=True,
                    capabilities=_default_capabilities(),
                    selected_backend=cls.backend_name,
                    available_backends=[cls.backend_name],
                    backend_priority=10,
                    raw_info={"direct_index": index, "device_info_ptr": pointer, "sdk_mode": "direct_dll"},
                )
            )
            log(f"[HIKROBOT][DIRECT][DEVICE][FOUND] index={index} ptr=0x{pointer:X}")
        return devices

    def open(self, device: CameraDeviceInfo) -> None:
        self._load_dll()
        index = _direct_index(device)
        device_list = self._enum_device_list()
        if index < 0 or index >= int(device_list.nDeviceNum):
            raise CameraBackendError(
                f"HIKROBOT direct camera index {index} is unavailable; found {int(device_list.nDeviceNum)} device(s)."
            )
        device_ptr = device_list.pDeviceInfo[index]
        handle = c_void_p()
        ret = self._dll.MV_CC_CreateHandle(byref(handle), c_void_p(device_ptr))
        if ret != 0:
            raise CameraBackendError(f"MV_CC_CreateHandle failed: {_ret_hex(ret)}", int(ret))
        opened = False
        try:
            ret = self._dll.MV_CC_OpenDevice(handle, MV_ACCESS_EXCLUSIVE, 0)
            if ret != 0:
                raise CameraBackendError(f"MV_CC_OpenDevice failed: {_ret_hex(ret)}", int(ret))
            opened = True
            self._handle = handle
            self._device = device
            self._set_enum_value("TriggerMode", 0, raise_on_fail=True)
            packet_size = int(self._dll.MV_CC_GetOptimalPacketSize(self._handle))
            if packet_size > 0:
                self._set_int_value("GevSCPSPacketSize", packet_size, raise_on_fail=True)
            self._opened = True
            self._last_error = ""
            self._log(f"[HIKROBOT][DIRECT][OPEN][OK] index={index} dll={self._dll_path}")
        except Exception:
            if opened:
                self._safe_call("MV_CC_CloseDevice")
            self._safe_call("MV_CC_DestroyHandle")
            self._handle = c_void_p()
            raise

    def close(self) -> None:
        self.stop_stream()
        if self._opened:
            self._safe_call("MV_CC_CloseDevice")
        if self._handle:
            self._safe_call("MV_CC_DestroyHandle")
        self._handle = c_void_p()
        self._opened = False
        self._frame_buffer = None
        self._frame_buffer_size = 0
        self._device = None

    def start_stream(self) -> None:
        if not self._opened:
            raise CameraBackendError("HIKROBOT direct camera is not open")
        ret = self._dll.MV_CC_StartGrabbing(self._handle)
        if ret != 0:
            raise CameraBackendError(f"MV_CC_StartGrabbing failed: {_ret_hex(ret)}", int(ret))
        self._streaming = True
        self._log("[HIKROBOT][DIRECT][STREAM][START]")

    def stop_stream(self) -> None:
        if self._streaming and self._dll is not None and self._handle:
            self._safe_call("MV_CC_StopGrabbing")
        self._streaming = False

    def read_frame(self, timeout_ms: int = 1000) -> FrameData:
        if not self._streaming:
            return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error="HIKROBOT direct camera is not streaming")
        buffer_size = int(getattr(self.config, "hikrobot_direct_buffer_mb", 16) or 16) * 1024 * 1024
        if self._frame_buffer is None or self._frame_buffer_size != buffer_size:
            self._frame_buffer = (c_ubyte * buffer_size)()
            self._frame_buffer_size = buffer_size
        buffer = self._frame_buffer
        info = DirectFrameInfo()
        ret = self._dll.MV_CC_GetOneFrameTimeout(self._handle, buffer, buffer_size, byref(info), int(timeout_ms))
        host_wall = time.time()
        host_monotonic = time.monotonic()
        if ret != 0:
            self._last_error = f"MV_CC_GetOneFrameTimeout failed: {_ret_hex(ret)}"
            return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error=self._last_error)

        try:
            width = int(info.nWidth)
            height = int(info.nHeight)
            pixel_type = int(info.enPixelType)
            reported_len = int(info.nFrameLen)
            actual_len = _expected_frame_len(width, height, pixel_type, reported_len)
            if width <= 0 or height <= 0 or actual_len <= 0:
                raise CameraBackendError(
                    f"Invalid HIKROBOT frame metadata: width={width}, height={height}, len={reported_len}, pixel={_ret_hex(pixel_type)}"
                )
            raw = np.ctypeslib.as_array(buffer)[:actual_len].copy()
            image, pixel_name = self._convert_frame(raw, width, height, pixel_type, actual_len)
            self._frame_id += 1
            hardware_frame_id = int(info.nFrameNum)
            device_ticks = (int(info.nDevTimeStampHigh) << 32) | int(info.nDevTimeStampLow)
            sdk_host_ticks = (int(info.nHostTimeStamp[0]) << 32) | int(info.nHostTimeStamp[1])
            self._last_width = width
            self._last_height = height
            self._last_pixel_format = pixel_name
            return FrameData(
                image=image,
                frame_id=hardware_frame_id or self._frame_id,
                timestamp=host_wall,
                width=width,
                height=height,
                pixel_format=pixel_name,
                source_backend=self.backend_name,
                device_unique_id=self._device.unique_id if self._device else "",
                valid=True,
                host_monotonic_timestamp=host_monotonic,
                hardware_frame_id=hardware_frame_id,
                hardware_timestamp_ticks=device_ticks,
                sdk_host_timestamp_ticks=sdk_host_ticks,
                lost_packet_count=int(info.nLostPacket),
                exposure_time_us=float(info.fExposureTime),
            )
        except Exception as exc:
            self._last_error = str(exc)
            return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error=self._last_error)

    def get_latest_frame(self) -> FrameData:
        return self.read_frame(timeout_ms=1)

    def get_device_info(self) -> CameraDeviceInfo | None:
        return self._device

    def get_capabilities(self) -> CameraCapabilities:
        return self._device.capabilities if self._device else _default_capabilities()

    def get_feature(self, name: str) -> Any:
        del name
        return None

    def set_feature(self, name: str, value: Any) -> None:
        key = _feature_name(name)
        try:
            if key == "TriggerMode":
                enum_value = _trigger_mode_value(value)
                self._set_enum_value(key, enum_value, raise_on_fail=True)
            elif key == "AcquisitionMode":
                enum_value = _acquisition_mode_value(value)
                self._set_enum_value(key, enum_value, raise_on_fail=True)
            elif key in {"Width", "Height", "OffsetX", "OffsetY", "GevSCPSPacketSize"}:
                self._set_int_value(key, int(value), raise_on_fail=True)
            elif key == "ExposureTime":
                self._set_enum_value("ExposureAuto", 0, raise_on_fail=True)
                self._set_float_value(key, float(value), raise_on_fail=True)
            elif key == "Gain":
                self._set_enum_value("GainAuto", 0, raise_on_fail=True)
                self._set_float_value(key, float(value), raise_on_fail=True)
            elif key == "AcquisitionFrameRate":
                self._set_bool_value("AcquisitionFrameRateEnable", True, raise_on_fail=True)
                self._set_float_value(key, float(value), raise_on_fail=True)
            else:
                raise CameraBackendError(f"HIKROBOT 参数 {name} 不可用")
        except Exception as exc:
            self._last_error = str(exc)
            raise

    def is_open(self) -> bool:
        return bool(self._opened)

    def is_streaming(self) -> bool:
        return bool(self._streaming)

    def get_last_error(self) -> str:
        return self._last_error

    def _load_dll(self) -> None:
        if self._dll is not None:
            return
        ok, reason, dll_path = self.is_available(self.config)
        if not ok or dll_path is None:
            raise CameraBackendError(reason)
        if hasattr(os, "add_dll_directory"):
            self._dll_dir_handle = os.add_dll_directory(str(dll_path.parent))
        dll = WinDLL(str(dll_path))
        dll.MV_CC_EnumDevices.argtypes = [c_uint, POINTER(DirectDeviceInfoList)]
        dll.MV_CC_EnumDevices.restype = c_int
        dll.MV_CC_CreateHandle.argtypes = [POINTER(c_void_p), c_void_p]
        dll.MV_CC_CreateHandle.restype = c_int
        dll.MV_CC_OpenDevice.argtypes = [c_void_p, c_uint, c_ushort]
        dll.MV_CC_OpenDevice.restype = c_int
        dll.MV_CC_CloseDevice.argtypes = [c_void_p]
        dll.MV_CC_CloseDevice.restype = c_int
        dll.MV_CC_DestroyHandle.argtypes = [c_void_p]
        dll.MV_CC_DestroyHandle.restype = c_int
        dll.MV_CC_StartGrabbing.argtypes = [c_void_p]
        dll.MV_CC_StartGrabbing.restype = c_int
        dll.MV_CC_StopGrabbing.argtypes = [c_void_p]
        dll.MV_CC_StopGrabbing.restype = c_int
        dll.MV_CC_GetOneFrameTimeout.argtypes = [c_void_p, POINTER(c_ubyte), c_uint, POINTER(DirectFrameInfo), c_uint]
        dll.MV_CC_GetOneFrameTimeout.restype = c_int
        dll.MV_CC_SetEnumValue.argtypes = [c_void_p, c_char_p, c_uint]
        dll.MV_CC_SetEnumValue.restype = c_int
        dll.MV_CC_SetIntValue.argtypes = [c_void_p, c_char_p, c_uint]
        dll.MV_CC_SetIntValue.restype = c_int
        dll.MV_CC_SetFloatValue.argtypes = [c_void_p, c_char_p, c_float]
        dll.MV_CC_SetFloatValue.restype = c_int
        if hasattr(dll, "MV_CC_SetBoolValue"):
            dll.MV_CC_SetBoolValue.argtypes = [c_void_p, c_char_p, c_bool]
            dll.MV_CC_SetBoolValue.restype = c_int
        dll.MV_CC_GetOptimalPacketSize.argtypes = [c_void_p]
        dll.MV_CC_GetOptimalPacketSize.restype = c_int
        if hasattr(dll, "MV_CC_ConvertPixelType"):
            dll.MV_CC_ConvertPixelType.argtypes = [c_void_p, POINTER(DirectPixelConvertParam)]
            dll.MV_CC_ConvertPixelType.restype = c_int
        self._dll = dll
        self._dll_path = dll_path
        self._log(f"[HIKROBOT][DIRECT][SDK] dll={dll_path}")

    def _enum_device_list(self) -> DirectDeviceInfoList:
        device_list = DirectDeviceInfoList()
        ret = self._dll.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, byref(device_list))
        if ret != 0:
            raise CameraBackendError(f"MV_CC_EnumDevices failed: {_ret_hex(ret)}", int(ret))
        return device_list

    def _set_enum_value(self, name: str, value: int, raise_on_fail: bool) -> None:
        ret = self._dll.MV_CC_SetEnumValue(self._handle, name.encode("ascii"), int(value))
        if ret != 0 and raise_on_fail:
            raise CameraBackendError(f"MV_CC_SetEnumValue {name} failed: {_ret_hex(ret)}", int(ret))

    def _set_int_value(self, name: str, value: int, raise_on_fail: bool) -> None:
        ret = self._dll.MV_CC_SetIntValue(self._handle, name.encode("ascii"), int(value))
        if ret != 0 and raise_on_fail:
            raise CameraBackendError(f"MV_CC_SetIntValue {name} failed: {_ret_hex(ret)}", int(ret))

    def _set_float_value(self, name: str, value: float, raise_on_fail: bool) -> None:
        ret = self._dll.MV_CC_SetFloatValue(self._handle, name.encode("ascii"), c_float(float(value)))
        if ret != 0 and raise_on_fail:
            raise CameraBackendError(f"MV_CC_SetFloatValue {name} failed: {_ret_hex(ret)}", int(ret))

    def _set_bool_value(self, name: str, value: bool, raise_on_fail: bool) -> None:
        setter = getattr(self._dll, "MV_CC_SetBoolValue", None)
        if setter is None:
            if raise_on_fail:
                raise CameraBackendError(f"MV_CC_SetBoolValue {name} is unavailable")
            return
        ret = setter(self._handle, name.encode("ascii"), c_bool(bool(value)))
        if ret != 0 and raise_on_fail:
            raise CameraBackendError(f"MV_CC_SetBoolValue {name} failed: {_ret_hex(ret)}", int(ret))

    def _safe_call(self, name: str) -> None:
        try:
            getattr(self._dll, name)(self._handle)
        except Exception:
            pass

    def _convert_frame(self, raw, width: int, height: int, pixel_type: int, frame_len: int):
        if pixel_type == PIXEL_MONO8:
            return raw.reshape((height, width)), "Mono8"
        if pixel_type == PIXEL_RGB8:
            rgb = raw.reshape((height, width, 3))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), "RGB8"
        if pixel_type == PIXEL_BGR8:
            return raw.reshape((height, width, 3)), "BGR8"
        bayer_codes = {
            PIXEL_BAYER_RG8: ("BayerRG8", cv2.COLOR_BayerRG2BGR),
            PIXEL_BAYER_GB8: ("BayerGB8", cv2.COLOR_BayerGB2BGR),
            PIXEL_BAYER_GR8: ("BayerGR8", cv2.COLOR_BayerGR2BGR),
            PIXEL_BAYER_BG8: ("BayerBG8", cv2.COLOR_BayerBG2BGR),
        }
        if pixel_type in bayer_codes:
            name, code = bayer_codes[pixel_type]
            return cv2.cvtColor(raw.reshape((height, width)), code), name
        converted = self._convert_with_mvs(raw, width, height, pixel_type, frame_len)
        if converted is not None:
            return converted, f"MVSConverted:{_ret_hex(pixel_type)}"
        raise CameraBackendError(f"Unsupported HIKROBOT pixel type: {_ret_hex(pixel_type)}")

    def _convert_with_mvs(self, raw, width: int, height: int, pixel_type: int, frame_len: int):
        if not hasattr(self._dll, "MV_CC_ConvertPixelType"):
            return None
        dst_size = int(width) * int(height) * 3
        dst = (c_ubyte * dst_size)()
        src = np.ascontiguousarray(raw)
        param = DirectPixelConvertParam()
        param.nWidth = int(width)
        param.nHeight = int(height)
        param.pSrcData = src.ctypes.data_as(POINTER(c_ubyte))
        param.nSrcDataLen = int(frame_len)
        param.enSrcPixelType = int(pixel_type)
        param.enDstPixelType = PIXEL_BGR8
        param.pDstBuffer = dst
        param.nDstBufferSize = int(dst_size)
        ret = self._dll.MV_CC_ConvertPixelType(self._handle, byref(param))
        if ret != 0:
            self._log(f"[HIKROBOT][DIRECT][CONVERT][WARN] code={_ret_hex(ret)} pixel={_ret_hex(pixel_type)}")
            return None
        out_len = int(param.nDstLen or dst_size)
        out = np.ctypeslib.as_array(dst)[:out_len].copy()
        return out[:dst_size].reshape((height, width, 3))


def _default_capabilities() -> CameraCapabilities:
    return CameraCapabilities(
        exposure=CameraFeatureCapability(True, False, True),
        exposure_auto=CameraFeatureCapability(True, False, True, "Off", available_values=["Off"]),
        gain=CameraFeatureCapability(True, False, True),
        gain_auto=CameraFeatureCapability(True, False, True, "Off", available_values=["Off"]),
        frame_rate=CameraFeatureCapability(True, False, True),
        width=CameraFeatureCapability(True, False, True),
        height=CameraFeatureCapability(True, False, True),
        offset_x=CameraFeatureCapability(True, False, True),
        offset_y=CameraFeatureCapability(True, False, True),
        pixel_format=CameraFeatureCapability(True, False, False),
        trigger_mode=CameraFeatureCapability(True, False, True, "Off", available_values=["Off", "On"]),
        packet_size=CameraFeatureCapability(True, False, True),
        acquisition_mode=CameraFeatureCapability(True, False, True, "Continuous", available_values=["Continuous"]),
    )


def _sdk_search_values(config: Any | None) -> list[str]:
    values: list[str] = []
    for name in ("hikrobot_mvs_sdk_path", "mvs_sdk_path"):
        value = str(getattr(config, name, "") or "").strip()
        if value:
            values.extend(item for item in value.split(os.pathsep) if item)
    values.extend(str(path) for path in (getattr(config, "sdk_paths", ()) or ()) if str(path).strip())
    values.append(os.environ.get("MVS_SDK_PATH", ""))
    return [value for value in dict.fromkeys(values) if value]


def _direct_index(device: CameraDeviceInfo) -> int:
    raw = device.raw_info if isinstance(device.raw_info, dict) else {}
    if "direct_index" in raw:
        return int(raw["direct_index"])
    parts = str(device.device_id or "").split(":")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return 0


def _expected_frame_len(width: int, height: int, pixel_type: int, reported_len: int) -> int:
    pixels = int(width) * int(height)
    if pixels <= 0:
        return 0
    if pixel_type in {PIXEL_MONO8, PIXEL_BAYER_GR8, PIXEL_BAYER_RG8, PIXEL_BAYER_GB8, PIXEL_BAYER_BG8}:
        return pixels
    if pixel_type in {PIXEL_RGB8, PIXEL_BGR8}:
        return pixels * 3
    if 0 < int(reported_len) < 512 * 1024 * 1024:
        return int(reported_len)
    return pixels * 3


def _feature_name(name: str) -> str:
    return {
        "exposure": "ExposureTime",
        "exposure_time": "ExposureTime",
        "gain": "Gain",
        "frame_rate": "AcquisitionFrameRate",
        "width": "Width",
        "height": "Height",
        "offset_x": "OffsetX",
        "offset_y": "OffsetY",
        "pixel_format": "PixelFormat",
        "trigger_mode": "TriggerMode",
        "acquisition_mode": "AcquisitionMode",
    }.get(str(name), str(name))


def _trigger_mode_value(value: Any) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"off", "0"}:
        return 0
    if normalized in {"on", "1"}:
        return 1
    return int(value)


def _acquisition_mode_value(value: Any) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"continuous", "2"}:
        # HIKROBOT MVS uses 2 for AcquisitionMode=Continuous.  Zero is
        # TriggerMode=Off and is not a valid continuous acquisition value.
        return 2
    return int(value)


def _ret_hex(ret: int) -> str:
    return f"0x{int(ret) & 0xFFFFFFFF:08X}"
