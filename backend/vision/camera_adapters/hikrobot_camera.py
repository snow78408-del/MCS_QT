from __future__ import annotations

import os
import platform
import sys
import threading
import time
from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof
from pathlib import Path
from typing import Any, Callable

try:
    import cv2
except Exception:
    cv2 = None
try:
    import numpy as np
except Exception:
    np = None

from ..models import (
    CameraConfig,
    CameraDevice,
    CameraParameterInfo,
    CameraStatus,
    FramePacket,
    HIKROBOT_DEVICE_TYPE,
)
from .base import BaseCameraAdapter, CameraAdapterError


SDK_MISSING_MESSAGE = "HIKROBOT MVS SDK was not detected. Please install MVS client and SDK development components."


def _decode_mvs_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        raw = value.split(b"\x00", 1)[0]
    else:
        try:
            raw = bytes(int(x) & 0xFF for x in value if int(x) != 0)
        except Exception:
            return str(value or "").strip()
    for encoding in ("utf-8", "gbk", "latin1"):
        try:
            return raw.decode(encoding).strip()
        except Exception:
            continue
    return raw.decode("latin1", errors="ignore").strip()


def _ip_from_int(value: int) -> str:
    try:
        n = int(value)
        return ".".join(str((n >> shift) & 0xFF) for shift in (24, 16, 8, 0))
    except Exception:
        return ""


class HikrobotSdkLoader:
    def __init__(
        self,
        configured_path: str = "",
        manual_path: str = "",
        extra_paths: list[str] | tuple[str, ...] | None = None,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.configured_path = configured_path
        self.manual_path = manual_path
        self.extra_paths = tuple(extra_paths or ())
        self._log = logger or (lambda _msg: None)
        self.sdk_root: Path | None = None
        self.import_dir: Path | None = None
        self.dll_path: Path | None = None
        self.dll_handles: list[Any] = []
        self.module: Any | None = None
        self.error = ""
        self.mvs_installation_found = False
        self.python_interface_path: Path | None = None
        self.mv_import_dir: Path | None = None
        self.cti_paths: list[Path] = []

    def candidate_roots(self) -> list[Path]:
        roots: list[Path] = []
        configured_values: list[str] = []
        if self.configured_path:
            configured_values.extend(str(self.configured_path).split(os.pathsep))
        configured_values.extend(str(path) for path in self.extra_paths if path)
        for value in (
            *configured_values,
            os.environ.get("MVS_SDK_PATH", ""),
            r"C:\Program Files (x86)\MVS",
            r"C:\Program Files (x86)\MVS\Development\Samples\Python",
            r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport",
            r"C:\Program Files\MVS",
            r"C:\Program Files\MVS\Development\Samples\Python",
            r"C:\Program Files\MVS\Development\Samples\Python\MvImport",
            r"C:\Program Files (x86)\Common Files\MVS",
            r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64",
            r"C:\Program Files\Common Files\MVS",
            r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64",
            str(Path.cwd() / "third_party" / "hikrobot"),
            self.manual_path,
            r"C:\Program Files",
            r"C:\Program Files (x86)",
        ):
            if value:
                path = Path(value).expanduser()
                if path not in roots:
                    roots.append(path)
        return roots

    def _find_files(self, root: Path, pattern: str) -> list[Path]:
        found: list[Path] = []
        direct = root / pattern
        if direct.exists():
            found.append(direct)
        try:
            found.extend(root.rglob(pattern))
        except Exception:
            pass
        return list(dict.fromkeys(found))

    def _find_file(self, root: Path, filename: str) -> Path | None:
        direct = root / filename
        if direct.exists():
            return direct
        try:
            for path in root.rglob(filename):
                return path
        except Exception:
            return None
        return None

    def find_sdk(self) -> bool:
        control_py = None
        dll_path = None
        mv_import_dir = None
        sdk_root = None
        cti_paths: list[Path] = []
        self.error = ""
        self.mvs_installation_found = False
        for root in self.candidate_roots():
            if not root.exists():
                continue
            root_text = str(root).lower()
            if "mvs" in root_text:
                self.mvs_installation_found = True
            cti_paths.extend(path for path in self._find_files(root, "*.cti") if "mvs" in str(path).lower() or "mv" in path.name.lower())
            if cti_paths:
                self.mvs_installation_found = True
            if control_py is None:
                control_py = self._find_file(root, "MvCameraControl_class.py")
                if control_py is not None:
                    sdk_root = root
                    self.python_interface_path = control_py
                    self.mvs_installation_found = True
            if dll_path is None:
                dll_path = self._find_file(root, "MvCameraControl.dll")
                if dll_path is not None:
                    self.mvs_installation_found = True
            if control_py is not None and mv_import_dir is None:
                if control_py.parent.name == "MvImport":
                    mv_import_dir = control_py.parent
                elif (control_py.parent / "MvImport").exists():
                    mv_import_dir = control_py.parent / "MvImport"
                else:
                    mv_import_dir = control_py.parent
            if control_py and mv_import_dir and dll_path:
                self.sdk_root = sdk_root or root
                self.import_dir = mv_import_dir
                self.mv_import_dir = mv_import_dir
                self.dll_path = dll_path
                self.cti_paths = list(dict.fromkeys(cti_paths))
                return True
        self.cti_paths = list(dict.fromkeys(cti_paths))
        if not self.mvs_installation_found:
            self.error = "MVS software was not found in default locations, MVS_SDK_PATH, or configured SDK paths."
        elif control_py is None:
            self.error = "MVS software was found, but development components or MvCameraControl_class.py were not found."
        elif mv_import_dir is None:
            self.error = "MvCameraControl_class.py was found, but no usable MvImport directory was found."
        elif dll_path is None:
            self.error = "MVS Python interface was found, but MvCameraControl.dll was not found."
        else:
            self.error = SDK_MISSING_MESSAGE
        return False

    def load(self) -> Any:
        if self.module is not None:
            return self.module
        if not self.find_sdk():
            self._log(f"[HIKROBOT][INSTALL] MVS installation found={self.mvs_installation_found}")
            self._log(f"[HIKROBOT][PYTHON] MvCameraControl_class.py path={self.python_interface_path or ''}")
            self._log(f"[HIKROBOT][DLL] MvCameraControl.dll path={self.dll_path or ''}")
            self._log(f"[HIKROBOT][SDK][MISSING] {self.error}")
            raise CameraAdapterError(self.error)
        assert self.import_dir is not None
        assert self.dll_path is not None
        self._log(f"[HIKROBOT][INSTALL] MVS installation found={self.mvs_installation_found}")
        self._log(f"[HIKROBOT][PYTHON] MvCameraControl_class.py path={self.python_interface_path or ''}")
        self._log(f"[HIKROBOT][DLL] MvCameraControl.dll path={self.dll_path}")
        python_arch = platform.architecture()[0]
        dll_text = str(self.dll_path).lower()
        sdk_arch = "32bit" if any(token in dll_text for token in ("x86", "win32", "32")) and not any(token in dll_text for token in ("x64", "win64", "64")) else "64bit"
        if python_arch != sdk_arch and ("x64" in dll_text or "win64" in dll_text or "x86" in dll_text or "win32" in dll_text):
            raise CameraAdapterError(f"Python and industrial camera SDK architectures do not match. Python={python_arch}, SDK={sdk_arch}.")
        search_dirs = {self.import_dir, self.import_dir.parent, self.dll_path.parent}
        for path in search_dirs:
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)
            if hasattr(os, "add_dll_directory"):
                try:
                    self.dll_handles.append(os.add_dll_directory(text))
                except Exception as exc:
                    self.error = f"Python interface was found, but DLL directory failed to load: {text}: {exc}"
                    self._log(f"[HIKROBOT][DLL] load=False path={text} error={exc}")
        try:
            import MvCameraControl_class as mvs  # type: ignore
        except Exception as exc:
            self.error = f"MVS Python interface was found, but importing MvCameraControl_class failed: {exc}"
            self._log("[HIKROBOT][IMPORT] success=False")
            self._log(f"[HIKROBOT][SDK][MISSING] {self.error}")
            raise CameraAdapterError(self.error) from exc
        self.module = mvs
        self._log("[HIKROBOT][IMPORT] success=True")
        self._log(f"[HIKROBOT][SDK] loaded=True python_path={self.import_dir} dll_dir={self.dll_path.parent}")
        return mvs

class HikrobotCameraAdapter(BaseCameraAdapter):
    def __init__(
        self,
        camera_config: CameraConfig | None = None,
        logger: Callable[[str], None] | None = None,
        sdk_path: str = "",
    ) -> None:
        self.config = camera_config or CameraConfig()
        self._log = logger or (lambda _msg: None)
        self._loader = HikrobotSdkLoader(
            configured_path=self.config.mvs_sdk_path or sdk_path,
            logger=self._log,
        )
        self._mvs: Any | None = None
        self._cam: Any | None = None
        self._device: CameraDevice | None = None
        self._device_info: Any | None = None
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._streaming = False
        self._opened = False
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_timestamp = 0.0
        self._frame_id = 0
        self._camera_connected = False
        self._consecutive_frame_failures = 0
        self._last_camera_error = ""
        self._width = 0
        self._height = 0
        self._pixel_format = ""
        self._frame_rate = 0.0
        self._failure_threshold = 10

    @classmethod
    def discover_devices(
        cls,
        sdk_path: str = "",
        logger: Callable[[str], None] | None = None,
    ) -> list[CameraDevice]:
        log = logger or (lambda _msg: None)
        loader = HikrobotSdkLoader(configured_path=sdk_path, logger=log)
        try:
            mvs = loader.load()
        except CameraAdapterError as exc:
            return [
                CameraDevice(
                    device_id="hikrobot:sdk_missing",
                    device_name="HIKROBOT MVS SDK",
                    device_type=HIKROBOT_DEVICE_TYPE,
                    manufacturer="HIKROBOT",
                    available=False,
                    error=str(exc),
                )
            ]

        log("[HIKROBOT][SDK] loaded=True")
        device_list = mvs.MV_CC_DEVICE_INFO_LIST()
        layer_type = int(getattr(mvs, "MV_GIGE_DEVICE", 0x00000001)) | int(getattr(mvs, "MV_USB_DEVICE", 0x00000004))
        log(f"[HIKROBOT][ENUM] transport_mask=0x{int(layer_type) & 0xFFFFFFFF:08X}")
        ret = mvs.MvCamera.MV_CC_EnumDevices(layer_type, device_list)
        log(f"[HIKROBOT][ENUM] return_code=0x{int(ret) & 0xFFFFFFFF:08X}")
        if ret != 0:
            return [
                CameraDevice(
                    device_id="hikrobot:enum_failed",
                    device_name="HIKROBOT MVS SDK",
                    device_type=HIKROBOT_DEVICE_TYPE,
                    manufacturer="HIKROBOT",
                    available=False,
                    error=f"HIKROBOT camera enumeration failed, error_code=0x{int(ret) & 0xFFFFFFFF:08X}",
                    sdk_path=str(loader.sdk_root or ""),
                )
            ]
        log(f"[HIKROBOT][ENUM] device_count={int(device_list.nDeviceNum)}")
        if int(device_list.nDeviceNum) == 0:
            return [
                CameraDevice(
                    device_id="hikrobot:no_devices",
                    device_name="HIKROBOT MVS SDK",
                    device_type=HIKROBOT_DEVICE_TYPE,
                    manufacturer="HIKROBOT",
                    available=False,
                    error="HIKROBOT SDK loaded successfully, but Python enumeration returned 0 devices. Check SDK version, transport mask, Python interface path, and close official preview windows.",
                    sdk_path=str(loader.sdk_root or ""),
                )
            ]
        devices: list[CameraDevice] = []
        info_type = getattr(mvs, "MV_CC_DEVICE_INFO")
        for idx in range(int(device_list.nDeviceNum)):
            try:
                raw_info = cast(device_list.pDeviceInfo[idx], POINTER(info_type)).contents
                device = cls._device_from_mvs_info(mvs, raw_info, idx, str(loader.sdk_root or ""))
                devices.append(device)
                log(
                    "[HIKROBOT][DEVICE][FOUND] "
                    f"model={device.model} sn={device.serial_number} transport={device.transport_type} ip={device.current_ip}"
                )
            except Exception as exc:
                devices.append(
                    CameraDevice(
                        device_id=f"hikrobot:index:{idx}",
                        device_name=f"HIKROBOT Device {idx}",
                        device_type=HIKROBOT_DEVICE_TYPE,
                        manufacturer="HIKROBOT",
                        available=False,
                        error=f"解析海康设备信息失败: {exc}",
                        device_index=idx,
                        sdk_path=str(loader.sdk_root or ""),
                    )
                )
        return devices

    @staticmethod
    def _device_from_mvs_info(mvs: Any, raw_info: Any, index: int, sdk_path: str = "") -> CameraDevice:
        layer_type = int(getattr(raw_info, "nTLayerType", 0))
        gige_type = int(getattr(mvs, "MV_GIGE_DEVICE", 0x00000001))
        usb_type = int(getattr(mvs, "MV_USB_DEVICE", 0x00000004))
        transport = "GigE" if layer_type == gige_type else "USB3" if layer_type == usb_type else str(layer_type)
        info = raw_info.SpecialInfo.stGigEInfo if transport == "GigE" else raw_info.SpecialInfo.stUsb3VInfo
        model = _decode_mvs_text(getattr(info, "chModelName", b""))
        serial = _decode_mvs_text(getattr(info, "chSerialNumber", b""))
        user_name = _decode_mvs_text(getattr(info, "chUserDefinedName", b""))
        manufacturer = _decode_mvs_text(getattr(info, "chManufacturerName", b"")) or "HIKROBOT"
        ip = _ip_from_int(getattr(info, "nCurrentIp", 0)) if transport == "GigE" else ""
        device_id = serial or f"index:{index}"
        display_name = user_name or model or f"HIKROBOT Device {index}"
        return CameraDevice(
            device_id=f"hikrobot:{device_id}",
            device_name=display_name,
            device_type=HIKROBOT_DEVICE_TYPE,
            transport_type=transport,
            manufacturer=manufacturer,
            model=model,
            serial_number=serial,
            user_defined_name=user_name,
            current_ip=ip,
            available=True,
            device_index=index,
            sdk_path=sdk_path,
            raw_info=raw_info,
        )

    def _ensure_sdk(self) -> Any:
        if self._mvs is None:
            self._mvs = self._loader.load()
        return self._mvs

    def _enum_available_devices(self) -> list[tuple[CameraDevice, Any]]:
        mvs = self._ensure_sdk()
        device_list = mvs.MV_CC_DEVICE_INFO_LIST()
        layer_type = int(getattr(mvs, "MV_GIGE_DEVICE", 0x00000001)) | int(getattr(mvs, "MV_USB_DEVICE", 0x00000004))
        ret = mvs.MvCamera.MV_CC_EnumDevices(layer_type, device_list)
        if ret != 0:
            raise CameraAdapterError(f"海康相机枚举失败，错误码: 0x{int(ret) & 0xFFFFFFFF:08X}", int(ret))
        info_type = getattr(mvs, "MV_CC_DEVICE_INFO")
        devices: list[tuple[CameraDevice, Any]] = []
        for idx in range(int(device_list.nDeviceNum)):
            raw_info = cast(device_list.pDeviceInfo[idx], POINTER(info_type)).contents
            devices.append((self._device_from_mvs_info(mvs, raw_info, idx, str(self._loader.sdk_root or "")), raw_info))
        return devices

    def _find_device(self, device_id: str) -> tuple[CameraDevice, Any]:
        key = str(device_id or "").strip()
        if key.startswith("hikrobot:"):
            key = key.split(":", 1)[1]
        for device, raw_info in self._enum_available_devices():
            values = {
                device.device_id,
                device.device_id.removeprefix("hikrobot:"),
                device.serial_number,
                str(device.device_index),
                f"index:{device.device_index}",
            }
            if key in values:
                return device, raw_info
        raise CameraAdapterError(f"未找到海康工业相机: {device_id}")

    def open(self, device_id: str, *, start_stream: bool = True) -> None:
        self.close()
        mvs = self._ensure_sdk()
        device: CameraDevice | None = None
        try:
            device, raw_info = self._find_device(device_id)
            cam = mvs.MvCamera()
            ret = cam.MV_CC_CreateHandle(raw_info)
            if ret != 0:
                raise CameraAdapterError(f"创建海康相机句柄失败，错误码: 0x{int(ret) & 0xFFFFFFFF:08X}", int(ret))
            self._cam = cam
            self._device = device
            self._device_info = raw_info

            exclusive = int(getattr(mvs, "MV_ACCESS_Exclusive", 1))
            ret = cam.MV_CC_OpenDevice(exclusive, 0)
            if ret != 0:
                raise CameraAdapterError(f"打开海康相机失败，错误码: 0x{int(ret) & 0xFFFFFFFF:08X}", int(ret))

            with self._lock:
                self._opened = True
                self._camera_connected = True
                self._last_camera_error = ""

            if device.transport_type == "GigE":
                self._configure_gige_packet_size(cam, device)

            self._refresh_basic_info()
            if not start_stream:
                # The unified manager applies user parameters between open and
                # start_stream. Keep the historical eager behavior by default,
                # while allowing modern adapters to configure safely first.
                self._log("[HIKROBOT][OPEN][OK] handle_ready=True")
                return

            self._set_enum_value("AcquisitionMode", "Continuous", raise_on_fail=False)
            self.set_trigger_mode(self.config.trigger_mode or "Off")
            self.configure(self.config)
            self.start_stream()

            deadline = time.monotonic() + 3.0
            valid = None
            while time.monotonic() < deadline:
                packet = self.read_frame()
                if packet.valid and packet.frame is not None:
                    valid = packet
                    break
                time.sleep(0.03)
            if valid is None:
                raise CameraAdapterError("海康工业相机打开后未获取到有效测试帧")

            self._log(
                "[HIKROBOT][OPEN][OK] "
                f"model={device.model} sn={device.serial_number} transport={device.transport_type} ip={device.current_ip}"
            )
        except Exception as exc:
            err = str(exc)
            code = getattr(exc, "error_code", None)
            if device is not None:
                self._log(
                    "[HIKROBOT][OPEN][FAIL] "
                    f"model={device.model} sn={device.serial_number} transport={device.transport_type} "
                    f"ip={device.current_ip} code={code} error={err}"
                )
            self.close()
            if isinstance(exc, CameraAdapterError):
                raise
            raise CameraAdapterError(err) from exc

    def _configure_gige_packet_size(self, cam: Any, device: CameraDevice) -> None:
        try:
            packet_size = int(cam.MV_CC_GetOptimalPacketSize())
            if packet_size > 0:
                ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
                if ret != 0:
                    self._log(
                        "[HIKROBOT][FRAME][FAIL] "
                        f"model={device.model} sn={device.serial_number} ip={device.current_ip} "
                        "设置推荐网络包大小失败；取流异常时请检查相机与网卡是否位于同一网段"
                    )
        except Exception as exc:
            self._log(f"[HIKROBOT][FRAME][FAIL] 获取 GigE 推荐包大小失败: {exc}")

    def close(self) -> None:
        self.stop_stream()
        cam = self._cam
        device = self._device
        if cam is not None:
            try:
                cam.MV_CC_CloseDevice()
            except Exception:
                pass
            try:
                cam.MV_CC_DestroyHandle()
            except Exception:
                pass
        with self._lock:
            self._cam = None
            self._device = None
            self._device_info = None
            self._opened = False
            self._camera_connected = False
            self._latest_frame = None
            self._streaming = False
        if device is not None:
            self._log(
                "[HIKROBOT][CLOSE] "
                f"model={device.model} sn={device.serial_number} transport={device.transport_type} ip={device.current_ip}"
            )

    def start_stream(self) -> None:
        with self._lock:
            if self._streaming and self._worker is not None and self._worker.is_alive():
                return
            cam = self._cam
            opened = self._opened
        if cam is None or not opened:
            raise CameraAdapterError("海康工业相机未打开")
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise CameraAdapterError(f"海康工业相机启动取流失败，错误码: 0x{int(ret) & 0xFFFFFFFF:08X}", int(ret))
        self._stop_event.clear()
        with self._lock:
            self._streaming = True
            self._camera_connected = True
            self._consecutive_frame_failures = 0
            self._last_camera_error = ""
            self._worker = threading.Thread(target=self._grab_loop, name="hikrobot-camera-grab", daemon=True)
            self._worker.start()
        device = self._device
        if device is not None:
            self._log(
                "[HIKROBOT][STREAM][START] "
                f"model={device.model} sn={device.serial_number} transport={device.transport_type} ip={device.current_ip}"
            )

    def stop_stream(self) -> None:
        self._stop_event.set()
        worker = self._worker
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=1.5)
        cam = self._cam
        if cam is not None:
            try:
                cam.MV_CC_StopGrabbing()
            except Exception:
                pass
        with self._lock:
            self._worker = None
            self._streaming = False

    def _grab_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame, pixel_format = self._capture_one_frame(timeout_ms=1000)
                now = time.time()
                with self._lock:
                    self._frame_id += 1
                    self._latest_frame = frame
                    self._latest_frame_timestamp = now
                    self._camera_connected = True
                    self._consecutive_frame_failures = 0
                    self._last_camera_error = ""
                    self._height = int(frame.shape[0])
                    self._width = int(frame.shape[1])
                    self._pixel_format = pixel_format
                if self._frame_id == 1:
                    device = self._device
                    if device is not None:
                        self._log(
                            "[HIKROBOT][FRAME][OK] "
                            f"model={device.model} sn={device.serial_number} transport={device.transport_type} ip={device.current_ip}"
                        )
            except Exception as exc:
                with self._lock:
                    self._consecutive_frame_failures += 1
                    self._last_camera_error = str(exc)
                    failures = self._consecutive_frame_failures
                    device = self._device
                    if failures >= self._failure_threshold:
                        self._camera_connected = False
                        self._latest_frame = None
                        self._streaming = False
                        self._stop_event.set()
                if device is not None:
                    message = (
                        "[HIKROBOT][DISCONNECTED] "
                        if failures >= self._failure_threshold
                        else "[HIKROBOT][FRAME][FAIL] "
                    )
                    self._log(
                        f"{message}model={device.model} sn={device.serial_number} "
                        f"transport={device.transport_type} ip={device.current_ip} error={exc}"
                    )
                    if device.transport_type == "GigE" and failures >= self._failure_threshold:
                        self._log("[HIKROBOT][FRAME][FAIL] GigE 连续取流失败，请检查相机与网卡是否位于同一网段")
                time.sleep(0.03)

    def read_frame(self) -> FramePacket:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with self._lock:
                if not self._camera_connected:
                    return FramePacket(
                        frame=None,
                        timestamp=time.time(),
                        frame_id=self._frame_id,
                        valid=False,
                        error=self._last_camera_error or "海康工业相机连接中断",
                    )
                if self._latest_frame is not None:
                    return FramePacket(
                        frame=self._latest_frame.copy(),
                        timestamp=self._latest_frame_timestamp,
                        frame_id=self._frame_id,
                    )
                error = self._last_camera_error
            time.sleep(0.02)
        return FramePacket(
            frame=None,
            timestamp=time.time(),
            frame_id=self._frame_id,
            valid=False,
            error=error or "Timed out waiting for HIKROBOT camera frame",
        )

    def test_camera(self, device_id: str, frame_count: int = 3):
        from dataclasses import asdict

        del asdict
        required = max(3, int(frame_count))
        try:
            self.open(device_id)
            self.set_trigger_mode("Off")
            self.start_stream()
            packets: list[FramePacket] = []
            seen_ids: set[int] = set()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and len(packets) < required:
                packet = self.read_frame()
                if packet.valid and packet.frame is not None and packet.frame_id not in seen_ids:
                    frame = packet.frame
                    if int(frame.shape[0]) > 0 and int(frame.shape[1]) > 0 and int(frame.size) > 0:
                        packets.append(packet)
                        seen_ids.add(int(packet.frame_id))
                time.sleep(0.02)
            if len(packets) < required:
                return __import__("backend.vision.models", fromlist=["CameraTestResult"]).CameraTestResult(
                    ok=False,
                    device_id=device_id,
                    error=f"Camera test failed: required {required} valid frames, got {len(packets)}",
                    frames_read=len(packets),
                )
            frame = packets[-1].frame
            status = self.get_status()
            return __import__("backend.vision.models", fromlist=["CameraTestResult"]).CameraTestResult(
                ok=True,
                device_id=device_id,
                message="Camera test succeeded",
                width=int(frame.shape[1]),
                height=int(frame.shape[0]),
                pixel_format=status.pixel_format,
                frame_rate=status.frame_rate,
                frames_read=len(packets),
            )
        except CameraAdapterError as exc:
            return __import__("backend.vision.models", fromlist=["CameraTestResult"]).CameraTestResult(
                ok=False,
                device_id=device_id,
                error=str(exc),
                error_code=exc.error_code,
            )
        except Exception as exc:
            return __import__("backend.vision.models", fromlist=["CameraTestResult"]).CameraTestResult(
                ok=False,
                device_id=device_id,
                error=str(exc),
            )
        finally:
            self.stop_stream()
            self.close()

    def _capture_one_frame(self, timeout_ms: int = 1000) -> tuple[np.ndarray, str]:
        cam = self._cam
        mvs = self._ensure_sdk()
        if cam is None:
            raise CameraAdapterError("HIKROBOT camera handle is not available")
        frame_out = mvs.MV_FRAME_OUT()
        memset(byref(frame_out), 0, sizeof(frame_out))
        ret = cam.MV_CC_GetImageBuffer(frame_out, int(timeout_ms))
        if ret != 0:
            raise CameraAdapterError(f"HIKROBOT camera frame grab failed, error_code=0x{int(ret) & 0xFFFFFFFF:08X}", int(ret))
        try:
            info = frame_out.stFrameInfo
            width = int(info.nWidth)
            height = int(info.nHeight)
            frame_len = int(info.nFrameLen)
            pixel_type = int(info.enPixelType)
            if width <= 0 or height <= 0 or frame_len <= 0:
                raise CameraAdapterError("HIKROBOT camera returned invalid frame metadata")
            raw = np.ctypeslib.as_array(frame_out.pBufAddr, shape=(frame_len,)).copy()
            return self._convert_frame(raw, width, height, pixel_type, frame_len)
        finally:
            try:
                cam.MV_CC_FreeImageBuffer(frame_out)
            except Exception:
                pass

    def _pixel_constants(self) -> dict[str, int]:
        mvs = self._ensure_sdk()
        return {
            "Mono8": int(getattr(mvs, "PixelType_Gvsp_Mono8", 0x01080001)),
            "BayerGR8": int(getattr(mvs, "PixelType_Gvsp_BayerGR8", 0x01080008)),
            "BayerRG8": int(getattr(mvs, "PixelType_Gvsp_BayerRG8", 0x01080009)),
            "BayerGB8": int(getattr(mvs, "PixelType_Gvsp_BayerGB8", 0x0108000A)),
            "BayerBG8": int(getattr(mvs, "PixelType_Gvsp_BayerBG8", 0x0108000B)),
            "RGB8": int(getattr(mvs, "PixelType_Gvsp_RGB8_Packed", 0x02180014)),
            "BGR8": int(getattr(mvs, "PixelType_Gvsp_BGR8_Packed", 0x02180015)),
        }

    def _convert_frame(
        self,
        raw: np.ndarray,
        width: int,
        height: int,
        pixel_type: int,
        frame_len: int,
    ) -> tuple[np.ndarray, str]:
        constants = self._pixel_constants()
        if pixel_type == constants["Mono8"]:
            return raw.reshape((height, width)), "Mono8"
        if pixel_type in {
            constants["BayerRG8"],
            constants["BayerGB8"],
            constants["BayerGR8"],
            constants["BayerBG8"],
        }:
            bayer = raw.reshape((height, width))
            code_map = {
                constants["BayerRG8"]: cv2.COLOR_BayerRG2BGR,
                constants["BayerGB8"]: cv2.COLOR_BayerGB2BGR,
                constants["BayerGR8"]: cv2.COLOR_BayerGR2BGR,
                constants["BayerBG8"]: cv2.COLOR_BayerBG2BGR,
            }
            name = next(k for k, v in constants.items() if v == pixel_type)
            return cv2.cvtColor(bayer, code_map[pixel_type]), name
        if pixel_type == constants["RGB8"]:
            rgb = raw.reshape((height, width, 3))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), "RGB8"
        if pixel_type == constants["BGR8"]:
            return raw.reshape((height, width, 3)), "BGR8"
        converted = self._convert_unsupported_pixel_type(raw, width, height, pixel_type, frame_len)
        if converted is None:
            raise CameraAdapterError(f"不支持的海康像素格式，且 MVS 像素格式转换失败: 0x{pixel_type:08X}")
        return converted, f"MVSConverted:0x{pixel_type:08X}"

    def _convert_unsupported_pixel_type(
        self,
        raw: np.ndarray,
        width: int,
        height: int,
        pixel_type: int,
        frame_len: int,
    ) -> np.ndarray | None:
        cam = self._cam
        mvs = self._ensure_sdk()
        convert_param_type = getattr(mvs, "MV_CC_PIXEL_CONVERT_PARAM", None)
        if cam is None or convert_param_type is None:
            return None
        try:
            dst_size = width * height * 3
            dst = (c_ubyte * dst_size)()
            src = np.ascontiguousarray(raw)
            param = convert_param_type()
            param.nWidth = width
            param.nHeight = height
            param.pSrcData = src.ctypes.data_as(POINTER(c_ubyte))
            param.nSrcDataLen = int(frame_len)
            param.enSrcPixelType = int(pixel_type)
            param.enDstPixelType = int(self._pixel_constants()["BGR8"])
            param.pDstBuffer = dst
            param.nDstBufferSize = dst_size
            ret = cam.MV_CC_ConvertPixelType(param)
            if ret != 0:
                return None
            out = np.ctypeslib.as_array(dst, shape=(dst_size,)).copy()
            return out.reshape((height, width, 3))
        except Exception:
            return None

    def get_device_info(self) -> CameraDevice | None:
        with self._lock:
            return self._device

    def _set_enum_value(self, name: str, value: str, raise_on_fail: bool = True) -> None:
        cam = self._cam
        if cam is None:
            return
        enum_value = 0 if str(value).lower() in {"off", "continuous"} else 1
        mvs = self._ensure_sdk()
        if name == "TriggerMode" and str(value).lower() == "off":
            enum_value = int(getattr(mvs, "MV_TRIGGER_MODE_OFF", 0))
        elif name == "TriggerMode":
            enum_value = int(getattr(mvs, "MV_TRIGGER_MODE_ON", 1))
        ret = cam.MV_CC_SetEnumValue(name, enum_value)
        if ret != 0 and raise_on_fail:
            raise CameraAdapterError(f"设置海康相机参数 {name}={value} 失败，错误码: 0x{int(ret) & 0xFFFFFFFF:08X}", int(ret))
        if ret == 0:
            self._log(f"[HIKROBOT][PARAM][SET] {name}={value}")

    def _set_float_value(self, name: str, value: float) -> None:
        cam = self._cam
        if cam is None:
            return
        info = self.get_parameter_info(name)
        if info.exists and info.writable:
            if info.min_value is not None and value < float(info.min_value):
                raise CameraAdapterError(f"{name} 超出允许范围: {info.min_value} - {info.max_value}")
            if info.max_value is not None and value > float(info.max_value):
                raise CameraAdapterError(f"{name} 超出允许范围: {info.min_value} - {info.max_value}")
        ret = cam.MV_CC_SetFloatValue(name, float(value))
        if ret != 0:
            raise CameraAdapterError(f"设置海康相机参数 {name}={value} 失败，错误码: 0x{int(ret) & 0xFFFFFFFF:08X}", int(ret))
        self._log(f"[HIKROBOT][PARAM][SET] {name}={value}")

    def _set_int_value(self, name: str, value: int) -> None:
        cam = self._cam
        if cam is None:
            return
        info = self.get_parameter_info(name)
        if info.exists and info.writable:
            if info.min_value is not None and value < int(info.min_value):
                raise CameraAdapterError(f"{name} 超出允许范围: {info.min_value} - {info.max_value}")
            if info.max_value is not None and value > int(info.max_value):
                raise CameraAdapterError(f"{name} 超出允许范围: {info.min_value} - {info.max_value}")
        ret = cam.MV_CC_SetIntValue(name, int(value))
        if ret != 0:
            raise CameraAdapterError(f"设置海康相机参数 {name}={value} 失败，错误码: 0x{int(ret) & 0xFFFFFFFF:08X}", int(ret))
        self._log(f"[HIKROBOT][PARAM][SET] {name}={value}")

    def get_parameter_info(self, name: str) -> CameraParameterInfo:
        cam = self._cam
        mvs = self._ensure_sdk()
        if cam is None:
            return CameraParameterInfo(name=name, exists=False, error="相机未打开")
        for type_name, method_name in (
            ("MVCC_FLOATVALUE", "MV_CC_GetFloatValue"),
            ("MVCC_INTVALUE_EX", "MV_CC_GetIntValueEx"),
            ("MVCC_INTVALUE", "MV_CC_GetIntValue"),
        ):
            value_type = getattr(mvs, type_name, None)
            method = getattr(cam, method_name, None)
            if value_type is None or method is None:
                continue
            try:
                value = value_type()
                ret = method(name, value)
                if ret == 0:
                    return CameraParameterInfo(
                        name=name,
                        exists=True,
                        readable=True,
                        writable=True,
                        min_value=getattr(value, "fMin", getattr(value, "nMin", None)),
                        max_value=getattr(value, "fMax", getattr(value, "nMax", None)),
                        current_value=getattr(value, "fCurValue", getattr(value, "nCurValue", None)),
                    )
            except Exception:
                continue
        return CameraParameterInfo(name=name, exists=False, error="Parameter does not exist or is not accessible")

    def set_exposure(self, exposure_time: float) -> None:
        self._set_enum_value("ExposureAuto", "Off", raise_on_fail=False)
        self._set_float_value("ExposureTime", exposure_time)

    def set_gain(self, gain: float) -> None:
        self._set_enum_value("GainAuto", "Off", raise_on_fail=False)
        self._set_float_value("Gain", gain)

    def set_frame_rate(self, frame_rate: float) -> None:
        self._set_enum_value("AcquisitionFrameRateEnable", "On", raise_on_fail=False)
        self._set_float_value("AcquisitionFrameRate", frame_rate)
        with self._lock:
            self._frame_rate = float(frame_rate)

    def set_resolution(
        self,
        width: int | None = None,
        height: int | None = None,
        offset_x: int | None = None,
        offset_y: int | None = None,
    ) -> None:
        if offset_x is not None:
            self._set_int_value("OffsetX", int(offset_x))
        if offset_y is not None:
            self._set_int_value("OffsetY", int(offset_y))
        if width is not None:
            self._set_int_value("Width", int(width))
        if height is not None:
            self._set_int_value("Height", int(height))

    def set_trigger_mode(self, trigger_mode: str) -> None:
        self._set_enum_value("TriggerMode", trigger_mode or "Off")

    def set_acquisition_mode(self, acquisition_mode: str) -> None:
        self._set_enum_value("AcquisitionMode", acquisition_mode or "Continuous")

    def _refresh_basic_info(self) -> None:
        for name, attr in (("Width", "_width"), ("Height", "_height")):
            info = self.get_parameter_info(name)
            if info.current_value is not None:
                setattr(self, attr, int(info.current_value))
        fps = self.get_parameter_info("AcquisitionFrameRate")
        if fps.current_value is not None:
            self._frame_rate = float(fps.current_value)
        pixel = self.get_parameter_info("PixelFormat")
        if pixel.current_value is not None:
            self._pixel_format = str(pixel.current_value)

    def is_open(self) -> bool:
        with self._lock:
            return bool(self._opened)

    def is_streaming(self) -> bool:
        with self._lock:
            return bool(self._streaming)

    def get_status(self) -> CameraStatus:
        with self._lock:
            device = self._device
            return CameraStatus(
                device_id=device.device_id if device else "",
                device_type=HIKROBOT_DEVICE_TYPE,
                is_open=self._opened,
                is_streaming=self._streaming,
                camera_connected=self._camera_connected,
                latest_frame_timestamp=self._latest_frame_timestamp,
                frame_id=self._frame_id,
                consecutive_frame_failures=self._consecutive_frame_failures,
                last_camera_error=self._last_camera_error,
                width=self._width,
                height=self._height,
                pixel_format=self._pixel_format,
                frame_rate=self._frame_rate,
            )
