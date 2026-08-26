from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from ..base import BaseCameraAdapter, CameraBackendError
from ..frame_converter import make_frame_data
from ..models import (
    CameraBackendStatus,
    CameraCapabilities,
    CameraDeviceInfo,
    CameraFeatureCapability,
    DEVICE_TYPE_INDUSTRIAL,
    FrameData,
    TRANSPORT_GENTL,
)


class GenTLCameraAdapter(BaseCameraAdapter):
    backend_name = "gentl"
    supported_manufacturers = ("GenICam", "GenTL")
    backend_priority = 30

    def __init__(self, config: Any | None = None, logger=None) -> None:
        super().__init__(config=config, logger=logger)
        self._harvester = None
        self._ia = None
        self._device: CameraDeviceInfo | None = None
        self._latest = FrameData(None, 0, 0.0, source_backend=self.backend_name)
        self._frame_id = 0
        self._streaming = False

    @classmethod
    def is_backend_available(cls, config: Any | None = None) -> tuple[bool, str]:
        try:
            from harvesters.core import Harvester  # noqa: F401
        except Exception as exc:
            return False, f"Harvester/GenTL 不可用: {exc}"
        paths = _cti_paths(config)
        return bool(paths), "GenTL Producer 可用" if paths else "未找到 GenTL Producer .cti 文件"

    @classmethod
    def check_backend(cls, config: Any | None = None, logger=None) -> CameraBackendStatus:
        paths = [str(path) for path in _cti_paths(config)]
        try:
            from harvesters.core import Harvester  # noqa: F401
            module_loaded = True
            error = ""
        except Exception as exc:
            module_loaded = False
            error = f"Harvester/GenTL 不可用: {exc}"
        if logger:
            logger(f"[GENTL][PYTHON] harvesters import={module_loaded}")
            logger(f"[GENTL][CTI] found_count={len(paths)}")
        if module_loaded and not paths:
            error = "已安装厂商软件，但项目未加载厂商GenTL Producer的CTI文件。"
        return CameraBackendStatus(
            backend_name=cls.backend_name,
            display_name="GenTL",
            sdk_loaded=module_loaded,
            python_module_loaded=module_loaded,
            cti_loaded=bool(paths),
            cti_paths=paths,
            backend_available=bool(module_loaded and paths),
            error=error,
        )

    @classmethod
    def get_backend_diagnostics(cls, config: Any | None = None) -> dict[str, Any]:
        return cls.check_backend(config).to_dict()

    @classmethod
    def discover_devices(cls, config: Any | None = None, logger=None) -> list[CameraDeviceInfo]:
        from harvesters.core import Harvester

        xml_cache_dir = _prepare_harvesters_xml_dir(config)
        h = Harvester()
        paths = _cti_paths(config)
        loaded_count = 0
        if logger:
            logger("[GENTL][PYTHON] harvesters import=True")
            logger(f"[GENTL][XML] cache_dir={xml_cache_dir}")
            logger(f"[CAMERA][GENTL][CTI] found={';'.join(str(p) for p in paths)}")
        for path in paths:
            try:
                h.add_file(str(path))
                loaded_count += 1
                if logger:
                    logger(f"[CAMERA][GENTL][CTI] loaded=True path={path}")
            except Exception as exc:
                if logger:
                    logger(f"[CAMERA][GENTL][CTI][FAILED] path={path} error={exc}")
        h.update()
        device_count = len(h.device_info_list)
        if logger:
            logger(f"[GENTL][CTI] loaded_count={loaded_count}")
            logger(f"[GENTL][ENUM] device_count={device_count}")
            logger(f"[CAMERA][GENTL][ENUM] device_count={device_count}")
        devices: list[CameraDeviceInfo] = []
        for index, info in enumerate(h.device_info_list):
            manufacturer = str(getattr(info, "vendor", "") or getattr(info, "manufacturer", "") or "GenTL")
            model = str(getattr(info, "model", "") or "")
            serial = str(getattr(info, "serial_number", "") or getattr(info, "serial", "") or "")
            producer = str(getattr(info, "tl_type", "") or "")
            unique = _unique_id(manufacturer, model, serial, producer or f"index:{index}")
            devices.append(
                CameraDeviceInfo(
                    device_id=f"gentl:{index}",
                    unique_id=unique,
                    backend_name=cls.backend_name,
                    manufacturer=manufacturer,
                    model=model,
                    serial_number=serial,
                    device_type=DEVICE_TYPE_INDUSTRIAL,
                    transport_type=TRANSPORT_GENTL,
                    gentl_producer=producer,
                    available=True,
                    capabilities=_empty_caps(),
                    available_backends=[cls.backend_name],
                    selected_backend=cls.backend_name,
                    backend_priority=cls.backend_priority,
                    raw_info=info,
                )
            )
        h.reset()
        return devices

    def open(self, device_info: CameraDeviceInfo) -> None:
        from harvesters.core import Harvester

        self.close()
        _prepare_harvesters_xml_dir(self.config)
        self._harvester = Harvester()
        try:
            paths = _cti_paths(self.config)
            if not paths:
                raise CameraBackendError("No GenTL Producer .cti file was found.")
            for path in paths:
                self._harvester.add_file(str(path))
            self._harvester.update()
            index = _resolve_device_index(device_info, self._harvester.device_info_list)
            self._ia = self._harvester.create(index)
            self._device = device_info
            self._device.capabilities = self.get_capabilities()
        except CameraBackendError:
            self.close()
            raise
        except IndexError as exc:
            self.close()
            raise CameraBackendError(
                "The selected GenTL camera is no longer available. Refresh the camera list and try again."
            ) from exc
        except UnicodeDecodeError as exc:
            self.close()
            raise CameraBackendError(_gentl_xml_encoding_error(exc)) from exc
        except Exception as exc:
            self.close()
            raise CameraBackendError(f"Failed to open GenTL camera: {exc}") from exc

    def close(self) -> None:
        self.stop_stream()
        if self._ia is not None:
            try:
                self._ia.destroy()
            except Exception:
                pass
        if self._harvester is not None:
            try:
                self._harvester.reset()
            except Exception:
                pass
        self._ia = None
        self._harvester = None
        self._device = None
        self._log("[CAMERA][CLOSE] backend=gentl")

    def start_stream(self) -> None:
        if self._ia is None:
            raise CameraBackendError("GenTL 相机未打开")
        self._ia.start()
        self._streaming = True
        self._log("[CAMERA][STREAM][START] backend=gentl")

    def stop_stream(self) -> None:
        if self._ia is not None and self._streaming:
            try:
                self._ia.stop()
            except Exception:
                pass
        self._streaming = False

    def read_frame(self, timeout_ms: int = 1000) -> FrameData:
        if self._ia is None or not self._streaming:
            return FrameData(None, self._frame_id, time.time(), source_backend=self.backend_name, valid=False, error="GenTL 未开始采集")
        try:
            with self._ia.fetch(timeout=timeout_ms / 1000.0) as buffer:
                component = buffer.payload.components[0]
                width = int(component.width)
                height = int(component.height)
                fmt = str(getattr(component, "data_format", "") or "Mono8")
                self._frame_id += 1
                self._latest = make_frame_data(
                    component.data,
                    width,
                    height,
                    fmt,
                    self._frame_id,
                    self.backend_name,
                    self._device.unique_id if self._device else "",
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
        node_map = getattr(getattr(self._ia, "remote_device", None), "node_map", None)
        return _caps_for_node_map(node_map) if node_map is not None else _empty_caps()

    def get_feature(self, name: str) -> Any:
        feature = _feature_node(self._ia, name)
        if feature is None or not _feature_readable(feature):
            return None
        return feature.value

    def set_feature(self, name: str, value: Any) -> None:
        feature = _feature_node(self._ia, name)
        if feature is None or not _feature_writable(feature) or not _feature_readable(feature):
            raise CameraBackendError(f"GenTL 参数 {name} 不支持可验证写入")
        try:
            feature.value = value
        except Exception as exc:
            self._last_error = str(exc)
            raise CameraBackendError(f"GenTL 参数 {name} 写入失败: {exc}") from exc

    def is_open(self) -> bool:
        return self._ia is not None

    def is_streaming(self) -> bool:
        return self._streaming


def _cti_paths(config: Any | None) -> list[Path]:
    values: list[str] = []
    env = os.environ.get("GENICAM_GENTL64_PATH", "")
    if env:
        values.extend(env.split(os.pathsep))
    values.extend(getattr(config, "gentl_producer_paths", []) or [])
    values.extend(
        [
            r"C:\Program Files\Basler\pylon\Runtime\x64",
            r"C:\Program Files\Teledyne\Spinnaker\cti64",
            r"C:\Program Files\Allied Vision\Vimba X\cti",
            r"C:\Program Files\Daheng Imaging\GalaxySDK\GenTL\Win64",
            r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64",
            r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64",
        ]
    )
    paths: list[Path] = []
    for value in values:
        path = Path(str(value)).expanduser()
        try:
            if path.is_file() and path.suffix.lower() == ".cti":
                paths.append(path)
            elif path.is_dir():
                paths.extend(path.rglob("*.cti"))
        except Exception:
            continue
    return list(dict.fromkeys(paths))


def _prepare_harvesters_xml_dir(config: Any | None) -> Path:
    configured = getattr(config, "gentl_xml_cache_dir", "") or os.environ.get("HARVESTERS_XML_FILE_DIR", "")
    cache_dir = Path(configured).expanduser() if configured else Path.cwd() / ".camera_cache" / "gentl_xml"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HARVESTERS_XML_FILE_DIR"] = str(cache_dir)
    os.environ["TMP"] = str(cache_dir)
    os.environ["TEMP"] = str(cache_dir)
    tempfile.tempdir = str(cache_dir)
    _patch_tempfile_mkdtemp_for_harvesters(cache_dir)
    _patch_harvesters_cleanup()
    return cache_dir


def _patch_tempfile_mkdtemp_for_harvesters(cache_dir: Path) -> None:
    if getattr(tempfile, "_camera_gentl_mkdtemp_patched", False):
        return
    original_mkdtemp = tempfile.mkdtemp

    def _camera_mkdtemp(suffix: str | None = None, prefix: str | None = None, dir: str | None = None) -> str:
        base_dir = Path(dir).expanduser() if dir else cache_dir
        base_dir.mkdir(parents=True, exist_ok=True)
        name_prefix = prefix or "tmp"
        name_suffix = suffix or ""
        for _ in range(100):
            candidate = base_dir / f"{name_prefix}{uuid.uuid4().hex}{name_suffix}"
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                return str(candidate)
            except FileExistsError:
                continue
        return original_mkdtemp(suffix=suffix or "", prefix=prefix or "tmp", dir=str(base_dir))

    tempfile._camera_gentl_original_mkdtemp = original_mkdtemp  # type: ignore[attr-defined]
    tempfile.mkdtemp = _camera_mkdtemp  # type: ignore[assignment]
    tempfile._camera_gentl_mkdtemp_patched = True  # type: ignore[attr-defined]


def _patch_harvesters_cleanup() -> None:
    try:
        import harvesters.core as harvesters_core
    except Exception:
        return
    module_cls = getattr(harvesters_core, "Module", None)
    if module_cls is None or getattr(module_cls, "_camera_gentl_cleanup_patched", False):
        return
    original_remove = module_cls._remove_intermediate_file
    original_retrieve = module_cls._retrieve_file_path

    def _safe_remove_intermediate_file(file_path: str) -> None:
        try:
            original_remove(file_path)
        except OSError:
            return

    def _safe_retrieve_file_path(*args, **kwargs):
        try:
            return original_retrieve(*args, **kwargs)
        except UnicodeDecodeError:
            return False, None

    module_cls._camera_gentl_original_remove_intermediate_file = original_remove
    module_cls._camera_gentl_original_retrieve_file_path = original_retrieve
    module_cls._remove_intermediate_file = staticmethod(_safe_remove_intermediate_file)
    module_cls._retrieve_file_path = staticmethod(_safe_retrieve_file_path)
    module_cls._camera_gentl_cleanup_patched = True


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


def _feature_node(ia: Any, name: str) -> Any:
    node_map = getattr(getattr(ia, "remote_device", None), "node_map", None)
    if node_map is None:
        return None
    node_name = _FEATURE_NODES.get(name, name)
    feature = getattr(node_map, node_name, None)
    if feature is None:
        getter = getattr(node_map, "get_node_by_name", None)
        if callable(getter):
            try:
                feature = getter(node_name)
            except Exception:
                return None
    return feature


def _feature_readable(feature: Any) -> bool:
    for name in ("is_readable", "is_readable_node"):
        status = getattr(feature, name, None)
        if callable(status):
            try:
                return bool(status())
            except Exception:
                return False
        if status is not None:
            return bool(status)
    return bool(getattr(feature, "readable", hasattr(feature, "value")))


def _feature_writable(feature: Any) -> bool:
    for name in ("is_writable", "is_writeable", "is_writable_node"):
        status = getattr(feature, name, None)
        if callable(status):
            try:
                return bool(status())
            except Exception:
                return False
        if status is not None:
            return bool(status)
    return bool(getattr(feature, "writable", hasattr(feature, "value")))


def _caps_for_node_map(node_map: Any) -> CameraCapabilities:
    values = {}
    for name in CameraCapabilities.__dataclass_fields__:
        feature = None
        if node_map is not None:
            node_name = _FEATURE_NODES.get(name, name)
            feature = getattr(node_map, node_name, None)
            if feature is None:
                getter = getattr(node_map, "get_node_by_name", None)
                if callable(getter):
                    try:
                        feature = getter(node_name)
                    except Exception:
                        feature = None
        readable = feature is not None and _feature_readable(feature)
        writable = readable and _feature_writable(feature)
        current = None
        if readable:
            try:
                current = feature.value
            except Exception:
                readable = False
                writable = False
        values[name] = CameraFeatureCapability(
            supported=bool(feature is not None and (readable or writable)),
            readable=readable,
            writable=writable,
            current_value=current,
            error="GenTL 参数不可验证写入" if feature is not None and not writable else ("GenTL 参数不可用" if feature is None else ""),
        )
    return CameraCapabilities(**values)


def _empty_caps() -> CameraCapabilities:
    return CameraCapabilities(**{
        name: CameraFeatureCapability(False, False, False, error="GenTL 参数能力需在相机打开后验证")
        for name in CameraCapabilities.__dataclass_fields__
    })


def _unique_id(manufacturer: str, model: str, serial: str, fallback: str) -> str:
    if manufacturer and model and serial:
        return f"{manufacturer}:{model}:{serial}"
    return f"gentl:{fallback}"


def _gentl_xml_encoding_error(exc: UnicodeDecodeError) -> str:
    return (
        "The GenTL camera description XML is not valid UTF-8, so Harvester cannot open this camera. "
        "Try the camera vendor backend/SDK first, or update/reinstall the vendor GenTL Producer (.cti). "
        f"Original decode error: {exc}"
    )


def _resolve_device_index(device_info: CameraDeviceInfo, device_info_list: Any) -> int:
    devices = list(device_info_list or [])
    if not devices:
        raise CameraBackendError("No GenTL camera was found. Check the camera connection, then refresh the device list.")

    selected_serial = str(device_info.serial_number or "")
    selected_unique_id = str(device_info.unique_id or "")
    for index, info in enumerate(devices):
        manufacturer = str(getattr(info, "vendor", "") or getattr(info, "manufacturer", "") or "GenTL")
        model = str(getattr(info, "model", "") or "")
        serial = str(getattr(info, "serial_number", "") or getattr(info, "serial", "") or "")
        producer = str(getattr(info, "tl_type", "") or "")
        if selected_serial and serial == selected_serial:
            return index
        if selected_unique_id and _unique_id(manufacturer, model, serial, producer or f"index:{index}") == selected_unique_id:
            return index

    try:
        selected_index = int(str(device_info.device_id).split(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise CameraBackendError(f"Invalid GenTL device id: {device_info.device_id}") from exc
    if 0 <= selected_index < len(devices):
        return selected_index

    raise CameraBackendError(
        f"The selected GenTL camera index {selected_index} is unavailable; {len(devices)} camera(s) were found. "
        "Refresh the camera list and try again."
    )
