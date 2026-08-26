from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any, Callable

from .config import CameraDiscoveryConfig, CameraSystemConfig
from .cameras.manager import CameraManager
from .cameras.models import FrameData


def _coerce_camera_system_config(config: Any | None) -> CameraSystemConfig:
    if isinstance(config, CameraSystemConfig):
        return config
    result = CameraSystemConfig()
    if config is None:
        return result
    # Keep custom config objects compatible while ensuring every current
    # CameraSystemConfig field survives service coercion.
    for item in fields(result):
        if hasattr(config, item.name):
            setattr(result, item.name, getattr(config, item.name))
    return result


class VisionCameraService:
    def __init__(
        self,
        camera_config: CameraSystemConfig | CameraDiscoveryConfig | None = None,
        logger: Callable[[str], None] | None = None,
        discovery_config: CameraDiscoveryConfig | None = None,
    ) -> None:
        self._log = logger or (lambda _msg: None)
        self.camera_config = _coerce_camera_system_config(camera_config or discovery_config)
        self.discovery_config = self.camera_config
        self.manager = CameraManager(config=self.camera_config, logger=self._log)

    def set_mvs_sdk_path(self, sdk_path: str) -> None:
        path = str(sdk_path or "").strip()
        self.camera_config.mvs_sdk_path = path
        self.camera_config.hikrobot_mvs_sdk_path = path

    def discover_cameras(self) -> list[dict[str, Any]]:
        return [device.to_dict() for device in self.manager.discover_all()]

    def refresh_cameras(self) -> list[dict[str, Any]]:
        return [device.to_dict() for device in self.manager.refresh_devices()]

    def discover_cameras_result(self) -> dict[str, Any]:
        return self.manager.discover_all_result().to_dict()

    def refresh_cameras_result(self) -> dict[str, Any]:
        return self.manager.refresh_devices_result().to_dict()

    def get_last_discovery_result(self) -> dict[str, Any]:
        return self.manager.get_last_discovery_result().to_dict()

    def get_camera_devices(self) -> list[dict[str, Any]]:
        return [device.to_dict() for device in self.manager.get_devices()]

    def discover_hikrobot_cameras(self) -> list[dict[str, Any]]:
        devices = self.manager.discover_all()
        return [device.to_dict() for device in devices if device.backend_name == "hikrobot"]

    def discover_usb_cameras(self) -> list[dict[str, Any]]:
        devices = self.manager.discover_all()
        return [device.to_dict() for device in devices if device.backend_name == "opencv"]

    def select_camera(self, unique_id: str, backend_name: str | None = None) -> dict[str, Any]:
        return self.manager.select_device(unique_id, backend_name).to_dict()

    def test_camera(
        self,
        unique_id: str | None = None,
        backend_name: str | None = None,
        camera_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if unique_id:
            self.manager.select_device(unique_id, backend_name)
        return asdict(self.manager.test_device(camera_config))

    def configure_camera(self, camera_config: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if isinstance(camera_config, str):
            self.manager.select_device(camera_config)
            return
        return self.manager.configure_selected(camera_config or {})

    def open_camera(self, unique_id: str | None = None, backend_name: str | None = None) -> None:
        if unique_id:
            self.manager.select_device(unique_id, backend_name)
        self.manager.open_selected()

    def start_camera_stream(self) -> None:
        self.manager.start_stream()

    def get_latest_frame(self) -> FrameData:
        return self.manager.get_latest_frame()

    def get_camera_status(self) -> dict[str, Any]:
        return asdict(self.manager.get_status())

    def stop_camera_stream(self) -> None:
        self.manager.stop_stream()

    def close_camera(self) -> None:
        self.manager.close_selected()


_default_service = VisionCameraService()


def discover_cameras() -> list[dict[str, Any]]:
    return _default_service.discover_cameras()


def refresh_cameras() -> list[dict[str, Any]]:
    return _default_service.refresh_cameras()


def discover_cameras_result() -> dict[str, Any]:
    return _default_service.discover_cameras_result()


def refresh_cameras_result() -> dict[str, Any]:
    return _default_service.refresh_cameras_result()


def get_camera_devices() -> list[dict[str, Any]]:
    return _default_service.get_camera_devices()


def discover_hikrobot_cameras() -> list[dict[str, Any]]:
    return _default_service.discover_hikrobot_cameras()


def discover_usb_cameras() -> list[dict[str, Any]]:
    return _default_service.discover_usb_cameras()


def select_camera(unique_id: str, backend_name: str | None = None) -> dict[str, Any]:
    return _default_service.select_camera(unique_id, backend_name)


def test_camera(
    unique_id: str | None = None,
    backend_name: str | None = None,
    camera_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _default_service.test_camera(unique_id, backend_name, camera_config)


def configure_camera(camera_config: dict[str, Any] | None = None) -> dict[str, Any] | None:
    return _default_service.configure_camera(camera_config)


def open_camera(unique_id: str | None = None, backend_name: str | None = None) -> None:
    _default_service.open_camera(unique_id, backend_name)


def start_camera_stream() -> None:
    _default_service.start_camera_stream()


def get_latest_frame() -> FrameData:
    return _default_service.get_latest_frame()


def get_camera_status() -> dict[str, Any]:
    return _default_service.get_camera_status()


def stop_camera_stream() -> None:
    _default_service.stop_camera_stream()


def close_camera() -> None:
    _default_service.close_camera()
