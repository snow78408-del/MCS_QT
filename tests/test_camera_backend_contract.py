from __future__ import annotations

from ctypes import c_void_p
from types import SimpleNamespace

import pytest

from backend.vision.cameras.adapters.alliedvision_camera import AlliedVisionCameraAdapter
from backend.vision.cameras.adapters.basler_camera import BaslerCameraAdapter
from backend.vision.cameras.adapters.daheng_camera import DahengCameraAdapter
from backend.vision.cameras.adapters.gentl_camera import GenTLCameraAdapter
from backend.vision.cameras.adapters.hikrobot_direct import DirectHikrobotDllCamera
from backend.vision.cameras.base import CameraBackendError
from backend.vision.service import VisionCameraService


class _FakePylonNode:
    def __init__(self, value, *, writable: bool = True) -> None:
        self.value = value
        self.readable = True
        self.writable = writable

    def GetValue(self):
        return self.value

    def SetValue(self, value):
        self.value = value

    def IsReadable(self):
        return self.readable

    def IsWritable(self):
        return self.writable


class _FakeGxFeature:
    def __init__(self, value, *, writable: bool = True) -> None:
        self.value = value
        self.writable = writable

    def get(self):
        return self.value

    def set(self, value):
        self.value = value

    def is_readable(self):
        return True

    def is_writable(self):
        return self.writable


class _FakeVimbaFeature:
    def __init__(self, value, *, writable: bool = True) -> None:
        self.value = value
        self.writable = writable

    def get(self):
        return self.value

    def set(self, value):
        self.value = value

    def is_readable(self):
        return True

    def is_writeable(self):
        return self.writable


class _FakeGenTLNode:
    def __init__(self, value, *, writable: bool = True) -> None:
        self.value = value
        self.readable = True
        self.writable = writable

    def is_readable(self):
        return self.readable

    def is_writable(self):
        return self.writable


@pytest.mark.parametrize(
    ("adapter", "camera", "missing_name"),
    [
        (
            BaslerCameraAdapter(),
            SimpleNamespace(ExposureTime=_FakePylonNode(1000.0)),
            "gain",
        ),
        (
            DahengCameraAdapter(),
            SimpleNamespace(ExposureTime=_FakeGxFeature(1000.0)),
            "gain",
        ),
        (
            AlliedVisionCameraAdapter(),
            SimpleNamespace(ExposureTime=_FakeVimbaFeature(1000.0)),
            "gain",
        ),
    ],
)
def test_sdk_adapters_derive_capabilities_from_accessible_nodes(adapter, camera, missing_name) -> None:
    adapter._camera = camera

    capabilities = adapter.get_capabilities()

    assert capabilities.exposure.supported
    assert capabilities.exposure.writable
    assert not getattr(capabilities, missing_name).supported
    with pytest.raises(CameraBackendError):
        adapter.set_feature(missing_name, 2.0)


@pytest.mark.parametrize(
    ("adapter", "node_map"),
    [
        (BaslerCameraAdapter(), SimpleNamespace(ExposureTime=_FakePylonNode(1000.0, writable=False))),
        (DahengCameraAdapter(), SimpleNamespace(ExposureTime=_FakeGxFeature(1000.0, writable=False))),
        (AlliedVisionCameraAdapter(), SimpleNamespace(ExposureTime=_FakeVimbaFeature(1000.0, writable=False))),
    ],
)
def test_sdk_adapters_reject_read_only_nodes(adapter, node_map) -> None:
    adapter._camera = node_map

    assert not adapter.get_capabilities().exposure.writable
    with pytest.raises(CameraBackendError):
        adapter.set_feature("exposure", 2000.0)


def test_gentl_capabilities_and_unsupported_write_are_node_based() -> None:
    node_map = SimpleNamespace(ExposureTime=_FakeGenTLNode(1000.0))
    adapter = GenTLCameraAdapter()
    adapter._ia = SimpleNamespace(remote_device=SimpleNamespace(node_map=node_map))

    capabilities = adapter.get_capabilities()

    assert capabilities.exposure.supported
    assert capabilities.exposure.writable
    assert not capabilities.gain.supported
    with pytest.raises(CameraBackendError):
        adapter.set_feature("gain", 2.0)


class _FakeDirectDll:
    def __init__(self, result: int = 0) -> None:
        self.result = result
        self.calls: list[tuple[str, int]] = []

    def MV_CC_SetEnumValue(self, _handle, name, value):
        self.calls.append((name.decode("ascii"), int(value)))
        return self.result


def test_hikrobot_direct_acquisition_continuous_uses_acquisition_enum() -> None:
    dll = _FakeDirectDll()
    adapter = DirectHikrobotDllCamera()
    adapter._dll = dll
    adapter._handle = c_void_p(1)

    adapter.set_feature("acquisition_mode", "Continuous")

    assert dll.calls == [("AcquisitionMode", 2)]


def test_hikrobot_direct_sdk_enum_write_failure_is_reported() -> None:
    adapter = DirectHikrobotDllCamera()
    adapter._dll = _FakeDirectDll(result=0x1234)
    adapter._handle = c_void_p(1)

    with pytest.raises(CameraBackendError, match="AcquisitionMode"):
        adapter.set_feature("acquisition_mode", "Continuous")


def test_vision_camera_service_preserves_system_config_fields() -> None:
    source = SimpleNamespace(
        sdk_paths=("/sdk",),
        gentl_xml_cache_dir="/tmp/xml",
        frame_timeout_ms=250,
        frame_failure_threshold=4,
        reconnect_attempts=5,
        reconnect_interval_s=0.25,
        test_frame_count=7,
    )

    service = VisionCameraService(camera_config=source)

    assert service.camera_config.sdk_paths == ("/sdk",)
    assert service.camera_config.gentl_xml_cache_dir == "/tmp/xml"
    assert service.camera_config.frame_timeout_ms == 250
    assert service.camera_config.frame_failure_threshold == 4
    assert service.camera_config.reconnect_attempts == 5
    assert service.camera_config.reconnect_interval_s == 0.25
    assert service.camera_config.test_frame_count == 7
