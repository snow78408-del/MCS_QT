from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend.vision.cameras.adapters.flir_camera import FlirCameraAdapter
from backend.vision.cameras.adapters.hikrobot_camera import HikrobotCameraAdapter
from backend.vision.cameras.models import CameraCapabilities, CameraDeviceInfo, CameraFeatureCapability
from backend.vision.cameras.registry import default_registry
from backend.vision.config import CameraSystemConfig, default_config
from backend.vision.run_vision import build_argument_parser


def test_allied_vision_is_the_registered_canonical_backend_name() -> None:
    registry = default_registry(config=CameraSystemConfig())

    adapter_class = registry.get_adapter_class("allied_vision")
    assert adapter_class.backend_name == "allied_vision"
    assert "alliedvision" not in registry._adapters


def test_standalone_tracker_default_matches_pipeline_config() -> None:
    args = build_argument_parser().parse_args(["--video", "sample.mp4"])

    assert args.tracker == default_config().tracker.tracker_type == "kalman"


class _FakePySpinNode:
    def __init__(self, value, kind: str, minimum=0, maximum=10000, increment=1) -> None:
        self.value = value
        self.kind = kind
        self.minimum = minimum
        self.maximum = maximum
        self.increment = increment
        self.readable = True
        self.writable = True
        self.enum_values = {"Off": 0, "On": 1, "Continuous": 2, "Mono8": 3}
        self.enum_names = {value: key for key, value in self.enum_values.items()}

    def GetValue(self):
        return self.value

    def SetValue(self, value) -> None:
        self.value = value

    def GetMin(self):
        return self.minimum

    def GetMax(self):
        return self.maximum

    def GetInc(self):
        return self.increment

    def GetCurrentEntry(self):
        return self

    def GetSymbolic(self):
        return self.enum_names.get(self.value, str(self.value))

    def GetEntryByName(self, name):
        if name not in self.enum_values:
            raise ValueError(name)
        return _FakePySpinEnumEntry(self.enum_values[name])

    def SetIntValue(self, value) -> None:
        self.value = value


class _FakePySpinEnumEntry:
    readable = True

    def __init__(self, value) -> None:
        self.value = value

    def GetValue(self):
        return self.value

    def GetSymbolic(self):
        return str(self.value)


class _FakePySpinNodeMap:
    def __init__(self) -> None:
        self.nodes = {
            "ExposureTime": _FakePySpinNode(1000.0, "float", 1.0, 10000.0),
            "Gain": _FakePySpinNode(2.0, "float", 0.0, 24.0),
            "AcquisitionFrameRate": _FakePySpinNode(30.0, "float", 1.0, 240.0),
            "Width": _FakePySpinNode(640, "int", 8, 4096),
            "Height": _FakePySpinNode(480, "int", 8, 4096),
            "OffsetX": _FakePySpinNode(0, "int", 0, 4096),
            "OffsetY": _FakePySpinNode(0, "int", 0, 4096),
            "PixelFormat": _FakePySpinNode(3, "enum"),
            "TriggerMode": _FakePySpinNode(0, "enum"),
            "AcquisitionMode": _FakePySpinNode(2, "enum"),
        }

    def GetNode(self, name):
        return self.nodes.get(name)


class _FakePySpin:
    CFloatPtr = staticmethod(lambda node: node)
    CIntegerPtr = staticmethod(lambda node: node)
    CEnumerationPtr = staticmethod(lambda node: node)
    CStringPtr = staticmethod(lambda node: node)

    @staticmethod
    def IsReadable(node):
        return bool(node and node.readable)

    @staticmethod
    def IsWritable(node):
        return bool(node and node.writable)


def test_flir_writes_features_and_verifies_readback() -> None:
    node_map = _FakePySpinNodeMap()
    camera = SimpleNamespace(GetNodeMap=lambda: node_map)
    adapter = FlirCameraAdapter()
    adapter._camera = camera

    with patch.dict(sys.modules, {"PySpin": _FakePySpin}):
        capabilities = adapter.get_capabilities()
        adapter.set_feature("exposure", 2200.0)
        adapter.set_feature("width", 800)
        adapter.set_feature("trigger_mode", "Off")

        assert capabilities.exposure.writable
        assert capabilities.width.writable
        assert capabilities.trigger_mode.writable
        assert adapter.get_feature("exposure") == 2200.0
        assert adapter.get_feature("width") == 800
        assert adapter.get_feature("trigger_mode") == "Off"


def test_flir_unavailable_feature_fails_instead_of_being_a_noop() -> None:
    node_map = _FakePySpinNodeMap()
    del node_map.nodes["Gain"]
    camera = SimpleNamespace(GetNodeMap=lambda: node_map)
    adapter = FlirCameraAdapter()
    adapter._camera = camera

    with patch.dict(sys.modules, {"PySpin": _FakePySpin}):
        assert not adapter.get_capabilities().gain.supported
        try:
            adapter.set_feature("gain", 3.0)
        except Exception as exc:
            assert "不可用" in str(exc)
        else:
            raise AssertionError("unsupported FLIR feature was silently accepted")


def test_hikrobot_modern_adapter_configures_before_starting_stream() -> None:
    events: list[str] = []

    class FakeLegacy:
        def __init__(self, *args, **kwargs) -> None:
            self._status = SimpleNamespace(pixel_format="Mono8", last_camera_error="")

        def open(self, device_id, *, start_stream=True) -> None:
            events.append("open")
            assert not start_stream

        def close(self) -> None:
            pass

        def start_stream(self) -> None:
            events.append("start_stream")

        def stop_stream(self) -> None:
            pass

        def read_frame(self):
            events.append("read_frame")
            return SimpleNamespace(
                frame=np.zeros((2, 2), dtype=np.uint8),
                frame_id=1,
                timestamp=1.0,
                valid=True,
                error="",
            )

        def get_status(self):
            return self._status

        def get_parameter_info(self, name):
            return SimpleNamespace(current_value=0)

        def set_resolution(self, **values) -> None:
            events.append("set_resolution")

        def set_trigger_mode(self, value) -> None:
            events.append("set_trigger_mode")

        def set_acquisition_mode(self, value) -> None:
            events.append("set_acquisition_mode")

        def set_exposure(self, value) -> None:
            events.append("set_exposure")

        def set_gain(self, value) -> None:
            events.append("set_gain")

        def set_frame_rate(self, value) -> None:
            events.append("set_frame_rate")

        def is_open(self):
            return True

        def is_streaming(self):
            return "start_stream" in events

    device = CameraDeviceInfo(
        device_id="hikrobot:0",
        unique_id="HIKROBOT:0",
        backend_name="hikrobot",
        capabilities=CameraCapabilities(
            width=CameraFeatureCapability(True, True, True),
            trigger_mode=CameraFeatureCapability(True, True, True),
            acquisition_mode=CameraFeatureCapability(True, True, True),
        ),
        available_backends=["hikrobot"],
        selected_backend="hikrobot",
    )
    module = sys.modules["backend.vision.cameras.adapters.hikrobot_camera"]
    with patch.object(module, "LegacyHikrobotAdapter", FakeLegacy):
        adapter = HikrobotCameraAdapter()
        adapter.open(device)
        adapter.set_feature("width", 800)
        adapter.set_feature("trigger_mode", "Off")
        adapter.start_stream()
        adapter.read_frame()

    assert events == ["open", "set_resolution", "set_trigger_mode", "start_stream", "read_frame"]
