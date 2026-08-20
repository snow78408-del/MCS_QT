from __future__ import annotations

import unittest

from backend.vision.cameras.manager import CameraManager
from backend.vision.cameras.models import CameraCapabilities, CameraFeatureCapability
from backend.vision.config import CameraSystemConfig


class _FakeCameraAdapter:
    backend_name = "fake"

    def __init__(self) -> None:
        writable = CameraFeatureCapability(
            supported=True,
            readable=True,
            writable=True,
            min_value=0,
            max_value=10000,
        )
        self.capabilities = CameraCapabilities(
            exposure=writable,
            gain=writable,
            frame_rate=writable,
            width=writable,
            height=writable,
        )
        self.values: dict[str, float | int] = {}

    def get_capabilities(self) -> CameraCapabilities:
        return self.capabilities

    def set_feature(self, name: str, value: float | int) -> None:
        self.values[name] = value

    def get_feature(self, name: str) -> float | int | None:
        return self.values.get(name)


class CameraParameterDownlinkTests(unittest.TestCase):
    def test_user_parameters_are_written_and_read_back(self) -> None:
        manager = CameraManager(config=CameraSystemConfig())
        adapter = _FakeCameraAdapter()
        requested = {
            "exposure": 2800.0,
            "gain": 1.5,
            "frame_rate": 120.0,
            "width": 720,
            "height": 540,
        }

        result = manager._configure_adapter(adapter, requested)

        self.assertEqual(adapter.values, requested)
        self.assertEqual(result["applied"], requested)
        self.assertEqual(result["readback"], requested)
        self.assertEqual(result["skipped"], {})


if __name__ == "__main__":
    unittest.main()
