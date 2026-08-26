from __future__ import annotations

import ctypes

from backend.vision.cameras.adapters.hikrobot_direct import DirectHikrobotDllCamera
from backend.vision.cameras.registry import default_registry


def test_hikrobot_adapter_imports_without_windows_ctypes() -> None:
    registry = default_registry()

    assert "hikrobot" in registry._adapters
    assert "hikrobot" not in registry._import_errors

    if not hasattr(ctypes, "WinDLL"):
        available, reason, dll_path = DirectHikrobotDllCamera.is_available()
        assert not available
        assert dll_path is None
        assert "requires Windows" in reason
