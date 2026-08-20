from __future__ import annotations

from typing import Any


# Application starting point for the currently used MV-CS004-10UM acquisition
# path (720x540 Mono8). Acquisition runs at 100 FPS for short multi-frame motion
# measurements. Preview publication remains independently capped at 30 FPS.
HIKROBOT_CS_DEFAULT_PARAMETERS: dict[str, float | int] = {
    "exposure": 3000.0,
    "gain": 0.0,
    "frame_rate": 100.0,
    "width": 720,
    "height": 540,
}


def resolve_camera_defaults(device: dict[str, Any] | None) -> tuple[str, dict[str, float | int]]:
    info = device or {}
    backend = str(info.get("selected_backend") or info.get("backend_name") or "").strip().lower()
    manufacturer = str(info.get("manufacturer") or "").strip().lower()
    model = str(info.get("model") or "").strip()
    if backend == "hikrobot" or "hikrobot" in manufacturer or "hikvision" in manufacturer:
        profile = "海康机器人 CS 系列"
        if model and model not in {"--", "MVS Device 0"}:
            profile = f"{profile}（{model}）"
        return profile, dict(HIKROBOT_CS_DEFAULT_PARAMETERS)
    return "通用工业相机（由设备能力校验）", {}


def normalize_camera_parameters(parameters: dict[str, Any] | None) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for name, raw in (parameters or {}).items():
        if raw in (None, ""):
            continue
        if name in {"width", "height"}:
            value = int(float(raw))
            if value <= 0:
                raise ValueError(f"{name} 必须是正整数")
            result[name] = value
        elif name in {"exposure", "frame_rate"}:
            value = float(raw)
            if value <= 0.0:
                raise ValueError(f"{name} 必须大于 0")
            result[name] = value
        elif name == "gain":
            value = float(raw)
            if value < 0.0:
                raise ValueError("gain 不能小于 0")
            result[name] = value
    return result
