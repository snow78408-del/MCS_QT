from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any

from backend.vision.config import ChannelRegionConfig, DetectorConfig

from .paths import user_data_dir


DEFAULT_VISION_TUNING_PATH = user_data_dir() / "config" / "vision_tuning_parameters.json"
VISION_TUNING_SCHEMA_VERSION = 1


class TuningLoadStatus(str, Enum):
    LOADED = "loaded"
    CREATED = "created"
    INVALID = "invalid"


@dataclass(frozen=True)
class TuningLoadResult:
    status: TuningLoadStatus
    detector: DetectorConfig
    channel_region: ChannelRegionConfig
    error: str = ""


class VisionTuningSettingsStore:
    """Strict, user-scoped persistence for the algorithm tuning page."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_VISION_TUNING_PATH

    def load_or_create(self) -> TuningLoadResult:
        defaults = self._defaults()
        if not self.path.exists():
            self.save(*defaults)
            return TuningLoadResult(TuningLoadStatus.CREATED, *defaults)

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            detector, channel = self._decode(payload)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            return TuningLoadResult(TuningLoadStatus.INVALID, *defaults, error=str(exc))
        return TuningLoadResult(TuningLoadStatus.LOADED, detector, channel)

    def save(self, detector: DetectorConfig, channel_region: ChannelRegionConfig) -> None:
        # Validate before replacing the last usable file.
        self._validate_section(asdict(detector), DetectorConfig(), "detector")
        self._validate_section(asdict(channel_region), ChannelRegionConfig(), "channel_region")
        payload = {
            "schema_version": VISION_TUNING_SCHEMA_VERSION,
            "detector": asdict(detector),
            "channel_region": asdict(channel_region),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def delete_and_create_defaults(self) -> tuple[DetectorConfig, ChannelRegionConfig]:
        self.path.unlink(missing_ok=True)
        defaults = self._defaults()
        self.save(*defaults)
        return defaults

    @staticmethod
    def _defaults() -> tuple[DetectorConfig, ChannelRegionConfig]:
        return DetectorConfig(), ChannelRegionConfig()

    def _decode(self, payload: Any) -> tuple[DetectorConfig, ChannelRegionConfig]:
        if not isinstance(payload, dict):
            raise ValueError("参数文件根节点必须是对象")
        expected_root = {"schema_version", "detector", "channel_region"}
        if set(payload) != expected_root:
            raise ValueError("参数文件字段与当前算法不一致")
        if payload["schema_version"] != VISION_TUNING_SCHEMA_VERSION:
            raise ValueError(f"不支持的参数版本：{payload['schema_version']!r}")

        detector_values = self._validate_section(payload["detector"], DetectorConfig(), "detector")
        channel_values = self._validate_section(
            payload["channel_region"], ChannelRegionConfig(), "channel_region"
        )
        return DetectorConfig(**detector_values), ChannelRegionConfig(**channel_values)

    @staticmethod
    def _validate_section(values: Any, defaults: object, section: str) -> dict[str, Any]:
        if not isinstance(values, dict):
            raise ValueError(f"{section} 必须是对象")
        expected = {item.name for item in fields(defaults)}
        if set(values) != expected:
            missing = sorted(expected - set(values))
            extra = sorted(set(values) - expected)
            details = []
            if missing:
                details.append("缺少 " + ", ".join(missing))
            if extra:
                details.append("多出 " + ", ".join(extra))
            raise ValueError(f"{section} 字段与当前算法不一致（{'；'.join(details)}）")

        checked: dict[str, Any] = {}
        for name in expected:
            value = values[name]
            default = getattr(defaults, name)
            if isinstance(default, bool):
                valid = isinstance(value, bool)
            elif isinstance(default, int):
                valid = isinstance(value, int) and not isinstance(value, bool)
            elif isinstance(default, float):
                valid = isinstance(value, (int, float)) and not isinstance(value, bool)
                valid = valid and math.isfinite(float(value))
                if valid:
                    value = float(value)
            elif isinstance(default, str):
                valid = isinstance(value, str)
            else:
                valid = type(value) is type(default)
            if not valid:
                raise ValueError(f"{section}.{name} 的类型或数值无效")
            checked[name] = value
        return checked
