from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

from .paths import user_data_dir


DEFAULT_SETTINGS_PATH = user_data_dir() / "config" / "frontend_settings.json"
SETTINGS_SCHEMA_VERSION = 3


class FrontendSettingsStore:
    """Small JSON-backed store for values entered in the frontend wizard."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH

    def load(self) -> dict[str, object]:
        for candidate in (self.path, self._backup_path):
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
                continue
            migrated = self._migrate(payload)
            if migrated is not None:
                return migrated
        return {}

    def save(self, settings: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {"schema_version": SETTINGS_SCHEMA_VERSION, "settings": dict(settings)},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        if self.path.is_file():
            shutil.copy2(self.path, self._backup_path)
        os.replace(temporary, self.path)

    @property
    def _backup_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".bak")

    @staticmethod
    def _migrate(payload: Any) -> dict[str, object] | None:
        if not isinstance(payload, dict):
            return None
        if "schema_version" not in payload:
            # Legacy schema stored settings directly at the JSON root.
            return FrontendSettingsStore._reconcile_generation_geometry(dict(payload))
        version = payload.get("schema_version")
        settings = payload.get("settings")
        if version in {1, 2, SETTINGS_SCHEMA_VERSION} and isinstance(settings, dict):
            return FrontendSettingsStore._reconcile_generation_geometry(dict(settings))
        return None

    @staticmethod
    def _positive_number(value: object) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0.0 else None

    @staticmethod
    def _reconcile_generation_geometry(settings: dict[str, object]) -> dict[str, object]:
        """Separate the retired 430-um view from current chip geometry.

        Older settings had only ``channel_width_um``.  In this installation an
        exact 430 um value describes the previous observation view.  It is
        retained as history while the generation-zone input starts at 50 um.
        Any explicit generation H/W value wins. A lone legacy width is still
        mirrored once for backward compatibility, but subsequent UI edits keep
        the visible width and out-of-plane depth independent.
        """
        roi_value = settings.get("recognition_roi")
        if not isinstance(roi_value, dict):
            return settings
        roi = dict(roi_value)
        legacy = FrontendSettingsStore._positive_number(roi.get("channel_width_um"))
        height = FrontendSettingsStore._positive_number(
            roi.get("generation_channel_height_um")
        )
        width = FrontendSettingsStore._positive_number(
            roi.get("generation_channel_width_um")
        )

        if legacy == 430.0 and height != width and (height == 430.0 or width == 430.0):
            explicit_current = next(
                (value for value in (width, height) if value is not None and value != 430.0),
                None,
            )
            if explicit_current is not None:
                if height == 430.0:
                    height = explicit_current
                if width == 430.0:
                    width = explicit_current

        if height is None and width is None:
            current = 50.0 if legacy == 430.0 else (legacy or 50.0)
            height = width = current
        elif height is None:
            height = width
        elif width is None:
            width = height

        if legacy is not None and legacy != width:
            roi.setdefault("previous_channel_width_um", legacy)
        roi["channel_width_um"] = float(width)
        roi["generation_channel_height_um"] = float(height)
        roi["generation_channel_width_um"] = float(width)
        roi.setdefault("generation_volume_correction", 1.0)
        settings["recognition_roi"] = roi
        return settings
