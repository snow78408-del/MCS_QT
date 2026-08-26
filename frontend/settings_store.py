from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

from .paths import user_data_dir


DEFAULT_SETTINGS_PATH = user_data_dir() / "config" / "frontend_settings.json"
SETTINGS_SCHEMA_VERSION = 2


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
            return dict(payload)
        version = payload.get("schema_version")
        settings = payload.get("settings")
        if version in {1, SETTINGS_SCHEMA_VERSION} and isinstance(settings, dict):
            return dict(settings)
        return None
