from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "data" / "frontend_settings.json"


class FrontendSettingsStore:
    """Small JSON-backed store for values entered in the frontend wizard."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH

    def load(self) -> dict[str, object]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def save(self, settings: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(dict(settings), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
