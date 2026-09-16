from __future__ import annotations

import os
import sys
from pathlib import Path


APP_DIR_NAME = "MicrofluidicControlSystem"


def user_data_dir() -> Path:
    override = str(os.environ.get("MCS_DATA_DIR", "")).strip()
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        # Keep application state separate from the source checkout. An explicit
        # deployment/test override still takes priority; do not silently write C:.
        return Path("D:/MCS_QT_Data")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / APP_DIR_NAME


def ensure_user_subdir(name: str) -> Path:
    path = user_data_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return path
