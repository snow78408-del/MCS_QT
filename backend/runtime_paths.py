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


def device_lock_dir() -> Path:
    """设备锁目录：**刻意不经过 MCS_DATA_DIR**。

    同一台物理设备必须对所有进程映射到同一把锁。若锁目录跟着数据目录走，两个用不同
    ``MCS_DATA_DIR`` 启动的进程会各自拿到一把锁，互斥直接失效——所以这里只认机器／用户级的
    运行时位置，与数据目录解耦。

    注意：不同**用户**之间不共享（Windows 用各自的 ``LOCALAPPDATA``，POSIX 按 uid 分目录），
    跨用户的设备争用不在本机制范围内。
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
        path = base / APP_DIR_NAME / "device-locks"
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Caches" / APP_DIR_NAME / "device-locks"
    else:
        base = Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp")
        path = base / f"{APP_DIR_NAME}-uid{os.getuid()}" / "device-locks"
    path.mkdir(parents=True, exist_ok=True)
    return path
