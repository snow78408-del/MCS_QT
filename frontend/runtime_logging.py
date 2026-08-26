from __future__ import annotations

import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .paths import ensure_user_subdir


class RuntimeLogger:
    def __init__(self, log_path: Path) -> None:
        self.path = log_path
        self._logger = logging.getLogger("microfluidic.runtime")

    def __call__(self, message: str) -> None:
        self._logger.info(str(message))


def create_runtime_logger() -> RuntimeLogger:
    log_dir = ensure_user_subdir("logs")
    _prune_runtime_logs(log_dir)
    log_path = log_dir / f"runtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s [%(threadName)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    logger = RuntimeLogger(log_path)
    logger(f"[APP][START] runtime log file={log_path}")
    return logger


def _prune_runtime_logs(log_dir: Path, *, max_files: int = 50, max_total_bytes: int = 250 * 1024 * 1024) -> None:
    resolved_dir = log_dir.resolve()
    files = sorted(
        (path for path in resolved_dir.glob("runtime_*.log*") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    retained_bytes = 0
    for index, path in enumerate(files):
        try:
            size = path.stat().st_size
            keep = index < max_files and retained_bytes + size <= max_total_bytes
            if keep:
                retained_bytes += size
                continue
            resolved = path.resolve()
            if resolved.parent == resolved_dir:
                resolved.unlink(missing_ok=True)
        except OSError:
            continue
