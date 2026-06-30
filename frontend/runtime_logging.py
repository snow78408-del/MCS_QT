from __future__ import annotations

import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


class RuntimeLogger:
    def __init__(self, log_path: Path) -> None:
        self.path = log_path
        self._logger = logging.getLogger("microfluidic.runtime")

    def __call__(self, message: str) -> None:
        self._logger.info(str(message))


def create_runtime_logger() -> RuntimeLogger:
    project_root = Path(__file__).resolve().parents[1]
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
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
