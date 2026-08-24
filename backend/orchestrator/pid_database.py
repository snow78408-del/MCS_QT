from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _iso_time(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(float(timestamp)).astimezone().isoformat(timespec="milliseconds")


@dataclass(slots=True)
class PIDFlowSample:
    sequence_no: int
    timestamp: float
    frame_id: int
    control_period_id: int
    q1_command_ul_min: float
    q2_command_ul_min: float
    q1_actual_ul_min: float | None
    q2_actual_ul_min: float | None
    target_diameter_um: float | None
    measured_diameter_um: float | None
    diameter_error_um: float
    droplet_speed_um_s: float | None
    adjustment: float
    pid_output: float
    kp: float
    ki: float
    kd: float
    adaptive_enabled: bool
    adaptive_active: bool
    feedback_frozen: bool
    reason: str


class PIDSessionRecorder:
    """Collect one PID run in memory and persist it only when requested."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._session_id: str | None = None
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._metadata: dict[str, Any] = {}
        self._samples: list[PIDFlowSample] = []
        self._active = False
        self._saved = False

    def begin_session(self, metadata: dict[str, Any] | None = None) -> str:
        with self._lock:
            if self._samples and not self._saved:
                raise RuntimeError("上一次 PID 数据尚未保存或放弃，不能开始新的 PID 运行")
            self._session_id = uuid.uuid4().hex
            self._started_at = time.time()
            self._stopped_at = None
            self._metadata = dict(metadata or {})
            self._samples = []
            self._active = True
            self._saved = False
            return self._session_id

    def record_sample(self, **values: Any) -> None:
        with self._lock:
            if not self._active:
                return
            self._samples.append(PIDFlowSample(sequence_no=len(self._samples) + 1, **values))

    def finish_session(self) -> None:
        with self._lock:
            if self._active:
                self._stopped_at = time.time()
            self._active = False

    def has_unsaved_data(self) -> bool:
        with self._lock:
            return bool(self._samples) and not self._saved

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session_id": self._session_id,
                "record_count": len(self._samples),
                "active": self._active,
                "saved": self._saved,
                "has_unsaved_data": bool(self._samples) and not self._saved,
                "started_at": self._started_at,
                "stopped_at": self._stopped_at,
            }

    def discard(self) -> None:
        with self._lock:
            self._session_id = None
            self._started_at = None
            self._stopped_at = None
            self._metadata = {}
            self._samples = []
            self._active = False
            self._saved = False

    def save_to_sqlite(self, database_path: str | Path) -> dict[str, Any]:
        raw_path = str(database_path).strip()
        if not raw_path:
            raise ValueError("数据库保存路径不能为空")
        path = Path(raw_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            if not self._session_id or not self._samples:
                raise RuntimeError("本次 PID 没有可保存的数据")
            session_id = self._session_id
            started_at = self._started_at
            stopped_at = self._stopped_at or time.time()
            metadata = dict(self._metadata)
            samples = [PIDFlowSample(**asdict(sample)) for sample in self._samples]

        with closing(sqlite3.connect(str(path))) as connection:
            with connection:
                connection.execute("PRAGMA foreign_keys = ON")
                connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS pid_sessions (
                    session_id TEXT PRIMARY KEY,
                    started_at REAL NOT NULL,
                    stopped_at REAL NOT NULL,
                    started_at_iso TEXT NOT NULL,
                    stopped_at_iso TEXT NOT NULL,
                    record_count INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pid_flow_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    sequence_no INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    timestamp_iso TEXT NOT NULL,
                    frame_id INTEGER NOT NULL,
                    control_period_id INTEGER NOT NULL,
                    q1_command_ul_min REAL NOT NULL,
                    q2_command_ul_min REAL NOT NULL,
                    q1_actual_ul_min REAL,
                    q2_actual_ul_min REAL,
                    target_diameter_um REAL,
                    measured_diameter_um REAL,
                    diameter_error_um REAL NOT NULL,
                    droplet_speed_um_s REAL,
                    adjustment REAL NOT NULL,
                    pid_output REAL NOT NULL,
                    kp REAL NOT NULL,
                    ki REAL NOT NULL,
                    kd REAL NOT NULL,
                    adaptive_enabled INTEGER NOT NULL,
                    adaptive_active INTEGER NOT NULL,
                    feedback_frozen INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    UNIQUE(session_id, sequence_no),
                    FOREIGN KEY(session_id) REFERENCES pid_sessions(session_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_pid_flow_samples_session_time
                ON pid_flow_samples(session_id, timestamp);
                """
            )
                connection.execute(
                """
                INSERT OR REPLACE INTO pid_sessions (
                    session_id, started_at, stopped_at, started_at_iso,
                    stopped_at_iso, record_count, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    float(started_at or stopped_at),
                    float(stopped_at),
                    _iso_time(started_at or stopped_at),
                    _iso_time(stopped_at),
                    len(samples),
                    json.dumps(metadata, ensure_ascii=False, default=str),
                ),
            )
                connection.execute("DELETE FROM pid_flow_samples WHERE session_id = ?", (session_id,))
                connection.executemany(
                """
                INSERT INTO pid_flow_samples (
                    session_id, sequence_no, timestamp, timestamp_iso, frame_id,
                    control_period_id, q1_command_ul_min, q2_command_ul_min,
                    q1_actual_ul_min, q2_actual_ul_min, target_diameter_um,
                    measured_diameter_um, diameter_error_um, droplet_speed_um_s,
                    adjustment, pid_output, kp, ki, kd, adaptive_enabled,
                    adaptive_active, feedback_frozen, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        session_id,
                        sample.sequence_no,
                        sample.timestamp,
                        _iso_time(sample.timestamp),
                        sample.frame_id,
                        sample.control_period_id,
                        sample.q1_command_ul_min,
                        sample.q2_command_ul_min,
                        sample.q1_actual_ul_min,
                        sample.q2_actual_ul_min,
                        sample.target_diameter_um,
                        sample.measured_diameter_um,
                        sample.diameter_error_um,
                        sample.droplet_speed_um_s,
                        sample.adjustment,
                        sample.pid_output,
                        sample.kp,
                        sample.ki,
                        sample.kd,
                        int(sample.adaptive_enabled),
                        int(sample.adaptive_active),
                        int(sample.feedback_frozen),
                        sample.reason,
                    )
                    for sample in samples
                ],
            )

        with self._lock:
            if self._session_id == session_id:
                self._saved = True

        return {"path": str(path.resolve()), "session_id": session_id, "record_count": len(samples)}
