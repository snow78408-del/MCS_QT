from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
import uuid
from collections import deque
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


@dataclass(frozen=True, slots=True)
class PIDReplaySample:
    sequence_no: int
    timestamp: float
    elapsed_s: float
    q1_flow_ul_min: float | None
    q2_flow_ul_min: float | None
    target_diameter_um: float | None
    measured_diameter_um: float | None
    droplet_speed_um_s: float | None


@dataclass(frozen=True, slots=True)
class PIDReplayData:
    database_path: str
    session_id: str
    started_at_iso: str
    stopped_at_iso: str
    metadata: dict[str, Any]
    q1_flow_source: str
    q2_flow_source: str
    samples: tuple[PIDReplaySample, ...]

    @property
    def duration_s(self) -> float:
        return float(self.samples[-1].elapsed_s) if self.samples else 0.0


class PIDSessionRecorder:
    """Collect one PID run in memory and persist it only when requested."""

    def __init__(self, max_samples: int = 100_000) -> None:
        self._lock = threading.RLock()
        self._session_id: str | None = None
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._metadata: dict[str, Any] = {}
        self._max_samples = max(1, int(max_samples))
        self._samples: deque[PIDFlowSample] = deque(maxlen=self._max_samples)
        self._next_sequence_no = 1
        self._dropped_sample_count = 0
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
            self._samples = deque(maxlen=self._max_samples)
            self._next_sequence_no = 1
            self._dropped_sample_count = 0
            self._active = True
            self._saved = False
            return self._session_id

    def record_sample(self, **values: Any) -> None:
        with self._lock:
            if not self._active:
                return
            if len(self._samples) == self._max_samples:
                self._dropped_sample_count += 1
            self._samples.append(
                PIDFlowSample(sequence_no=self._next_sequence_no, **values)
            )
            self._next_sequence_no += 1

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
                "max_samples": self._max_samples,
                "dropped_sample_count": self._dropped_sample_count,
            }

    def discard(self) -> None:
        with self._lock:
            self._session_id = None
            self._started_at = None
            self._stopped_at = None
            self._metadata = {}
            self._samples = deque(maxlen=self._max_samples)
            self._next_sequence_no = 1
            self._dropped_sample_count = 0
            self._active = False
            self._saved = False

    def save_to_sqlite(self, database_path: str | Path) -> dict[str, Any]:
        raw_path = str(database_path).strip()
        if not raw_path:
            raise ValueError("数据库保存路径不能为空")
        path = Path(raw_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            if self._active:
                raise RuntimeError("PID 仍在运行，请先停止实验再保存数据")
            if not self._session_id or not self._samples:
                raise RuntimeError("本次 PID 没有可保存的数据")
            session_id = self._session_id
            started_at = self._started_at
            stopped_at = self._stopped_at or time.time()
            metadata = dict(self._metadata)
            metadata["dropped_sample_count"] = self._dropped_sample_count
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


def load_pid_replay(database_path: str | Path) -> PIDReplayData:
    """Load the newest PID session from a recorder SQLite file in read-only mode."""
    raw_path = str(database_path).strip()
    if not raw_path:
        raise ValueError("PID 数据库路径不能为空")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PID 数据库不存在：{path}")

    uri = f"{path.as_uri()}?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            required_tables = {"pid_sessions", "pid_flow_samples"}
            if not required_tables.issubset(tables):
                raise ValueError("所选文件不是有效的 PID 数据库：缺少会话或采样表")

            required_columns = {
                "session_id",
                "sequence_no",
                "timestamp",
                "q1_command_ul_min",
                "q2_command_ul_min",
                "q1_actual_ul_min",
                "q2_actual_ul_min",
                "measured_diameter_um",
                "droplet_speed_um_s",
            }
            sample_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(pid_flow_samples)")
            }
            missing_columns = required_columns - sample_columns
            if missing_columns:
                raise ValueError(
                    "PID 数据库版本不兼容，缺少字段：" + ", ".join(sorted(missing_columns))
                )

            target_column = (
                "target_diameter_um"
                if "target_diameter_um" in sample_columns
                else "NULL AS target_diameter_um"
            )

            session = connection.execute(
                """
                SELECT session_id, started_at_iso, stopped_at_iso, metadata_json
                FROM pid_sessions
                ORDER BY stopped_at DESC, started_at DESC
                LIMIT 1
                """
            ).fetchone()
            if session is None:
                raise ValueError("PID 数据库中没有实验会话")

            rows = connection.execute(
                f"""
                SELECT sequence_no, timestamp, q1_command_ul_min,
                       q2_command_ul_min, q1_actual_ul_min, q2_actual_ul_min,
                       {target_column},
                       measured_diameter_um, droplet_speed_um_s
                FROM pid_flow_samples
                WHERE session_id = ?
                ORDER BY sequence_no
                """,
                (str(session["session_id"]),),
            ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"无法读取 PID SQLite 数据库：{exc}") from exc

    if not rows:
        raise ValueError("最新实验会话中没有 PID 采样数据")

    timestamps = [_finite_float(row["timestamp"], "timestamp") for row in rows]
    first_timestamp = timestamps[0]
    q1_has_actual = any(_optional_finite_float(row["q1_actual_ul_min"]) is not None for row in rows)
    q2_has_actual = any(_optional_finite_float(row["q2_actual_ul_min"]) is not None for row in rows)
    samples: list[PIDReplaySample] = []
    for row, timestamp in zip(rows, timestamps):
        samples.append(
            PIDReplaySample(
                sequence_no=int(row["sequence_no"]),
                timestamp=timestamp,
                elapsed_s=max(0.0, timestamp - first_timestamp),
                q1_flow_ul_min=_selected_flow(row, "q1", q1_has_actual),
                q2_flow_ul_min=_selected_flow(row, "q2", q2_has_actual),
                target_diameter_um=_optional_finite_float(row["target_diameter_um"]),
                measured_diameter_um=_optional_finite_float(row["measured_diameter_um"]),
                droplet_speed_um_s=_optional_finite_float(row["droplet_speed_um_s"]),
            )
        )

    try:
        metadata_raw = json.loads(str(session["metadata_json"] or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        metadata_raw = {}
    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
    return PIDReplayData(
        database_path=str(path),
        session_id=str(session["session_id"]),
        started_at_iso=str(session["started_at_iso"] or ""),
        stopped_at_iso=str(session["stopped_at_iso"] or ""),
        metadata=metadata,
        q1_flow_source="device_parameter_estimate" if q1_has_actual else "command",
        q2_flow_source="device_parameter_estimate" if q2_has_actual else "command",
        samples=tuple(samples),
    )


def _selected_flow(row: sqlite3.Row, channel: str, has_actual: bool) -> float | None:
    if has_actual:
        return _optional_finite_float(row[f"{channel}_actual_ul_min"])
    return _optional_finite_float(row[f"{channel}_command_ul_min"])


def _finite_float(value: Any, field_name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"PID 数据字段 {field_name} 包含 NaN 或 Inf")
    return converted


def _optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None
