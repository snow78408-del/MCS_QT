from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path

import pytest

from backend.orchestrator.pid_database import PIDSessionRecorder


def _sample(timestamp: float, q1: float, q2: float, *, frozen: bool = False) -> dict:
    return {
        "timestamp": timestamp,
        "frame_id": 12,
        "control_period_id": 3,
        "q1_command_ul_min": q1,
        "q2_command_ul_min": q2,
        "q1_actual_ul_min": q1 - 0.1,
        "q2_actual_ul_min": q2 - 0.1,
        "target_diameter_um": 50.0,
        "measured_diameter_um": 52.0,
        "diameter_error_um": -2.0,
        "droplet_speed_um_s": 225.5,
        "adjustment": -0.25,
        "pid_output": -0.2,
        "kp": 0.1,
        "ki": 0.01,
        "kd": 0.0,
        "adaptive_enabled": True,
        "adaptive_active": True,
        "feedback_frozen": frozen,
        "reason": "test",
    }


def test_pid_session_is_saved_to_user_selected_sqlite_database() -> None:
    database_path = Path(__file__).with_name(f"_pid_data_{uuid.uuid4().hex}.sqlite")
    recorder = PIDSessionRecorder()
    try:
        session_id = recorder.begin_session({"operator": "test"})
        now = time.time()
        recorder.record_sample(**_sample(now, 60.0, 30.0))
        recorder.record_sample(**_sample(now + 1.0, 59.5, 30.25, frozen=True))
        recorder.finish_session()

        result = recorder.save_to_sqlite(database_path)

        assert result["session_id"] == session_id
        assert result["record_count"] == 2
        assert recorder.status()["saved"] is True
        with closing(sqlite3.connect(database_path)) as connection:
            session = connection.execute(
                "SELECT record_count, metadata_json FROM pid_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            samples = connection.execute(
                """
                SELECT q1_command_ul_min, q2_command_ul_min, droplet_speed_um_s,
                       feedback_frozen
                FROM pid_flow_samples
                WHERE session_id = ? ORDER BY sequence_no
                """,
                (session_id,),
            ).fetchall()
        assert session is not None and session[0] == 2 and '"operator": "test"' in session[1]
        assert samples == [(60.0, 30.0, 225.5, 0), (59.5, 30.25, 225.5, 1)]
    finally:
        if database_path.exists():
            database_path.unlink()


def test_unsaved_session_must_be_resolved_before_next_run() -> None:
    recorder = PIDSessionRecorder()
    recorder.begin_session()
    recorder.record_sample(**_sample(time.time(), 60.0, 30.0))
    recorder.finish_session()

    with pytest.raises(RuntimeError, match="尚未保存或放弃"):
        recorder.begin_session()

    recorder.discard()
    recorder.begin_session()
    assert recorder.status()["record_count"] == 0
