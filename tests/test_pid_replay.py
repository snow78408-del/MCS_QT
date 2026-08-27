from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from backend.orchestrator.pid_database import PIDReplayData, PIDReplaySample
from frontend.pid_replay import _build_response_chart


def test_replay_response_chart_contains_target_step_and_measurements() -> None:
    application = QApplication.instance() or QApplication([])
    replay = PIDReplayData(
        database_path="test.sqlite",
        session_id="session",
        started_at_iso="",
        stopped_at_iso="",
        metadata={},
        q1_flow_source="command",
        q2_flow_source="command",
        samples=(
            PIDReplaySample(1, 100.0, 0.0, 60.0, 30.0, 50.0, 49.0, 200.0),
            PIDReplaySample(2, 101.0, 1.0, 61.0, 29.0, 60.0, 54.0, 210.0),
        ),
    )

    chart = _build_response_chart(replay)

    series_by_name = {series.name(): series for series in chart.series()}
    assert set(series_by_name) == {
        "液滴平均直径",
        "目标液滴直径（阶跃）",
        "液滴平均速度",
    }
    target_series = series_by_name["目标液滴直径（阶跃）"]
    assert target_series.pen().style() == Qt.PenStyle.DashLine
    assert [(point.x(), point.y()) for point in target_series.points()] == [
        (0.0, 50.0),
        (1.0, 60.0),
    ]
    assert application is not None
