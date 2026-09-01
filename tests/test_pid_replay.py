from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from backend.orchestrator.pid_database import PIDReplayData, PIDReplaySample
from frontend.pid_replay import _build_channel_flow_chart, _build_response_chart


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


def test_replay_separates_q1_q2_and_highlights_each_flow_change() -> None:
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
            PIDReplaySample(1, 100.0, 0.0, 60.0, 20.0, 50.0, 49.0, 200.0),
            PIDReplaySample(2, 101.0, 1.0, 60.0, 20.0, 50.0, 49.0, 200.0),
            PIDReplaySample(3, 102.0, 2.0, 65.0, 18.0, 50.0, 49.0, 200.0),
            PIDReplaySample(4, 103.0, 3.0, 62.0, 18.0, 50.0, 49.0, 200.0),
        ),
    )

    q1_chart = _build_channel_flow_chart(replay, "q1")
    q2_chart = _build_channel_flow_chart(replay, "q2")

    assert "泵1（Q1）" in q1_chart.title()
    assert "变化 2 次" in q1_chart.title()
    assert "泵2（Q2）" in q2_chart.title()
    assert "变化 1 次" in q2_chart.title()
    q1_series = {series.name(): series for series in q1_chart.series()}
    q2_series = {series.name(): series for series in q2_chart.series()}
    assert set(q1_series) == {"Q1 控制指令值", "Q1 变化点"}
    assert set(q2_series) == {"Q2 控制指令值", "Q2 变化点"}
    assert [(point.x(), point.y()) for point in q1_series["Q1 变化点"].points()] == [
        (2.0, 65.0),
        (3.0, 62.0),
    ]
    assert [(point.x(), point.y()) for point in q2_series["Q2 变化点"].points()] == [
        (2.0, 18.0),
    ]
    assert application is not None
