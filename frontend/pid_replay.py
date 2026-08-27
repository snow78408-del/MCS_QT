from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from backend.orchestrator.pid_database import PIDReplayData, PIDReplaySample


_SERIES_COLORS = {
    "q1": QColor("#ef4444"),
    "q2": QColor("#2563eb"),
    "target": QColor("#f59e0b"),
    "diameter": QColor("#16a34a"),
    "speed": QColor("#9333ea"),
}


class PIDReplayDialog(QDialog):
    """Read-only replay of one persisted PID session."""

    def __init__(self, replay: PIDReplayData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.replay = replay
        self.setWindowTitle("PID 实验数据复现")
        self.resize(1180, 820)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                f"会话 {replay.session_id[:12]}    采样 {len(replay.samples)} 个控制周期    "
                f"时长 {replay.duration_s:.1f} s\n"
                f"开始：{replay.started_at_iso or '--'}    结束：{replay.stopped_at_iso or '--'}"
            )
        )

        notice = QLabel(
            "Q1/Q2 优先显示泵设备参数换算值；数据库没有回读值时显示控制指令。"
            "这些曲线不代表经流量传感器验证的物理流量。"
        )
        notice.setWordWrap(True)
        notice.setStyleSheet("color:#92400e;background:#fffbeb;padding:6px;border-radius:4px")
        layout.addWidget(notice)

        self.flow_chart = _build_flow_chart(replay)
        self.response_chart = _build_response_chart(replay)
        layout.addWidget(_chart_view(self.flow_chart), 1)
        layout.addWidget(_chart_view(self.response_chart), 1)

        buttons = QHBoxLayout()
        buttons.addStretch()
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)


def _build_flow_chart(replay: PIDReplayData) -> QChart:
    chart = QChart()
    chart.setTitle("泵流量随时间变化")
    axis_x = _time_axis(replay.duration_s)
    axis_y = QValueAxis()
    axis_y.setTitleText("流量（μL/min）")
    axis_y.setLabelFormat("%.2f")

    q1_label = _flow_label("Q1", replay.q1_flow_source)
    q2_label = _flow_label("Q2", replay.q2_flow_source)
    q1 = _line_series(q1_label, _SERIES_COLORS["q1"], replay.samples, "q1_flow_ul_min")
    q2 = _line_series(q2_label, _SERIES_COLORS["q2"], replay.samples, "q2_flow_ul_min")
    _add_axes(chart, axis_x, axis_y)
    _attach_series(chart, q1, axis_x, axis_y)
    _attach_series(chart, q2, axis_x, axis_y)
    flow_values = [sample.q1_flow_ul_min for sample in replay.samples]
    flow_values.extend(sample.q2_flow_ul_min for sample in replay.samples)
    axis_y.setRange(*_value_range(flow_values))
    chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
    return chart


def _build_response_chart(replay: PIDReplayData) -> QChart:
    chart = QChart()
    chart.setTitle("液滴响应随时间变化")
    axis_x = _time_axis(replay.duration_s)
    diameter_axis = QValueAxis()
    diameter_axis.setTitleText("平均直径（μm）")
    diameter_axis.setLabelFormat("%.2f")
    speed_axis = QValueAxis()
    speed_axis.setTitleText("平均速度（μm/s）")
    speed_axis.setLabelFormat("%.1f")

    diameter = _line_series(
        "液滴平均直径",
        _SERIES_COLORS["diameter"],
        replay.samples,
        "measured_diameter_um",
    )
    target = _line_series(
        "目标液滴直径（阶跃）",
        _SERIES_COLORS["target"],
        replay.samples,
        "target_diameter_um",
        line_style=Qt.PenStyle.DashLine,
    )
    speed = _line_series(
        "液滴平均速度",
        _SERIES_COLORS["speed"],
        replay.samples,
        "droplet_speed_um_s",
    )
    chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
    chart.addAxis(diameter_axis, Qt.AlignmentFlag.AlignLeft)
    chart.addAxis(speed_axis, Qt.AlignmentFlag.AlignRight)
    _attach_series(chart, diameter, axis_x, diameter_axis)
    _attach_series(chart, target, axis_x, diameter_axis)
    _attach_series(chart, speed, axis_x, speed_axis)
    diameter_values = [sample.measured_diameter_um for sample in replay.samples]
    diameter_values.extend(sample.target_diameter_um for sample in replay.samples)
    diameter_axis.setRange(*_value_range(diameter_values))
    speed_axis.setRange(*_value_range(sample.droplet_speed_um_s for sample in replay.samples))
    chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
    return chart


def _time_axis(duration_s: float) -> QValueAxis:
    axis = QValueAxis()
    axis.setTitleText("实验时间（s）")
    axis.setLabelFormat("%.1f")
    axis.setRange(0.0, max(1.0, float(duration_s)))
    axis.setTickCount(7)
    return axis


def _line_series(
    name: str,
    color: QColor,
    samples: Iterable[PIDReplaySample],
    value_field: str,
    line_style: Qt.PenStyle = Qt.PenStyle.SolidLine,
) -> QLineSeries:
    series = QLineSeries()
    series.setName(name)
    pen = QPen(color)
    pen.setWidthF(2.2)
    pen.setStyle(line_style)
    series.setPen(pen)
    for sample in samples:
        value = getattr(sample, value_field)
        if value is not None:
            series.append(float(sample.elapsed_s), float(value))
    return series


def _add_axes(chart: QChart, axis_x: QValueAxis, axis_y: QValueAxis) -> None:
    chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
    chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)


def _attach_series(chart: QChart, series: QLineSeries, axis_x: QValueAxis, axis_y: QValueAxis) -> None:
    chart.addSeries(series)
    series.attachAxis(axis_x)
    series.attachAxis(axis_y)


def _chart_view(chart: QChart) -> QChartView:
    view = QChartView(chart)
    view.setRenderHint(QPainter.RenderHint.Antialiasing)
    view.setMinimumHeight(300)
    return view


def _flow_label(channel: str, source: str) -> str:
    if source == "device_parameter_estimate":
        return f"{channel} 设备参数换算值"
    return f"{channel} 控制指令值"


def _value_range(values: Iterable[float | None]) -> tuple[float, float]:
    finite_values = [float(value) for value in values if value is not None]
    if not finite_values:
        return 0.0, 1.0
    minimum = min(finite_values)
    maximum = max(finite_values)
    if minimum == maximum:
        margin = max(1.0, abs(minimum) * 0.05)
    else:
        margin = (maximum - minimum) * 0.08
    return minimum - margin, maximum + margin
