from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QScatterSeries, QValueAxis
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from backend.orchestrator.pid_database import PIDReplayData, PIDReplaySample


_SERIES_COLORS = {
    "q1": QColor("#ef4444"),
    "q2": QColor("#2563eb"),
    "q1_change": QColor("#991b1b"),
    "q2_change": QColor("#1e3a8a"),
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
        self.resize(1320, 900)

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

        self.q1_flow_chart = _build_channel_flow_chart(replay, "q1")
        self.q2_flow_chart = _build_channel_flow_chart(replay, "q2")
        self.response_chart = _build_response_chart(replay)
        flow_charts = QHBoxLayout()
        flow_charts.addWidget(_chart_view(self.q1_flow_chart), 1)
        flow_charts.addWidget(_chart_view(self.q2_flow_chart), 1)
        layout.addLayout(flow_charts, 1)
        layout.addWidget(_chart_view(self.response_chart), 1)

        buttons = QHBoxLayout()
        buttons.addStretch()
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)


def _build_channel_flow_chart(replay: PIDReplayData, channel: str) -> QChart:
    if channel not in {"q1", "q2"}:
        raise ValueError("channel must be q1 or q2")
    channel_name = channel.upper()
    pump_name = "泵1" if channel == "q1" else "泵2"
    value_field = f"{channel}_flow_ul_min"
    source = replay.q1_flow_source if channel == "q1" else replay.q2_flow_source
    values = [getattr(sample, value_field) for sample in replay.samples]
    changes = _flow_change_points(replay.samples, value_field)
    finite_values = [float(value) for value in values if value is not None]
    value_summary = "无有效流速数据"
    if finite_values:
        value_summary = (
            f"范围 {min(finite_values):.2f}–{max(finite_values):.2f} μL/min · "
            f"变化 {len(changes)} 次"
        )

    chart = QChart()
    chart.setTitle(f"{pump_name}（{channel_name}）流速变化｜{value_summary}")
    axis_x = _time_axis(replay.duration_s)
    axis_y = QValueAxis()
    axis_y.setTitleText("流量（μL/min）")
    axis_y.setLabelFormat("%.2f")

    flow = _line_series(
        _flow_label(channel_name, source),
        _SERIES_COLORS[channel],
        replay.samples,
        value_field,
        width=3.0,
    )
    change_markers = QScatterSeries()
    change_markers.setName(f"{channel_name} 变化点")
    change_markers.setMarkerSize(12.0)
    change_markers.setColor(_SERIES_COLORS[f"{channel}_change"])
    change_markers.setBorderColor(QColor("#ffffff"))
    for elapsed_s, value in changes:
        change_markers.append(elapsed_s, value)

    _add_axes(chart, axis_x, axis_y)
    _attach_series(chart, flow, axis_x, axis_y)
    _attach_series(chart, change_markers, axis_x, axis_y)
    axis_y.setRange(*_value_range(values))
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
    width: float = 2.2,
) -> QLineSeries:
    series = QLineSeries()
    series.setName(name)
    pen = QPen(color)
    pen.setWidthF(float(width))
    pen.setStyle(line_style)
    series.setPen(pen)
    for sample in samples:
        value = getattr(sample, value_field)
        if value is not None:
            series.append(float(sample.elapsed_s), float(value))
    return series


def _flow_change_points(
    samples: Iterable[PIDReplaySample],
    value_field: str,
) -> list[tuple[float, float]]:
    changes: list[tuple[float, float]] = []
    previous: float | None = None
    for sample in samples:
        raw_value = getattr(sample, value_field)
        if raw_value is None:
            continue
        value = float(raw_value)
        if previous is not None and abs(value - previous) > 1e-9:
            changes.append((float(sample.elapsed_s), value))
        previous = value
    return changes


def _add_axes(chart: QChart, axis_x: QValueAxis, axis_y: QValueAxis) -> None:
    chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
    chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)


def _attach_series(chart: QChart, series, axis_x: QValueAxis, axis_y: QValueAxis) -> None:
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
