from __future__ import annotations

import logging
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import cv2
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from backend.vision.config import ChannelRegionConfig, DetectorConfig
from backend.vision.tuning import PipelineStage, TuningFrame, inspect_frame, read_video_frames

from .vision_tuning_store import TuningLoadStatus, VisionTuningSettingsStore


_LOGGER = logging.getLogger(__name__)
_INSPECTION_TIMEOUT_MS = 15_000

_INTEGER_PARAMETERS = {
    "channel_region.sample_frames",
    "channel_region.work_max_width",
    "channel_region.work_max_height",
    "channel_region.canny_low",
    "channel_region.canny_high",
    "channel_region.hough_threshold",
    "channel_region.max_lines",
}

# Algorithm-safe slider ranges: (minimum, maximum, step). Text entry remains
# available for exact values, while sliders cover the useful operating range.
_PARAMETER_RANGES: dict[str, tuple[float, float, float]] = {
    "min_radius": (1.0, 300.0, 1.0),
    "max_radius": (2.0, 300.0, 1.0),
    "min_center_distance": (1.0, 300.0, 1.0),
    "sensitivity": (0.0, 1.0, 0.01),
    "radius_adjustment_percent": (-20.0, 20.0, 1.0),
    "channel_region.sample_frames": (1.0, 48.0, 1.0),
    "channel_region.min_confidence": (0.0, 1.0, 0.01),
    "channel_region.frequency_window_ratio": (0.005, 0.20, 0.005),
    "channel_region.min_frequency_region_thickness_ratio": (0.005, 0.20, 0.005),
    "channel_region.min_frequency_frame_support": (0.0, 1.0, 0.05),
    "channel_region.min_region_contrast": (0.0, 1.0, 0.01),
    "channel_region.full_region_contrast": (0.01, 1.0, 0.01),
    "channel_region.min_region_coverage": (0.0, 1.0, 0.01),
    "channel_region.min_coverage_advantage": (0.0, 1.0, 0.01),
    "channel_region.work_max_width": (160.0, 3840.0, 20.0),
    "channel_region.work_max_height": (120.0, 2160.0, 20.0),
    "channel_region.canny_low": (0.0, 254.0, 1.0),
    "channel_region.canny_high": (1.0, 255.0, 1.0),
    "channel_region.hough_threshold": (8.0, 300.0, 1.0),
    "channel_region.min_line_length_ratio": (0.1, 1.0, 0.01),
    "channel_region.max_line_gap_ratio": (0.0, 0.5, 0.01),
    "channel_region.max_lines": (4.0, 120.0, 1.0),
    "channel_region.parallel_tolerance_degrees": (0.5, 30.0, 0.5),
    "channel_region.min_width_ratio": (0.01, 0.8, 0.01),
    "channel_region.max_width_ratio": (0.1, 1.0, 0.01),
    "channel_region.max_separation_variation_ratio": (0.01, 0.5, 0.01),
    "channel_region.high_frequency_weight": (0.0, 1.0, 0.01),
    "channel_region.straightness_weight": (0.0, 1.0, 0.01),
    "channel_region.geometry_weight": (0.0, 1.0, 0.01),
}


def _config_value(
    detector_config: DetectorConfig,
    channel_config: ChannelRegionConfig,
    key: str,
) -> int | float:
    if key.startswith("channel_region."):
        return getattr(channel_config, key.split(".", 1)[1])
    return getattr(detector_config, key)


def _validate_tuning_configs(
    detector_config: DetectorConfig,
    channel_config: ChannelRegionConfig,
) -> None:
    """Reject values known to make OpenCV invalid or pathologically expensive."""
    for key, (minimum, maximum, _step) in _PARAMETER_RANGES.items():
        value = _config_value(detector_config, channel_config, key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"参数 {key} 必须是数值")
        if not math.isfinite(float(value)):
            raise ValueError(f"参数 {key} 必须是有限数值")
        if not minimum <= float(value) <= maximum:
            raise ValueError(f"参数 {key} 必须在 {minimum:g}–{maximum:g} 之间")

    relationships = (
        (detector_config.min_radius, detector_config.max_radius, "最小半径不能大于最大半径"),
        (channel_config.min_width_ratio, channel_config.max_width_ratio, "最小管宽比例不能大于最大管宽比例"),
    )
    for lower, upper, message in relationships:
        if float(lower) > float(upper):
            raise ValueError(message)
    if channel_config.canny_low >= channel_config.canny_high:
        raise ValueError("Canny 低阈值必须小于高阈值")


class _InspectionSignals(QObject):
    succeeded = Signal(int, object)
    failed = Signal(int, object, str)
    finished = Signal(int)


class _InspectionWorker(QRunnable):
    def __init__(self, request_id: int, task: Callable[[], object]) -> None:
        super().__init__()
        self.request_id = request_id
        self.task = task
        self.signals = _InspectionSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.signals.succeeded.emit(self.request_id, self.task())
        except Exception as exc:
            self.signals.failed.emit(self.request_id, exc, traceback.format_exc())
        finally:
            self.signals.finished.emit(self.request_id)


def _pixmap(image) -> QPixmap:
    if image.ndim == 2:
        converted = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = converted.shape
    qimage = QImage(converted.data, width, height, 3 * width, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


class _NoWheelSlider(QSlider):
    """Prevent accidental parameter changes when scrolling over a slider."""

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        event.ignore()


class NumberSlider(QWidget):
    value_changed = Signal(object)

    def __init__(
        self,
        value: int | float,
        minimum: float,
        maximum: float,
        step: float,
        integer: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.step = max(1e-9, float(step))
        self.integer = bool(integer)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.slider = _NoWheelSlider(Qt.Horizontal)
        self.slider.setRange(0, int(round((self.maximum - self.minimum) / self.step)))
        self.slider.setMinimumWidth(150)
        self.slider.setToolTip(
            f"滑条范围 {self.minimum:g}–{self.maximum:g}，步长 {self.step:g}；松开后刷新"
        )
        self.input = QLineEdit()
        self.input.setMaximumWidth(76)
        self.input.setAlignment(Qt.AlignRight)
        self.input.setToolTip("可直接输入精确数值；停止输入 500ms 后刷新")
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.input)

        self._set_controls(float(value))
        self.slider.valueChanged.connect(self._slider_changed)
        self.slider.sliderReleased.connect(lambda: self.value_changed.emit(self._slider_value()))
        self.input.textChanged.connect(self.value_changed.emit)
        self.input.editingFinished.connect(self._input_finished)

    def _slider_value(self) -> int | float:
        value = self.minimum + self.slider.value() * self.step
        return int(round(value)) if self.integer else float(value)

    def _format(self, value: float) -> str:
        if self.integer:
            return str(int(round(value)))
        decimals = 0
        step = self.step
        while decimals < 4 and abs(step - round(step)) > 1e-9:
            step *= 10.0
            decimals += 1
        return f"{value:.{decimals}f}"

    def _set_controls(self, value: float) -> None:
        bounded = min(self.maximum, max(self.minimum, float(value)))
        position = int(round((bounded - self.minimum) / self.step))
        self.slider.setValue(position)
        self.input.setText(self._format(float(value)))

    def _slider_changed(self, _position: int) -> None:
        # Updating the displayed value is safe while dragging, but detection
        # must only be triggered by sliderReleased below. This prevents a
        # costly redraw from interrupting an in-progress drag.
        value = self._slider_value()
        self.input.blockSignals(True)
        self.input.setText(self._format(float(value)))
        self.input.blockSignals(False)

    def _input_finished(self) -> None:
        try:
            value = float(self.input.text())
        except ValueError:
            return
        normalized = int(round(value)) if self.integer else float(value)
        bounded = min(self.maximum, max(self.minimum, float(normalized)))
        position = int(round((bounded - self.minimum) / self.step))
        self.slider.blockSignals(True)
        self.slider.setValue(position)
        self.slider.blockSignals(False)
        self.input.blockSignals(True)
        self.input.setText(self._format(float(normalized)))
        self.input.blockSignals(False)
        self.value_changed.emit(normalized)


class _StageImage(QLabel):
    double_clicked = Signal()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.double_clicked.emit()
        super().mouseDoubleClickEvent(event)


class StageCard(QGroupBox):
    parameter_changed = Signal(str, object)
    parameter_reset = Signal(str)
    expanded_changed = Signal(bool)

    def __init__(
        self,
        stage: PipelineStage,
        controls: list[dict[str, Any]],
        parent=None,
        expanded: bool = False,
    ) -> None:
        super().__init__(stage.name, parent)
        self.stage = stage
        self.setMinimumWidth(390)
        # Cards should keep their own natural height.  Otherwise QGridLayout
        # stretches the shorter card when its neighbour's parameters open.
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout = QVBoxLayout(self)

        image = _StageImage()
        image.setAlignment(Qt.AlignCenter)
        image.setToolTip("双击图像放大查看")
        image.double_clicked.connect(self._zoom)
        image.setMinimumHeight(220)
        image.setStyleSheet("background:#111827;border-radius:5px")
        image.setPixmap(_pixmap(stage.image).scaled(440, 270, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(image)

        description = QLabel(stage.description)
        description.setWordWrap(True)
        description.setStyleSheet("font-weight:400;color:#475569")
        layout.addWidget(description)

        if controls:
            self.parameter_toggle = QPushButton()
            self.parameter_toggle.setObjectName("stageParameterToggle")
            self.parameter_toggle.setCheckable(True)
            self.parameter_toggle.setChecked(expanded)
            self.parameter_toggle.setCursor(Qt.PointingHandCursor)
            self.parameter_toggle.setMinimumHeight(38)
            self.parameter_toggle.setStyleSheet(
                "QPushButton#stageParameterToggle {"
                " background:#e7effc; color:#123f82; border:1px solid #7899cc;"
                " border-radius:7px; padding:8px 11px; text-align:left; font-weight:700;"
                "}"
                "QPushButton#stageParameterToggle:hover {"
                " background:#d7e6fb; border-color:#2563eb; color:#102f61;"
                "}"
                "QPushButton#stageParameterToggle:checked {"
                " background:#1d4ed8; border-color:#1e3a8a; color:#ffffff;"
                "}"
                "QPushButton#stageParameterToggle:checked:hover { background:#1e40af; }"
                "QPushButton#stageParameterToggle:focus { border:2px solid #f59e0b; padding:7px 10px; }"
            )
            self.parameter_toggle.setAccessibleName(f"{stage.name} 参数设置")
            self._update_parameter_toggle(expanded)
            self.parameter_toggle.toggled.connect(self._toggle_parameters)
            layout.addWidget(self.parameter_toggle)

            self.parameter_panel = QWidget()
            self.parameter_panel.setVisible(expanded)
            form = QFormLayout(self.parameter_panel)
            form.setContentsMargins(0, 4, 0, 0)
            for spec in controls:
                widget = self._control_widget(spec)
                modified = bool(spec.get("modified", False))
                label = QLabel(str(spec["label"]))
                if modified:
                    label.setStyleSheet("color:#dc2626")
                    label.setToolTip("此参数已偏离打开页面时的原设定")

                # Keep a stable, text-free restore affordance beside every
                # label.  The fixed slot prevents a redraw from moving the
                # control being edited when a parameter becomes modified.
                reset = QPushButton()
                reset.setFixedSize(14, 14)
                reset.setEnabled(modified)
                reset.setAccessibleName(f"恢复参数 {spec['label']}")
                reset.setToolTip("恢复此参数的原设定" if modified else "参数未修改")
                reset.setStyleSheet(
                    "QPushButton { border: none; border-radius: 7px; "
                    f"background-color: {'#f59e0b' if modified else '#cbd5e1'}; }}"
                    "QPushButton:disabled { background-color: #cbd5e1; }"
                )
                reset.clicked.connect(
                    lambda _checked=False, name=str(spec["key"]): self.parameter_reset.emit(name)
                )
                label_row = QWidget()
                label_layout = QHBoxLayout(label_row)
                label_layout.setContentsMargins(0, 0, 4, 0)
                label_layout.setSpacing(4)
                label_layout.addWidget(label, 1)
                label_layout.addWidget(reset)
                form.addRow(label_row, widget)
            layout.addWidget(self.parameter_panel)

        if stage.parameters:
            parameters = QLabel("当前操作：" + stage.parameters)
            parameters.setWordWrap(True)
            parameters.setStyleSheet("font-weight:400;color:#1d4ed8")
            layout.addWidget(parameters)
        if stage.statistics:
            statistics = QLabel("结果：" + stage.statistics)
            statistics.setWordWrap(True)
            statistics.setStyleSheet("font-weight:600;color:#166534")
            layout.addWidget(statistics)

    def _update_parameter_toggle(self, expanded: bool) -> None:
        self.parameter_toggle.setText("－  参数设置（点击收起）" if expanded else "＋  参数设置（点击展开）")
        self.parameter_toggle.setAccessibleDescription(
            "参数设置已展开，点击收起" if expanded else "参数设置已收起，点击展开"
        )

    def _toggle_parameters(self, expanded: bool) -> None:
        self.parameter_panel.setVisible(expanded)
        self._update_parameter_toggle(expanded)
        self.expanded_changed.emit(expanded)

    def _control_widget(self, spec: dict[str, Any]) -> QWidget:
        key = str(spec["key"])
        kind = str(spec.get("kind", "number"))
        value = spec.get("value")
        if kind == "check":
            widget = QCheckBox(str(spec.get("text", "启用此步骤")))
            widget.setChecked(bool(value))
            widget.toggled.connect(lambda checked, name=key: self.parameter_changed.emit(name, checked))
            return widget
        if kind == "choice":
            widget = QComboBox()
            for label, option_value in spec["options"]:
                widget.addItem(label, option_value)
            index = widget.findData(value)
            widget.setCurrentIndex(max(0, index))
            widget.currentIndexChanged.connect(
                lambda _index, combo=widget, name=key: self.parameter_changed.emit(name, combo.currentData())
            )
            return widget

        slider_range = _PARAMETER_RANGES.get(key)
        if slider_range is not None:
            minimum, maximum, step = slider_range
            widget = NumberSlider(
                value,
                minimum,
                maximum,
                step,
                key in _INTEGER_PARAMETERS,
            )
            widget.value_changed.connect(
                lambda changed, name=key: self.parameter_changed.emit(name, changed)
            )
            return widget

        widget = QLineEdit(str(value))
        widget.setMaximumWidth(180)
        widget.textChanged.connect(lambda text, name=key: self.parameter_changed.emit(name, text))
        return widget

    def _zoom(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(self.stage.name)
        dialog.resize(1000, 760)
        layout = QVBoxLayout(dialog)
        image = QLabel()
        image.setAlignment(Qt.AlignCenter)
        image.setPixmap(_pixmap(self.stage.image).scaled(940, 650, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(image, 1)
        details = QLabel(
            "\n".join(
                value
                for value in (self.stage.description, self.stage.parameters, self.stage.statistics)
                if value
            )
        )
        details.setWordWrap(True)
        layout.addWidget(details)
        dialog.exec()


class TuningWindow(QWidget):
    def __init__(
        self,
        initial_video: str = "",
        parent: QWidget | None = None,
        on_sample_loaded: Callable[[str], None] | None = None,
        settings_store: VisionTuningSettingsStore | None = None,
        on_parameters_saved: Callable[[DetectorConfig, ChannelRegionConfig], object] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("液滴识别算法调参工作台")
        self.resize(1280, 820)
        self.frames: list[TuningFrame] = []
        self.frame_pos = 0
        self._on_sample_loaded = on_sample_loaded
        self._settings_store = settings_store
        self._on_parameters_saved = on_parameters_saved
        self._expanded_stages: set[int] = set()
        detector_config, channel_config, self._initial_parameter_status = self._load_initial_parameters()
        self.original_config = DetectorConfig(**vars(detector_config))
        self.current_config = DetectorConfig(**vars(detector_config))
        self.original_channel_config = ChannelRegionConfig(**vars(channel_config))
        self.current_channel_config = ChannelRegionConfig(**vars(channel_config))
        self._last_good_config = DetectorConfig(**vars(self.current_config))
        self._last_good_channel_config = ChannelRegionConfig(**vars(self.current_channel_config))
        self._last_good_stages: list[PipelineStage] = []
        self._config_revision = 0
        self._sample_revision = 0
        self._request_sequence = 0
        self._active_request: int | None = None
        self._request_context: dict[int, tuple[int, int, int, DetectorConfig, ChannelRegionConfig]] = {}
        self._workers: dict[int, _InspectionWorker] = {}
        self._pending_refresh = False
        self._timed_out_requests: set[int] = set()
        self._inspection_pool = QThreadPool.globalInstance()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(500)
        self._refresh_timer.timeout.connect(self._redraw)
        self._inspection_timeout = QTimer(self)
        self._inspection_timeout.setSingleShot(True)
        self._inspection_timeout.setInterval(_INSPECTION_TIMEOUT_MS)
        self._inspection_timeout.timeout.connect(self._inspection_timed_out)
        self._build(initial_video)
        if self._initial_parameter_status:
            self.status.setText(self._initial_parameter_status)

    def _load_initial_parameters(self) -> tuple[DetectorConfig, ChannelRegionConfig, str]:
        defaults = (DetectorConfig(), ChannelRegionConfig())
        if self._settings_store is None:
            return defaults[0], defaults[1], ""
        try:
            result = self._settings_store.load_or_create()
            if result.status is not TuningLoadStatus.INVALID:
                _validate_tuning_configs(result.detector, result.channel_region)
                message = "已创建并加载默认算法参数。" if result.status is TuningLoadStatus.CREATED else "已加载用户算法参数。"
                return result.detector, result.channel_region, message
            error = result.error
        except Exception as exc:
            error = str(exc)

        answer = QMessageBox.question(
            self,
            "算法参数版本不兼容",
            "用户算法参数与当前算法格式不一致，可能由算法更新导致。\n"
            f"详情：{error}\n\n是否删除老版本参数并重建默认参数？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            try:
                detector, channel = self._settings_store.delete_and_create_defaults()
                return detector, channel, "已删除老版本参数，并重建默认算法参数。"
            except Exception as exc:
                QMessageBox.warning(self, "参数重建失败", str(exc))
                return defaults[0], defaults[1], "参数文件重建失败，本次使用默认算法参数。"
        return defaults[0], defaults[1], "已保留老版本参数文件，本次使用默认算法参数。"

    def _build(self, initial_video: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("样本"))
        self.video = QLineEdit(initial_video)
        self.video.setPlaceholderText("选择本地视频或图像文件")
        toolbar.addWidget(self.video, 2)
        browse = QPushButton("选择样本…")
        browse.clicked.connect(self._browse)
        toolbar.addWidget(browse)
        self.video.editingFinished.connect(self._load)
        previous = QPushButton("上一帧")
        previous.clicked.connect(lambda: self.slider.setValue(max(0, self.slider.value() - 1)))
        toolbar.addWidget(previous)
        self.slider = _NoWheelSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setMinimumWidth(160)
        self.slider.valueChanged.connect(self._slider_changed)
        toolbar.addWidget(self.slider, 1)
        next_frame = QPushButton("下一帧")
        next_frame.clicked.connect(lambda: self.slider.setValue(min(self.slider.maximum(), self.slider.value() + 1)))
        toolbar.addWidget(next_frame)
        save = QPushButton("保存参数")
        save.clicked.connect(self._save)
        toolbar.addWidget(save)
        self.reset = QPushButton("回退参数修改")
        self.reset.setEnabled(False)
        self.reset.clicked.connect(self._reset_parameters)
        toolbar.addWidget(self.reset)
        layout.addLayout(toolbar)

        self.status = QLabel("等待加载样本；参数修改或步骤开关变化后，将在 500ms 内自动刷新。")
        self.status.setStyleSheet("color:#475569;padding:2px 4px")
        layout.addWidget(self.status)

        self.stage_container = QWidget()
        self.stage_grid = QGridLayout(self.stage_container)
        self.stage_grid.setContentsMargins(6, 6, 6, 6)
        self.stage_grid.setSpacing(12)
        placeholder = QLabel("请选择视频或图像样本")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setMinimumHeight(480)
        placeholder.setStyleSheet("background:#111827;color:#cbd5e1;border-radius:6px")
        self.stage_grid.addWidget(placeholder, 0, 0, 1, 2)
        self.stage_scroll = QScrollArea()
        self.stage_scroll.setWidgetResizable(True)
        self.stage_scroll.setWidget(self.stage_container)
        layout.addWidget(self.stage_scroll, 1)
        if initial_video:
            QTimer.singleShot(0, self._load)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择样本",
            "",
            "样本 (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)",
        )
        if path:
            self.video.setText(path)
            self._load()

    def _load(self) -> None:
        self._sample_revision += 1
        self._refresh_timer.stop()
        try:
            path = self.video.text().strip()
            suffix = Path(path).suffix.lower()
            if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                image = cv2.imread(path)
                if image is None:
                    raise ValueError(f"无法读取图像：{path}")
                self.frames = [TuningFrame(0, image)]
            else:
                self.frames = read_video_frames(path, max_frames=80)
            self.slider.blockSignals(True)
            self.slider.setRange(0, len(self.frames) - 1)
            self.slider.setEnabled(len(self.frames) > 1)
            self.slider.setValue(0)
            self.slider.blockSignals(False)
            self.frame_pos = 0
            if self._on_sample_loaded is not None:
                self._on_sample_loaded(path)
            self._redraw()
        except Exception as exc:
            self.frames = []
            self._pending_refresh = False
            self.slider.setEnabled(False)
            self.status.setText(f"加载失败：{exc}")
            QMessageBox.warning(self, "加载失败", str(exc))

    def _slider_changed(self, value: int) -> None:
        self.frame_pos = value
        self._refresh_timer.stop()
        if self.frames:
            self._redraw()

    def _parameter_owner(self, key: str) -> tuple[object, str]:
        if key.startswith("channel_region."):
            return self.current_channel_config, key.split(".", 1)[1]
        return self.current_config, key

    def _original_parameter_owner(self, key: str) -> tuple[object, str]:
        if key.startswith("channel_region."):
            return self.original_channel_config, key.split(".", 1)[1]
        return self.original_config, key

    def _parameter_changed(self, key: str, raw_value: object) -> None:
        try:
            owner, field = self._parameter_owner(key)
            if not hasattr(owner, field):
                return
            if isinstance(raw_value, str) and key in _INTEGER_PARAMETERS:
                setattr(owner, field, int(float(raw_value)))
            elif isinstance(raw_value, str):
                current = getattr(owner, field)
                setattr(owner, field, float(raw_value) if isinstance(current, float) else raw_value)
            else:
                setattr(owner, field, raw_value)
        except (TypeError, ValueError):
            self.status.setText(f"参数 {key} 尚未输入完成")
            return
        except Exception as exc:
            _LOGGER.exception("更新调参字段 %s 失败", key)
            self.status.setText(f"参数 {key} 更新失败：{exc}")
            return
        self._config_revision += 1
        self.reset.setEnabled(self._has_modified_parameters())
        if self.frames:
            self.status.setText("参数已修改，正在等待自动刷新…")
            self._refresh_timer.start()

    def _reset_parameter(self, key: str) -> None:
        owner, field = self._parameter_owner(key)
        original, original_field = self._original_parameter_owner(key)
        if not hasattr(original, original_field):
            return
        setattr(owner, field, getattr(original, original_field))
        self._config_revision += 1
        self.reset.setEnabled(self._has_modified_parameters())
        if self.frames:
            self.status.setText(f"参数 {key} 已恢复原设定")
            self._redraw()
        else:
            self.status.setText(f"参数 {key} 已恢复原设定")

    def _has_modified_parameters(self) -> bool:
        detector_modified = any(
            getattr(self.current_config, field) != getattr(self.original_config, field)
            for field in vars(self.original_config)
        )
        channel_modified = any(
            getattr(self.current_channel_config, field) != getattr(self.original_channel_config, field)
            for field in vars(self.original_channel_config)
        )
        return detector_modified or channel_modified

    def _reset_parameters(self) -> None:
        self.current_config = DetectorConfig(**vars(self.original_config))
        self.current_channel_config = ChannelRegionConfig(**vars(self.original_channel_config))
        self._config_revision += 1
        self.reset.setEnabled(False)
        if self.frames:
            self._redraw()
        else:
            self.status.setText("参数已回退为原设定")

    def _controls_for_stage(self, index: int) -> list[dict[str, Any]]:
        config = self.current_config
        channel = self.current_channel_config
        controls: dict[int, list[dict[str, Any]]] = {
            0: [
                self._check("channel_region.enabled", "启用管道区域检定", channel.enabled),
                self._number("channel_region.sample_frames", "启动采样帧数", channel.sample_frames),
            ],
            1: [
                self._number("channel_region.frequency_window_ratio", "局部高频窗口比例", channel.frequency_window_ratio),
                self._number("channel_region.min_frequency_region_thickness_ratio", "最小高频区域厚度比例", channel.min_frequency_region_thickness_ratio),
                self._number("channel_region.min_frequency_frame_support", "最低帧持续比例", channel.min_frequency_frame_support),
                self._number("channel_region.canny_low", "界线 Canny 低阈值", channel.canny_low),
                self._number("channel_region.canny_high", "界线 Canny 高阈值", channel.canny_high),
                self._number("channel_region.work_max_width", "最大工作宽度", channel.work_max_width),
                self._number("channel_region.work_max_height", "最大工作高度", channel.work_max_height),
            ],
            2: [
                self._number("channel_region.hough_threshold", "直线累加阈值", channel.hough_threshold),
                self._number("channel_region.min_line_length_ratio", "最短直线比例", channel.min_line_length_ratio),
                self._number("channel_region.max_line_gap_ratio", "最大断线间隙比例", channel.max_line_gap_ratio),
                self._number("channel_region.max_lines", "最大候选直线数", channel.max_lines),
                self._number("channel_region.parallel_tolerance_degrees", "平行角容差（°）", channel.parallel_tolerance_degrees),
            ],
            3: [
                self._number("channel_region.min_confidence", "最低可信度", channel.min_confidence),
                self._number("channel_region.min_width_ratio", "最小管宽比例", channel.min_width_ratio),
                self._number("channel_region.max_width_ratio", "最大管宽比例", channel.max_width_ratio),
                self._number("channel_region.max_separation_variation_ratio", "间距变化容差", channel.max_separation_variation_ratio),
                self._number("channel_region.min_region_contrast", "最低内外高频对比", channel.min_region_contrast),
                self._number("channel_region.full_region_contrast", "满分内外高频对比", channel.full_region_contrast),
                self._number("channel_region.min_region_coverage", "最低高频区域覆盖率", channel.min_region_coverage),
                self._number("channel_region.min_coverage_advantage", "最低内外覆盖率差", channel.min_coverage_advantage),
                self._number("channel_region.high_frequency_weight", "高低频区域权重", channel.high_frequency_weight),
                self._number("channel_region.straightness_weight", "直线性质权重", channel.straightness_weight),
                self._number("channel_region.geometry_weight", "区域几何权重", channel.geometry_weight),
            ],
            10: [
                self._check("enable_hough_candidates", "启用全帧 Hough", config.enable_hough_candidates),
                self._number("min_radius", "最小半径", config.min_radius),
                self._number("max_radius", "最大半径", config.max_radius),
                self._number("min_center_distance", "最小圆心距离", config.min_center_distance),
                self._number("sensitivity", "识别敏感度", config.sensitivity),
            ],
            7: [
                self._number(
                    "radius_adjustment_percent",
                    "液滴整体尺寸调节（%）",
                    config.radius_adjustment_percent,
                ),
            ],
            11: [
                self._number(
                    "radius_adjustment_percent",
                    "液滴整体尺寸调节（%）",
                    config.radius_adjustment_percent,
                ),
            ],
        }
        return controls.get(index, [])

    def _check(self, key: str, label: str, value: bool) -> dict[str, Any]:
        return {
            "key": key,
            "label": label,
            "kind": "check",
            "value": value,
            "text": "执行此步骤",
            "modified": value != getattr(self._original_parameter_owner(key)[0], self._original_parameter_owner(key)[1]),
        }

    def _number(self, key: str, label: str, value: int | float) -> dict[str, Any]:
        return {
            "key": key,
            "label": label,
            "kind": "number",
            "value": value,
            "modified": value != getattr(self._original_parameter_owner(key)[0], self._original_parameter_owner(key)[1]),
        }

    def _redraw(self) -> None:
        if not self.frames:
            return
        if self._active_request is not None:
            self._pending_refresh = True
            self.status.setText("算法仍在后台计算；已记录最新参数，完成后自动刷新。")
            return

        detector_config = DetectorConfig(**vars(self.current_config))
        channel_config = ChannelRegionConfig(**vars(self.current_channel_config))
        try:
            _validate_tuning_configs(detector_config, channel_config)
        except Exception as exc:
            self._rollback_after_failure(f"参数无效：{exc}")
            return

        frame_position = self.frame_pos
        sample_revision = self._sample_revision
        config_revision = self._config_revision
        frame = self.frames[frame_position].image
        channel_frames = [item.image for item in self.frames]
        self._request_sequence += 1
        request_id = self._request_sequence

        def task() -> object:
            return inspect_frame(
                frame,
                detector_config,
                channel_config=channel_config,
                channel_frames=channel_frames,
            )

        worker = _InspectionWorker(request_id, task)
        worker.signals.succeeded.connect(self._inspection_succeeded)
        worker.signals.failed.connect(self._inspection_failed)
        worker.signals.finished.connect(self._inspection_finished)
        self._workers[request_id] = worker
        self._request_context[request_id] = (
            sample_revision,
            config_revision,
            frame_position,
            detector_config,
            channel_config,
        )
        self._active_request = request_id
        self._pending_refresh = False
        self.status.setText("正在后台运行识别算法…")
        self._inspection_timeout.start()
        self._inspection_pool.start(worker)

    def _request_is_stale(self, request_id: int) -> bool:
        context = self._request_context.get(request_id)
        if context is None:
            return True
        sample_revision, config_revision, frame_position, _detector, _channel = context
        return (
            sample_revision != self._sample_revision
            or config_revision != self._config_revision
            or frame_position != self.frame_pos
            or not self.frames
        )

    def _inspection_succeeded(self, request_id: int, payload: object) -> None:
        if request_id != self._active_request:
            return
        timed_out = request_id in self._timed_out_requests
        stale = self._request_is_stale(request_id)
        context = self._request_context.get(request_id)
        self._complete_active_request(request_id)
        if timed_out or stale or context is None:
            self._start_pending_refresh()
            return

        try:
            result, stages = payload
            self._show_stages(stages)
        except Exception as exc:
            _LOGGER.exception("显示识别算法结果失败")
            self._rollback_after_failure(f"结果显示失败：{exc}")
            return

        _sample_revision, _config_revision, frame_position, detector_config, channel_config = context
        self._last_good_config = DetectorConfig(**vars(detector_config))
        self._last_good_channel_config = ChannelRegionConfig(**vars(channel_config))
        self._last_good_stages = list(stages)
        frame_index = self.frames[frame_position].index
        self.status.setText(
            f"帧 {frame_index}（{frame_position + 1}/{len(self.frames)}）：检测到 {len(result.centers)} 个液滴"
        )
        self._start_pending_refresh()

    def _inspection_failed(self, request_id: int, exc: object, details: str) -> None:
        if request_id != self._active_request:
            return
        timed_out = request_id in self._timed_out_requests
        stale = self._request_is_stale(request_id)
        self._complete_active_request(request_id)
        _LOGGER.error("调参识别任务失败：%s\n%s", exc, details)
        if timed_out or stale:
            self._start_pending_refresh()
            return
        self._rollback_after_failure(f"识别失败：{exc}")

    def _inspection_finished(self, request_id: int) -> None:
        self._workers.pop(request_id, None)
        if request_id != self._active_request:
            self._request_context.pop(request_id, None)
            self._timed_out_requests.discard(request_id)

    def _inspection_timed_out(self) -> None:
        request_id = self._active_request
        if request_id is None:
            return
        self._timed_out_requests.add(request_id)
        if not self._request_is_stale(request_id):
            self._rollback_to_last_good()
            self._pending_refresh = False
            self.status.setText(
                "识别算法运行超过 15 秒，已回退到上次有效参数；后台任务结束前界面仍可继续操作。"
            )

    def _complete_active_request(self, request_id: int) -> None:
        if request_id != self._active_request:
            return
        self._inspection_timeout.stop()
        self._active_request = None
        self._request_context.pop(request_id, None)
        self._timed_out_requests.discard(request_id)

    def _start_pending_refresh(self) -> None:
        if self._active_request is not None or not self._pending_refresh or not self.frames:
            return
        self._pending_refresh = False
        QTimer.singleShot(0, self._redraw)

    def _rollback_to_last_good(self) -> None:
        self.current_config = DetectorConfig(**vars(self._last_good_config))
        self.current_channel_config = ChannelRegionConfig(**vars(self._last_good_channel_config))
        self._config_revision += 1
        self.reset.setEnabled(self._has_modified_parameters())
        if self._last_good_stages:
            try:
                self._show_stages(self._last_good_stages)
            except Exception:
                _LOGGER.exception("恢复上次成功的算法画面失败")

    def _rollback_after_failure(self, message: str) -> None:
        self._pending_refresh = False
        self._rollback_to_last_good()
        self.status.setText(f"{message}；已回退到上次有效参数。")

    def _show_stages(self, stages: list[PipelineStage]) -> None:
        scroll_position = self.stage_scroll.verticalScrollBar().value()
        while self.stage_grid.count():
            item = self.stage_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for index, stage in enumerate(stages):
            card = StageCard(
                stage,
                self._controls_for_stage(index),
                self.stage_container,
                expanded=index in self._expanded_stages,
            )
            card.expanded_changed.connect(
                lambda expanded, stage_index=index: self._set_stage_expanded(stage_index, expanded)
            )
            card.parameter_changed.connect(self._parameter_changed)
            card.parameter_reset.connect(self._reset_parameter)
            # Top alignment keeps this card independent from a taller card in
            # the same row, so opening one panel never stretches its neighbour.
            self.stage_grid.addWidget(card, index // 2, index % 2, Qt.AlignTop)
        self.stage_grid.setRowStretch((len(stages) + 1) // 2, 1)
        QTimer.singleShot(0, lambda: self.stage_scroll.verticalScrollBar().setValue(scroll_position))

    def _set_stage_expanded(self, index: int, expanded: bool) -> None:
        if expanded:
            self._expanded_stages.add(index)
        else:
            self._expanded_stages.discard(index)

    def _save(self) -> None:
        if self._settings_store is None:
            QMessageBox.warning(self, "保存失败", "未配置算法参数存储位置。")
            return
        try:
            _validate_tuning_configs(self.current_config, self.current_channel_config)
            self._settings_store.save(self.current_config, self.current_channel_config)
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            return

        apply_error = ""
        if self._on_parameters_saved is not None:
            try:
                self._on_parameters_saved(
                    DetectorConfig(**vars(self.current_config)),
                    ChannelRegionConfig(**vars(self.current_channel_config)),
                )
            except Exception as exc:
                apply_error = str(exc)

        self.original_config = DetectorConfig(**vars(self.current_config))
        self.original_channel_config = ChannelRegionConfig(**vars(self.current_channel_config))
        self.reset.setEnabled(False)
        if apply_error:
            self.status.setText(f"算法参数已保存，但未能应用到采样识别：{apply_error}")
            QMessageBox.warning(
                self,
                "实时应用失败",
                f"参数文件已经保存，但实时采样识别未更新：\n{apply_error}",
            )
        elif self._on_parameters_saved is not None:
            self.status.setText("算法参数已保存并应用到采样识别，从下一采样帧开始生效。")
        else:
            self.status.setText("算法参数已保存，并替代原用户参数。")


def main(video: str = "") -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = TuningWindow(video, settings_store=VisionTuningSettingsStore())
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
