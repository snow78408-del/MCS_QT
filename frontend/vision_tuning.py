from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import cv2
from PySide6.QtCore import Qt, QTimer, Signal
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
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from backend.vision.algorithm_profiles import AlgorithmProfileStore
from backend.vision.algorithms import get_algorithm, list_algorithms
from backend.vision.tuning import PipelineStage, TuningFrame, read_video_frames


_INTEGER_PARAMETERS = {
    "gaussian_blur_size",
    "adaptive_threshold_block_size",
    "morphology_open_kernel",
    "morphology_close_kernel",
    "contour_close_kernel",
    "hough_edge_neighborhood",
    "hough_refresh_interval",
    "edge_ownership_search_radius",
    "watershed_max_markers",
    "local_hough_max_regions",
}

# Algorithm-safe slider ranges: (minimum, maximum, step). Text entry remains
# available for exact values, while sliders cover the useful operating range.
_PARAMETER_RANGES: dict[str, tuple[float, float, float]] = {
    "gaussian_blur_size": (1.0, 31.0, 2.0),
    "contour_work_scale": (0.25, 1.0, 0.05),
    "adaptive_threshold_block_size": (3.0, 101.0, 2.0),
    "adaptive_threshold_c": (-20.0, 30.0, 0.5),
    "morphology_open_kernel": (1.0, 15.0, 2.0),
    "morphology_close_kernel": (1.0, 21.0, 2.0),
    "background_difference_threshold": (1.0, 80.0, 1.0),
    "background_learning_rate": (0.0, 0.25, 0.005),
    "contour_canny_low": (1.0, 250.0, 1.0),
    "contour_canny_high": (2.0, 400.0, 1.0),
    "contour_close_kernel": (1.0, 15.0, 2.0),
    "contour_min_circularity": (0.0, 1.0, 0.01),
    "contour_min_axis_ratio": (0.0, 1.0, 0.01),
    "contour_min_edge_support": (0.0, 1.0, 0.01),
    "contour_min_area_fill_ratio": (0.0, 1.0, 0.01),
    "watershed_peak_ratio": (0.2, 0.95, 0.01),
    "watershed_min_peak_radius_ratio": (0.1, 1.0, 0.01),
    "watershed_max_markers": (2.0, 30.0, 1.0),
    "local_hough_padding_ratio": (0.0, 2.0, 0.05),
    "local_hough_max_regions": (1.0, 40.0, 1.0),
    "hough_refresh_interval": (0.0, 30.0, 1.0),
    "hough_param1": (10.0, 300.0, 1.0),
    "hough_dp": (1.0, 3.0, 0.05),
    "hough_min_distance": (0.0, 200.0, 1.0),
    "hough_param2": (5.0, 100.0, 1.0),
    "min_radius": (1.0, 300.0, 1.0),
    "max_radius": (2.0, 300.0, 1.0),
    "hough_min_radius": (1.0, 300.0, 1.0),
    "hough_max_radius": (2.0, 300.0, 1.0),
    "hough_edge_support_threshold": (0.0, 1.0, 0.01),
    "hough_edge_neighborhood": (0.0, 8.0, 1.0),
    "expected_radius": (0.0, 300.0, 1.0),
    "expected_radius_tolerance_ratio": (0.0, 1.0, 0.01),
    "edge_ownership_search_radius": (1.0, 10.0, 1.0),
    "edge_ownership_margin": (0.0, 5.0, 0.05),
    "edge_ownership_min_ratio": (0.0, 1.0, 0.01),
    "candidate_min_edge_support": (0.0, 1.0, 0.01),
    "candidate_min_visible_circle_ratio": (0.0, 1.0, 0.01),
    "candidate_full_circle_ratio": (0.0, 1.0, 0.01),
}


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

    def __init__(self, stage: PipelineStage, controls: list[dict[str, Any]], parent=None) -> None:
        super().__init__(stage.name, parent)
        self.stage = stage
        self.setMinimumWidth(390)
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
            form = QFormLayout()
            form.setContentsMargins(0, 4, 0, 0)
            for spec in controls:
                widget = self._control_widget(spec)
                widget.setEnabled(bool(spec.get("editable", True)))
                form.addRow(self._parameter_label(spec), widget)
            layout.addLayout(form)

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

    def _parameter_label(self, spec: dict[str, Any]) -> QWidget:
        key = str(spec["key"])
        modified = bool(spec.get("modified", False))
        row = QWidget()
        row.setObjectName(f"parameter-label-{key}")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        label = QLabel(str(spec["label"]))
        if modified:
            label.setStyleSheet("color:#dc2626")
            label.setToolTip("此参数已偏离打开页面时的原设定")
        layout.addWidget(label)

        # The slot always keeps the same size, while the icon itself is only
        # shown for a modified value. This prevents a reset action from being
        # inserted into the slider row and shifting the card layout.
        icon_slot = QWidget()
        icon_slot.setFixedSize(24, 24)
        icon_layout = QHBoxLayout(icon_slot)
        icon_layout.setContentsMargins(1, 1, 1, 1)
        reset = QToolButton()
        reset.setObjectName(f"parameter-reset-{key}")
        reset.setText("↶")
        reset.setAccessibleName(f"恢复{spec['label']}的原设定")
        reset.setToolTip("恢复此参数的原设定")
        reset.setAutoRaise(True)
        reset.setFixedSize(22, 22)
        reset.setStyleSheet(
            "QToolButton{color:#dc2626;border:0;font-size:16px;padding:0}"
            "QToolButton:hover{background:#fee2e2;border-radius:4px}"
        )
        reset.clicked.connect(lambda _checked=False, name=key: self.parameter_reset.emit(name))
        reset.setVisible(modified and bool(spec.get("editable", True)))
        icon_layout.addWidget(reset)
        layout.addWidget(icon_slot)
        return row

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
                (key in _INTEGER_PARAMETERS) or (isinstance(value, int) and not isinstance(value, bool)),
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
        algorithm_store: AlgorithmProfileStore | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("液滴识别算法调参工作台")
        self.resize(1280, 820)
        self.frames: list[TuningFrame] = []
        self.frame_pos = 0
        self.algorithm_store = algorithm_store or AlgorithmProfileStore()
        self.current_profile = self.algorithm_store.active_profile()
        self.current_plugin = get_algorithm(self.current_profile.plugin_id)
        self.original_config = self.current_plugin.build_config(self.current_profile.parameters)
        self.current_config = self.current_plugin.build_config(self.current_profile.parameters)
        self._profile_dirty = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(500)
        self._refresh_timer.timeout.connect(self._redraw)
        self._build(initial_video)

    def _build(self, initial_video: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        algorithm_bar = QHBoxLayout()
        algorithm_bar.addWidget(QLabel("整体算法"))
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.setMinimumWidth(260)
        self.algorithm_combo.currentIndexChanged.connect(self._algorithm_selected)
        algorithm_bar.addWidget(self.algorithm_combo, 1)
        self.new_algorithm = QPushButton("新建…")
        self.new_algorithm.clicked.connect(self._create_algorithm)
        algorithm_bar.addWidget(self.new_algorithm)
        self.copy_algorithm = QPushButton("复制…")
        self.copy_algorithm.clicked.connect(self._copy_algorithm)
        algorithm_bar.addWidget(self.copy_algorithm)
        self.rename_algorithm = QPushButton("重命名…")
        self.rename_algorithm.clicked.connect(self._rename_algorithm)
        algorithm_bar.addWidget(self.rename_algorithm)
        self.delete_algorithm = QPushButton("删除")
        self.delete_algorithm.clicked.connect(self._delete_algorithm)
        algorithm_bar.addWidget(self.delete_algorithm)
        self.import_algorithm = QPushButton("导入…")
        self.import_algorithm.clicked.connect(self._import_algorithm)
        algorithm_bar.addWidget(self.import_algorithm)
        self.export_algorithm = QPushButton("导出…")
        self.export_algorithm.clicked.connect(self._export_algorithm)
        algorithm_bar.addWidget(self.export_algorithm)
        self.activate_algorithm = QPushButton("设为运行算法")
        self.activate_algorithm.clicked.connect(self._activate_algorithm)
        algorithm_bar.addWidget(self.activate_algorithm)
        layout.addLayout(algorithm_bar)
        self._reload_algorithm_combo(self.current_profile.profile_id)

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
        self.save_parameters = QPushButton("保存参数")
        self.save_parameters.clicked.connect(self._save)
        toolbar.addWidget(self.save_parameters)
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

    def _reload_algorithm_combo(self, selected_id: str) -> None:
        self.algorithm_combo.blockSignals(True)
        self.algorithm_combo.clear()
        for profile in self.algorithm_store.profiles():
            suffix = "  [内置只读]" if profile.protected else ""
            if profile.profile_id == self.algorithm_store.active_profile_id:
                suffix += "  [运行中]"
            self.algorithm_combo.addItem(profile.name + suffix, profile.profile_id)
        index = self.algorithm_combo.findData(selected_id)
        self.algorithm_combo.setCurrentIndex(max(0, index))
        self.algorithm_combo.blockSignals(False)
        self._sync_algorithm_actions()

    def _algorithm_selected(self, _index: int) -> None:
        profile_id = self.algorithm_combo.currentData()
        if not profile_id or profile_id == self.current_profile.profile_id:
            return
        if self._profile_dirty:
            answer = QMessageBox.question(
                self,
                "放弃未保存参数？",
                "当前算法有未保存的参数修改。切换算法将放弃这些修改，是否继续？",
            )
            if answer != QMessageBox.StandardButton.Yes:
                self._reload_algorithm_combo(self.current_profile.profile_id)
                return
        self._select_profile(str(profile_id))

    def _select_profile(self, profile_id: str) -> None:
        self.current_profile = self.algorithm_store.get(profile_id)
        self.current_plugin = get_algorithm(self.current_profile.plugin_id)
        self.original_config = self.current_plugin.build_config(self.current_profile.parameters)
        self.current_config = self.current_plugin.build_config(self.current_profile.parameters)
        self._profile_dirty = False
        self.reset.setEnabled(False)
        self._reload_algorithm_combo(profile_id)
        self._sync_algorithm_actions()
        if self.frames:
            self._redraw()
        else:
            self.status.setText(f"已选择算法：{self.current_profile.name}")

    def _sync_algorithm_actions(self) -> None:
        protected = self.current_profile.protected
        active = self.current_profile.profile_id == self.algorithm_store.active_profile_id
        self.rename_algorithm.setEnabled(not protected)
        self.delete_algorithm.setEnabled(not protected and not active)
        self.save_parameters.setEnabled(not protected)
        self.activate_algorithm.setEnabled(not active)
        self.activate_algorithm.setText("当前运行算法" if active else "设为运行算法")

    def _create_algorithm(self) -> None:
        plugins = list(list_algorithms())
        labels = [f"{item.display_name} — {item.description}" for item in plugins]
        selected, ok = QInputDialog.getItem(self, "新建算法", "选择算法实现", labels, 0, False)
        if not ok:
            return
        plugin = plugins[labels.index(selected)]
        name, ok = QInputDialog.getText(self, "新建算法", "算法名称")
        if not ok:
            return
        try:
            profile = self.algorithm_store.create(name, plugin.plugin_id)
            self._select_profile(profile.profile_id)
        except Exception as exc:
            QMessageBox.warning(self, "新建失败", str(exc))

    def _copy_algorithm(self) -> None:
        suggested = self.algorithm_store.next_copy_name(self.current_profile.profile_id)
        name, ok = QInputDialog.getText(self, "复制算法", "新算法名称", text=suggested)
        if not ok:
            return
        try:
            profile = self.algorithm_store.duplicate(self.current_profile.profile_id, name)
            self._select_profile(profile.profile_id)
        except Exception as exc:
            QMessageBox.warning(self, "复制失败", str(exc))

    def _rename_algorithm(self) -> None:
        name, ok = QInputDialog.getText(self, "重命名算法", "算法名称", text=self.current_profile.name)
        if not ok:
            return
        try:
            profile = self.algorithm_store.rename(self.current_profile.profile_id, name)
            self._select_profile(profile.profile_id)
        except Exception as exc:
            QMessageBox.warning(self, "重命名失败", str(exc))

    def _delete_algorithm(self) -> None:
        if QMessageBox.question(self, "删除算法", f"确定删除“{self.current_profile.name}”吗？") != QMessageBox.StandardButton.Yes:
            return
        try:
            self.algorithm_store.delete(self.current_profile.profile_id)
            self._select_profile(self.algorithm_store.active_profile_id)
        except Exception as exc:
            QMessageBox.warning(self, "删除失败", str(exc))

    def _import_algorithm(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "导入算法", "", "算法 JSON (*.json)")
        if not path:
            return
        try:
            profile = self.algorithm_store.import_profile(path)
            self._select_profile(profile.profile_id)
        except Exception as exc:
            QMessageBox.warning(self, "导入失败", str(exc))

    def _export_algorithm(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出算法",
            f"{self.current_profile.name}.json",
            "算法 JSON (*.json)",
        )
        if not path:
            return
        try:
            self.algorithm_store.export_profile(self.current_profile.profile_id, path)
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", str(exc))

    def _activate_algorithm(self) -> None:
        try:
            if self._profile_dirty and not self._save():
                return
            self.algorithm_store.activate(self.current_profile.profile_id)
            self._reload_algorithm_combo(self.current_profile.profile_id)
            self.status.setText(f"已将“{self.current_profile.name}”设为正式运行算法；下次初始化时生效。")
        except Exception as exc:
            QMessageBox.warning(self, "启用失败", str(exc))

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
            self._redraw()
        except Exception as exc:
            self.frames = []
            self.slider.setEnabled(False)
            QMessageBox.warning(self, "加载失败", str(exc))

    def _slider_changed(self, value: int) -> None:
        self.frame_pos = value
        self._refresh_timer.stop()
        if self.frames:
            self._redraw()

    def _parameter_changed(self, key: str, raw_value: object) -> None:
        if self.current_profile.protected:
            self.status.setText("内置算法为只读；请先复制为新算法后再调参")
            return
        try:
            if isinstance(raw_value, str) and hasattr(self.current_config, key):
                current = getattr(self.current_config, key)
                if isinstance(current, bool):
                    setattr(self.current_config, key, raw_value.strip().lower() in {"1", "true", "yes", "on"})
                elif isinstance(current, int):
                    setattr(self.current_config, key, int(float(raw_value)))
                elif isinstance(current, float):
                    setattr(self.current_config, key, float(raw_value))
                else:
                    setattr(self.current_config, key, raw_value)
            elif hasattr(self.current_config, key):
                setattr(self.current_config, key, raw_value)
            else:
                return
        except (TypeError, ValueError):
            self.status.setText(f"参数 {key} 尚未输入完成")
            return
        self._profile_dirty = self._has_modified_parameters()
        self.reset.setEnabled(self._profile_dirty)
        if self.frames:
            self.status.setText("参数已修改，正在等待自动刷新…")
            self._refresh_timer.start()

    def _reset_parameter(self, key: str) -> None:
        if self.current_profile.protected or not hasattr(self.original_config, key):
            return
        setattr(self.current_config, key, getattr(self.original_config, key))
        self._profile_dirty = self._has_modified_parameters()
        self.reset.setEnabled(self._profile_dirty)
        if self.frames:
            self.status.setText(f"参数 {key} 已恢复原设定")
            self._redraw()
        else:
            self.status.setText(f"参数 {key} 已恢复原设定")

    def _has_modified_parameters(self) -> bool:
        return self.current_plugin.serialize_config(self.current_config) != self.current_plugin.serialize_config(
            self.original_config
        )

    def _reset_parameters(self) -> None:
        self.current_config = self.current_plugin.build_config(
            self.current_plugin.serialize_config(self.original_config)
        )
        self._profile_dirty = False
        self.reset.setEnabled(False)
        if self.frames:
            self._redraw()
        else:
            self.status.setText("参数已回退为原设定")

    def _controls_for_stage(self, index: int) -> list[dict[str, Any]]:
        controls: list[dict[str, Any]] = []
        for parameter in self.current_plugin.parameters:
            if parameter.stage_index != index or not hasattr(self.current_config, parameter.key):
                continue
            value = getattr(self.current_config, parameter.key)
            controls.append(
                {
                    "key": parameter.key,
                    "label": parameter.label,
                    "kind": parameter.kind,
                    "value": value,
                    "text": parameter.text,
                    "modified": value != getattr(self.original_config, parameter.key),
                    "editable": not self.current_profile.protected,
                }
            )
        return controls

    def _redraw(self) -> None:
        if not self.frames:
            return
        try:
            result, stages = self.current_plugin.inspector(
                self.frames[self.frame_pos].image,
                self.current_config,
            )
            self._show_stages(stages)
            frame_index = self.frames[self.frame_pos].index
            self.status.setText(
                f"帧 {frame_index}（{self.frame_pos + 1}/{len(self.frames)}）：检测到 {len(result.centers)} 个液滴"
            )
        except Exception as exc:
            self.status.setText(f"识别失败：{exc}")

    def _show_stages(self, stages: list[PipelineStage]) -> None:
        scroll_position = self.stage_scroll.verticalScrollBar().value()
        while self.stage_grid.count():
            item = self.stage_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for index, stage in enumerate(stages):
            card = StageCard(stage, self._controls_for_stage(index), self.stage_container)
            card.parameter_changed.connect(self._parameter_changed)
            card.parameter_reset.connect(self._reset_parameter)
            self.stage_grid.addWidget(card, index // 2, index % 2)
        self.stage_grid.setRowStretch((len(stages) + 1) // 2, 1)
        QTimer.singleShot(0, lambda: self.stage_scroll.verticalScrollBar().setValue(scroll_position))

    def _save(self) -> bool:
        try:
            profile = self.algorithm_store.update_parameters(
                self.current_profile.profile_id,
                self.current_plugin.serialize_config(self.current_config),
            )
            self.current_profile = profile
            self.original_config = self.current_plugin.build_config(profile.parameters)
            self._profile_dirty = False
            self.reset.setEnabled(False)
            self.status.setText(f"算法“{profile.name}”的参数已保存")
            if self.frames:
                self._redraw()
            return True
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            return False


def main(video: str = "") -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = TuningWindow(video)
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
