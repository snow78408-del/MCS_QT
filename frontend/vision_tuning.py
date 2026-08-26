from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
    QScrollArea, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from backend.vision.config import DetectorConfig
from backend.vision.tuning import (
    SEARCHABLE_FIELDS, PipelineStage, grid_search, inspect_frame, read_video_frames,
)


class SearchWorker(QObject):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, path: str, max_frames: int, expected: float, base: DetectorConfig, grid: dict):
        super().__init__()
        self.path, self.max_frames, self.expected, self.base, self.grid = path, max_frames, expected, base, grid

    def run(self) -> None:
        try:
            frames = read_video_frames(self.path, self.max_frames)
            results = grid_search(frames, self.base, self.grid, self.expected)
            self.done.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))


def _pixmap(image) -> QPixmap:
    if image.ndim == 2:
        converted = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = converted.shape
    qimage = QImage(converted.data, width, height, 3 * width, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


class StageCard(QGroupBox):
    def __init__(self, stage: PipelineStage, parent=None) -> None:
        super().__init__(stage.name, parent)
        self.stage = stage
        self.setMinimumWidth(275)
        layout = QVBoxLayout(self)
        image = QLabel()
        image.setAlignment(Qt.AlignCenter)
        image.setMinimumHeight(170)
        image.setStyleSheet("background:#111827;border-radius:5px")
        image.setPixmap(_pixmap(stage.image).scaled(300, 185, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(image)
        description = QLabel(stage.description)
        description.setWordWrap(True)
        description.setStyleSheet("font-weight:400;color:#475569")
        layout.addWidget(description)
        if stage.parameters:
            parameters = QLabel("参数：" + stage.parameters)
            parameters.setWordWrap(True)
            parameters.setStyleSheet("font-weight:400;color:#1d4ed8")
            layout.addWidget(parameters)
        if stage.statistics:
            statistics = QLabel("结果：" + stage.statistics)
            statistics.setWordWrap(True)
            statistics.setStyleSheet("font-weight:600;color:#166534")
            layout.addWidget(statistics)
        zoom = QPushButton("放大查看")
        zoom.clicked.connect(self._zoom)
        layout.addWidget(zoom)

    def _zoom(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(self.stage.name)
        dialog.resize(1000, 760)
        layout = QVBoxLayout(dialog)
        image = QLabel()
        image.setAlignment(Qt.AlignCenter)
        image.setPixmap(_pixmap(self.stage.image).scaled(940, 650, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(image, 1)
        details = QLabel("\n".join(value for value in (self.stage.description, self.stage.parameters, self.stage.statistics) if value))
        details.setWordWrap(True)
        layout.addWidget(details)
        dialog.exec()


class TuningWindow(QWidget):
    def __init__(self, initial_video: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("液滴识别算法调参工作台")
        self.resize(1280, 820)
        self.frames = []
        self.frame_pos = 0
        self.current_config = DetectorConfig()
        self.worker_thread: QThread | None = None
        self.worker: SearchWorker | None = None
        self._fields: dict[str, QLineEdit] = {}
        self._build(initial_video)

    def _build(self, initial_video: str) -> None:
        root = self
        outer = QHBoxLayout(root)
        outer.setSpacing(14)
        control_panel = QWidget()
        controls = QVBoxLayout(control_panel)
        controls.setContentsMargins(0, 0, 0, 0)
        preview = QVBoxLayout()
        source = QGroupBox("1. 视频样本"); source_form = QFormLayout(source)
        self.video = QLineEdit(initial_video); browse = QPushButton("浏览…"); browse.clicked.connect(self._browse)
        row = QHBoxLayout(); row.addWidget(self.video); row.addWidget(browse); source_form.addRow("视频", row)
        self.max_frames = QSpinBox(); self.max_frames.setRange(1, 2000); self.max_frames.setValue(80); source_form.addRow("搜索帧数", self.max_frames)
        self.expected = QLineEdit("1"); source_form.addRow("期望液滴数/帧", self.expected)
        load = QPushButton("加载视频样本"); load.clicked.connect(self._load); source_form.addRow("", load); controls.addWidget(source)

        params = QGroupBox("2. 检测参数（修改后点击重新识别）"); form = QFormLayout(params)
        editable = ("min_radius", "max_radius", "circularity_threshold", "gaussian_blur_size", "morphology_open_kernel", "morphology_close_kernel", "candidate_min_edge_support", "candidate_full_circle_ratio", "hough_param2")
        for key in editable:
            field = QLineEdit(str(getattr(self.current_config, key))); self._fields[key] = field; form.addRow(key, field)
        redraw = QPushButton("重新识别当前帧"); redraw.clicked.connect(self._redraw); form.addRow("", redraw); controls.addWidget(params)

        search = QGroupBox("3. 自动参数搜索（无人工标注的启发式评分）"); search_form = QFormLayout(search)
        search_form.addRow(QLabel("每行格式：参数=值1,值2；只搜索下方支持的参数"))
        self.grid_text = QPlainTextEdit("min_radius=12,18,24\nmax_radius=32,40,50\ncircularity_threshold=0.12,0.2\nhough_param2=24,28,32")
        self.grid_text.setMinimumHeight(115); search_form.addRow(self.grid_text)
        self.search_button = QPushButton("开始自动搜索"); self.search_button.clicked.connect(self._search); search_form.addRow("", self.search_button)
        controls.addWidget(search); controls.addStretch()
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setMinimumWidth(440)
        control_scroll.setWidget(control_panel)
        outer.addWidget(control_scroll, 0)

        stage_header = QLabel("算法处理步骤总览（点击“重新识别当前帧”后更新）")
        stage_header.setStyleSheet("font-size:18px;font-weight:700")
        preview.addWidget(stage_header)
        self.stage_container = QWidget()
        self.stage_grid = QGridLayout(self.stage_container)
        self.stage_grid.setContentsMargins(6, 6, 6, 6)
        self.stage_grid.setSpacing(12)
        self.stage_placeholder = QLabel("请选择视频并加载样本")
        self.stage_placeholder.setAlignment(Qt.AlignCenter)
        self.stage_placeholder.setMinimumHeight(420)
        self.stage_placeholder.setStyleSheet("background:#111827;color:#cbd5e1;border-radius:6px")
        self.stage_grid.addWidget(self.stage_placeholder, 0, 0, 1, 2)
        stage_scroll = QScrollArea()
        stage_scroll.setWidgetResizable(True)
        stage_scroll.setWidget(self.stage_container)
        preview.addWidget(stage_scroll, 1)
        self.slider = QSlider(Qt.Horizontal); self.slider.setRange(0, 0); self.slider.valueChanged.connect(self._slider_changed); preview.addWidget(self.slider)
        nav = QHBoxLayout(); prev = QPushButton("上一帧"); prev.clicked.connect(lambda: self.slider.setValue(max(0, self.slider.value()-1))); nxt = QPushButton("下一帧"); nxt.clicked.connect(lambda: self.slider.setValue(min(self.slider.maximum(), self.slider.value()+1))); nav.addWidget(prev); nav.addWidget(nxt)
        self.status = QLabel("等待加载"); nav.addWidget(self.status, 1)
        save = QPushButton("保存当前参数 JSON"); save.clicked.connect(self._save); nav.addWidget(save); preview.addLayout(nav)
        self.results = QPlainTextEdit(); self.results.setReadOnly(True); preview.addWidget(self.results, 0)
        outer.addLayout(preview, 1)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频 (*.mp4 *.avi *.mov *.mkv)")
        if path: self.video.setText(path)

    def _load(self) -> None:
        try:
            self.frames = read_video_frames(self.video.text().strip(), self.max_frames.value())
            self.slider.setRange(0, len(self.frames) - 1); self.frame_pos = 0; self._redraw()
            self.status.setText(f"已加载 {len(self.frames)} 帧")
        except Exception as exc:
            QMessageBox.warning(self, "加载失败", str(exc))

    def _slider_changed(self, value: int) -> None:
        self.frame_pos = value
        if self.frames: self._redraw()

    def _config_from_fields(self) -> DetectorConfig:
        config = DetectorConfig()
        for key, field in self._fields.items():
            value = float(field.text())
            if key in {"gaussian_blur_size", "morphology_open_kernel", "morphology_close_kernel"}:
                value = int(value)
            setattr(config, key, value)
        return config

    def _redraw(self) -> None:
        if not self.frames: return
        try:
            config = self._config_from_fields()
            result, stages = inspect_frame(self.frames[self.frame_pos].image, config)
            self._show_stages(stages)
            self.status.setText(f"帧 {self.frames[self.frame_pos].index}：检测到 {len(result.centers)} 个液滴；已生成 {len(stages)} 个处理步骤")
        except Exception as exc:
            self.status.setText(f"识别失败：{exc}")

    def _show_stages(self, stages: list[PipelineStage]) -> None:
        while self.stage_grid.count():
            item = self.stage_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for index, stage in enumerate(stages):
            self.stage_grid.addWidget(StageCard(stage, self.stage_container), index // 2, index % 2)
        self.stage_grid.setRowStretch((len(stages) + 1) // 2, 1)

    def _parse_grid(self) -> dict:
        grid = {}
        for line in self.grid_text.toPlainText().splitlines():
            if not line.strip(): continue
            key, raw = line.split("=", 1); key = key.strip()
            if key not in SEARCHABLE_FIELDS: raise ValueError(f"不支持搜索参数：{key}")
            values = [float(item.strip()) for item in raw.split(",") if item.strip()]
            if not values: raise ValueError(f"参数没有候选值：{key}")
            if key in {"gaussian_blur_size", "morphology_open_kernel", "morphology_close_kernel"}: values = [int(v) for v in values]
            grid[key] = values
        if not grid: raise ValueError("请至少输入一个搜索参数")
        return grid

    def _search(self) -> None:
        try:
            if not self.frames: self._load()
            if not self.frames: return
            base = self._config_from_fields(); grid = self._parse_grid(); expected = float(self.expected.text())
            combinations = 1
            for values in grid.values(): combinations *= len(values)
            if combinations > 500: raise ValueError("搜索组合超过 500 组，请减少候选值")
            if expected < 0: raise ValueError("期望液滴数不能小于 0")
        except Exception as exc:
            QMessageBox.warning(self, "搜索参数错误", str(exc)); return
        self.search_button.setEnabled(False); self.status.setText("自动搜索中，请稍候…")
        self.worker_thread = QThread(self); self.worker = SearchWorker(self.video.text().strip(), self.max_frames.value(), expected, base, grid)
        self.worker.moveToThread(self.worker_thread); self.worker_thread.started.connect(self.worker.run); self.worker.done.connect(self._search_done); self.worker.failed.connect(self._search_failed); self.worker.done.connect(self.worker_thread.quit); self.worker.failed.connect(self.worker_thread.quit); self.worker_thread.finished.connect(self.worker.deleteLater); self.worker_thread.finished.connect(self.worker_thread.deleteLater); self.worker_thread.start()

    def _search_done(self, results) -> None:
        self.search_button.setEnabled(True)
        if not results: return
        best = results[0]; self.results.setPlainText("最佳参数：\n" + json.dumps({"score": best.score, "parameters": best.parameters, "evaluation": asdict(best.evaluation)}, ensure_ascii=False, indent=2) + "\n\n前 10 名：\n" + "\n".join(f"{i+1}. score={r.score:.4f} {r.parameters}" for i, r in enumerate(results[:10])))
        for key, value in best.parameters.items(): self._fields[key].setText(str(value))
        self._redraw(); self.status.setText(f"搜索完成，共评估 {len(results)} 组")

    def _search_failed(self, message: str) -> None:
        self.search_button.setEnabled(True); self.status.setText("搜索失败"); QMessageBox.warning(self, "搜索失败", message)

    def _save(self) -> None:
        try: values = asdict(self._config_from_fields())
        except Exception as exc: QMessageBox.warning(self, "保存失败", str(exc)); return
        path, _ = QFileDialog.getSaveFileName(self, "保存检测参数", "droplet_detector_tuning.json", "JSON (*.json)")
        if path: Path(path).write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")


def main(video: str = "") -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = TuningWindow(video); window.show(); raise SystemExit(app.exec())


if __name__ == "__main__": main()
