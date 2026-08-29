from __future__ import annotations

import base64
from collections import deque
import json
import multiprocessing as mp
import shutil
import sys
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QLockFile, QObject, QPoint, QRect, QRunnable, QSize, Qt, QThread, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QCloseEvent, QFont, QFontDatabase, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QInputDialog, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSpinBox,
    QSplitter, QStackedWidget, QStyle, QToolButton, QVBoxLayout, QWidget)

from backend.orchestrator import BayesianOptimizationConfig, OrchestratorService
from backend.orchestrator.models import SystemConfig
from backend.vision.calibration import load_calibration
from .config import (
    APP_TITLE,
    DEFAULT_BO_Q1_RANGE,
    DEFAULT_BO_Q2_RANGE,
    DEFAULT_CONTROL_INTERVAL_MS,
    DEFAULT_REFRESH_INTERVAL_MS,
    MAX_CONTROL_INTERVAL_MS,
    MIN_CONTROL_INTERVAL_MS,
)
from .runtime_logging import create_runtime_logger
from .settings_store import FrontendSettingsStore
from .vision_tuning import TuningWindow
from .vision_tuning_store import VisionTuningSettingsStore
from .paths import ensure_user_subdir
from .pid_replay import PIDReplayDialog


def jsonable(value):
    if value is None:
        return None
    if is_dataclass(value):
        return {k: jsonable(v) for k, v in asdict(value).items()}
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


class WorkerSignals(QObject):
    succeeded = Signal(object)
    failed = Signal(object)
    finished = Signal()


class Worker(QRunnable):
    def __init__(self, task: Callable[[], object]):
        super().__init__()
        self.task, self.signals = task, WorkerSignals()

    @Slot()
    def run(self):
        try:
            self.signals.succeeded.emit(self.task())
        except Exception as exc:
            self.signals.failed.emit(exc)
        finally:
            self.signals.finished.emit()


def set_button_role(button: QPushButton | QToolButton, role: str) -> None:
    """Assign a semantic role consumed by the application stylesheet."""
    button.setProperty("role", role)


class CollapsibleSection(QWidget):
    def __init__(self, title: str, parent=None, *, expanded: bool = False):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.toggle = QToolButton()
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toggle.setProperty("role", "section")
        self._title = title
        self.body = QWidget()
        self.body.setVisible(expanded)
        self.toggle.toggled.connect(self._set_expanded)
        layout.addWidget(self.toggle)
        layout.addWidget(self.body)
        self._update_text(expanded)

    def _set_expanded(self, expanded: bool) -> None:
        self.body.setVisible(expanded)
        self._update_text(expanded)

    def _update_text(self, expanded: bool) -> None:
        self.toggle.setText(("▾  " if expanded else "▸  ") + self._title)


class StatusModule(QWidget):
    """Compact key-metric view with opt-in full diagnostic text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.summary = QLabel("等待数据…")
        self.summary.setObjectName("metricSummary")
        self.summary.setWordWrap(True)
        self.summary.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(self.summary, 1)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.details.setVisible(False)
        self.details.setMaximumHeight(170)
        layout.addWidget(self.details)
        self.toggle = QToolButton()
        self.toggle.setCheckable(True)
        self.toggle.setText("查看详情")
        self.toggle.setProperty("role", "link")
        self.toggle.toggled.connect(self._toggle_details)
        layout.addWidget(self.toggle, 0, Qt.AlignRight)

    def _toggle_details(self, checked: bool) -> None:
        self.details.setVisible(checked)
        self.toggle.setText("收起详情" if checked else "查看详情")

    def setPlainText(self, text: str) -> None:
        self._text = str(text)
        self.details.setPlainText(self._text)
        lines = [line.strip() for line in self._text.splitlines() if line.strip()]
        self.summary.setText("\n".join(lines[:6]) or "等待数据…")

    def toPlainText(self) -> str:
        return self._text


class HealthCard(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("healthCard")
        self.setProperty("health", "idle")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)
        header = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setObjectName("healthTitle")
        self.badge = QLabel("等待")
        self.badge.setObjectName("healthBadge")
        header.addWidget(title_label)
        header.addStretch()
        header.addWidget(self.badge)
        self.detail = QLabel("尚无状态数据")
        self.detail.setObjectName("healthDetail")
        self.detail.setWordWrap(True)
        layout.addLayout(header)
        layout.addWidget(self.detail)

    def set_status(self, status: str, badge: str, detail: str) -> None:
        self.setProperty("health", status)
        self.badge.setText(badge)
        self.detail.setText(detail)
        self.style().unpolish(self)
        self.style().polish(self)


class StatusPoller(QObject):
    snapshotReady = Signal(object)
    failed = Signal(object)
    finished = Signal()

    def __init__(self, fetch: Callable[[], object], interval_ms: int, blocking_fetch: bool = False):
        super().__init__(); self.fetch=fetch; self.interval_s=max(0.05,int(interval_ms)/1000.0); self.blocking_fetch=blocking_fetch; self._stop=threading.Event()

    @Slot()
    def run(self):
        try:
            while not self._stop.is_set():
                started=time.monotonic()
                try: self.snapshotReady.emit(self.fetch())
                except Exception as exc: self.failed.emit(exc)
                if self.blocking_fetch: continue
                remaining=self.interval_s-(time.monotonic()-started)
                self._stop.wait(max(0.0,remaining))
        finally: self.finished.emit()

    def stop(self): self._stop.set()


class Page(QWidget):
    def __init__(self, app: "FrontendApp"):
        super().__init__()
        self.app = app

    def on_show(self): pass
    def on_hide(self): pass

    @staticmethod
    def field(form, label, value=""):
        edit = QLineEdit(str(value)); form.addRow(label, edit); return edit

    @staticmethod
    def number_field(
        form: QFormLayout,
        label: str,
        value: float,
        minimum: float,
        maximum: float,
        *,
        decimals: int = 2,
        suffix: str = "",
        integer: bool = False,
    ) -> QDoubleSpinBox | QSpinBox:
        if integer:
            spin = QSpinBox()
            spin.setRange(int(minimum), int(maximum))
            spin.setValue(int(value))
        else:
            spin = QDoubleSpinBox()
            spin.setDecimals(decimals)
            spin.setRange(float(minimum), float(maximum))
            spin.setValue(float(value))
            spin.setSingleStep(10 ** -min(decimals, 3))
        spin.setAccelerated(True)
        spin.setKeyboardTracking(False)
        if suffix:
            spin.setSuffix(f" {suffix}")
        form.addRow(label, spin)
        return spin

    @staticmethod
    def summary_box(title: str) -> tuple[QGroupBox, QVBoxLayout]:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setSpacing(10)
        return box, layout


class RoiImageLabel(QLabel):
    roiChanged = Signal(float, float, float, float)
    wallLinesChanged = Signal(object)

    def __init__(self):
        super().__init__("完成相机测试后，可在测试帧上拖拽选择 ROI")
        self.setAlignment(Qt.AlignCenter); self.setMinimumSize(520,300)
        self.setStyleSheet("background:#080c12;color:#94a3b8;border:1px solid #334155;border-radius:6px")
        self._source=QPixmap(); self._start=QPoint(); self._selection=QRect(); self._dragging=False; self._line_candidates=[]; self._selected_line_ids=[]; self._line_click_tolerance=24.0

    def set_image(self, data: bytes):
        image=QImage.fromData(data)
        if image.isNull(): raise ValueError("测试帧图像无法解析")
        self._source=QPixmap.fromImage(image); self._selection=QRect(); self.update()

    def set_hough_lines(self,candidates,selected=None):
        self._line_candidates=[dict(line) for line in (candidates or [])]
        selected=list(selected or []); self._selected_line_ids=[]
        for chosen in selected:
            best=None; best_score=1e9
            for line in self._line_candidates:
                direct=sum(abs(float(line.get(k,0))-float(chosen.get(k,0))) for k in ("x1","y1","x2","y2")); reverse=abs(float(line.get("x1",0))-float(chosen.get("x2",0)))+abs(float(line.get("y1",0))-float(chosen.get("y2",0)))+abs(float(line.get("x2",0))-float(chosen.get("x1",0)))+abs(float(line.get("y2",0))-float(chosen.get("y1",0))); score=min(direct,reverse)
                if score<best_score: best,best_score=line,score
            if best is not None and best_score<0.12: self._selected_line_ids.append(int(best.get("id",0)))
        self.update()

    def set_line_click_tolerance(self, pixels):
        self._line_click_tolerance=max(6.0,min(80.0,float(pixels)))

    def image_rect(self):
        if self._source.isNull(): return QRect()
        size=self._source.size(); size.scale(self.size(),Qt.KeepAspectRatio)
        return QRect((self.width()-size.width())//2,(self.height()-size.height())//2,size.width(),size.height())

    def paintEvent(self,event):
        super().paintEvent(event)
        if self._source.isNull(): return
        painter=QPainter(self); target=self.image_rect(); painter.drawPixmap(target,self._source)
        if not self._selection.isNull():
            painter.fillRect(self._selection,QColor(43,109,229,45)); painter.setPen(QPen(QColor("#38bdf8"),2)); painter.drawRect(self._selection)
        painter.setPen(QPen(QColor("#f97316"),4))
        for line in self._line_candidates:
            if int(line.get("id",0)) not in self._selected_line_ids: continue
            p1,p2=self._candidate_points(line,target); painter.drawLine(p1,p2)

    def mousePressEvent(self,event:QMouseEvent):
        target=self.image_rect()
        point=event.position().toPoint()
        # A fitted wall can lie on the first or last image row. Accept clicks
        # in the adjacent letterbox within the configured line tolerance.
        hit_target=target.adjusted(
            -int(self._line_click_tolerance),
            -int(self._line_click_tolerance),
            int(self._line_click_tolerance),
            int(self._line_click_tolerance),
        )
        if event.button()==Qt.LeftButton and hit_target.contains(point):
            self._start=self._clamp_to_image(point,target); self._selection=QRect(self._start,self._start); self._dragging=True; self.update()

    def mouseMoveEvent(self,event:QMouseEvent):
        if self._dragging:
            point=event.position().toPoint(); target=self.image_rect(); point.setX(max(target.left(),min(target.right(),point.x()))); point.setY(max(target.top(),min(target.bottom(),point.y())))
            self._selection=QRect(self._start,point).normalized(); self.update()

    def mouseReleaseEvent(self,event:QMouseEvent):
        if not self._dragging: return
        self._dragging=False; target=self.image_rect(); selected=self._selection.intersected(target)
        end=self._clamp_to_image(event.position().toPoint(),target); dx=end.x()-self._start.x(); dy=end.y()-self._start.y(); movement=(dx*dx+dy*dy)**0.5
        click_drag_limit=max(12.0,min(32.0,self._line_click_tolerance*1.25))
        if movement<=click_drag_limit and self._select_nearest_line(end): self._selection=QRect(); self.update(); return
        if selected.width()<5 or selected.height()<5: self._selection=QRect(); self.update(); return
        x0=(selected.left()-target.left())/target.width(); y0=(selected.top()-target.top())/target.height(); x1=(selected.right()-target.left()+1)/target.width(); y1=(selected.bottom()-target.top()+1)/target.height()
        self.roiChanged.emit(max(0,x0),max(0,y0),min(1,x1),min(1,y1)); self.update()

    @staticmethod
    def _candidate_points(line,target):
        width=max(1,target.width()-1); height=max(1,target.height()-1)
        return (
            QPoint(target.left()+int(round(float(line["x1"])*width)),target.top()+int(round(float(line["y1"])*height))),
            QPoint(target.left()+int(round(float(line["x2"])*width)),target.top()+int(round(float(line["y2"])*height))),
        )

    @staticmethod
    def _clamp_to_image(point,target):
        return QPoint(
            max(target.left(),min(target.right(),point.x())),
            max(target.top(),min(target.bottom(),point.y())),
        )

    def _select_nearest_line(self,point):
        target=self.image_rect()
        if not target.contains(point) or not self._line_candidates:return False
        def distance(line):
            p1,p2=self._candidate_points(line,target); ax=float(p1.x()); ay=float(p1.y()); bx=float(p2.x()); by=float(p2.y()); px=float(point.x()); py=float(point.y()); dx=bx-ax; dy=by-ay; denom=dx*dx+dy*dy
            t=0.0 if denom<=1e-9 else max(0.0,min(1.0,((px-ax)*dx+(py-ay)*dy)/denom)); return ((px-(ax+t*dx))**2+(py-(ay+t*dy))**2)**0.5
        line=min(self._line_candidates,key=distance)
        if distance(line)>self._line_click_tolerance:return False
        line_id=int(line.get("id",0))
        if line_id in self._selected_line_ids:self._selected_line_ids.remove(line_id)
        else:
            if len(self._selected_line_ids)>=2:self._selected_line_ids.pop(0)
            self._selected_line_ids.append(line_id)
        chosen=[{k:float(line[k]) for k in ("x1","y1","x2","y2")} for line in self._line_candidates if int(line.get("id",0)) in self._selected_line_ids]
        self.wallLinesChanged.emit(chosen)
        self.update()
        return True


class HoughParametersDialog(QDialog):
    parametersApplied = Signal(object)

    def __init__(self, values, validator, parent=None):
        super().__init__(parent)
        self._validator = validator
        self.setWindowTitle("霍夫直线识别参数")
        self.setModal(True)
        self.setMinimumWidth(410)
        layout = QVBoxLayout(self)
        hint = QLabel("调整完成后点击“应用并重新识别”。弹窗关闭后，再在完整测试图上选择管壁线。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        form = QFormLayout()
        self.fields = {}
        entries = (
            ("canny_low", "Canny 低阈值", values.get("canny_low", 35)),
            ("canny_high", "Canny 高阈值", values.get("canny_high", 100)),
            ("hough_threshold", "霍夫累加阈值（0=自动）", values.get("hough_threshold", 0)),
            ("min_line_length_percent", "最短线长度（图宽 %）", float(values.get("min_line_length_ratio", .55)) * 100),
            ("max_line_gap_percent", "最大间断（图宽 %）", float(values.get("max_line_gap_ratio", .16)) * 100),
            ("max_tilt_degrees", "最大倾斜角（°）", values.get("max_tilt_degrees", 33)),
            ("merge_distance_px", "相近线合并距离（px）", values.get("merge_distance_px", 4)),
            ("click_tolerance_px", "直线点击容差（px）", values.get("click_tolerance_px", 24)),
        )
        for key, label, value in entries:
            display = f"{value:g}" if isinstance(value, (int, float)) else str(value)
            edit = QLineEdit(display)
            self.fields[key] = edit
            form.addRow(label, edit)
        layout.addLayout(form)
        actions = QHBoxLayout()
        actions.addStretch()
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        apply_button = QPushButton("应用并重新识别")
        apply_button.clicked.connect(self._apply)
        actions.addWidget(cancel)
        actions.addWidget(apply_button)
        layout.addLayout(actions)

    def _apply(self):
        try:
            raw = {
                "canny_low": int(float(self.fields["canny_low"].text())),
                "canny_high": int(float(self.fields["canny_high"].text())),
                "hough_threshold": int(float(self.fields["hough_threshold"].text())),
                "min_line_length_ratio": float(self.fields["min_line_length_percent"].text()) / 100.0,
                "max_line_gap_ratio": float(self.fields["max_line_gap_percent"].text()) / 100.0,
                "max_tilt_degrees": float(self.fields["max_tilt_degrees"].text()),
                "merge_distance_px": float(self.fields["merge_distance_px"].text()),
                "click_tolerance_px": float(self.fields["click_tolerance_px"].text()),
                "max_lines": 32,
            }
            values = self._validator(raw)
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "霍夫参数错误", str(exc))
            return
        self.parametersApplied.emit(values)
        self.accept()


class ParameterPage(Page):
    def __init__(self, app):
        super().__init__(app); cfg = app.frontend_config
        layout = QVBoxLayout(self); layout.addWidget(app.title("基础参数", "设定控制目标与光学标定"))
        box = QGroupBox("参数设定"); form = QFormLayout(box)
        self.target = self.number_field(form, "目标液滴直径", cfg.get("target_diameter", 60), 0.01, 10000, decimals=2, suffix="μm")
        self.mag = self.number_field(form, "总放大倍率", cfg.get("magnification", 10), 0.01, 10000, decimals=3, suffix="×")
        self.pixel = self.number_field(form, "相机像元尺寸", cfg.get("camera_pixel_size_um", 6.9), 0.001, 1000, decimals=4, suffix="μm")
        self.interval = self.number_field(
            form,
            f"控制周期（最低 {MIN_CONTROL_INTERVAL_MS} ms）",
            int(cfg.get("control_interval_ms", DEFAULT_CONTROL_INTERVAL_MS)),
            MIN_CONTROL_INTERVAL_MS,
            MAX_CONTROL_INTERVAL_MS,
            suffix="ms",
            integer=True,
        )
        calibration_row=QWidget(); calibration_layout=QHBoxLayout(calibration_row); calibration_layout.setContentsMargins(0,0,0,0)
        self.calibration_path=QLineEdit(str(cfg.get("calibration_path",""))); calibration_button=QPushButton("选择…"); set_button_role(calibration_button,"secondary"); calibration_button.clicked.connect(self._browse_calibration)
        calibration_layout.addWidget(self.calibration_path,1); calibration_layout.addWidget(calibration_button); form.addRow("版本化标定 JSON（可选）",calibration_row)
        hint=QLabel("没有标定文件可留空；将使用像元尺寸÷放大倍率，仅允许预览，闭环控制前需补充标定 JSON。"); hint.setWordWrap(True); hint.setObjectName("formHint"); form.addRow("", hint)
        button = QPushButton("保存并进入相机配置"); button.clicked.connect(self.submit); form.addRow("", button)

        summary, summary_layout = self.summary_box("配置摘要")
        self.scale_summary = QLabel()
        self.scale_summary.setWordWrap(True)
        self.calibration_summary = QLabel()
        self.calibration_summary.setWordWrap(True)
        next_hint = QLabel("下一步将选择实时相机或本地视频，并完成测试取帧和 ROI 设置。")
        next_hint.setWordWrap(True)
        next_hint.setObjectName("formHint")
        summary_layout.addWidget(self.scale_summary)
        summary_layout.addWidget(self.calibration_summary)
        summary_layout.addSpacing(8)
        summary_layout.addWidget(next_hint)
        summary_layout.addStretch()
        content = QHBoxLayout()
        content.addWidget(box, 3)
        content.addWidget(summary, 2)
        layout.addLayout(content)
        layout.addStretch()
        for field in (self.target,self.mag,self.pixel,self.interval): field.valueChanged.connect(self._refresh_summary)
        self.calibration_path.textChanged.connect(self._refresh_summary)
        self._refresh_summary()

    def _refresh_summary(self, *_args):
        scale=self.pixel.value()/max(self.mag.value(),0.001)
        self.scale_summary.setText(f"预估比例\n{scale:.6f} μm/px\n\n目标直径\n{self.target.value():.2f} μm")
        path=self.calibration_path.text().strip()
        self.calibration_summary.setText("标定状态\n已选择版本化标定文件" if path else "标定状态\n仅可预览；闭环控制前需要标定 JSON")
        self.calibration_summary.setProperty("status","ok" if path else "warning")
        self.calibration_summary.style().unpolish(self.calibration_summary); self.calibration_summary.style().polish(self.calibration_summary)

    def _browse_calibration(self):
        path,_=QFileDialog.getOpenFileName(self,"选择像素标定文件",self.calibration_path.text(),"JSON 文件 (*.json)")
        if path:self.calibration_path.setText(path)

    def submit(self):
        try:
            target, mag, pixel = self.target.value(), self.mag.value(), self.pixel.value()
            interval = self.interval.value()
            if min(target, mag, pixel) <= 0 or not MIN_CONTROL_INTERVAL_MS<=interval<=MAX_CONTROL_INTERVAL_MS: raise ValueError(f"光学参数必须大于 0，实时控制周期必须为 {MIN_CONTROL_INTERVAL_MS}–{MAX_CONTROL_INTERVAL_MS} ms")
            calibration_path=self.calibration_path.text().strip(); calibration={}
            # 标定文件是闭环控制的前置条件，但不应阻止用户先进入视频/预览步骤。
            scale=pixel/mag
            if calibration_path:
                record=load_calibration(calibration_path); calibration=record.to_dict(); scale=record.pixel_to_micron
        except (ValueError,OSError) as exc: return self.app.error("参数错误", str(exc))
        self.app.save(target_diameter=target, magnification=mag, camera_pixel_size_um=pixel,
                      pixel_to_micron=scale, control_interval_ms=interval,
                      calibration_path=calibration_path,calibration=calibration)
        self.app.show_page("video")


class VideoPage(Page):
    def __init__(self, app):
        super().__init__(app); cfg=app.frontend_config; params=dict(cfg.get("camera_parameters",{}) or {})
        layout=QVBoxLayout(self); layout.setSpacing(10); layout.addWidget(app.title("相机识别与读写","按顺序完成设备发现、参数测试、同步取帧和 ROI 选择"))

        source_box=QGroupBox("第 1 步 · 选择来源与设备"); source_row=QHBoxLayout(source_box)
        self.mode=QComboBox(); self.mode.addItem("实时相机","camera"); self.mode.addItem("本地视频","file"); self.mode.setCurrentIndex(1 if cfg.get("video_source_type")=="file" else 0)
        self.source=QLineEdit(str(cfg.get("video_source",""))); self.source.setPlaceholderText("扫描相机后自动填写设备 ID，或选择本地视频")
        self.browse_button=QPushButton("浏览文件"); set_button_role(self.browse_button,"secondary"); self.browse_button.clicked.connect(self.browse)
        self.scan_button=QPushButton("扫描相机"); set_button_role(self.scan_button,"secondary"); self.scan_button.clicked.connect(self.scan_cameras)
        source_row.addWidget(QLabel("来源")); source_row.addWidget(self.mode); source_row.addWidget(self.source,1); source_row.addWidget(self.browse_button); source_row.addWidget(self.scan_button); layout.addWidget(source_box)

        work=QSplitter(Qt.Horizontal)
        settings=QGroupBox("第 2 步 · 设备与参数"); form=QFormLayout(settings); form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow); self.settings_form=form
        self.devices=QComboBox(); self.devices.setMinimumContentsLength(24); self.devices.currentIndexChanged.connect(self.camera_selected); form.addRow("发现的相机",self.devices)
        self.backend=QComboBox(); self.backend.addItems(["","hikrobot","opencv","gentl","basler","daheng","flir","allied_vision"]); self.backend.setCurrentText(str(cfg.get("camera_backend",""))); form.addRow("相机后端",self.backend)
        self.exposure=self.number_field(form,"曝光时间",params.get("exposure",3000),1,10000000,decimals=1,suffix="μs"); self.gain=self.number_field(form,"增益",params.get("gain",0),0,1000,decimals=2); self.fps=self.number_field(form,"目标帧率",params.get("frame_rate",100),0.1,10000,decimals=1,suffix="FPS")
        self.frame_width=self.number_field(form,"图像宽度",params.get("width",720),16,16384,integer=True,suffix="px"); self.frame_height=self.number_field(form,"图像高度",params.get("height",540),16,16384,integer=True,suffix="px")
        roi=dict(cfg.get("recognition_roi",{}) or {}); self._roi_user_modified=bool(roi.get("user_defined",False)); self._selected_wall_lines=[dict(line) for line in list(roi.get("wall_lines",[]) or [])[:2]]; self._last_test_preview_b64=""
        self.advanced_dialog=QDialog(self); self.advanced_dialog.setWindowTitle("高级识别、管道检定与 ROI 设置"); self.advanced_dialog.setMinimumSize(780,520)
        advanced_layout=QVBoxLayout(self.advanced_dialog); advanced_layout.setContentsMargins(18,16,18,16); advanced_layout.setSpacing(12)
        advanced_title=QLabel("高级识别、管道检定与 ROI 设置"); advanced_title.setStyleSheet("font-size:20px;font-weight:700")
        advanced_intro=QLabel("按实验需要分别配置三个区域。ROI 也可以在测试帧上直接拖拽或点击两条管壁线生成。"); advanced_intro.setObjectName("formHint"); advanced_intro.setWordWrap(True)
        advanced_layout.addWidget(advanced_title); advanced_layout.addWidget(advanced_intro)
        advanced_grid=QGridLayout(); advanced_grid.setHorizontalSpacing(12); advanced_grid.setVerticalSpacing(12)

        roi_box=QGroupBox("1 · ROI 识别范围"); roi_form=QFormLayout(roi_box); roi_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.roi_on=QCheckBox("启用 ROI，仅识别框选区域"); self.roi_on.setChecked(bool(roi.get("enabled",False))); roi_form.addRow("",self.roi_on)
        values=[float(roi.get(k,d))*100 for k,d in (("x_start_ratio",0),("y_start_ratio",0),("x_end_ratio",1),("y_end_ratio",1))]; self.roi=self.field(roi_form,"左、上、右、下",",".join(f"{v:g}" for v in values)); self.roi.setPlaceholderText("例如：10, 20, 90, 80"); self.roi.textEdited.connect(self._mark_roi_modified); self.roi.editingFinished.connect(self._analyze_user_roi)
        roi_hint=QLabel("数值单位为图像百分比；必须满足左 < 右、上 < 下。测试取帧后优先直接在图像上选择。"); roi_hint.setObjectName("formHint"); roi_hint.setWordWrap(True); roi_form.addRow("",roi_hint)

        region_box=QGroupBox("2 · 自动管道区域检定"); region_form=QFormLayout(region_box); region_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.channel_region_on=QCheckBox("启动时自动检定管道区域"); self.channel_region_on.setChecked(bool(roi.get("channel_region_enabled",True))); region_form.addRow("",self.channel_region_on)
        self.channel_region_samples=self.number_field(region_form,"采样帧数",roi.get("channel_region_sample_frames",12),1,48,integer=True,suffix="帧")
        region_hint=QLabel("使用多帧估计稳定管道区域；检定失败时自动回退整帧，不会阻止预览。"); region_hint.setObjectName("formHint"); region_hint.setWordWrap(True); region_form.addRow("",region_hint)

        calibration_box=QGroupBox("3 · 已知管道宽度标定"); calibration_form=QFormLayout(calibration_box); calibration_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.channel_cal_on=QCheckBox("使用已知内宽标定 μm/px"); self.channel_cal_on.setChecked(bool(roi.get("channel_calibration_enabled",True))); calibration_form.addRow("",self.channel_cal_on)
        self.channel_width=self.number_field(calibration_form,"管道实际内宽",roi.get("channel_width_um",430.0),0.01,100000,decimals=2,suffix="μm")
        calibration_hint=QLabel("请完整框住两条管道内壁，ROI 四周仅留约 3–5 px。该标定只在启动阶段运行。"); calibration_hint.setWordWrap(True); calibration_hint.setObjectName("formHint"); calibration_form.addRow("",calibration_hint)

        advanced_grid.addWidget(roi_box,0,0,1,2); advanced_grid.addWidget(region_box,1,0); advanced_grid.addWidget(calibration_box,1,1); advanced_grid.setColumnStretch(0,1); advanced_grid.setColumnStretch(1,1)
        advanced_layout.addLayout(advanced_grid,1)
        dialog_actions=QHBoxLayout(); dialog_actions.addStretch(); close_advanced=QPushButton("完成"); close_advanced.clicked.connect(self.advanced_dialog.accept); dialog_actions.addWidget(close_advanced); advanced_layout.addLayout(dialog_actions)
        self.advanced_summary=QLabel(); self.advanced_summary.setObjectName("advancedSummary"); self.advanced_summary.setWordWrap(True)
        advanced_button=QPushButton("配置高级识别与标定…"); set_button_role(advanced_button,"secondary"); advanced_button.clicked.connect(self.advanced_dialog.open)
        advanced_entry=QVBoxLayout(); advanced_entry.setContentsMargins(0,0,0,0); advanced_entry.setSpacing(7); advanced_entry.addWidget(self.advanced_summary); advanced_entry.addWidget(advanced_button)
        advanced_entry_widget=QWidget(); advanced_entry_widget.setLayout(advanced_entry); form.addRow("高级设置",advanced_entry_widget)
        self.test_button=QPushButton("写入参数并同步测试取帧"); set_button_role(self.test_button,"secondary"); self.test_button.clicked.connect(self.test_camera); form.addRow("",self.test_button)
        save=QPushButton("保存配置并进入初始化"); save.clicked.connect(self.submit); form.addRow("",save); work.addWidget(settings)

        right=QWidget(); right_layout=QVBoxLayout(right); right_layout.setContentsMargins(0,0,0,0)
        preview_box=QGroupBox("第 3 步 · 测试帧与 ROI 划选"); preview_layout=QVBoxLayout(preview_box); self.preview=RoiImageLabel(); self.preview.roiChanged.connect(self.roi_drawn); self.preview.wallLinesChanged.connect(self.wall_lines_selected); preview_layout.addWidget(self.preview,1)
        hough=dict(cfg.get("hough_line_parameters",{}) or {})
        try:self._hough_values=self._validate_hough_parameters(hough)
        except (TypeError,ValueError):self._hough_values=self._validate_hough_parameters({})
        hough_actions=QHBoxLayout(); hough_actions.addStretch(); hough_button=QPushButton("调整霍夫参数…"); set_button_role(hough_button,"secondary"); hough_button.clicked.connect(self._open_hough_parameters); hough_actions.addWidget(hough_button); preview_layout.addLayout(hough_actions)
        self.preview.set_line_click_tolerance(self._hough_values["click_tolerance_px"])
        self.roi_status=QLabel("测试取帧后会叠加霍夫直线。单击直线附近即可选取，两条橙色线表示已选管壁；拖拽空白区域可使用普通矩形 ROI。霍夫阈值为 0 时自动取图像宽度的 10%。"); self.roi_status.setWordWrap(True); preview_layout.addWidget(self.roi_status); right_layout.addWidget(preview_box,3)
        result_box=QGroupBox("连接结果与诊断"); result_layout=QVBoxLayout(result_box); self.camera_result=QPlainTextEdit(); self.camera_result.setReadOnly(True); self.camera_result.setMaximumHeight(150); result_layout.addWidget(self.camera_result)
        result_actions=QHBoxLayout(); log_button=QPushButton("导出日志"); set_button_role(log_button,"secondary"); log_button.clicked.connect(self.export_log); result_actions.addStretch(); result_actions.addWidget(log_button); result_layout.addLayout(result_actions); right_layout.addWidget(result_box,1); work.addWidget(right)
        work.setStretchFactor(0,1); work.setStretchFactor(1,3); work.setSizes([330,760]); layout.addWidget(work,1)
        self._camera_only_fields=(self.devices,self.backend,self.exposure,self.gain,self.fps,self.frame_width,self.frame_height,self.test_button)
        for control in (self.roi_on,self.channel_region_on,self.channel_cal_on): control.toggled.connect(self._refresh_advanced_summary)
        for control in (self.channel_region_samples,self.channel_width): control.valueChanged.connect(self._refresh_advanced_summary)
        self.roi.textChanged.connect(self._refresh_advanced_summary)
        for control in (self.exposure,self.gain,self.fps,self.frame_width,self.frame_height): control.valueChanged.connect(self._invalidate_camera_verification)
        self.backend.currentTextChanged.connect(self._invalidate_camera_verification)
        self.source.textChanged.connect(self._invalidate_camera_verification)
        self.mode.currentIndexChanged.connect(self._source_mode_changed)
        self._refresh_advanced_summary()
        self._source_mode_changed()

    def _refresh_advanced_summary(self, *_args):
        roi_text=self.roi.text().strip() if self.roi_on.isChecked() else "整帧"
        region="开启" if self.channel_region_on.isChecked() else "关闭"
        calibration=f"{self.channel_width.value():g} μm" if self.channel_cal_on.isChecked() else "关闭"
        self.advanced_summary.setText(f"ROI：{roi_text}  ·  自动检定：{region}  ·  宽度标定：{calibration}")

    def _invalidate_camera_verification(self, *_args):
        self.app.device_verification["camera"]=False

    def _source_mode_changed(self, *_args):
        self._invalidate_camera_verification()
        is_camera=self.mode.currentData()=="camera"
        self.scan_button.setVisible(is_camera)
        self.browse_button.setVisible(not is_camera)
        self.source.setPlaceholderText("扫描相机后自动填写设备 ID" if is_camera else "选择本地视频文件")
        for field in self._camera_only_fields:
            field.setVisible(is_camera)
            label=self.settings_form.labelForField(field)
            if label is not None: label.setVisible(is_camera)
        self.roi_status.setText("实时相机模式：请扫描设备并完成同步测试取帧。" if is_camera else "本地视频模式：相机通信参数已隐藏；ROI 将在视频准备阶段应用。")

    def export_log(self):
        source=Path(self.app.runtime_logger.path); target,_=QFileDialog.getSaveFileName(self,"保存相机运行日志",source.name,"日志文件 (*.log);;所有文件 (*)")
        if not target: return
        try:
            for handler in __import__("logging").getLogger().handlers: handler.flush()
            shutil.copy2(source,target); self.camera_result.appendPlainText(f"日志已保存: {target}")
        except Exception as exc: self.app.error("日志保存失败",str(exc))

    @staticmethod
    def _validate_hough_parameters(raw):
        defaults = {
            "canny_low": 35,
            "canny_high": 100,
            "hough_threshold": 0,
            "min_line_length_ratio": 0.55,
            "max_line_gap_ratio": 0.16,
            "max_tilt_degrees": 33.0,
            "merge_distance_px": 4.0,
            "click_tolerance_px": 24.0,
            "max_lines": 32,
        }
        defaults.update(dict(raw or {}))
        values = {
            "canny_low": int(float(defaults["canny_low"])),
            "canny_high": int(float(defaults["canny_high"])),
            "hough_threshold": int(float(defaults["hough_threshold"])),
            "min_line_length_ratio": float(defaults["min_line_length_ratio"]),
            "max_line_gap_ratio": float(defaults["max_line_gap_ratio"]),
            "max_tilt_degrees": float(defaults["max_tilt_degrees"]),
            "merge_distance_px": float(defaults["merge_distance_px"]),
            "click_tolerance_px": float(defaults["click_tolerance_px"]),
            "max_lines": max(2, min(200, int(float(defaults.get("max_lines", 32))))),
        }
        if not 0 <= values["canny_low"] < values["canny_high"] <= 255:
            raise ValueError("Canny 参数必须满足 0 ≤ 低阈值 < 高阈值 ≤ 255")
        if values["hough_threshold"] < 0:
            raise ValueError("霍夫阈值必须大于等于 0；输入 0 表示自动")
        if not 5 <= values["min_line_length_ratio"] * 100 <= 100:
            raise ValueError("最短线长度必须在 5%–100% 之间")
        if not 0 <= values["max_line_gap_ratio"] * 100 <= 100:
            raise ValueError("最大间断必须在 0%–100% 之间")
        if not 0 <= values["max_tilt_degrees"] <= 89:
            raise ValueError("最大倾角必须在 0°–89° 之间")
        if not 0 <= values["merge_distance_px"] <= 100:
            raise ValueError("合并距离必须在 0–100 px 之间")
        if not 6 <= values["click_tolerance_px"] <= 80:
            raise ValueError("点击容差必须在 6–80 px 之间")
        return values

    def _hough_parameters(self):
        values = self._validate_hough_parameters(self._hough_values)
        self.preview.set_line_click_tolerance(values["click_tolerance_px"])
        self.app.save(hough_line_parameters=values)
        return values

    def _open_hough_parameters(self):
        dialog = HoughParametersDialog(
            self._hough_values,
            self._validate_hough_parameters,
            self,
        )
        dialog.parametersApplied.connect(self._apply_hough_parameters)
        dialog.exec()

    def _apply_hough_parameters(self, values):
        self._hough_values = self._validate_hough_parameters(values)
        self.preview.set_line_click_tolerance(self._hough_values["click_tolerance_px"])
        self.app.save(hough_line_parameters=self._hough_values)
        if self._last_test_preview_b64:
            self._reanalyze_hough_lines(self._hough_values)
        else:
            self.roi_status.setText("霍夫参数已保存；完成测试取帧后将使用新参数识别直线。")

    def _reanalyze_hough_lines(self, parameters=None):
        if not self._last_test_preview_b64:
            return self.app.error("没有测试帧", "请先执行“写入参数并同步测试取帧”")
        try:
            roi = self._roi_payload()
            hough_parameters = self._validate_hough_parameters(
                parameters if parameters is not None else self._hough_values
            )
        except ValueError as exc:
            return self.app.error("霍夫参数错误", str(exc))
        self._selected_wall_lines = []
        roi["wall_lines"] = []
        self.preview.set_hough_lines(self.preview._line_candidates, [])
        self.roi_status.setText("正在使用新参数重新执行霍夫直线识别…")
        self.app.task(
            lambda: self.app.orchestrator.analyze_channel_calibration_preview(
                self._last_test_preview_b64,
                roi,
                roi["channel_width_um"],
                float(self.app.frontend_config.get("pixel_to_micron", 1.0)),
                hough_parameters,
            ),
            self._channel_analysis_done,
            lambda exc: self.roi_status.setText(f"霍夫直线识别失败：{exc}"),
        )

    def _channel_region_sample_count(self):
        sample_frames=int(self.channel_region_samples.value())
        if not 1<=sample_frames<=48: raise ValueError("自动检定采样帧数必须在 1–48 之间")
        return sample_frames

    def roi_drawn(self,x0,y0,x1,y1):
        self._roi_user_modified=True; self._selected_wall_lines=[]; self.preview.set_hough_lines(self.preview._line_candidates,[]); self.roi_on.setChecked(True); values=(x0*100,y0*100,x1*100,y1*100); self.roi.setText(",".join(f"{v:.2f}" for v in values))
        try: channel_width=float(self.channel_width.value()); sample_frames=self._channel_region_sample_count()
        except ValueError as exc: self.roi_status.setText(str(exc)); return
        roi={"enabled":True,"x_start_ratio":x0,"y_start_ratio":y0,"x_end_ratio":x1,"y_end_ratio":y1,"user_defined":True,"wall_lines":[],"channel_region_enabled":self.channel_region_on.isChecked(),"channel_region_sample_frames":sample_frames,"channel_calibration_enabled":self.channel_cal_on.isChecked(),"channel_width_um":channel_width}; self.app.save(recognition_roi=roi)
        self.roi_status.setText(f"已选择 ROI：左 {values[0]:.2f}% / 上 {values[1]:.2f}% / 右 {values[2]:.2f}% / 下 {values[3]:.2f}%。请确认两条内壁均完整可见。")
        self._analyze_user_roi()

    def _mark_roi_modified(self,*_args):
        self._roi_user_modified=True; self._selected_wall_lines=[]

    def wall_lines_selected(self,lines):
        self._selected_wall_lines=[dict(line) for line in (lines or [])]
        if len(self._selected_wall_lines)<2:
            self.roi_status.setText(f"已选择 {len(self._selected_wall_lines)}/2 条管壁线，请继续点击另一条管壁。")
            return
        self._roi_user_modified=True; self.roi_on.setChecked(True)
        xs=[float(line[k]) for line in self._selected_wall_lines for k in ("x1","x2")]; ys=[float(line[k]) for line in self._selected_wall_lines for k in ("y1","y2")]
        coords=(max(0,min(xs)-.01),max(0,min(ys)-.01),min(1,max(xs)+.01),min(1,max(ys)+.01)); self.roi.setText(",".join(f"{v*100:.2f}" for v in coords))
        try: payload=self._roi_payload()
        except ValueError as exc:self.roi_status.setText(str(exc));return
        self.app.save(recognition_roi=payload); self.roi_status.setText("已选择两条管壁，正在计算倾斜 ROI；后续监控与识别画面将自动摆正。"); self._analyze_user_roi()

    def _roi_payload(self):
        coords=[float(x.strip())/100 for x in self.roi.text().split(",")]
        if len(coords)!=4 or not (0<=coords[0]<coords[2]<=1 and 0<=coords[1]<coords[3]<=1): raise ValueError("ROI 范围无效")
        width=float(self.channel_width.value())
        if width<=0: raise ValueError("管道内宽必须大于 0 μm")
        sample_frames=self._channel_region_sample_count()
        return {"enabled":self.roi_on.isChecked(),"x_start_ratio":coords[0],"y_start_ratio":coords[1],"x_end_ratio":coords[2],"y_end_ratio":coords[3],"user_defined":self._roi_user_modified,"wall_lines":[dict(line) for line in self._selected_wall_lines] if len(self._selected_wall_lines)==2 else [],"channel_region_enabled":self.channel_region_on.isChecked(),"channel_region_sample_frames":sample_frames,"channel_calibration_enabled":self.channel_cal_on.isChecked(),"channel_width_um":width}

    def _analyze_user_roi(self):
        if not self._last_test_preview_b64 or not self._roi_user_modified: return
        try: roi=self._roi_payload(); hough_parameters=self._hough_parameters()
        except ValueError as exc: self.roi_status.setText(str(exc)); return
        self.roi_status.setText("正在按用户 ROI 重新拟合管壁…")
        self.app.task(lambda:self.app.orchestrator.analyze_channel_calibration_preview(self._last_test_preview_b64,roi,roi["channel_width_um"],float(self.app.frontend_config.get("pixel_to_micron",1.0)),hough_parameters),self._channel_analysis_done,lambda exc:self.roi_status.setText(f"ROI 分析失败，保留用户设置：{exc}"))

    def _channel_analysis_done(self,analysis):
        analysis=dict(analysis or {}); overlay=analysis.get("overlay_png_base64")
        if overlay:
            try:self.preview.set_image(base64.b64decode(overlay))
            except Exception as exc:self.app.runtime_logger(f"[CAMERA][UI][CHANNEL_OVERLAY][ERROR] {exc}")
        hough_lines=analysis.get("hough_lines") or []
        self.preview.set_hough_lines(hough_lines,self._selected_wall_lines)
        applied=dict(analysis.get("roi") or {})
        if analysis.get("auto_suggested") and applied and not self._roi_user_modified:
            values=[float(applied.get(k,d))*100 for k,d in (("x_start_ratio",0),("y_start_ratio",0),("x_end_ratio",1),("y_end_ratio",1))]
            self.roi.setText(",".join(f"{v:.2f}" for v in values)); self.roi_on.setChecked(True)
            try: sample_frames=self._channel_region_sample_count()
            except ValueError as exc: self.roi_status.setText(str(exc)); return
            applied["channel_region_enabled"]=self.channel_region_on.isChecked(); applied["channel_region_sample_frames"]=sample_frames; applied["channel_calibration_enabled"]=self.channel_cal_on.isChecked(); applied["channel_width_um"]=float(self.channel_width.value()); applied["user_defined"]=False; self.app.save(recognition_roi=applied)
        if analysis.get("ok"):
            source="用户 ROI" if analysis.get("used_user_roi") else "自动 ROI"
            self.roi_status.setText(f"{source} 管壁拟合成功：内宽 {float(analysis.get('channel_width_px')):.2f} px，比例 {float(analysis.get('pixel_to_micron')):.6f} μm/px；共显示 {len(hough_lines)} 条候选线。单击候选线附近即可选择，橙色表示已选。")
        else:
            self.roi_status.setText(f"管壁拟合未通过：{analysis.get('reason') or '未找到可信管壁'}。当前显示 {len(hough_lines)} 条候选线；可以调低霍夫阈值/最短线长度后重新识别。已保留用户 ROI 和光学比例 {float(analysis.get('pixel_to_micron',1.0)):.6f} μm/px。")

    def scan_cameras(self):
        self.app.runtime_logger("[CAMERA][UI][DISCOVERY][BEGIN]")
        self.camera_result.setPlainText("正在扫描所有相机后端…")
        self.app.task(self.app.orchestrator.discover_cameras, self.scan_done, lambda exc:self.camera_result.setPlainText(f"扫描失败: {exc}"))

    def scan_done(self, result):
        devices=list((result or {}).get("devices") or (result or {}).get("deduplicated_devices") or [])
        self.devices.blockSignals(True); self.devices.clear()
        for device in devices:
            uid=str(device.get("unique_id") or device.get("device_id") or ""); backend=str(device.get("selected_backend") or device.get("backend_name") or "")
            label=" | ".join(filter(None,[str(device.get("manufacturer") or "未知厂商"),str(device.get("model") or "未知型号"),str(device.get("serial_number") or uid),backend]))
            self.devices.addItem(label,{"id":uid,"backend":backend,"device":device})
        self.devices.blockSignals(False)
        self.camera_result.setPlainText(f"发现 {len(devices)} 台相机\n"+json.dumps({k:v for k,v in (result or {}).items() if k not in {"devices","deduplicated_devices","raw_devices"}},ensure_ascii=False,indent=2,default=str))
        self.app.runtime_logger(f"[CAMERA][UI][DISCOVERY][END] count={len(devices)}")
        if devices: self.devices.setCurrentIndex(0); self.camera_selected()

    def camera_selected(self, _index=0):
        data=self.devices.currentData()
        if not data: return
        self.source.setText(str(data["id"])); self.backend.setCurrentText(str(data["backend"])); self.mode.setCurrentIndex(0)
        self.camera_result.setPlainText(json.dumps(data["device"],ensure_ascii=False,indent=2))

    def test_camera(self):
        data=self.devices.currentData()
        if not data: return self.app.error("未选择相机","请先扫描并选择一台相机")
        try: params={"exposure":self.exposure.value(),"gain":self.gain.value(),"frame_rate":self.fps.value(),"width":self.frame_width.value(),"height":self.frame_height.value()}; roi_request=self._roi_payload(); hough_parameters=self._hough_parameters()
        except ValueError as exc: return self.app.error("相机参数错误",str(exc))
        self.camera_result.setPlainText("正在连接相机、写入参数、采集测试帧并回读…")
        def operation():
            self.app.runtime_logger(f"[CAMERA][UI][TEST][BEGIN] id={data['id']} backend={data['backend']} requested={params}")
            selected=self.app.orchestrator.select_camera(data["id"],data["backend"] or None)
            self.app.runtime_logger(f"[CAMERA][UI][SELECT][OK] selected={selected}")
            tested=self.app.orchestrator.test_camera(params)
            preview=tested.get("preview_png_base64")
            if preview:
                try: tested["channel_calibration_analysis"]=self.app.orchestrator.analyze_channel_calibration_preview(preview,roi_request,roi_request["channel_width_um"],float(self.app.frontend_config.get("pixel_to_micron",1.0)),hough_parameters)
                except Exception as exc: tested["channel_calibration_analysis"]={"ok":False,"reason":str(exc),"pixel_to_micron":float(self.app.frontend_config.get("pixel_to_micron",1.0)),"fallback_to_configured_scale":True}
            self.app.runtime_logger(f"[CAMERA][UI][TEST][END] ok={tested.get('ok')} frames={tested.get('frames_read')} readback={tested.get('parameter_readback')} error={tested.get('error')}")
            return {"selected":selected,"test":tested}
        self.app.task(operation,self.camera_test_done,lambda exc:self.camera_result.setPlainText(f"相机读写失败: {exc}"))

    def camera_test_done(self, result):
        test=(result or {}).get("test") or {}; display=dict(result or {}); display_test=dict(test)
        if display_test.get("preview_png_base64"): display_test["preview_png_base64"]="<测试帧图像已显示在下方>"
        if isinstance(display_test.get("channel_calibration_analysis"),dict) and display_test["channel_calibration_analysis"].get("overlay_png_base64"): display_test["channel_calibration_analysis"]=dict(display_test["channel_calibration_analysis"]); display_test["channel_calibration_analysis"]["overlay_png_base64"]="<管壁与 ROI 标注图已显示在下方>"
        display["test"]=display_test; self.camera_result.setPlainText(json.dumps(display,ensure_ascii=False,indent=2,default=str))
        if test.get("ok"):
            readback=dict(test.get("parameter_readback") or test.get("applied_parameters") or {})
            self.app.save(video_source_type="camera",video_source=self.source.text(),camera_backend=self.backend.currentText(),camera_parameters=readback)
            preview=test.get("preview_png_base64")
            if preview:
                try: self._last_test_preview_b64=preview; self._channel_analysis_done(test.get("channel_calibration_analysis") or {"ok":False,"reason":"未返回管壁分析结果","pixel_to_micron":float(self.app.frontend_config.get("pixel_to_micron",1.0))})
                except Exception as exc: self.app.runtime_logger(f"[CAMERA][UI][PREVIEW][ERROR] {exc}")
            self.app.device_verification["camera"]=True
        else:
            self.app.device_verification["camera"]=False
            self.app.error("相机测试失败",str(test.get("error") or test.get("message") or "设备未返回有效数据"))

    def browse(self):
        name, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)")
        if name: self.source.setText(name); self.mode.setCurrentIndex(1)

    def submit(self):
        try:
            mode, source = str(self.mode.currentData()), self.source.text().strip()
            if mode == "file" and not Path(source).is_file(): raise ValueError("请选择有效的视频文件")
            roi_config=self._roi_payload(); channel_width=float(roi_config["channel_width_um"])
            if self.channel_cal_on.isChecked() and not self.roi_on.isChecked(): raise ValueError("使用管道标定前必须启用并框选 ROI")
            params = {"exposure":self.exposure.value(), "gain":self.gain.value(), "frame_rate":self.fps.value(), "width":self.frame_width.value(), "height":self.frame_height.value()}; hough_parameters=self._hough_parameters()
        except ValueError as exc: return self.app.error("视频参数错误", str(exc))
        self.app.save(video_source_type=mode, video_source=source, camera_backend=self.backend.currentText(), camera_parameters=params,
                      recognition_roi=roi_config, hough_line_parameters=hough_parameters)
        self.app.show_page("init")


class PumpPage(Page):
    def __init__(self, app):
        super().__init__(app); cfg=app.frontend_config
        layout=QVBoxLayout(self); layout.addWidget(app.title("泵机识别与读写","扫描串口、验证泵协议、读取通道并安全写入回读"))
        box=QGroupBox("泵机通信"); form=QFormLayout(box)
        self.ports=QComboBox(); form.addRow("发现的串口",self.ports)
        scan=QPushButton("扫描串口设备"); set_button_role(scan,"secondary"); scan.clicked.connect(self.scan_ports); form.addRow("",scan)
        self.address=self.number_field(form,"泵地址",cfg.get("pump_address",1),1,31,integer=True); self.baud=self.number_field(form,"波特率",cfg.get("pump_baudrate",1200),1,2000000,integer=True,suffix="baud")
        self.parity=QComboBox(); self.parity.addItems(["N","E"]); self.parity.setCurrentText(str(cfg.get("pump_parity","N"))); form.addRow("校验位",self.parity)
        self.q1=self.number_field(form,"写入 Q1",cfg.get("initial_q1",50),0.01,5000,decimals=2,suffix="μL/min"); self.q2=self.number_field(form,"写入 Q2",cfg.get("initial_q2",20),0.01,5000,decimals=2,suffix="μL/min")
        actions=QWidget(); row=QHBoxLayout(actions); row.setContentsMargins(0,0,0,0)
        read=QPushButton("连接并读取全部数据"); set_button_role(read,"secondary"); read.clicked.connect(self.read_pump); write=QPushButton("写入 Q1/Q2 并回读校验"); set_button_role(write,"warning"); write.clicked.connect(self.write_pump); row.addWidget(read); row.addWidget(write); form.addRow("",actions)
        self.result=QPlainTextEdit(); self.result.setReadOnly(True); form.addRow("识别与读写结果",self.result)
        layout.addWidget(box)
        self.ports.currentIndexChanged.connect(self._invalidate_pump_verification)
        self.address.valueChanged.connect(self._invalidate_pump_verification)
        self.baud.valueChanged.connect(self._invalidate_pump_verification)
        self.parity.currentTextChanged.connect(self._invalidate_pump_verification)

    def _invalidate_pump_verification(self, *_args):
        self.app.device_verification["pump"]=False

    def on_show(self):
        if self.ports.count()==0: self.scan_ports()

    def scan_ports(self):
        self._invalidate_pump_verification()
        self.result.setPlainText("正在枚举串口设备…")
        def operation():
            from serial.tools import list_ports
            return [{"device":p.device,"description":p.description,"manufacturer":p.manufacturer,"product":p.product,"serial_number":p.serial_number,"vid":p.vid,"pid":p.pid,"hwid":p.hwid} for p in list_ports.comports()]
        self.app.task(operation,self.ports_done,lambda exc:self.result.setPlainText(f"串口扫描失败: {exc}"))

    def ports_done(self, ports):
        configured=str(self.app.frontend_config.get("pump_port","")).upper(); self.ports.clear()
        ranked=sorted(ports,key=lambda p:(0 if any(x in str(p).lower() for x in ("usb","serial","ch340","cp210","ftdi")) else 1,str(p.get("device"))))
        for port in ranked:
            label=f"{port['device']} | {port.get('description') or '串口设备'}"; self.ports.addItem(label,port)
            if str(port["device"]).upper()==configured: self.ports.setCurrentIndex(self.ports.count()-1)
        self.result.setPlainText(f"发现 {len(ranked)} 个串口候选。识别泵机需要继续执行协议读取。\n"+json.dumps(ranked,ensure_ascii=False,indent=2,default=str))

    def values(self):
        data=self.ports.currentData()
        if not data: raise ValueError("未发现或未选择串口设备")
        address=self.address.value(); baud=self.baud.value(); q1=self.q1.value(); q2=self.q2.value()
        if not 1<=address<=31 or min(baud,q1,q2)<=0 or max(q1,q2)>5000: raise ValueError("泵地址必须为 1–31，流量必须在 (0, 5000] μL/min")
        if q1 < q2 + 0.2: raise ValueError("油相 Q1 必须至少比水相 Q2 大 0.2 uL/min")
        return {"port":str(data["device"]),"address":address,"baudrate":baud,"parity":self.parity.currentText(),"q1":q1,"q2":q2}

    def read_pump(self):
        try: values=self.values()
        except ValueError as exc: return self.app.error("泵机参数错误",str(exc))
        self.result.setPlainText("正在打开串口并用 RSS/RSE/RSP 协议识别泵机…")
        def operation():
            service=self.app.orchestrator.pump_service; service.disconnect(); cfg=service.serial_config; cfg.port=values["port"]; cfg.address=values["address"]; cfg.baudrate=values["baudrate"]; cfg.parity=values["parity"]
            state=service.connect_and_probe()
            channels={str(ch):jsonable(service.read_rsp(ch)) for ch in (1,2,3,4)} if state.comm_established else {}
            return {"recognized_as_pump":bool(state.comm_established),"connection":jsonable(state),"system_setup":jsonable(service.read_rss()) if state.comm_established else None,"run_state":jsonable(service.read_rse()) if state.comm_established else None,"channels":channels,"connected_parity":service.client.connected_parity}
        self.app.task(operation,lambda result:self.pump_done("读取",values,result),lambda exc:self.result.setPlainText(f"泵机读取失败: {exc}"))

    def write_pump(self):
        try: values=self.values()
        except ValueError as exc: return self.app.error("泵机参数错误",str(exc))
        answer=QMessageBox.question(self,"确认写入","将写入 Q1/Q2，执行回读校验，并短暂启动后立即安全停止。是否继续？")
        if answer!=QMessageBox.StandardButton.Yes: return
        self.result.setPlainText("正在写入泵参数并回读校验…")
        self.app.task(lambda:self.app.orchestrator.run_pump_interaction_test(**values),lambda result:self.pump_done("写入",values,result),lambda exc:self.result.setPlainText(f"泵机写入失败: {exc}"))

    def pump_done(self, action, values, result):
        recognized=bool((result or {}).get("recognized_as_pump",(result or {}).get("ok",False)))
        self.app.device_verification["pump"]=recognized
        self.result.setPlainText(f"{action}完成；泵机协议识别: {'成功' if recognized else '失败'}\n"+json.dumps(result,ensure_ascii=False,indent=2,default=str))
        if recognized: self.app.save(pump_port=values["port"],pump_address=values["address"],pump_baudrate=values["baudrate"],pump_parity=values["parity"],initial_q1=values["q1"],initial_q2=values["q2"])
        else: self.app.error("未识别到泵机","串口可以打开，但设备没有返回有效泵协议数据。请检查地址、波特率和线缆。")


class InitPage(Page):
    def __init__(self, app):
        super().__init__(app)
        layout=QVBoxLayout(self)
        layout.addWidget(app.title("系统初始化","确认已验证配置并启动视频、泵通信和控制服务"))
        self.readiness=QLabel("正在检查配置…"); self.readiness.setObjectName("statusAlert"); self.readiness.setWordWrap(True); layout.addWidget(self.readiness)
        cards=QGridLayout(); cards.setSpacing(12)
        self.source_card=HealthCard("视频与相机")
        self.roi_card=HealthCard("ROI 与标定")
        self.pump_card=HealthCard("泵机通信")
        self.flow_card=HealthCard("初始流量")
        cards.addWidget(self.source_card,0,0); cards.addWidget(self.roi_card,0,1); cards.addWidget(self.pump_card,1,0); cards.addWidget(self.flow_card,1,1)
        cards.setColumnStretch(0,1); cards.setColumnStretch(1,1); layout.addLayout(cards)
        actions=QHBoxLayout()
        self.status=QLabel("未初始化"); self.status.setObjectName("runtimeStateBadge"); actions.addWidget(self.status); actions.addStretch()
        self.back_button=QPushButton("返回泵机配置"); set_button_role(self.back_button,"secondary"); self.back_button.clicked.connect(self._go_back); actions.addWidget(self.back_button)
        self.button=QPushButton("确认配置并初始化系统"); self.button.clicked.connect(self.initialize); actions.addWidget(self.button)
        layout.addLayout(actions); layout.addStretch()
        self._back_target="pump"

    def on_show(self):
        cfg=self.app.frontend_config
        verification=dict(getattr(self.app,"device_verification",{}) or {})
        source_type=str(cfg.get("video_source_type","") or "")
        source_value=str(cfg.get("video_source","") or "")
        is_file=source_type=="file"
        self._back_target="video" if is_file else "pump"
        self.back_button.setText("返回相机与 ROI 配置" if is_file else "返回泵机配置")
        file_available=is_file and bool(source_value) and Path(source_value).is_file()
        camera_verified=bool(verification.get("camera",False))
        if is_file and file_available:self.source_card.set_status("ok","文件可用",f"本地视频 · {Path(source_value).name}")
        elif is_file and source_value:self.source_card.set_status("error","文件不可用",f"已保存路径不存在 · {source_value}")
        elif not is_file and source_value and camera_verified:self.source_card.set_status("ok","本次已验证",f"实时相机 · 设备 {source_value} · 后端 {cfg.get('camera_backend','--') or '--'}")
        elif not is_file and source_value:self.source_card.set_status("warning","仅保存配置",f"设备 {source_value} 尚未在本次启动中测试取帧")
        else:self.source_card.set_status("error","缺失","尚未选择视频来源")
        roi=dict(cfg.get("recognition_roi",{}) or {})
        roi_state="已启用" if roi.get("enabled") else "整帧识别"
        calibration=dict(cfg.get("calibration",{}) or {})
        channel_cal=bool(roi.get("channel_calibration_enabled",False))
        if calibration:self.roi_card.set_status("ok","配置可用",f"{roi_state} · 已保存版本化像素标定")
        elif channel_cal:self.roi_card.set_status("warning","待初始化检定",f"{roi_state} · 配置的已知管宽 {float(roi.get('channel_width_um',430.0)):g} μm 尚未实际检定")
        else:self.roi_card.set_status("warning","仅保存配置",f"{roi_state} · 闭环前需要版本化标定")
        port=str(cfg.get("pump_port","") or "").strip().upper()
        pump_verified=bool(verification.get("pump",False))
        if port and pump_verified:self.pump_card.set_status("ok","本次已验证",f"{port} · 地址 {int(cfg.get('pump_address',1))} · {int(cfg.get('pump_baudrate',1200))} baud · {cfg.get('pump_parity','N')}")
        elif port:self.pump_card.set_status("warning","仅保存配置",f"{port} 尚未在本次启动中完成泵协议读取与回读校验")
        elif is_file:self.pump_card.set_status("idle","本地模式","本地视频预览允许不连接泵机")
        else:self.pump_card.set_status("error","未验证","请返回泵机页面完成协议识别与回读校验")
        q1=float(cfg.get("initial_q1",50)); q2=float(cfg.get("initial_q2",20))
        flow_ok=0<q1<=5000 and 0<q2<=5000 and q1>=q2+0.2
        self.flow_card.set_status("idle" if flow_ok else "error","已保存" if flow_ok else "无效",f"Q1 {q1:g} / Q2 {q2:g} μL/min · 初始化时才会写入泵机")
        source_ready=file_available if is_file else (bool(source_value) and camera_verified)
        pump_ready=is_file or (bool(port) and pump_verified)
        ready=source_ready and pump_ready and flow_ok
        missing=[]
        if not source_ready:missing.append("本次相机测试取帧" if not is_file else "有效的本地视频文件")
        if not pump_ready:missing.append("本次泵协议读取与回读校验")
        calibration_note="；当前缺少版本化标定，只能完成预览初始化，实验前检查会禁止闭环" if not calibration and not channel_cal else ""
        readiness_text="本次会话的设备验证已完成，可以初始化" if ready else "不能初始化，尚缺："+"、".join(missing or ["有效配置"])
        self.readiness.setProperty("status","ok" if ready else "error"); self.readiness.setText(readiness_text+calibration_note); self.readiness.style().unpolish(self.readiness); self.readiness.style().polish(self.readiness)
        self.button.setEnabled(ready)

    def _go_back(self):self.app.show_page(self._back_target)

    def initialize(self):
        cfg=self.app.frontend_config
        try:
            q1,q2=float(cfg.get("initial_q1",50)),float(cfg.get("initial_q2",20)); address,baud=int(cfg.get("pump_address",1)),int(cfg.get("pump_baudrate",1200)); port=str(cfg.get("pump_port","") or "").strip().upper()
            if min(q1,q2,baud)<=0 or max(q1,q2)>5000 or not 1<=address<=31: raise ValueError("泵地址必须为 1–31，流量必须在 (0, 5000] μL/min")
            if q1<q2+0.2: raise ValueError("油相 Q1 必须至少比水相 Q2 大 0.2 μL/min")
            verification=dict(getattr(self.app,"device_verification",{}) or {})
            if cfg.get("video_source_type")=="file":
                if not Path(str(cfg.get("video_source","") or "")).is_file(): raise ValueError("本地视频文件不存在")
            elif not verification.get("camera",False): raise ValueError("相机尚未在本次启动中成功测试取帧")
            if cfg.get("video_source_type")!="file" and (not port or not verification.get("pump",False)): raise ValueError("泵机尚未在本次启动中完成协议识别与回读校验")
        except ValueError as exc: return self.app.error("初始化参数错误",str(exc))
        self.app.save(initial_q1=q1,initial_q2=q2,pump_port=port,pump_address=address,pump_baudrate=baud,pump_parity=str(cfg.get("pump_parity","N")))
        self.status.setText("初始化中…"); self.app.task(self.app.configure_prepare_initialize,self.done,self.failed,self.button)

    def done(self,_=None): self.status.setText("初始化完成"); self.app.show_page("monitor")
    def failed(self,exc): self.status.setText("初始化失败"); self.app.error("初始化失败",str(exc))


class DropletGalleryDialog(QDialog):
    """Closable, non-modal full-frame evidence view for one completed period."""

    def __init__(self, gallery, parent=None):
        super().__init__(parent)
        self.setWindowTitle("上一控制周期完整识别帧")
        self.resize(1180, 760)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        layout = QVBoxLayout(self)
        period_id = int(gallery.get("period_id", 0) or 0)
        frames = list(gallery.get("frames", []) or [])
        unique_droplet_count = int(gallery.get("droplet_count", 0) or 0)
        reason = str(gallery.get("reason", "") or "")
        summary = QLabel(
            f"控制周期：{period_id or '--'}    采样识别帧：{len(frames)}    "
            f"不同有效液滴轨迹：{unique_droplet_count}\n"
            "绿色圆圈表示该帧有效液滴，橙色圆圈表示该帧同时穿过计数线，蓝色细线为计数线。"
            + (f"\n说明：{reason}" if reason and reason != "ok" else "")
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget(scroll)
        grid = QGridLayout(content)
        grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        if not frames:
            empty = QLabel("该周期没有可显示的完整采样识别帧。")
            empty.setAlignment(Qt.AlignCenter)
            grid.addWidget(empty, 0, 0)
        for index, item in enumerate(frames):
            frame_id = int(item.get("frame_id", 0) or 0)
            valid_count = int(item.get("valid_droplet_count", 0) or 0)
            crossed_count = int(item.get("crossed_droplet_count", 0) or 0)
            card = QGroupBox(f"采样帧 #{frame_id}")
            card_layout = QVBoxLayout(card)
            image_label = QLabel("图像不可用")
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setMinimumSize(500, 280)
            image_label.setStyleSheet("background:#080c12;color:#94a3b8")
            try:
                image = QImage.fromData(base64.b64decode(item.get("image_jpeg_base64", "")))
                if not image.isNull():
                    image_label.setPixmap(
                        QPixmap.fromImage(image).scaled(
                            520, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation
                        )
                    )
            except Exception:
                pass
            average = item.get("average_diameter_um")
            average_text = "--" if average is None else f"{float(average):.2f} μm"
            track_ids = ", ".join(str(value) for value in item.get("valid_track_ids", []) or []) or "无"
            detail = QLabel(
                f"有效液滴：{valid_count}    本帧穿线：{crossed_count}    "
                f"平均直径：{average_text}\n有效轨迹 ID：{track_ids}"
            )
            detail.setAlignment(Qt.AlignCenter)
            detail.setWordWrap(True)
            card_layout.addWidget(image_label)
            card_layout.addWidget(detail)
            grid.addWidget(card, index // 2, index % 2)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        close_row = QHBoxLayout()
        close_row.addStretch()
        close_row.addWidget(close_button)
        layout.addLayout(close_row)


class MonitorPage(Page):
    def __init__(self, app):
        super().__init__(app)
        self.last_frame = -1
        self._display_times = deque(maxlen=120)
        self._status_thread = None
        self._status_poller = None
        self._last_control_status_ts = 0.0
        self._gallery_dialog = None
        self._replay_dialog = None
        layout = QVBoxLayout(self)
        layout.addWidget(app.title("运行监控", "视频、识别、泵机与 PID 状态分别刷新"))
        control_bar=QFrame(); control_bar.setObjectName("controlBar"); control_layout=QVBoxLayout(control_bar); control_layout.setContentsMargins(10,8,10,8); control_layout.setSpacing(7)
        lifecycle=QHBoxLayout(); lifecycle.setSpacing(7)
        self.state_badge=QLabel("等待状态"); self.state_badge.setObjectName("runtimeStateBadge"); lifecycle.addWidget(self.state_badge)
        self.action_buttons={}
        initialize_button=QPushButton("初始化"); set_button_role(initialize_button,"secondary"); initialize_button.clicked.connect(lambda:app.task(app.configure_prepare_initialize)); lifecycle.addWidget(initialize_button); self.action_buttons["初始化"]=initialize_button
        start_button=QPushButton("开始"); start_button.clicked.connect(lambda:app.task(app.orchestrator.start)); lifecycle.addWidget(start_button); self.action_buttons["开始"]=start_button
        self.pause_button=QPushButton("暂停"); set_button_role(self.pause_button,"secondary"); self.pause_button.clicked.connect(self._pause_or_resume); lifecycle.addWidget(self.pause_button); self.action_buttons["暂停/继续"]=self.pause_button
        self.stop_button=QPushButton("停止"); set_button_role(self.stop_button,"danger"); self.stop_button.clicked.connect(self._stop_pid); lifecycle.addWidget(self.stop_button)
        safety_button=QPushButton("复位安全锁"); set_button_role(safety_button,"warning"); safety_button.clicked.connect(lambda:app.task(app.orchestrator.reset_safety_latch)); lifecycle.addWidget(safety_button); self.action_buttons["复位安全锁"]=safety_button
        lifecycle.addStretch()
        self.fps_label=QLabel("视频显示：0.0 / 30 FPS"); self.fps_label.setObjectName("fpsLabel"); lifecycle.addWidget(self.fps_label)
        tools=QHBoxLayout(); tools.setSpacing(7); tools.addWidget(QLabel("实验工具"))
        self.preflight_button=QPushButton("实验前检查"); set_button_role(self.preflight_button,"secondary"); self.preflight_button.clicked.connect(self._run_preflight); tools.addWidget(self.preflight_button)
        self.bo_button=QPushButton("BO 寻优"); set_button_role(self.bo_button,"warning"); self.bo_button.clicked.connect(self._start_bo); tools.addWidget(self.bo_button)
        self.target_button=QPushButton("更改目标"); set_button_role(self.target_button,"secondary"); self.target_button.clicked.connect(self._change_target_diameter); tools.addWidget(self.target_button)
        tools.addSpacing(12); tools.addWidget(QLabel("数据"))
        self.save_data_button=QPushButton("保存数据"); set_button_role(self.save_data_button,"secondary"); self.save_data_button.clicked.connect(self._save_pid_data); tools.addWidget(self.save_data_button)
        self.import_data_button=QPushButton("导入数据"); set_button_role(self.import_data_button,"secondary"); self.import_data_button.clicked.connect(self._import_pid_data); tools.addWidget(self.import_data_button)
        self.gallery_button = QPushButton("上一周期识别帧")
        set_button_role(self.gallery_button,"secondary")
        self.gallery_button.clicked.connect(self._show_last_period_droplets)
        tools.addWidget(self.gallery_button); tools.addStretch()
        control_layout.addLayout(lifecycle); control_layout.addLayout(tools); layout.addWidget(control_bar)

        row=QSplitter(Qt.Horizontal)
        self.video=QLabel("等待视频帧")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(480,320)
        self.video.setStyleSheet("background:#080c12;color:#8b98aa;border-radius:8px")
        row.addWidget(self.video)

        modules = QWidget(self)
        module_grid = QGridLayout(modules)
        module_grid.setContentsMargins(0, 0, 0, 0)
        self.system_panel = self._make_status_module(module_grid, "系统状态", 0, 0)
        self.vision_panel = self._make_status_module(module_grid, "液滴识别", 0, 1)
        self.pump_panel = self._make_status_module(module_grid, "泵状态", 1, 0)
        self.pid_panel = self._make_status_module(module_grid, "PID 控制", 1, 1)
        module_grid.setRowStretch(0, 1)
        module_grid.setRowStretch(1, 1)
        module_grid.setColumnStretch(0, 1)
        module_grid.setColumnStretch(1, 1)
        row.addWidget(modules)
        row.setStretchFactor(0,6); row.setStretchFactor(1,5); row.setSizes([620,480])
        layout.addWidget(row,1)

        self.video_timer=QTimer(self)
        self.video_timer.setTimerType(Qt.PreciseTimer)
        self.video_timer.setInterval(33)
        self.video_timer.timeout.connect(self.refresh_video)
        self.system_timer = QTimer(self)
        self.system_timer.setInterval(1000)
        self.system_timer.timeout.connect(self.refresh_system_panel)
        self.vision_timer = QTimer(self)
        self.vision_timer.setInterval(400)
        self.vision_timer.timeout.connect(self.refresh_vision_panel)
        self.pump_timer = QTimer(self)
        self.pump_timer.setInterval(700)
        self.pump_timer.timeout.connect(self.refresh_pump_panel)
        self._frame_error=""

    @staticmethod
    def _make_status_module(grid, title, row, column):
        box = QGroupBox(title)
        box_layout = QVBoxLayout(box)
        panel = StatusModule()
        panel.setMinimumSize(190, 150)
        box_layout.addWidget(panel)
        grid.addWidget(box, row, column)
        return panel

    def _pause_or_resume(self):
        try: snapshot=self.app.orchestrator.get_snapshot()
        except Exception as exc: return self.app.error("状态读取失败",str(exc))
        state=str(getattr(getattr(snapshot,"system_state",None),"value",getattr(snapshot,"system_state",""))).upper()
        operation=self.app.orchestrator.resume if state=="PAUSED" else self.app.orchestrator.pause
        self.app.task(operation,disable=self.pause_button)

    def on_show(self):
        self.video_timer.start()
        self.system_timer.start()
        self.vision_timer.start()
        self.pump_timer.start()
        self.refresh()
        self._start_status_thread()

    def on_hide(self):
        self.video_timer.stop()
        self.system_timer.stop()
        self.vision_timer.stop()
        self.pump_timer.stop()
        self._stop_status_thread()

    def _show_last_period_droplets(self):
        self.app.task(
            self.app.orchestrator.get_last_control_period_droplets,
            self._open_droplet_gallery,
            disable=self.gallery_button,
        )

    def _run_preflight(self):
        self.app.task(self.app.orchestrator.run_preflight_check,self._show_preflight,disable=self.preflight_button)

    def _show_preflight(self,result):
        issues=list((result or {}).get("issues",[]) or [])
        if not issues:
            QMessageBox.information(self,"实验前检查","检查通过：状态、停泵确认、像素标定和泵通信满足启动条件。")
            return
        QMessageBox.warning(self,"实验前检查未通过","\n".join(f"• {item}" for item in issues))

    def _start_bo(self):
        try:
            snapshot=self.app.orchestrator.get_snapshot()
            config=getattr(snapshot,"config",None)
            if config is None:
                raise RuntimeError("请先完成参数配置和系统初始化")
            target=float(config.target_diameter)
        except Exception as exc:
            QMessageBox.warning(self,"无法启动 BO",str(exc)); return

        dialog=QDialog(self); dialog.setWindowTitle("安全 BO 工作点寻优")
        layout=QVBoxLayout(dialog)
        note=QLabel("仅填写实际实验测得的泵到液滴响应延迟；串口应答时间不能作为响应延迟。BO 期间 PID 与前馈不写泵。")
        note.setWordWrap(True); layout.addWidget(note)
        form=QFormLayout(); fields={}
        defaults={
            "q1_min":DEFAULT_BO_Q1_RANGE[0], "q1_max":DEFAULT_BO_Q1_RANGE[1],
            "q2_min":DEFAULT_BO_Q2_RANGE[0], "q2_max":DEFAULT_BO_Q2_RANGE[1],
            "delay":"", "uncertainty":"", "settling":"", "source":"阶跃实验",
        }
        for key,label in (("q1_min","Q1 下限 (uL/min)"),("q1_max","Q1 上限 (uL/min)"),("q2_min","Q2 下限 (uL/min)"),("q2_max","Q2 上限 (uL/min)"),("delay","实测响应延迟 (ms)"),("uncertainty","延迟不确定度/安全余量 (ms)"),("settling","候选点稳定时间 (ms)"),("source","延迟测量来源")):
            edit=QLineEdit(str(defaults[key])); fields[key]=edit; form.addRow(label,edit)
        layout.addLayout(form)
        buttons=QHBoxLayout(); cancel=QPushButton("取消"); start=QPushButton("开始寻优")
        cancel.clicked.connect(dialog.reject); start.clicked.connect(dialog.accept)
        buttons.addStretch(); buttons.addWidget(cancel); buttons.addWidget(start); layout.addLayout(buttons)
        if dialog.exec()!=QDialog.DialogCode.Accepted: return
        try:
            delay=float(fields["delay"].text())
            bo_config=BayesianOptimizationConfig(
                target_diameter_um=target,
                q1_min=float(fields["q1_min"].text()), q1_max=float(fields["q1_max"].text()),
                q2_min=float(fields["q2_min"].text()), q2_max=float(fields["q2_max"].text()),
                measured_response_delay_ms=delay,
                response_delay_uncertainty_ms=float(fields["uncertainty"].text()),
                settling_time_ms=float(fields["settling"].text()),
                response_delay_source=fields["source"].text().strip(),
            )
        except Exception as exc:
            QMessageBox.warning(self,"BO 参数无效",str(exc)); return
        self.app.task(lambda:self.app.orchestrator.start_optimization(bo_config),disable=self.bo_button)

    def _open_droplet_gallery(self, gallery):
        if self._gallery_dialog is not None:
            self._gallery_dialog.close()
        self._gallery_dialog = DropletGalleryDialog(dict(gallery or {}), self)
        self._gallery_dialog.destroyed.connect(
            lambda _object=None: setattr(self, "_gallery_dialog", None)
        )
        self._gallery_dialog.show()
        self._gallery_dialog.raise_()
        self._gallery_dialog.activateWindow()

    def _stop_pid(self):
        self.app.task(self.app.orchestrator.stop, self._after_pid_stopped, disable=self.stop_button)

    def _change_target_diameter(self):
        try:
            snapshot=self.app.orchestrator.get_snapshot()
            config=getattr(snapshot,"config",None)
            current=float(getattr(config,"target_diameter",self.app.frontend_config.get("target_diameter",60.0)))
        except Exception as exc:
            QMessageBox.warning(self,"无法读取目标",str(exc))
            return
        dialog=QInputDialog(self)
        dialog.setWindowTitle("更改目标液滴直径")
        dialog.setLabelText("新的目标液滴平均直径（μm）：")
        dialog.setInputMode(QInputDialog.InputMode.DoubleInput)
        dialog.setDoubleRange(0.001,1_000_000.0)
        dialog.setDoubleDecimals(3)
        dialog.setDoubleValue(current)
        dialog.setOkButtonText("确定")
        dialog.setCancelButtonText("取消")
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        target=dialog.doubleValue()
        self.app.task(
            lambda:self.app.orchestrator.update_target_diameter(float(target)),
            self._target_diameter_updated,
            disable=self.target_button,
        )

    def _target_diameter_updated(self,result):
        target=float(result.get("target_diameter_um"))
        previous=float(result.get("previous_target_diameter_um"))
        self.app.save(target_diameter=target)
        self.refresh_system_panel()
        self.show_status(self.app.orchestrator.get_snapshot())
        self.app.runtime_logger(
            f"[PID][TARGET][UI] {previous:.3f}um -> {target:.3f}um; "
            "effective from next control period"
        )

    def _after_pid_stopped(self, _result=None):
        status=self.app.orchestrator.get_pid_session_data_status(); count=int(status.get("record_count",0) or 0); summary=self._experiment_summary(count)
        if count <= 0 or not bool(status.get("has_unsaved_data",False)):
            QMessageBox.information(self,"实验结束摘要",summary+"\n本次运行没有尚未保存的 PID 数据。")
            return
        answer=QMessageBox.question(
            self,
            "保存本次 PID 数据",
            summary+"\n是否保存到 SQLite 数据库？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.No:
            self.app.orchestrator.discard_pid_session_data()
            return
        self._choose_and_save_pid_data()

    def _save_pid_data(self):
        status=self.app.orchestrator.get_pid_session_data_status()
        if bool(status.get("active",False)):
            QMessageBox.warning(self,"无法保存","PID 仍在运行，请先停止实验再保存数据。")
            return
        if int(status.get("record_count",0) or 0) <= 0:
            QMessageBox.information(self,"没有数据","当前没有可保存的 PID 控制周期数据。")
            return
        if not bool(status.get("has_unsaved_data",False)):
            QMessageBox.information(self,"无需保存","当前 PID 数据已经保存。")
            return
        self._choose_and_save_pid_data()

    def _choose_and_save_pid_data(self):
        default_dir=Path(str(self.app.frontend_config.get("pid_database_dir",ensure_user_subdir("experiments"))))
        default_name=f"pid_data_{time.strftime('%Y%m%d_%H%M%S')}.sqlite"
        selected,_=QFileDialog.getSaveFileName(
            self,
            "选择 PID 数据库保存位置",
            str(default_dir/default_name),
            "SQLite 数据库 (*.sqlite *.db)",
        )
        if not selected:
            QMessageBox.information(self,"尚未保存","本次数据仍保留在内存中；再次点击“停止”可以重新选择保存位置。")
            return
        database_path=Path(selected)
        if not database_path.suffix:
            database_path=database_path.with_suffix(".sqlite")
        self.app.save(pid_database_dir=str(database_path.parent))
        self.app.task(
            lambda:self.app.orchestrator.save_pid_session_data(str(database_path)),
            self._pid_data_saved,
            disable=self.save_data_button,
        )

    def _import_pid_data(self):
        default_dir=Path(str(self.app.frontend_config.get("pid_database_dir",ensure_user_subdir("experiments"))))
        selected,_=QFileDialog.getOpenFileName(
            self,
            "选择已保存的 PID 数据",
            str(default_dir),
            "SQLite 数据库 (*.sqlite *.db);;所有文件 (*)",
        )
        if not selected:
            return
        database_path=Path(selected)
        self.app.save(pid_database_dir=str(database_path.parent))
        self.app.task(
            lambda:self.app.orchestrator.load_pid_replay(str(database_path)),
            self._open_pid_replay,
            disable=self.import_data_button,
        )

    def _open_pid_replay(self,replay):
        if self._replay_dialog is not None:
            self._replay_dialog.close()
        self._replay_dialog=PIDReplayDialog(replay,self)
        self._replay_dialog.destroyed.connect(
            lambda _object=None:setattr(self,"_replay_dialog",None)
        )
        self._replay_dialog.show()
        self._replay_dialog.raise_()
        self._replay_dialog.activateWindow()

    def _experiment_summary(self,count):
        try:snapshot=self.app.orchestrator.get_snapshot()
        except Exception:return f"PID 已停止，本次共记录 {count} 个控制周期。"
        recognition=getattr(snapshot,"recognition",None); pump=getattr(snapshot,"pump_state",None)
        total=int(getattr(recognition,"total_droplet_count",0) or 0); cv=getattr(recognition,"raw_frame_diameter_cv",None)
        cv_text="--" if cv is None else f"{float(cv):.2f}%"
        physical="是" if bool(getattr(pump,"physical_flow_measured",False)) else "否（设备回读仅为参数换算值）"
        return f"PID 已停止，本次记录 {count} 个控制周期；累计液滴 {total}；末周期原始 CV {cv_text}；物理流量测量：{physical}。"

    def _pid_data_saved(self, result):
        QMessageBox.information(
            self,
            "保存完成",
            f"已保存 {int(result.get('record_count',0))} 条 PID 数据：\n{result.get('path','')}",
        )

    def _start_status_thread(self):
        self._stop_status_thread(); interval=int(self.app.frontend_config.get("control_interval_ms",500) or 500)
        self._status_thread=QThread(self); self._status_thread.setObjectName("monitor-status-control-period")
        waiter=getattr(self.app.orchestrator,"wait_for_control_snapshot",None)
        fetch=self._fetch_control_status if callable(waiter) else self.app.orchestrator.get_snapshot
        self._status_poller=StatusPoller(fetch,interval,blocking_fetch=callable(waiter)); self._status_poller.moveToThread(self._status_thread)
        self._status_thread.started.connect(self._status_poller.run); self._status_poller.snapshotReady.connect(self.show_status); self._status_poller.failed.connect(lambda exc:self._set_panel(self.pid_panel,f"控制状态读取失败：{exc}")); self._status_poller.finished.connect(self._status_thread.quit); self._status_thread.start()

    def _stop_status_thread(self):
        if self._status_poller is not None: self._status_poller.stop()
        if self._status_thread is not None and self._status_thread.isRunning(): self._status_thread.quit(); self._status_thread.wait(1500)
        self._status_poller=None; self._status_thread=None

    def _fetch_control_status(self):
        interval=float(self.app.frontend_config.get("control_interval_ms",500) or 500)/1000.0
        snap=self.app.orchestrator.wait_for_control_snapshot(self._last_control_status_ts,timeout=min(0.5,max(0.1,interval*1.5)))
        control=getattr(snap,"control",None); self._last_control_status_ts=float(getattr(control,"timestamp",self._last_control_status_ts) or self._last_control_status_ts)
        return snap

    @Slot(object)
    def show_status(self,snap):
        """The blocking control-period feed updates only the PID module."""
        self._set_panel(self.pid_panel, self._pid_text(snap))

    def refresh(self):
        """Compatibility hook; each module still owns its normal refresh timer."""
        self.refresh_system_panel()
        self.refresh_vision_panel()
        self.refresh_pump_panel()
        try:
            self.show_status(self.app.orchestrator.get_snapshot())
        except Exception as exc:
            self._set_panel(self.pid_panel, f"控制状态读取失败：{exc}")
        self.refresh_video()

    @staticmethod
    def _set_panel(panel, text):
        if panel.toPlainText() != text:
            panel.setPlainText(text)

    def refresh_system_panel(self):
        try:
            snapshot=self.app.orchestrator.get_snapshot()
            self._set_panel(self.system_panel, self._system_text(snapshot))
            self._update_operation_matrix(snapshot)
        except Exception as exc:
            self._set_panel(self.system_panel, f"系统状态读取失败：{exc}")

    def _update_operation_matrix(self, snapshot):
        state=str(getattr(getattr(snapshot,"system_state",None),"value",getattr(snapshot,"system_state",""))).upper()
        allowed={
            "初始化":{"IDLE","CONFIGURED","VIDEO_READY","STOPPED","ERROR"},
            "开始":{"INITIALIZED","PAUSED","STOPPED"},
            "暂停/继续":{"OPTIMIZING","STABILIZING","RUNNING","PAUSED"},
            "复位安全锁":{"ERROR"},
        }
        for name,button in self.action_buttons.items():
            button.setEnabled(state in allowed[name])
        state_cn, _yes, _number=self._display_helpers(snapshot)
        self.state_badge.setText(state_cn)
        self.state_badge.setProperty("state","error" if state=="ERROR" else ("active" if state in {"OPTIMIZING","STABILIZING","RUNNING"} else "idle"))
        self.state_badge.style().unpolish(self.state_badge); self.state_badge.style().polish(self.state_badge)
        self.pause_button.setText("继续" if state=="PAUSED" else "暂停")
        config=getattr(snapshot,"config",None)
        source_type=str(getattr(config,"video_source_type","") or "").strip().lower()
        realtime=source_type not in {"file","local","local_video","video"}
        self.bo_button.setEnabled(realtime and state in {"INITIALIZED","PAUSED","STOPPED"})
        self.stop_button.setEnabled(state in {"OPTIMIZING","STABILIZING","RUNNING","PAUSED","INITIALIZING","ERROR"})
        self.target_button.setEnabled(state in {"CONFIGURED","VIDEO_READY","INITIALIZED","RUNNING","PAUSED","STOPPED"})
        self.app.refresh_navigation(snapshot)

    def refresh_vision_panel(self):
        try:
            self._set_panel(self.vision_panel, self._vision_text(self.app.orchestrator.get_snapshot()))
        except Exception as exc:
            self._set_panel(self.vision_panel, f"识别状态读取失败：{exc}")

    def refresh_pump_panel(self):
        try:
            self._set_panel(self.pump_panel, self._pump_text(self.app.orchestrator.get_snapshot()))
        except Exception as exc:
            self._set_panel(self.pump_panel, f"泵机状态读取失败：{exc}")

    def refresh_video(self):
        try: frame=self.app.orchestrator.get_video_frame_snapshot()
        except Exception as exc:
            frame=None; self._show_frame_error(f"实时帧读取失败：{exc}")
        if frame is None: return self._show_frame_error("等待后端发布视频帧…")
        fid=int(getattr(frame,"frame_id",0) or getattr(frame,"preview_frame_id",0) or 0)
        if fid==self.last_frame: return
        try:
            binary=getattr(frame,"frame_jpeg",None) or getattr(frame,"frame_pgm",None); payload=getattr(frame,"frame_png_base64",None)
            if binary: data=bytes(binary)
            elif payload: data=base64.b64decode(payload)
            else: return self._show_frame_error(str(getattr(frame,"reason","") or "视频服务尚未返回图像数据"))
            image=QImage.fromData(data)
            if image.isNull(): raise ValueError(f"Qt 无法解码帧数据（{len(data)} 字节）")
            pix=QPixmap.fromImage(image).scaled(self.video.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation); self.video.setPixmap(pix); self.last_frame=fid; self._frame_error=""; self.video.setToolTip(f"实时帧 #{fid} · {image.width()}×{image.height()}")
            now=time.monotonic(); self._display_times.append(now)
            while self._display_times and self._display_times[0]<now-1.0: self._display_times.popleft()
            fps=(len(self._display_times)-1)/max(0.001,self._display_times[-1]-self._display_times[0]) if len(self._display_times)>1 else 0.0; self.fps_label.setText(f"视频显示：{fps:.1f} / 30 FPS")
        except Exception as exc: self._show_frame_error(f"视频帧解码失败：{exc}")

    def _show_frame_error(self,message):
        if message==self._frame_error: return
        self._frame_error=message; self.video.setText(message); self.app.runtime_logger(f"[MONITOR][FRAME] {message}")

    @staticmethod
    def _display_helpers(snap):
        state_map={"idle":"空闲","configured":"参数已配置","video_ready":"视频已就绪","initializing":"正在初始化","initialized":"初始化完成","optimizing":"BO 寻优中","stabilizing":"最优点稳定保持中","running":"运行中","paused":"已暂停","stopping":"正在停止","stopped":"已停止","error":"错误"}
        state=getattr(getattr(snap,"system_state",None),"value",getattr(snap,"system_state","--")); state_cn=state_map.get(str(state).lower(),str(state))
        yes=lambda value:"是" if value else "否"
        number=lambda value,digits=2:"--" if value is None else f"{float(value):.{digits}f}"
        return state_cn, yes, number

    @staticmethod
    def _system_text(snap):
        state_cn, _yes, _number = MonitorPage._display_helpers(snap)
        optimization=dict(getattr(snap,"optimization",None) or {})
        bo_text="未启动"
        if optimization:
            bo_text=(f"{optimization.get('phase','--')}，有效观测 {optimization.get('observation_count',0)}，"
                     f"确认 {optimization.get('confirmation_count',0)}；{optimization.get('reason','') or '无说明'}")
        return "\n".join([
            f"运行状态：{state_cn}",
            f"BO 状态：{bo_text}",
            f"提示信息：{getattr(snap,'message','') or '无'}",
            f"错误信息：{getattr(snap,'error','') or '无'}",
        ])

    @staticmethod
    def _vision_text(snap):
        rec = getattr(snap, "recognition", None)
        _state, yes, number = MonitorPage._display_helpers(snap)
        if rec is None:
            return "等待识别数据…"
        calibration_map={"disabled":"未启用","collecting":"标定中","calibrated":"已完成","user_config":"采用用户设置","failed":"失败"}
        calibration_status=calibration_map.get(str(getattr(rec,'channel_calibration_status','disabled')),str(getattr(rec,'channel_calibration_status','--')))
        return "\n".join([
            f"控制周期：{getattr(rec,'control_period_id',0)}",
            f"当前帧编号：{getattr(rec,'frame_id',0)}",
            f"本周期穿线液滴：{getattr(rec,'frame_droplet_count',0)}",
            f"累计穿线液滴：{getattr(rec,'total_droplet_count',0)}",
            f"平均直径：{number(getattr(rec,'avg_diameter',None))} μm",
            f"液滴速度：{number(getattr(rec,'average_droplet_speed_um_s',None))} μm/s",
            f"速度样本数：{getattr(rec,'speed_sample_count',0)}",
            f"累计液滴级单珠率：{number(getattr(rec,'single_cell_rate',None))}%",
            f"本周期液滴级单珠率：{number(getattr(rec,'frame_single_cell_rate',None))}%",
            f"原始/筛选后直径 CV：{number(getattr(rec,'raw_frame_diameter_cv',None))}% / {number(getattr(rec,'frame_diameter_cv',None))}%",
            f"筛选规则：{getattr(rec,'filtering_rule','none')}",
            f"可用于控制：{yes(getattr(rec,'valid_for_control',False))}",
            f"管道标定：{calibration_status}",
            f"管道内宽：{number(getattr(rec,'channel_width_px',None))} px",
            f"比例：{number(getattr(rec,'pixel_to_micron',None),6)} μm/px",
            f"比例来源/标定 ID：{getattr(rec,'scale_source','--')} / {getattr(rec,'calibration_id','') or '--'}",
            f"硬件帧号：{getattr(rec,'hardware_frame_id',0) or '--'}",
            f"采集/处理：{number(getattr(rec,'capture_fps',0),1)} / {number(getattr(rec,'processing_fps',0),1)} FPS",
            f"主机取帧后识别延迟：{number(getattr(rec,'recognition_latency_ms',0),1)} ms",
            f"说明：{getattr(rec,'reason','') or '无'}",
        ])

    @staticmethod
    def _pump_text(snap):
        pump = getattr(snap, "pump_state", None)
        _state, yes, number = MonitorPage._display_helpers(snap)
        if pump is None:
            return "等待泵机数据…"
        channels = dict(getattr(pump, "channels", {}) or {})
        q1_channel = channels.get("Q1") or channels.get("q1")
        q2_channel = channels.get("Q2") or channels.get("q2")
        def channel_status(channel):
            if channel is None:
                return "等待状态"
            normal = (
                bool(getattr(channel, "enabled", False))
                and bool(getattr(channel, "running", False))
                and bool(getattr(channel, "communication_ok", False))
            )
            return "正常运行" if normal else "未正常运行"
        pump_normal = (
            bool(getattr(pump, "connected", False))
            and bool(getattr(pump, "comm_established", False))
            and bool(getattr(pump, "fully_ready", False))
            and bool(getattr(pump, "running", False))
        )
        return "\n".join([
            f"泵机：{'正常运行' if pump_normal else '未正常运行'}",
            f"Q1：{channel_status(q1_channel)}",
            f"Q2：{channel_status(q2_channel)}",
            f"设定值 Q1/Q2：{number(getattr(pump,'q1',None))} / {number(getattr(pump,'q2',None))} μL/min",
            f"设备参数换算值 Q1/Q2：{number(getattr(pump,'q1_actual',None))} / {number(getattr(pump,'q2_actual',None))} μL/min",
            f"物理流量已测量：{yes(getattr(pump,'physical_flow_measured',False))}",
            f"泵物理响应延迟：{number(getattr(pump,'pump_response_delay_ms',None),1)} ms（{getattr(pump,'pump_response_measurement_status','unmeasured')}）",
        ])

    @staticmethod
    def _pid_text(snap):
        ctrl = getattr(snap, "control", None)
        config = getattr(snap, "config", None)
        _state, yes, number = MonitorPage._display_helpers(snap)
        if ctrl is None:
            return "等待控制数据…"
        adaptive_status = "已接入"
        if bool(getattr(ctrl, "adaptive_active", False)):
            adaptive_status += "，本周期已调参"
        elif bool(getattr(ctrl, "adaptive_enabled", False)):
            adaptive_status += "，正在预热/等待有效样本"
        else:
            adaptive_status = "未启用"
        return "\n".join([
            f"当前设定目标：{number(getattr(config,'target_diameter',None),3)} μm",
            f"本周期采用目标：{number(getattr(ctrl,'target_diameter_um',None),3)} μm",
            f"直径误差：{number(getattr(ctrl,'diameter_error',None))} μm",
            f"Q1/Q2 指令：{number(getattr(ctrl,'q1_command',None))} / {number(getattr(ctrl,'q2_command',None))} μL/min",
            f"PID / 前馈 / 最终输出：{number(getattr(ctrl,'pid_output',None),4)} / {number(getattr(ctrl,'feedforward_output',None),4)} / {number(getattr(ctrl,'final_output',None),4)}",
            f"执行器饱和：{yes(getattr(ctrl,'actuator_saturated',False))}",
            f"控制权所有者：{getattr(ctrl,'control_owner','--')}",
            f"控制模式：{getattr(ctrl,'control_mode','--')}",
            f"自适应状态：{adaptive_status}",
            f"自适应说明：{getattr(ctrl,'adaptive_reason','') or '无'}",
            f"Kp / Ki / Kd：{number(getattr(ctrl,'kp',None),6)} / {number(getattr(ctrl,'ki',None),6)} / {number(getattr(ctrl,'kd',None),6)}",
            f"Q1/Q2 调节倍率：{number(getattr(ctrl,'q1_output_gain',None),2)} : {number(getattr(ctrl,'q2_output_gain',None),2)}",
            f"请求/实际分配输出：{number(getattr(ctrl,'requested_output',None),4)} / {number(getattr(ctrl,'realized_output',None),4)}",
            f"BO/PID 工作点 Q1/Q2：{number(getattr(ctrl,'operating_point_q1',None))} / {number(getattr(ctrl,'operating_point_q2',None))} μL/min",
            f"前馈状态：{yes(getattr(ctrl,'feedforward_active',False))}；{getattr(ctrl,'feedforward_reason','') or '无'}",
            f"反馈冻结：{yes(getattr(ctrl,'freeze_feedback',False))}",
            f"控制说明：{getattr(ctrl,'reason','') or '无'}",
        ])

    @staticmethod
    def chinese_status(snap):
        """Backward-compatible combined rendering used by older callers."""
        return "\n\n".join([
            "【系统状态】\n" + MonitorPage._system_text(snap),
            "【液滴识别】\n" + MonitorPage._vision_text(snap),
            "【泵机状态】\n" + MonitorPage._pump_text(snap),
            "【PID 控制】\n" + MonitorPage._pid_text(snap),
        ])


class StatusPage(Page):
    def __init__(self, app):
        super().__init__(app)
        layout=QVBoxLayout(self)
        layout.addWidget(app.title("系统状态","先查看关键健康状态，需要时再展开完整 SystemSnapshot"))
        self.cards={}
        card_grid=QGridLayout(); card_grid.setSpacing(10)
        for index,(key,title) in enumerate((("system","系统"),("vision","液滴识别"),("camera","视频源"),("pump","泵机"),("pid","PID 控制"))):
            card=HealthCard(title); self.cards[key]=card; card_grid.addWidget(card,index//3,index%3)
        for column in range(3): card_grid.setColumnStretch(column,1)
        layout.addLayout(card_grid)
        self.alert=QLabel("等待系统快照…"); self.alert.setObjectName("statusAlert"); self.alert.setWordWrap(True); layout.addWidget(self.alert)
        actions=QHBoxLayout()
        self.updated=QLabel("最后更新：--"); self.updated.setObjectName("mutedText"); actions.addWidget(self.updated); actions.addStretch()
        self.raw_toggle=QPushButton("展开原始 JSON"); set_button_role(self.raw_toggle,"secondary"); self.raw_toggle.setCheckable(True); self.raw_toggle.toggled.connect(self._toggle_raw); actions.addWidget(self.raw_toggle)
        copy_button=QPushButton("复制 JSON"); set_button_role(copy_button,"secondary"); copy_button.clicked.connect(self._copy_json); actions.addWidget(copy_button)
        export_button=QPushButton("导出 JSON"); set_button_role(export_button,"secondary"); export_button.clicked.connect(self._export_json); actions.addWidget(export_button)
        layout.addLayout(actions)
        self.text=QPlainTextEdit(); self.text.setReadOnly(True); self.text.setVisible(False); layout.addWidget(self.text,1)
        layout.addStretch(1)
        self.timer=QTimer(self); self.timer.setInterval(app.refresh_interval_ms); self.timer.timeout.connect(self.refresh)
    def on_show(self): self.timer.start(); self.refresh()
    def on_hide(self): self.timer.stop()

    def _toggle_raw(self, expanded):
        self.text.setVisible(expanded)
        self.raw_toggle.setText("收起原始 JSON" if expanded else "展开原始 JSON")

    def _copy_json(self):
        QApplication.clipboard().setText(self.text.toPlainText())
        self.alert.setText("完整 SystemSnapshot JSON 已复制到剪贴板。")

    def _export_json(self):
        target,_=QFileDialog.getSaveFileName(self,"导出系统状态","system_snapshot.json","JSON 文件 (*.json)")
        if not target:return
        try: Path(target).write_text(self.text.toPlainText(),encoding="utf-8")
        except OSError as exc:return self.app.error("导出失败",str(exc))
        self.alert.setText(f"系统状态已导出：{target}")

    @staticmethod
    def _overview(snapshot):
        state=str(getattr(getattr(snapshot,"system_state",None),"value",getattr(snapshot,"system_state","--"))).upper()
        error=str(getattr(snapshot,"error","") or "")
        message=str(getattr(snapshot,"message","") or "")
        state_cn,yes,number=MonitorPage._display_helpers(snapshot)
        system_status="error" if error or state=="ERROR" else ("ok" if state in {"INITIALIZED","RUNNING","PAUSED","STOPPED"} else ("active" if state in {"INITIALIZING","OPTIMIZING","STABILIZING","STOPPING"} else "idle"))
        system_badge="异常" if system_status=="error" else ("进行中" if system_status=="active" else ("正常" if system_status=="ok" else "等待"))
        recognition=getattr(snapshot,"recognition",None)
        if recognition is None: vision=("idle","等待","尚无识别数据")
        elif bool(getattr(recognition,"valid_for_control",False)): vision=("ok","有效",f"平均直径 {number(getattr(recognition,'avg_diameter',None))} μm · 本周期 {getattr(recognition,'frame_droplet_count',0)} 个")
        else: vision=("warning","暂不可控",str(getattr(recognition,"reason","") or "识别数据尚未达到控制条件"))
        frame=getattr(snapshot,"frame",None)
        if frame is None: camera=("idle","等待","尚无视频帧")
        elif bool(getattr(frame,"valid",False)): camera=("ok","在线",f"帧 #{getattr(frame,'frame_id',0)} · {getattr(frame,'width',0)}×{getattr(frame,'height',0)}")
        else: camera=("warning","无效帧",str(getattr(frame,"reason","") or "视频帧不可用"))
        pump_state=getattr(snapshot,"pump_state",None)
        if pump_state is None: pump=("idle","等待","尚无泵机状态")
        elif str(getattr(pump_state,"last_error","") or ""): pump=("error","异常",str(getattr(pump_state,"last_error","")))
        elif bool(getattr(pump_state,"connected",False)) and bool(getattr(pump_state,"comm_established",False)): pump=("ok","已连接",f"Q1/Q2 {number(getattr(pump_state,'q1',None))} / {number(getattr(pump_state,'q2',None))} μL/min")
        else: pump=("warning","未就绪",str(getattr(pump_state,"last_update_reason","") or "泵机尚未建立完整通信"))
        control=getattr(snapshot,"control",None)
        if control is None: pid=("idle","等待","尚无控制周期数据")
        elif bool(getattr(control,"suggested_stop",False)): pid=("error","建议停止",str(getattr(control,"reason","") or "控制器建议安全停止"))
        elif bool(getattr(control,"freeze_feedback",False)): pid=("warning","反馈冻结",str(getattr(control,"reason","") or "当前周期未更新反馈"))
        else: pid=("ok","运行正常",f"误差 {number(getattr(control,'diameter_error',None))} μm · 输出 {number(getattr(control,'final_output',None),4)}")
        alert=error or message or "系统没有报告异常。"
        return {"system":(system_status,system_badge,f"{state_cn} · {message or '无附加信息'}"),"vision":vision,"camera":camera,"pump":pump,"pid":pid,"alert":alert}

    def refresh(self):
        try:
            snapshot=self.app.orchestrator.get_snapshot(); overview=self._overview(snapshot)
            self.text.setPlainText(json.dumps(jsonable(snapshot),ensure_ascii=False,indent=2))
            for key,card in self.cards.items(): card.set_status(*overview[key])
            has_error=overview["system"][0]=="error"
            self.alert.setProperty("status","error" if has_error else "ok"); self.alert.setText(("需要处理：" if has_error else "当前提示：")+overview["alert"]); self.alert.style().unpolish(self.alert); self.alert.style().polish(self.alert)
            timestamp=float(getattr(snapshot,"timestamp",0) or 0); stamp=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(timestamp)) if timestamp>1_000_000_000 else time.strftime("%Y-%m-%d %H:%M:%S")
            self.updated.setText(f"最后更新：{stamp}")
            self.app.refresh_navigation(snapshot)
        except Exception as exc:
            self.text.setPlainText(str(exc)); self.alert.setProperty("status","error"); self.alert.setText(f"状态读取失败：{exc}")


class TuningPage(Page):
    """Detector-only workbench embedded directly in the main page."""

    def __init__(self, app):
        super().__init__(app)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.addWidget(app.title("液滴识别算法调参", "只处理本地视频或图像，不连接相机、泵机、跟踪或 PID"))
        tuning_sample = str(app.frontend_config.get("tuning_sample", ""))
        if not tuning_sample:
            video = str(app.frontend_config.get("video_source", ""))
            tuning_sample = video if app.frontend_config.get("video_source_type") == "file" else ""
        self.workbench = TuningWindow(
            tuning_sample,
            self,
            app.save_tuning_sample,
            settings_store=app.vision_tuning_settings_store,
        )
        layout.addWidget(self.workbench, 1)


class FrontendApp(QMainWindow):
    def __init__(self, orchestrator=None, settings_store=None):
        super().__init__(); self.setWindowTitle(APP_TITLE); self.resize(1360,860); self.setMinimumSize(QSize(1180,700))
        self.runtime_logger=create_runtime_logger(); self.orchestrator=orchestrator or OrchestratorService(logger=self.runtime_logger); self.settings_store=settings_store or FrontendSettingsStore(); self.vision_tuning_settings_store=VisionTuningSettingsStore(); self.frontend_config=self.settings_store.load(); self.device_verification={"camera":False,"pump":False}; self.refresh_interval_ms=DEFAULT_REFRESH_INTERVAL_MS
        self.pool=QThreadPool.globalInstance(); self.workers=set(); self.current=None; self._build(); self.show_page("parameter")

    def _build(self):
        root=QWidget(); self.setCentralWidget(root); outer=QHBoxLayout(root); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)
        self.nav_panel=QFrame(); self.nav_panel.setObjectName("navPanel"); self.nav_panel.setFixedWidth(238)
        nav_layout=QVBoxLayout(self.nav_panel); nav_layout.setContentsMargins(12,14,12,14); nav_layout.setSpacing(10)
        self.nav_toggle=QPushButton("收起导航"); self.nav_toggle.setObjectName("navToggle"); self.nav_toggle.setToolTip("收起左侧导航栏"); self.nav_toggle.clicked.connect(self._toggle_nav)
        nav_layout.addWidget(self.nav_toggle)
        self.nav=QListWidget(); self.nav.setObjectName("nav"); self.nav.setSpacing(2); nav_layout.addWidget(self.nav,1)
        self.stack=QStackedWidget(); outer.addWidget(self.nav_panel); outer.addWidget(self.stack,1)
        self._nav_collapsed=False
        self._nav_entries=(("parameter","基础参数",1),("video","相机识别与读写",2),("pump","泵机识别与读写",3),("init","系统初始化",4),("monitor","运行监控",5),("status","系统状态",None),("tuning","液滴算法调参",None))
        self.pages={"parameter":ParameterPage(self),"video":VideoPage(self),"pump":PumpPage(self),"init":InitPage(self),"monitor":MonitorPage(self),"status":StatusPage(self),"tuning":TuningPage(self)}
        self._nav_items={}; self._nav_headers=[]
        icon_map={"parameter":QStyle.SP_FileDialogDetailedView,"video":QStyle.SP_ComputerIcon,"pump":QStyle.SP_DriveNetIcon,"init":QStyle.SP_BrowserReload,"monitor":QStyle.SP_MediaPlay,"status":QStyle.SP_MessageBoxInformation,"tuning":QStyle.SP_FileDialogContentsView}
        for group,keys in (("实验流程",("parameter","video","pump","init","monitor")),("工具与诊断",("status","tuning"))):
            header=QListWidgetItem(group); header.setFlags(Qt.ItemIsEnabled); header.setData(Qt.UserRole,"__header__"); header.setSizeHint(QSize(210,30)); font=header.font(); font.setBold(True); header.setFont(font); header.setForeground(QColor("#8fa4bf")); self.nav.addItem(header); self._nav_headers.append(header)
            for key,label,step in self._nav_entries:
                if key not in keys: continue
                item=QListWidgetItem(self.style().standardIcon(icon_map[key]),label); item.setData(Qt.UserRole,key); item.setData(Qt.UserRole+1,label); item.setData(Qt.UserRole+2,step); item.setToolTip(label); item.setSizeHint(QSize(210,46)); self.nav.addItem(item); self._nav_items[key]=item; self.stack.addWidget(self.pages[key])
        self.nav.currentItemChanged.connect(lambda current,_: current and self.show_page(str(current.data(Qt.UserRole))))
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#f4f7fb;color:#172033;font-size:14px}
            #navPanel{background:#152238}
            #navToggle{background:#223554;color:#dbe7f7;border:1px solid #3b5275;border-radius:8px;padding:9px 12px;text-align:left}
            #navToggle:hover{background:#2b6de5;color:white}
            #nav{background:transparent;color:#dbe7f7;border:0;padding:2px 0;outline:0}
            #nav::item{border-radius:8px;padding:8px 10px;margin:1px 0}
            #nav::item:hover{background:#203554}
            #nav::item:selected{background:#2b6de5;color:white}
            #nav::item:disabled{color:#71839a}
            QGroupBox{background:white;border:1px solid #dce3ed;border-radius:10px;margin-top:14px;padding:18px;font-weight:600}
            QGroupBox::title{subcontrol-origin:margin;left:14px;padding:0 6px}
            QLineEdit,QComboBox,QPlainTextEdit,QDoubleSpinBox,QSpinBox{background:white;border:1px solid #cbd5e1;border-radius:6px;padding:7px;selection-background-color:#2b6de5}
            QLineEdit:focus,QComboBox:focus,QPlainTextEdit:focus,QDoubleSpinBox:focus,QSpinBox:focus{border:2px solid #2b6de5;padding:6px}
            QLineEdit:read-only,QPlainTextEdit:read-only{background:#f8fafc}
            QPushButton,QToolButton{background:#2b6de5;color:white;border:1px solid #2b6de5;border-radius:6px;padding:8px 14px}
            QPushButton:hover,QToolButton:hover{background:#1f5fcf;border-color:#1f5fcf}
            QPushButton:pressed,QToolButton:pressed{background:#194fae}
            QPushButton:focus,QToolButton:focus{border:2px solid #0f3f99;padding:7px 13px}
            QPushButton:disabled,QToolButton:disabled{background:#e2e8f0;color:#94a3b8;border-color:#e2e8f0}
            QPushButton[role="secondary"]{background:white;color:#2457a7;border:1px solid #9bb7e6}
            QPushButton[role="secondary"]:hover{background:#eef4ff;border-color:#2b6de5}
            QPushButton[role="danger"]{background:#c93838;color:white;border-color:#c93838}
            QPushButton[role="danger"]:hover{background:#a92626;border-color:#a92626}
            QPushButton[role="warning"]{background:#fff7e8;color:#8a4b08;border:1px solid #e8b35d}
            QPushButton[role="warning"]:hover{background:#ffedc7;border-color:#d89224}
            QToolButton[role="section"]{background:#eef4ff;color:#2457a7;border:1px solid #c7d8f3;text-align:left;padding:9px 12px}
            QToolButton[role="link"]{background:transparent;color:#2b6de5;border:0;padding:3px 6px}
            #controlBar{background:white;border:1px solid #dce3ed;border-radius:10px}
            #runtimeStateBadge{background:#edf2f7;color:#526071;border-radius:10px;padding:5px 10px;font-weight:700}
            #runtimeStateBadge[state="active"]{background:#e7f0ff;color:#1f5fcf}
            #runtimeStateBadge[state="error"]{background:#fde8e8;color:#a92626}
            #fpsLabel,#mutedText,#formHint{color:#64748b}
            #advancedSummary{background:#f8fafc;color:#42526a;border:1px solid #dce3ed;border-radius:7px;padding:8px}
            #metricSummary{background:#f8fafc;border:1px solid #e2e8f0;border-radius:7px;padding:9px;line-height:1.3}
            #healthCard{background:white;border:1px solid #dce3ed;border-left:4px solid #94a3b8;border-radius:9px}
            #healthCard[health="ok"]{border-left-color:#24935a}
            #healthCard[health="active"]{border-left-color:#2b6de5}
            #healthCard[health="warning"]{border-left-color:#d89224}
            #healthCard[health="error"]{border-left-color:#c93838}
            #healthTitle{font-weight:700}
            #healthBadge{background:#edf2f7;border-radius:9px;padding:3px 8px;font-weight:700}
            #healthDetail{color:#526071}
            #statusAlert{background:#edf7f1;color:#226943;border:1px solid #b9dfc9;border-radius:7px;padding:10px}
            #statusAlert[status="error"]{background:#fdecec;color:#9e2929;border-color:#efb6b6}
            QLabel[status="warning"]{color:#9a5a0a}
            QLabel[status="ok"]{color:#247548}
        """)
        self.refresh_navigation()

    def _toggle_nav(self):
        self._nav_collapsed=not self._nav_collapsed
        width=68 if self._nav_collapsed else 238
        self.nav_panel.setFixedWidth(width)
        self.nav_toggle.setText("展开" if self._nav_collapsed else "收起导航")
        self.nav_toggle.setToolTip("展开左侧导航栏" if self._nav_collapsed else "收起左侧导航栏")
        for header in self._nav_headers: header.setHidden(self._nav_collapsed)
        for key,item in self._nav_items.items():
            label=str(item.data(Qt.UserRole+1)); step=item.data(Qt.UserRole+2)
            item.setText("" if self._nav_collapsed else self._navigation_label(key,label,step))
            item.setSizeHint(QSize(42 if self._nav_collapsed else 210,46))
        self.nav.repaint()

    def title(self,heading,subtitle):
        widget=QFrame(); layout=QVBoxLayout(widget); layout.setContentsMargins(0,2,0,2); layout.setSpacing(2); label=QLabel(heading); label.setStyleSheet("font-size:24px;font-weight:700"); detail=QLabel(subtitle); detail.setStyleSheet("color:#64748b"); layout.addWidget(label); layout.addWidget(detail); return widget

    def _navigation_label(self,key,label,step):
        item=self._nav_items.get(key); status=str(item.data(Qt.UserRole+3) or "") if item else ""
        prefix=f"{step}  " if step else ""
        suffix={"complete":"  ✓","active":"  ●","error":"  !","locked":"  ·"}.get(status,"")
        return prefix+label+suffix

    def refresh_navigation(self,snapshot=None):
        cfg=self.frontend_config
        state=""
        if snapshot is not None: state=str(getattr(getattr(snapshot,"system_state",None),"value",getattr(snapshot,"system_state",""))).upper()
        base=all(cfg.get(key) not in (None,"") for key in ("target_diameter","pixel_to_micron","control_interval_ms"))
        video=bool(cfg.get("video_source_type") and cfg.get("video_source"))
        pump=bool(cfg.get("pump_port")) or cfg.get("video_source_type")=="file"
        initialized=state in {"INITIALIZED","OPTIMIZING","STABILIZING","RUNNING","PAUSED","STOPPING","STOPPED"}
        monitor_available=initialized or state=="ERROR"
        running=state in {"OPTIMIZING","STABILIZING","RUNNING","PAUSED"}
        status_map={"parameter":"complete" if base else "active","video":"complete" if video else ("active" if base else "locked"),"pump":"complete" if pump else ("active" if video else "locked"),"init":"error" if state=="ERROR" else ("complete" if initialized else ("active" if video and pump else "locked")),"monitor":"error" if state=="ERROR" else ("active" if running else ("complete" if state=="STOPPED" else ("locked" if not initialized else "active"))),"status":"error" if state=="ERROR" else "","tuning":""}
        enabled={"parameter":True,"video":base,"pump":video,"init":video and pump,"monitor":monitor_available,"status":True,"tuning":True}
        for key,item in self._nav_items.items():
            item.setData(Qt.UserRole+3,status_map[key])
            item.setFlags((Qt.ItemIsSelectable|Qt.ItemIsEnabled) if enabled[key] else Qt.NoItemFlags)
            item.setText("" if self._nav_collapsed else self._navigation_label(key,str(item.data(Qt.UserRole+1)),item.data(Qt.UserRole+2)))
    def show_page(self,key):
        if key not in self.pages:return
        if self.current: self.current.on_hide()
        page=self.pages[key]; self.stack.setCurrentWidget(page); self.current=page; page.on_show()
        item=self._nav_items.get(key)
        if item is not None:
            self.nav.blockSignals(True); self.nav.setCurrentItem(item); self.nav.blockSignals(False)
    def save_tuning_sample(self, path: str):
        self.save(tuning_sample=path)

    def save(self,**values):
        self.frontend_config.update(values)
        try: self.settings_store.save(self.frontend_config)
        except Exception as exc: self.runtime_logger(f"[APP][SETTINGS][ERROR] {exc}")
        if hasattr(self,"_nav_items"): self.refresh_navigation()
    def build_system_config(self):
        cfg=self.frontend_config; required=("target_diameter","pixel_to_micron","video_source_type","video_source","initial_q1","initial_q2","control_interval_ms"); missing=[k for k in required if k not in cfg]
        if missing: raise ValueError(f"缺少配置字段: {', '.join(missing)}")
        return SystemConfig(target_diameter=float(cfg["target_diameter"]),pixel_to_micron=float(cfg["pixel_to_micron"]),video_source_type=str(cfg["video_source_type"]),video_source=str(cfg["video_source"]),initial_q1=float(cfg["initial_q1"]),initial_q2=float(cfg["initial_q2"]),control_interval_ms=int(cfg["control_interval_ms"]),pump_port=str(cfg.get("pump_port","")),pump_address=int(cfg.get("pump_address",1)),pump_baudrate=int(cfg.get("pump_baudrate",1200)),pump_parity=str(cfg.get("pump_parity","N")),mvs_sdk_path=str(cfg.get("mvs_sdk_path","")),camera_backend=str(cfg.get("camera_backend","")),camera_parameters=dict(cfg.get("camera_parameters",{}) or {}),recognition_roi=dict(cfg.get("recognition_roi",{}) or {}),calibration=dict(cfg.get("calibration",{}) or {}))
    def configure_prepare_initialize(self):
        cfg=self.build_system_config(); self.orchestrator.configure(cfg); self.orchestrator.prepare_video(); self.orchestrator.initialize_system()
    def task(self,task,on_success=None,on_error=None,disable=None):
        worker=Worker(task); self.workers.add(worker)
        if disable: disable.setEnabled(False)
        worker.signals.succeeded.connect(on_success or (lambda _:None)); worker.signals.failed.connect(on_error or (lambda exc:self.error("操作失败",str(exc))))
        def cleanup():
            if disable: disable.setEnabled(True)
            self.workers.discard(worker)
        worker.signals.finished.connect(cleanup); self.pool.start(worker)
    def error(self,title,message): QMessageBox.critical(self,title,message)
    def closeEvent(self,event:QCloseEvent):
        tuning_page = self.pages.get("tuning")
        if isinstance(tuning_page, TuningPage) and tuning_page.workbench is not None:
            tuning_page.workbench.close()
        try: self.orchestrator.stop()
        except Exception: pass
        if self.orchestrator.has_unsaved_pid_session_data():
            status=self.orchestrator.get_pid_session_data_status(); count=int(status.get("record_count",0) or 0)
            answer=QMessageBox.question(
                self,
                "保存 PID 数据",
                f"还有 {count} 条本次 PID 数据尚未保存。\n退出前是否保存？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer == QMessageBox.StandardButton.Yes:
                default_dir=Path(str(self.frontend_config.get("pid_database_dir",ensure_user_subdir("experiments"))))
                default_name=f"pid_data_{time.strftime('%Y%m%d_%H%M%S')}.sqlite"
                selected,_=QFileDialog.getSaveFileName(self,"选择 PID 数据库保存位置",str(default_dir/default_name),"SQLite 数据库 (*.sqlite *.db)")
                if not selected:
                    event.ignore()
                    return
                database_path=Path(selected)
                if not database_path.suffix: database_path=database_path.with_suffix(".sqlite")
                try:
                    result=self.orchestrator.save_pid_session_data(str(database_path)); self.save(pid_database_dir=str(database_path.parent))
                    QMessageBox.information(self,"保存完成",f"已保存 {int(result.get('record_count',0))} 条 PID 数据：\n{result.get('path','')}")
                except Exception as exc:
                    self.error("PID 数据保存失败",str(exc)); event.ignore(); return
            else:
                self.orchestrator.discard_pid_session_data()
        if self.current is not None: self.current.on_hide()
        event.accept()


def _configure_application_font(application: QApplication) -> None:
    """Use an installed UI font instead of Qt's unresolved generic alias."""
    installed = set(QFontDatabase.families())
    for family in (".AppleSystemUIFont", "Segoe UI", "Noto Sans CJK SC", "Arial"):
        if family in installed:
            application.setFont(QFont(family, 14))
            return


def main():
    mp.freeze_support(); application=QApplication.instance() or QApplication(sys.argv); _configure_application_font(application); application.setApplicationName(APP_TITLE)
    instance_lock=QLockFile(str(ensure_user_subdir("runtime") / "application.lock")); instance_lock.setStaleLockTime(0)
    if not instance_lock.tryLock(100):
        QMessageBox.critical(None,"程序已在运行","检测到另一个程序实例；为避免串口和数据文件冲突，本次启动已取消。")
        raise SystemExit(2)
    window=FrontendApp(); window.show(); raise SystemExit(application.exec())

if __name__ == "__main__": main()
