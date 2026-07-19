from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk

try:
    from backend.orchestrator.models import RecognitionSnapshot, SystemSnapshot
except Exception:  # pragma: no cover
    from ...backend.orchestrator.models import RecognitionSnapshot, SystemSnapshot

from .ui_update import set_var_if_changed


class RecognitionPanel(ttk.LabelFrame):
    FRAME_TIMEOUT_S = 2.0

    def __init__(self, parent):
        super().__init__(parent, text="当前帧液滴指标")
        self.frame_id_var = tk.StringVar(value="--")
        self.frame_count_var = tk.StringVar(value="--")
        self.single_count_var = tk.StringVar(value="--")
        self.avg_diameter_var = tk.StringVar(value="--")
        self.single_cell_rate_var = tk.StringVar(value="--")
        self.std_var = tk.StringVar(value="--")
        self.cv_var = tk.StringVar(value="--")
        self.uniformity_var = tk.StringVar(value="--")
        self.valid_var = tk.StringVar(value="--")
        self.reason_var = tk.StringVar(value="--")
        self.new_crossing_var = tk.StringVar(value="--")
        self.total_count_var = tk.StringVar(value="--")
        self.video_mode_var = tk.StringVar(value="--")
        self.video_source_var = tk.StringVar(value="--")
        self.resolution_var = tk.StringVar(value="--")
        self.timestamp_var = tk.StringVar(value="--")
        self._build()

    def _build(self) -> None:
        rows = [
            ("当前 frame_id", self.frame_id_var),
            ("当前帧液滴数量", self.frame_count_var),
            ("当前帧单胞数量", self.single_count_var),
            ("当前帧平均直径", self.avg_diameter_var),
            ("当前帧单胞率", self.single_cell_rate_var),
            ("当前帧直径标准差", self.std_var),
            ("当前帧直径变异率", self.cv_var),
            ("液滴均匀程度", self.uniformity_var),
            ("识别结果有效", self.valid_var),
            ("状态", self.reason_var),
            ("新增通过液滴", self.new_crossing_var),
            ("累计通过液滴", self.total_count_var),
            ("视频模式", self.video_mode_var),
            ("视频来源", self.video_source_var),
            ("视频分辨率", self.resolution_var),
            ("更新时间", self.timestamp_var),
        ]
        for i, (name, var) in enumerate(rows):
            ttk.Label(self, text=f"{name}:").grid(row=i, column=0, sticky="w", padx=6, pady=3)
            ttk.Label(self, textvariable=var, wraplength=280).grid(row=i, column=1, sticky="w", padx=6, pady=3)
        self.columnconfigure(1, weight=1)

    @staticmethod
    def _format_float(value: float | None, suffix: str = "", digits: int = 2) -> str:
        if value is None:
            return "--"
        return f"{float(value):.{digits}f}{suffix}"

    def _clear_current_frame_metrics(self, reason: str) -> None:
        set_var_if_changed(self.frame_count_var, "0")
        set_var_if_changed(self.single_count_var, "0")
        set_var_if_changed(self.avg_diameter_var, "--")
        set_var_if_changed(self.single_cell_rate_var, "--")
        set_var_if_changed(self.std_var, "--")
        set_var_if_changed(self.cv_var, "--")
        set_var_if_changed(self.uniformity_var, reason)
        set_var_if_changed(self.valid_var, "否")
        set_var_if_changed(self.reason_var, reason)

    def update_recognition(self, rec: RecognitionSnapshot | None, *, synced: bool = True) -> None:
        if rec is None:
            self._clear_current_frame_metrics("暂无识别结果")
            return

        stale = (time.time() - float(rec.timestamp)) > self.FRAME_TIMEOUT_S
        set_var_if_changed(self.frame_id_var, str(rec.frame_id))
        set_var_if_changed(self.new_crossing_var, str(rec.new_crossing_count))
        set_var_if_changed(self.total_count_var, str(rec.total_droplet_count))
        set_var_if_changed(self.video_mode_var, rec.video_source_type or "--")
        set_var_if_changed(self.video_source_var, rec.video_source or "--")
        set_var_if_changed(self.timestamp_var, f"{rec.timestamp:.3f}")
        if rec.frame_width > 0 and rec.frame_height > 0:
            set_var_if_changed(self.resolution_var, f"{rec.frame_width} x {rec.frame_height}")
        else:
            set_var_if_changed(self.resolution_var, "--")

        if not synced:
            self._clear_current_frame_metrics("数据同步中")
            return
        if stale:
            self._clear_current_frame_metrics("图像数据已过期")
            return
        if rec.frame_droplet_count <= 0:
            self._clear_current_frame_metrics(rec.reason or "当前无有效液滴")
            return

        set_var_if_changed(self.frame_count_var, str(rec.frame_droplet_count))
        set_var_if_changed(self.single_count_var, str(rec.frame_single_cell_count))
        set_var_if_changed(self.avg_diameter_var, self._format_float(rec.frame_avg_diameter, " μm"))
        set_var_if_changed(self.single_cell_rate_var, self._format_float(rec.frame_single_cell_rate, " %"))
        set_var_if_changed(self.std_var, self._format_float(rec.frame_diameter_std, " μm"))
        set_var_if_changed(self.cv_var, self._format_float(rec.frame_diameter_cv, " %"))
        uniformity = rec.uniformity_status or ("有效" if rec.uniformity_valid else "样本不足")
        if rec.uniformity_reason:
            uniformity = f"{uniformity}（{rec.uniformity_reason}）"
        set_var_if_changed(self.uniformity_var, uniformity)
        set_var_if_changed(self.valid_var, "是" if rec.valid_for_control else "否")
        set_var_if_changed(self.reason_var, rec.reason or rec.control_reason or "--")

    def update_snapshot(self, snapshot: SystemSnapshot | None) -> None:
        if snapshot is None:
            self.update_recognition(None)
            return
        rec = snapshot.recognition
        synced = True
        if snapshot.frame is not None and rec is not None:
            synced = int(snapshot.frame.frame_id) == int(rec.frame_id)
        self.update_recognition(rec, synced=synced)
