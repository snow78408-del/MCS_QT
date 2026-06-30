from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk

try:
    from backend.orchestrator.models import RecognitionSnapshot, SystemSnapshot
except Exception:  # pragma: no cover
    from ...backend.orchestrator.models import RecognitionSnapshot, SystemSnapshot


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
        self.frame_count_var.set("0")
        self.single_count_var.set("0")
        self.avg_diameter_var.set("--")
        self.single_cell_rate_var.set("--")
        self.std_var.set("--")
        self.cv_var.set("--")
        self.uniformity_var.set(reason)
        self.valid_var.set("否")
        self.reason_var.set(reason)

    def update_recognition(self, rec: RecognitionSnapshot | None, *, synced: bool = True) -> None:
        if rec is None:
            self._clear_current_frame_metrics("暂无识别结果")
            return

        stale = (time.time() - float(rec.timestamp)) > self.FRAME_TIMEOUT_S
        self.frame_id_var.set(str(rec.frame_id))
        self.new_crossing_var.set(str(rec.new_crossing_count))
        self.total_count_var.set(str(rec.total_droplet_count))
        self.video_mode_var.set(rec.video_source_type or "--")
        self.video_source_var.set(rec.video_source or "--")
        self.timestamp_var.set(f"{rec.timestamp:.3f}")
        if rec.frame_width > 0 and rec.frame_height > 0:
            self.resolution_var.set(f"{rec.frame_width} x {rec.frame_height}")
        else:
            self.resolution_var.set("--")

        if not synced:
            self._clear_current_frame_metrics("数据同步中")
            return
        if stale:
            self._clear_current_frame_metrics("图像数据已过期")
            return
        if rec.frame_droplet_count <= 0:
            self._clear_current_frame_metrics(rec.reason or "当前无有效液滴")
            return

        self.frame_count_var.set(str(rec.frame_droplet_count))
        self.single_count_var.set(str(rec.frame_single_cell_count))
        self.avg_diameter_var.set(self._format_float(rec.frame_avg_diameter, " μm"))
        self.single_cell_rate_var.set(self._format_float(rec.frame_single_cell_rate, " %"))
        self.std_var.set(self._format_float(rec.frame_diameter_std, " μm"))
        self.cv_var.set(self._format_float(rec.frame_diameter_cv, " %"))
        uniformity = rec.uniformity_status or ("有效" if rec.uniformity_valid else "样本不足")
        if rec.uniformity_reason:
            uniformity = f"{uniformity}（{rec.uniformity_reason}）"
        self.uniformity_var.set(uniformity)
        self.valid_var.set("是" if rec.valid_for_control else "否")
        self.reason_var.set(rec.reason or rec.control_reason or "--")

    def update_snapshot(self, snapshot: SystemSnapshot | None) -> None:
        if snapshot is None:
            self.update_recognition(None)
            return
        rec = snapshot.recognition
        synced = True
        if snapshot.frame is not None and rec is not None:
            synced = int(snapshot.frame.frame_id) == int(rec.frame_id)
        self.update_recognition(rec, synced=synced)
