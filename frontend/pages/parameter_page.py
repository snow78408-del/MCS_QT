from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk


DEFAULT_CAMERA_PIXEL_SIZE_UM = 6.9
DEFAULT_MAGNIFICATION = 10.0


class ParameterPage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.target_var = tk.StringVar(value="60")
        self.magnification_var = tk.StringVar(value=f"{DEFAULT_MAGNIFICATION:g}")
        self.camera_pixel_size_var = tk.StringVar(value=f"{DEFAULT_CAMERA_PIXEL_SIZE_UM:g}")
        self.interval_var = tk.StringVar(value="300")
        self._build()

    def _build(self) -> None:
        card = ttk.LabelFrame(self, text="参数设定")
        card.pack(fill="x", padx=24, pady=24)

        ttk.Label(card, text="目标液滴平均直径").grid(row=0, column=0, padx=8, pady=8, sticky="w")
        ttk.Entry(card, textvariable=self.target_var, width=24).grid(row=0, column=1, padx=8, pady=8, sticky="w")

        ttk.Label(card, text="总放大倍率").grid(row=1, column=0, padx=8, pady=8, sticky="w")
        ttk.Entry(card, textvariable=self.magnification_var, width=24).grid(
            row=1, column=1, padx=8, pady=8, sticky="w"
        )

        ttk.Label(card, text="相机像元尺寸(μm)").grid(row=2, column=0, padx=8, pady=8, sticky="w")
        ttk.Entry(card, textvariable=self.camera_pixel_size_var, width=24).grid(
            row=2, column=1, padx=8, pady=8, sticky="w"
        )

        ttk.Label(card, text="控制周期(ms)").grid(row=3, column=0, padx=8, pady=8, sticky="w")
        ttk.Entry(card, textvariable=self.interval_var, width=24).grid(row=3, column=1, padx=8, pady=8, sticky="w")

        ttk.Button(card, text="下一步", command=self._next_step).grid(row=4, column=1, padx=8, pady=16, sticky="e")

    def _next_step(self) -> None:
        try:
            target = float(self.target_var.get().strip())
            magnification = float(self.magnification_var.get().strip())
            camera_pixel_size_um = float(self.camera_pixel_size_var.get().strip())
            interval = int(float(self.interval_var.get().strip()))
            if target <= 0:
                raise ValueError("目标液滴平均直径必须大于 0")
            if magnification <= 0:
                raise ValueError("总放大倍率必须大于 0")
            if camera_pixel_size_um <= 0:
                raise ValueError("相机像元尺寸必须大于 0")
            if interval <= 0:
                raise ValueError("控制周期必须大于 0")
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return

        pixel = camera_pixel_size_um / magnification
        self.app.frontend_config["target_diameter"] = target
        self.app.frontend_config["magnification"] = magnification
        self.app.frontend_config["camera_pixel_size_um"] = camera_pixel_size_um
        self.app.frontend_config["pixel_to_micron"] = pixel
        self.app.frontend_config["control_interval_ms"] = interval
        self.app.show_page("video_source")
