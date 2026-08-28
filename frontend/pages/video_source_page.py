from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from backend.vision.camera_profiles import normalize_camera_parameters, resolve_camera_defaults


class VideoSourcePage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        saved = app.frontend_config
        saved_camera_parameters = saved.get("camera_parameters", {})
        saved_recognition_roi = saved.get("recognition_roi", {})
        camera_parameters = dict(saved_camera_parameters) if isinstance(saved_camera_parameters, dict) else {}
        recognition_roi = dict(saved_recognition_roi) if isinstance(saved_recognition_roi, dict) else {}
        self._saved_camera_parameters = camera_parameters
        saved_mode = str(saved.get("video_source_type", "camera") or "camera")
        self.mode_var = tk.StringVar(value=saved_mode if saved_mode in {"camera", "file"} else "camera")
        self.file_var = tk.StringVar(value=str(saved.get("video_source", "")) if saved_mode == "file" else "")
        self.device_var = tk.StringVar(value="")
        self.backend_var = tk.StringVar(value="")
        self.vendor_var = tk.StringVar(value="--")
        self.model_var = tk.StringVar(value="--")
        self.serial_var = tk.StringVar(value="--")
        self.device_type_var = tk.StringVar(value="--")
        self.transport_var = tk.StringVar(value="--")
        self.ip_var = tk.StringVar(value="--")
        self.sdk_status_var = tk.StringVar(value="未扫描")
        self.status_var = tk.StringVar(value="未测试")
        self.error_var = tk.StringVar(value="")
        self.exposure_var = tk.StringVar(value=str(camera_parameters.get("exposure", 3000)))
        self.gain_var = tk.StringVar(value=str(camera_parameters.get("gain", 0)))
        self.frame_rate_var = tk.StringVar(value=str(camera_parameters.get("frame_rate", 100)))
        self.width_var = tk.StringVar(value=str(camera_parameters.get("width", 720)))
        self.height_var = tk.StringVar(value=str(camera_parameters.get("height", 540)))
        self.camera_profile_var = tk.StringVar(value="海康机器人 CS 系列默认参数")
        self.roi_enabled_var = tk.BooleanVar(value=bool(recognition_roi.get("enabled", False)))
        self.roi_x0_var = tk.StringVar(value=str(float(recognition_roi.get("x_start_ratio", 0)) * 100))
        self.roi_y0_var = tk.StringVar(value=str(float(recognition_roi.get("y_start_ratio", 0)) * 100))
        self.roi_x1_var = tk.StringVar(value=str(float(recognition_roi.get("x_end_ratio", 1)) * 100))
        self.roi_y1_var = tk.StringVar(value=str(float(recognition_roi.get("y_end_ratio", 1)) * 100))
        self._devices: list[dict[str, object]] = []
        self._display_to_device: dict[str, dict[str, object]] = {}
        self._last_discovery_result: dict[str, object] = {}
        self._selected_test_ok = False
        self._tested_camera_parameters: dict[str, object] | None = None
        self._preview_photo = None
        self._roi_drag_start: tuple[int, int] | None = None
        self._roi_rect_id: int | None = None
        self._page_canvas: tk.Canvas | None = None
        self._page_scrollbar: ttk.Scrollbar | None = None
        self._page_body: ttk.Frame | None = None
        self._mousewheel_bound = False
        self._build()

    def _build(self) -> None:
        self._page_canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self._page_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._page_canvas.yview)
        self._page_canvas.configure(yscrollcommand=self._page_scrollbar.set)
        self._page_scrollbar.pack(side="right", fill="y")
        self._page_canvas.pack(side="left", fill="both", expand=True)

        self._page_body = ttk.Frame(self._page_canvas)
        body_window = self._page_canvas.create_window((0, 0), window=self._page_body, anchor="nw")
        self._page_body.bind(
            "<Configure>",
            lambda _event: self._page_canvas.configure(scrollregion=self._page_canvas.bbox("all")),
        )
        self._page_canvas.bind(
            "<Configure>",
            lambda event: self._page_canvas.itemconfigure(body_window, width=event.width),
        )
        self._page_canvas.bind("<Enter>", lambda _event: self._bind_mousewheel())
        self._page_canvas.bind("<Leave>", lambda _event: self._unbind_mousewheel())
        self._page_body.bind("<Enter>", lambda _event: self._bind_mousewheel())
        self._page_body.bind("<Leave>", lambda _event: self._unbind_mousewheel())

        root = ttk.LabelFrame(self._page_body, text="视频来源选择")
        root.pack(fill="both", expand=True, padx=24, pady=24)
        root.columnconfigure(1, weight=1)

        ttk.Radiobutton(root, text="实时摄像头", value="camera", variable=self.mode_var, command=self._toggle).grid(
            row=0, column=0, padx=8, pady=8, sticky="w"
        )
        ttk.Radiobutton(root, text="本地视频文件", value="file", variable=self.mode_var, command=self._toggle).grid(
            row=0, column=1, padx=8, pady=8, sticky="w"
        )

        self.scan_btn = ttk.Button(root, text="扫描设备", command=self._scan_devices)
        self.refresh_btn = ttk.Button(root, text="刷新设备", command=self._scan_devices)
        self.scan_btn.grid(row=1, column=0, padx=8, pady=6, sticky="w")
        self.refresh_btn.grid(row=1, column=2, padx=8, pady=6, sticky="w")

        ttk.Label(root, text="设备").grid(row=2, column=0, padx=8, pady=6, sticky="w")
        self.device_combo = ttk.Combobox(root, textvariable=self.device_var, state="readonly", width=88)
        self.device_combo.grid(row=2, column=1, columnspan=2, padx=8, pady=6, sticky="ew")
        self.device_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_device_selected())

        ttk.Label(root, text="当前后端").grid(row=3, column=0, padx=8, pady=6, sticky="w")
        self.backend_combo = ttk.Combobox(root, textvariable=self.backend_var, state="readonly", width=28)
        self.backend_combo.grid(row=3, column=1, padx=8, pady=6, sticky="w")
        self.backend_combo.bind("<<ComboboxSelected>>", lambda _event: self._mark_untested())

        info = ttk.LabelFrame(root, text="设备信息")
        info.grid(row=4, column=0, columnspan=3, padx=8, pady=8, sticky="ew")
        for col in range(4):
            info.columnconfigure(col, weight=1)
        self._info_row(info, 0, "厂商", self.vendor_var, "型号", self.model_var)
        self._info_row(info, 1, "序列号", self.serial_var, "相机类型", self.device_type_var)
        self._info_row(info, 2, "接口类型", self.transport_var, "IP地址", self.ip_var)
        self._info_row(info, 3, "SDK状态", self.sdk_status_var, "错误信息", self.error_var)

        params = ttk.LabelFrame(root, text="相机参数")
        params.grid(row=5, column=0, columnspan=3, padx=8, pady=8, sticky="ew")
        for idx, (label, var) in enumerate(
            (
                ("曝光(μs)", self.exposure_var),
                ("增益(dB)", self.gain_var),
                ("帧率(fps)", self.frame_rate_var),
                ("宽度(px)", self.width_var),
                ("高度(px)", self.height_var),
            )
        ):
            ttk.Label(params, text=label).grid(row=0, column=idx * 2, padx=6, pady=6, sticky="w")
            entry = ttk.Entry(params, textvariable=var, width=10)
            entry.grid(row=0, column=idx * 2 + 1, padx=6, pady=6, sticky="w")
            entry.bind("<KeyRelease>", lambda _event: self._mark_untested())
        ttk.Label(params, textvariable=self.camera_profile_var).grid(
            row=1, column=0, columnspan=8, padx=6, pady=(2, 6), sticky="w"
        )
        ttk.Button(params, text="恢复推荐值", command=self._restore_camera_defaults).grid(
            row=1, column=8, columnspan=2, padx=6, pady=(2, 6), sticky="e"
        )

        self.test_btn = ttk.Button(root, text="写入参数并测试取帧", command=self._test_camera)
        self.test_btn.grid(row=6, column=0, padx=8, pady=8, sticky="w")
        ttk.Label(root, textvariable=self.status_var).grid(row=6, column=1, padx=8, pady=8, sticky="w")
        self.preview_canvas = tk.Canvas(root, width=760, height=560, background="black", highlightthickness=1)
        self.preview_canvas.grid(row=8, column=0, columnspan=3, padx=8, pady=8, sticky="w")
        self.preview_canvas.create_text(380, 280, text="预览画面（取帧后可拖框选择 ROI）", fill="white", tags="placeholder")
        self.preview_canvas.bind("<ButtonPress-1>", self._roi_drag_begin)
        self.preview_canvas.bind("<B1-Motion>", self._roi_drag_move)
        self.preview_canvas.bind("<ButtonRelease-1>", self._roi_drag_end)

        roi = ttk.LabelFrame(root, text="液滴识别区域 ROI（预览画面百分比）")
        roi.grid(row=7, column=0, columnspan=3, padx=8, pady=8, sticky="ew")
        ttk.Checkbutton(roi, text="仅识别指定区域", variable=self.roi_enabled_var).grid(row=0, column=0, padx=6, pady=6)
        for column, (label, variable) in enumerate((("左", self.roi_x0_var), ("上", self.roi_y0_var), ("右", self.roi_x1_var), ("下", self.roi_y1_var)), start=1):
            ttk.Label(roi, text=f"{label}(%)").grid(row=0, column=column * 2 - 1, padx=(6, 2), pady=6)
            ttk.Entry(roi, textvariable=variable, width=7).grid(row=0, column=column * 2, padx=(2, 6), pady=6)
        ttk.Label(roi, text="例如左20、上10、右80、下90；关闭时识别整个画面").grid(row=1, column=0, columnspan=9, padx=6, pady=(0, 6), sticky="w")
        ttk.Button(roi, text="全画面识别", command=self._use_full_frame).grid(row=0, column=9, padx=6, pady=6)

        diag = ttk.LabelFrame(root, text="后端诊断")
        diag.grid(row=9, column=0, columnspan=3, padx=8, pady=8, sticky="nsew")
        diag.columnconfigure(0, weight=1)
        root.rowconfigure(9, weight=1)
        self.diagnostic_text = tk.Text(diag, height=8, wrap="word", state="disabled")
        self.diagnostic_text.grid(row=0, column=0, sticky="nsew")
        diag_scroll = ttk.Scrollbar(diag, command=self.diagnostic_text.yview)
        diag_scroll.grid(row=0, column=1, sticky="ns")
        self.diagnostic_text.configure(yscrollcommand=diag_scroll.set)

        self.file_label = ttk.Label(root, text="本地视频路径")
        self.file_entry = ttk.Entry(root, textvariable=self.file_var, width=72)
        self.file_btn = ttk.Button(root, text="浏览", command=self._browse_file)
        self.file_label.grid(row=10, column=0, padx=8, pady=8, sticky="w")
        self.file_entry.grid(row=10, column=1, padx=8, pady=8, sticky="ew")
        self.file_btn.grid(row=10, column=2, padx=8, pady=8, sticky="w")

        ttk.Button(root, text="上一步", command=lambda: self.app.show_page("parameter")).grid(
            row=11, column=0, padx=8, pady=16, sticky="w"
        )
        ttk.Button(root, text="下一步", command=self._next_step).grid(row=11, column=2, padx=8, pady=16, sticky="e")
        self._toggle()
        self._bind_mousewheel()

    def _bind_mousewheel(self) -> None:
        if self._mousewheel_bound:
            return
        for widget in self._scroll_widgets():
            widget.bind("<MouseWheel>", self._on_mousewheel, add="+")
            widget.bind("<Button-4>", self._on_mousewheel, add="+")
            widget.bind("<Button-5>", self._on_mousewheel, add="+")
        self._mousewheel_bound = True

    def _unbind_mousewheel(self) -> None:
        return

    def _scroll_widgets(self):
        stack = [self]
        while stack:
            widget = stack.pop()
            yield widget
            stack.extend(widget.winfo_children())

    def _on_mousewheel(self, event) -> str:
        if self._page_canvas is None:
            return "break"
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -1 * int(event.delta / 120)
        self._page_canvas.yview_scroll(delta, "units")
        return "break"

    def on_hide(self) -> None:
        self._unbind_mousewheel()

    def _info_row(self, parent, row, label_a, var_a, label_b, var_b) -> None:
        ttk.Label(parent, text=label_a).grid(row=row, column=0, padx=8, pady=4, sticky="w")
        ttk.Label(parent, textvariable=var_a).grid(row=row, column=1, padx=8, pady=4, sticky="w")
        ttk.Label(parent, text=label_b).grid(row=row, column=2, padx=8, pady=4, sticky="w")
        ttk.Label(parent, textvariable=var_b, wraplength=360).grid(row=row, column=3, padx=8, pady=4, sticky="w")

    def _toggle(self) -> None:
        is_camera = self.mode_var.get() == "camera"
        camera_state = "normal" if is_camera else "disabled"
        combo_state = "readonly" if is_camera else "disabled"
        for widget in (self.scan_btn, self.refresh_btn, self.test_btn):
            widget.configure(state=camera_state)
        self.device_combo.configure(state=combo_state)
        self.backend_combo.configure(state=combo_state)
        file_state = "disabled" if is_camera else "normal"
        self.file_entry.configure(state=file_state)
        self.file_btn.configure(state=file_state)

    def _browse_file(self) -> None:
        path = filedialog.askopenfilename(
            title="选择本地视频文件",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")],
        )
        if path:
            self.file_var.set(path)

    def _scan_devices(self) -> None:
        self.scan_btn.configure(state="disabled")
        self.refresh_btn.configure(state="disabled")
        self.sdk_status_var.set("正在扫描所有相机后端...")
        self.error_var.set("")
        self._set_diagnostics("正在扫描，请关闭厂商官方相机软件及其预览窗口，避免设备被独占。")
        self._selected_test_ok = False

        def worker() -> None:
            try:
                result = self.app.orchestrator.discover_cameras()
                self.after(0, lambda: self._apply_discovery_result(result))
            except Exception as exc:
                self.after(0, lambda error=exc: self._scan_failed(error))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_discovery_result(self, result: dict[str, object]) -> None:
        self._last_discovery_result = result
        devices = list(result.get("devices", []) or [])
        self._devices = [dict(device) for device in devices if isinstance(device, dict)]
        self._display_to_device = {self._format_device(device): device for device in self._devices}
        values = list(self._display_to_device.keys())
        self.device_combo.configure(values=values)
        self._set_diagnostics(self._format_backend_diagnostics(result))
        if values:
            saved_id = str(self.app.frontend_config.get("video_source", "") or "")
            selected = next(
                (label for label, device in self._display_to_device.items()
                 if str(device.get("unique_id", "") or "") == saved_id),
                values[0],
            )
            self.device_var.set(selected)
            self.sdk_status_var.set(f"发现 {len(values)} 个设备")
            self._on_device_selected()
        else:
            self.device_var.set("")
            self.backend_combo.configure(values=[])
            self.backend_var.set("")
            self.sdk_status_var.set("未发现设备，请查看后端诊断")
            self.error_var.set(self._first_backend_error(result))
        self.scan_btn.configure(state="normal")
        self.refresh_btn.configure(state="normal")

    def _scan_failed(self, exc: Exception) -> None:
        self.sdk_status_var.set("扫描失败")
        self.error_var.set(str(exc))
        self._set_diagnostics(f"扫描失败：{exc}")
        self.scan_btn.configure(state="normal")
        self.refresh_btn.configure(state="normal")

    def _set_diagnostics(self, text: str) -> None:
        self.diagnostic_text.configure(state="normal")
        self.diagnostic_text.delete("1.0", "end")
        self.diagnostic_text.insert("1.0", text)
        self.diagnostic_text.configure(state="disabled")

    def _format_backend_diagnostics(self, result: dict[str, object]) -> str:
        lines: list[str] = []
        statuses = list(result.get("backend_statuses", []) or [])
        if statuses:
            for status in statuses:
                if not isinstance(status, dict):
                    continue
                name = str(status.get("backend_name", "") or "--")
                display = self._friendly_backend_name(name)
                available = "可用" if status.get("backend_available") else "不可用"
                count = int(status.get("raw_device_count", 0) or 0)
                reason = str(status.get("error", "") or "")
                cti_paths = list(status.get("cti_paths", []) or [])
                detail = f"{display}：{available}，发现 {count} 台设备"
                if name == "gentl":
                    detail += f"，已加载 CTI {len(cti_paths)} 个"
                if reason:
                    detail += f"，原因：{reason}"
                lines.append(detail)
        raw_count = int(result.get("raw_device_count", 0) or 0)
        final_count = int(result.get("final_device_count", 0) or 0)
        lines.append(f"原始设备数：{raw_count}，去重后设备数：{final_count}")
        errors = list(result.get("errors", []) or [])
        if errors:
            lines.append("错误：")
            lines.extend(str(item) for item in errors)
        return "\n".join(lines)

    def _first_backend_error(self, result: dict[str, object]) -> str:
        for status in list(result.get("backend_statuses", []) or []):
            if isinstance(status, dict) and status.get("error"):
                return str(status.get("error"))
        return "未发现设备；请查看后端诊断。"

    def _friendly_backend_name(self, backend: str) -> str:
        return {
            "hikrobot": "海康MVS",
            "basler": "Basler",
            "daheng": "大恒",
            "flir": "FLIR",
            "allied_vision": "Allied Vision",
            "gentl": "GenTL",
            "opencv": "OpenCV",
        }.get(backend, backend or "--")

    def _format_device(self, device: dict[str, object]) -> str:
        dtype = str(device.get("device_type", "") or "")
        vendor = str(device.get("manufacturer", "") or "")
        model = str(device.get("model", "") or device.get("user_defined_name", "") or "")
        serial = str(device.get("serial_number", "") or "")
        transport = str(device.get("transport_type", "") or "Unknown")
        backend = str(device.get("selected_backend", "") or device.get("backend_name", "") or "")
        if dtype == "usb_camera":
            return f"[普通摄像头][UVC] {model or 'Camera'}"
        label = "[工业相机]"
        if backend == "gentl":
            label += "[GenTL]"
        elif vendor:
            label += f"[{vendor}]"
        label += f"[{transport}] {model or '--'}"
        if serial:
            label += f" SN:{serial}"
        return label

    def _selected_device(self) -> dict[str, object] | None:
        return self._display_to_device.get(self.device_var.get())

    def _selected_industrial_error(self, device: dict[str, object]) -> str:
        dtype = str(device.get("device_type", "") or "").strip().lower()
        backend = str(self.backend_var.get() or device.get("selected_backend", "") or device.get("backend_name", ""))
        backend = backend.strip().lower()
        if dtype != "industrial_camera":
            return f"Realtime monitoring must use an industrial camera; selected device_type={dtype or '--'}."
        if backend == "opencv":
            return "Realtime monitoring must use the Hikrobot MVS SDK backend, not OpenCV/USB."
        return ""

    def _on_device_selected(self) -> None:
        device = self._selected_device()
        self._mark_untested()
        if device is None:
            return
        self.vendor_var.set(str(device.get("manufacturer", "") or "--"))
        self.model_var.set(str(device.get("model", "") or "--"))
        self.serial_var.set(str(device.get("serial_number", "") or "--"))
        self.device_type_var.set(self._friendly_type(str(device.get("device_type", "") or "")))
        self.transport_var.set(str(device.get("transport_type", "") or "--"))
        self.ip_var.set(str(device.get("ip_address", "") or "--"))
        backends = list(device.get("available_backends", []) or [device.get("backend_name", "")])
        self.backend_combo.configure(values=backends)
        self.backend_var.set(str(device.get("selected_backend", "") or (backends[0] if backends else "")))
        self.sdk_status_var.set("可用后端: " + ", ".join(str(b) for b in backends))
        self.error_var.set(str(device.get("error", "") or ""))
        self._restore_camera_defaults()
        saved_id = str(self.app.frontend_config.get("video_source", "") or "")
        if str(device.get("unique_id", "") or "") == saved_id and self._saved_camera_parameters:
            for key, variable in (
                ("exposure", self.exposure_var),
                ("gain", self.gain_var),
                ("frame_rate", self.frame_rate_var),
                ("width", self.width_var),
                ("height", self.height_var),
            ):
                if key in self._saved_camera_parameters:
                    variable.set(str(self._saved_camera_parameters[key]))
            self._mark_untested()

    def _friendly_type(self, dtype: str) -> str:
        if dtype == "industrial_camera":
            return "工业相机"
        if dtype == "usb_camera":
            return "普通 USB 摄像头"
        return "未知相机"

    def _mark_untested(self) -> None:
        self._selected_test_ok = False
        self._tested_camera_parameters = None
        self.status_var.set("未测试")
        self._preview_photo = None
        if hasattr(self, "preview_canvas"):
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(380, 280, text="预览画面（取帧后可拖框选择 ROI）", fill="white", tags="placeholder")

    def _roi_drag_begin(self, event) -> None:
        if self._preview_photo is None:
            return
        x = max(0, min(self._preview_photo.width() - 1, int(event.x)))
        y = max(0, min(self._preview_photo.height() - 1, int(event.y)))
        self._roi_drag_start = (x, y)
        if self._roi_rect_id is not None:
            self.preview_canvas.delete(self._roi_rect_id)
        self._roi_rect_id = self.preview_canvas.create_rectangle(x, y, x, y, outline="#00ff66", width=2)

    def _roi_drag_move(self, event) -> None:
        if self._roi_drag_start is None or self._roi_rect_id is None or self._preview_photo is None:
            return
        x = max(0, min(self._preview_photo.width() - 1, int(event.x)))
        y = max(0, min(self._preview_photo.height() - 1, int(event.y)))
        self.preview_canvas.coords(self._roi_rect_id, *self._roi_drag_start, x, y)

    def _roi_drag_end(self, event) -> None:
        if self._roi_drag_start is None or self._preview_photo is None:
            return
        width, height = self._preview_photo.width(), self._preview_photo.height()
        x0, y0 = self._roi_drag_start
        x1 = max(0, min(width - 1, int(event.x)))
        y1 = max(0, min(height - 1, int(event.y)))
        self._roi_drag_start = None
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        if right - left < 5 or bottom - top < 5:
            return
        self.roi_x0_var.set(f"{left / width * 100.0:.1f}")
        self.roi_y0_var.set(f"{top / height * 100.0:.1f}")
        self.roi_x1_var.set(f"{right / width * 100.0:.1f}")
        self.roi_y1_var.set(f"{bottom / height * 100.0:.1f}")
        self.roi_enabled_var.set(True)

    def _use_full_frame(self) -> None:
        self.roi_enabled_var.set(False)
        self.roi_x0_var.set("0")
        self.roi_y0_var.set("0")
        self.roi_x1_var.set("100")
        self.roi_y1_var.set("100")
        if self._roi_rect_id is not None:
            self.preview_canvas.delete(self._roi_rect_id)
            self._roi_rect_id = None

    def _camera_params(self) -> dict[str, object]:
        return normalize_camera_parameters({
            "exposure": self.exposure_var.get().strip(),
            "gain": self.gain_var.get().strip(),
            "frame_rate": self.frame_rate_var.get().strip(),
            "width": self.width_var.get().strip(),
            "height": self.height_var.get().strip(),
        })

    def _recognition_roi(self) -> dict[str, float | bool]:
        values = [float(var.get().strip()) / 100.0 for var in (self.roi_x0_var, self.roi_y0_var, self.roi_x1_var, self.roi_y1_var)]
        x0, y0, x1, y1 = values
        if not all(0.0 <= value <= 1.0 for value in values) or x1 <= x0 or y1 <= y0:
            raise ValueError("ROI 必须在 0–100% 内，且右/下必须大于左/上")
        return {"enabled": bool(self.roi_enabled_var.get()), "x_start_ratio": x0, "y_start_ratio": y0, "x_end_ratio": x1, "y_end_ratio": y1}

    def _restore_camera_defaults(self) -> None:
        profile_name, defaults = resolve_camera_defaults(self._selected_device())
        self.camera_profile_var.set(f"参数模板：{profile_name}")
        if defaults:
            self.exposure_var.set(str(defaults["exposure"]))
            self.gain_var.set(str(defaults["gain"]))
            self.frame_rate_var.set(str(defaults["frame_rate"]))
            self.width_var.set(str(defaults["width"]))
            self.height_var.set(str(defaults["height"]))
        self._mark_untested()

    def _test_camera(self) -> None:
        device = self._selected_device()
        if device is None:
            messagebox.showerror("输入错误", "请先扫描并选择相机设备")
            return
        try:
            parameters = self._camera_params()
        except ValueError as exc:
            messagebox.showwarning("参数错误", str(exc))
            return
        unique_id = str(device.get("unique_id", "") or "")
        backend = self.backend_var.get().strip()
        self.status_var.set("正在写入相机参数并测试取帧...")
        self.error_var.set("")
        self.test_btn.configure(state="disabled")

        def worker() -> None:
            try:
                self.app.orchestrator.select_camera(unique_id, backend or None)
                result = self.app.orchestrator.test_camera(parameters)
                result["_requested_parameters"] = parameters
                self.after(0, lambda: self._apply_test_result(result))
            except Exception as exc:
                self.after(0, lambda error=exc: self._test_failed(error))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_test_result(self, result: dict[str, object]) -> None:
        ok = bool(result.get("ok", False))
        self._selected_test_ok = ok
        if ok:
            width = int(result.get("width", 0) or 0)
            height = int(result.get("height", 0) or 0)
            fmt = str(result.get("pixel_format", "") or "--")
            frames = int(result.get("frames_read", 0) or 0)
            self.status_var.set(f"测试成功：{width} x {height}, {fmt}, {frames} 帧")
            self._tested_camera_parameters = dict(result.get("_requested_parameters", {}) or {})
            applied_parameters = dict(result.get("applied_parameters", {}) or {})
            missing_parameters = sorted(set(self._tested_camera_parameters) - set(applied_parameters))
            if missing_parameters:
                self._selected_test_ok = False
                self.status_var.set("相机参数未完整下发")
                self.error_var.set(
                    "当前适配器未写入参数：" + ", ".join(missing_parameters)
                )
            requested_width = int(self._tested_camera_parameters.get("width", 0) or 0)
            requested_height = int(self._tested_camera_parameters.get("height", 0) or 0)
            if requested_width and requested_height and (width != requested_width or height != requested_height):
                self._selected_test_ok = False
                self.status_var.set(
                    f"参数未完全生效：请求 {requested_width} x {requested_height}，实际 {width} x {height}"
                )
                self.error_var.set("请恢复推荐值或按相机支持的分辨率修改后重新测试")
            preview = result.get("preview_png_base64")
            if preview:
                self._preview_photo = tk.PhotoImage(data=str(preview))
                self.preview_canvas.configure(width=self._preview_photo.width(), height=self._preview_photo.height())
                self.preview_canvas.delete("all")
                self.preview_canvas.create_image(0, 0, image=self._preview_photo, anchor="nw")
                self._roi_rect_id = None
        else:
            self.status_var.set("测试失败")
            self.error_var.set(str(result.get("error", "") or "测试取帧失败"))
        self.test_btn.configure(state="normal")

    def _test_failed(self, exc: Exception) -> None:
        self._selected_test_ok = False
        self.status_var.set("测试失败")
        self.error_var.set(str(exc))
        self.test_btn.configure(state="normal")

    def _next_step(self) -> None:
        if self.mode_var.get() == "camera":
            device = self._selected_device()
            if device is None:
                messagebox.showerror("输入错误", "请先扫描并选择相机设备")
                return
            industrial_error = self._selected_industrial_error(device)
            if industrial_error:
                messagebox.showerror("Camera source error", industrial_error)
                return
            if not self._selected_test_ok:
                messagebox.showerror("输入错误", "请先执行“写入参数并测试取帧”，成功后才能进入实时监控")
                return
            try:
                camera_parameters = self._camera_params()
            except (TypeError, ValueError) as exc:
                messagebox.showerror("相机参数错误", str(exc))
                return
            if camera_parameters != (self._tested_camera_parameters or {}):
                messagebox.showerror("相机参数已变化", "参数修改后必须重新执行测试取帧，确认下发成功")
                return
            self.app.frontend_config["video_source_type"] = "camera"
            self.app.frontend_config["video_source"] = str(device.get("unique_id", "") or "")
            self.app.frontend_config["camera_backend"] = self.backend_var.get().strip()
            self.app.frontend_config["camera_device"] = dict(device)
            self.app.frontend_config["camera_parameters"] = camera_parameters
            try:
                self.app.frontend_config["recognition_roi"] = self._recognition_roi()
            except ValueError as exc:
                messagebox.showerror("识别区域错误", str(exc))
                return
            logger = getattr(self.app, "runtime_logger", None)
            if callable(logger):
                logger(
                    "[UI][CAMERA][CONFIRMED] "
                    f"backend={self.app.frontend_config['camera_backend']} "
                    f"unique_id={self.app.frontend_config['video_source']}"
                )
        else:
            path = self.file_var.get().strip()
            if not path or not os.path.isfile(path):
                messagebox.showerror("输入错误", "本地视频模式必须选择有效文件路径")
                return
            self.app.frontend_config["video_source_type"] = "file"
            self.app.frontend_config["video_source"] = path
            self.app.frontend_config["camera_backend"] = ""
            try:
                self.app.frontend_config["recognition_roi"] = self._recognition_roi()
            except ValueError as exc:
                messagebox.showerror("识别区域错误", str(exc))
                return
        self.app.update_frontend_config()
        self.app.show_page("init")
