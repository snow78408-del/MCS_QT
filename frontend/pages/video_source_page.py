from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


class VideoSourcePage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.mode_var = tk.StringVar(value="camera")
        self.file_var = tk.StringVar(value="")
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
        self.exposure_var = tk.StringVar(value="")
        self.gain_var = tk.StringVar(value="")
        self.frame_rate_var = tk.StringVar(value="")
        self.width_var = tk.StringVar(value="")
        self.height_var = tk.StringVar(value="")
        self._devices: list[dict[str, object]] = []
        self._display_to_device: dict[str, dict[str, object]] = {}
        self._last_discovery_result: dict[str, object] = {}
        self._selected_test_ok = False
        self._preview_photo = None
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
                ("曝光", self.exposure_var),
                ("增益", self.gain_var),
                ("帧率", self.frame_rate_var),
                ("宽度", self.width_var),
                ("高度", self.height_var),
            )
        ):
            ttk.Label(params, text=label).grid(row=0, column=idx * 2, padx=6, pady=6, sticky="w")
            ttk.Entry(params, textvariable=var, width=10).grid(row=0, column=idx * 2 + 1, padx=6, pady=6, sticky="w")

        self.test_btn = ttk.Button(root, text="测试取帧", command=self._test_camera)
        self.test_btn.grid(row=6, column=0, padx=8, pady=8, sticky="w")
        ttk.Label(root, textvariable=self.status_var).grid(row=6, column=1, padx=8, pady=8, sticky="w")
        self.preview_label = ttk.Label(root, text="预览画面")
        self.preview_label.grid(row=7, column=0, columnspan=3, padx=8, pady=8, sticky="w")

        diag = ttk.LabelFrame(root, text="后端诊断")
        diag.grid(row=8, column=0, columnspan=3, padx=8, pady=8, sticky="nsew")
        diag.columnconfigure(0, weight=1)
        root.rowconfigure(8, weight=1)
        self.diagnostic_text = tk.Text(diag, height=8, wrap="word", state="disabled")
        self.diagnostic_text.grid(row=0, column=0, sticky="nsew")
        diag_scroll = ttk.Scrollbar(diag, command=self.diagnostic_text.yview)
        diag_scroll.grid(row=0, column=1, sticky="ns")
        self.diagnostic_text.configure(yscrollcommand=diag_scroll.set)

        self.file_label = ttk.Label(root, text="本地视频路径")
        self.file_entry = ttk.Entry(root, textvariable=self.file_var, width=72)
        self.file_btn = ttk.Button(root, text="浏览", command=self._browse_file)
        self.file_label.grid(row=9, column=0, padx=8, pady=8, sticky="w")
        self.file_entry.grid(row=9, column=1, padx=8, pady=8, sticky="ew")
        self.file_btn.grid(row=9, column=2, padx=8, pady=8, sticky="w")

        ttk.Button(root, text="上一步", command=lambda: self.app.show_page("parameter")).grid(
            row=10, column=0, padx=8, pady=16, sticky="w"
        )
        ttk.Button(root, text="下一步", command=self._next_step).grid(row=10, column=2, padx=8, pady=16, sticky="e")
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
                self.after(0, lambda: self._scan_failed(exc))

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
            self.device_var.set(values[0])
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

    def _friendly_type(self, dtype: str) -> str:
        if dtype == "industrial_camera":
            return "工业相机"
        if dtype == "usb_camera":
            return "普通 USB 摄像头"
        return "未知相机"

    def _mark_untested(self) -> None:
        self._selected_test_ok = False
        self.status_var.set("未测试")
        self._preview_photo = None
        if hasattr(self, "preview_label"):
            self.preview_label.configure(image="", text="预览画面")

    def _camera_params(self) -> dict[str, object]:
        return {
            "exposure": self.exposure_var.get().strip(),
            "gain": self.gain_var.get().strip(),
            "frame_rate": self.frame_rate_var.get().strip(),
            "width": self.width_var.get().strip(),
            "height": self.height_var.get().strip(),
        }

    def _test_camera(self) -> None:
        device = self._selected_device()
        if device is None:
            messagebox.showerror("输入错误", "请先扫描并选择相机设备")
            return
        unique_id = str(device.get("unique_id", "") or "")
        backend = self.backend_var.get().strip()
        self.status_var.set("正在测试取帧...")
        self.error_var.set("")
        self.test_btn.configure(state="disabled")

        def worker() -> None:
            try:
                self.app.orchestrator.select_camera(unique_id, backend or None)
                result = self.app.orchestrator.test_camera()
                self.after(0, lambda: self._apply_test_result(result))
            except Exception as exc:
                self.after(0, lambda: self._test_failed(exc))

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
            preview = result.get("preview_png_base64")
            if preview:
                self._preview_photo = tk.PhotoImage(data=str(preview))
                self.preview_label.configure(image=self._preview_photo, text="")
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
                messagebox.showerror("输入错误", "请先执行测试取帧，成功后才能进入实时监控")
                return
            self.app.frontend_config["video_source_type"] = "camera"
            self.app.frontend_config["video_source"] = str(device.get("unique_id", "") or "")
            self.app.frontend_config["camera_backend"] = self.backend_var.get().strip()
            self.app.frontend_config["camera_device"] = dict(device)
            self.app.frontend_config["camera_parameters"] = self._camera_params()
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
        self.app.show_page("init")
