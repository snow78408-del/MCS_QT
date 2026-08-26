from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk

try:
    from backend.orchestrator.state import SystemState
except Exception:  # pragma: no cover
    from ...backend.orchestrator.state import SystemState

from ..components.control_buttons import ControlButtons
from ..components.pump_panel import PumpPanel
from ..components.recognition_panel import RecognitionPanel
from ..components.status_panel import StatusPanel
from ..components.ui_update import set_var_if_changed
from ..video_process import VideoProcessController


class MonitorPage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._video_process: VideoProcessController | None = None
        self._status_poll_job = None
        self._status_worker: threading.Thread | None = None
        self._status_stop_event = threading.Event()
        self._status_snapshot_lock = threading.Lock()
        self._pending_status_snapshot = None
        self._last_button_state: str | None = None
        self._start_pending = False
        self._scrollregion_pending = False
        self._status_refresh_interval_ms = 100
        self._sidebar_visible = True
        self.calibration_var = tk.StringVar(value="未标定")

        self.err_var = tk.StringVar(value="--")
        self.adjust_var = tk.StringVar(value="--")
        self.freeze_var = tk.StringVar(value="--")
        self.stop_var = tk.StringVar(value="--")
        self.q1_cmd_var = tk.StringVar(value="--")
        self.q2_cmd_var = tk.StringVar(value="--")
        self.q1_actual_var = tk.StringVar(value="--")
        self.q2_actual_var = tk.StringVar(value="--")
        self.ch1_exec_var = tk.StringVar(value="--")
        self.ch2_exec_var = tk.StringVar(value="--")
        self.reason_var = tk.StringVar(value="--")

        self.pid_mode_var = tk.StringVar(value="--")
        self.pid_gains_var = tk.StringVar(value="--")
        self.adaptive_var = tk.StringVar(value="--")
        self.adaptive_reason_var = tk.StringVar(value="--")
        self.output_gain_var = tk.StringVar(value="--")
        self.feedforward_var = tk.StringVar(value="--")
        self.pid_output_var = tk.StringVar(value="--")
        self.feedforward_output_var = tk.StringVar(value="--")
        self.final_output_var = tk.StringVar(value="--")
        self.model_state_var = tk.StringVar(value="--")
        self.model_confidence_var = tk.StringVar(value="--")
        self.predicted_change_var = tk.StringVar(value="--")
        self.model_version_var = tk.StringVar(value="--")
        self.model_sample_count_var = tk.StringVar(value="--")
        self.video_mode_var = tk.StringVar(value="--")
        self.video_source_var = tk.StringVar(value="--")
        self.video_res_var = tk.StringVar(value="--")

        self._canvas: tk.Canvas | None = None
        self._content: ttk.Frame | None = None
        self._content_window_id: int | None = None
        self._build()

    def _build(self) -> None:
        self._canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self._content = ttk.Frame(self._canvas)
        self._content_window_id = self._canvas.create_window((0, 0), window=self._content, anchor="nw")
        self._content.bind("<Configure>", self._on_content_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_mousewheel()

        top = ttk.Frame(self._content)
        top.pack(fill="x", padx=10, pady=(6, 4))
        ttk.Button(top, text="参数页", command=lambda: self.app.show_page("parameter")).pack(side="left")
        ttk.Button(top, text="状态页", command=lambda: self.app.show_page("status")).pack(side="left", padx=4)
        ttk.Button(top, text="打开独立视频窗口", command=self._open_video_window).pack(side="left", padx=4)
        self.sidebar_btn = ttk.Button(top, text="隐藏状态侧栏", command=self._toggle_sidebar)
        self.sidebar_btn.pack(side="right", padx=4)
        self.calibration_btn = ttk.Button(top, text="自动标定识别(3秒)", command=self._auto_calibrate)
        self.calibration_btn.pack(side="right", padx=4)
        ttk.Label(top, textvariable=self.calibration_var).pack(side="right", padx=4)

        self.buttons = ControlButtons(self._content)
        self.buttons.pack(fill="x", padx=10, pady=(2, 6))
        self.buttons.bind_actions(
            on_init=self._on_init,
            on_start=self._on_start,
            on_pause=self._on_pause,
            on_resume=self._on_resume,
            on_stop=self._on_stop,
        )

        main = ttk.Panedwindow(self._content, orient="horizontal")
        main.pack(fill="both", expand=True, padx=10, pady=6)

        video_frame = ttk.LabelFrame(main, text="工业相机实时视频画面")
        main.add(video_frame, weight=5)
        video_frame.configure(width=980, height=700)
        ttk.Label(
            video_frame,
            text="实时画面已移至独立窗口，避免监控页状态布局刷新影响画面。",
            anchor="center",
        ).pack(fill="both", expand=True, padx=6, pady=6)
        ttk.Button(video_frame, text="打开/置顶独立视频窗口", command=self._open_video_window).pack(pady=(0, 12))

        meta = ttk.Frame(video_frame)
        meta.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(meta, text="视频模式:").grid(row=0, column=0, sticky="w", padx=3, pady=1)
        ttk.Label(meta, textvariable=self.video_mode_var).grid(row=0, column=1, sticky="w", padx=3, pady=1)
        ttk.Label(meta, text="视频来源:").grid(row=0, column=2, sticky="w", padx=3, pady=1)
        ttk.Label(meta, textvariable=self.video_source_var).grid(row=0, column=3, sticky="w", padx=3, pady=1)
        ttk.Label(meta, text="分辨率:").grid(row=0, column=4, sticky="w", padx=3, pady=1)
        ttk.Label(meta, textvariable=self.video_res_var).grid(row=0, column=5, sticky="w", padx=3, pady=1)
        meta.columnconfigure(3, weight=1)

        self._main_pane = main
        self._sidebar = ttk.Frame(main)
        main.add(self._sidebar, weight=1)
        self.recognition_panel = RecognitionPanel(self._sidebar)
        self.recognition_panel.pack(fill="x", expand=False, pady=(0, 6))
        self.pump_panel = PumpPanel(self._sidebar)
        self.pump_panel.pack(fill="x", expand=False, pady=(6, 0))

        self.status_panel = StatusPanel(self._sidebar)
        self.status_panel.pack(fill="x", expand=False, pady=(6, 0))

        ctrl_frame = ttk.LabelFrame(self._sidebar, text="PID 控制结果")
        ctrl_frame.pack(fill="x", expand=False, pady=(6, 0))
        rows = [
            ("PID mode", self.pid_mode_var),
            ("kp / ki / kd", self.pid_gains_var),
            ("adaptive active", self.adaptive_var),
            ("adaptive reason", self.adaptive_reason_var),
            ("Q1 / Q2 output gain", self.output_gain_var),
            ("feedforward active", self.feedforward_var),
            ("PID output", self.pid_output_var),
            ("feedforward output", self.feedforward_output_var),
            ("final output", self.final_output_var),
            ("model state", self.model_state_var),
            ("model confidence", self.model_confidence_var),
            ("predicted diameter change", self.predicted_change_var),
            ("model version", self.model_version_var),
            ("recorded samples", self.model_sample_count_var),            ("直径误差", self.err_var),
            ("PID 调节量", self.adjust_var),
            ("反馈冻结", self.freeze_var),
            ("建议停机", self.stop_var),
            ("Q1 指令", self.q1_cmd_var),
            ("Q1 设备参数换算值（非物理流量）", self.q1_actual_var),
            ("CH1执行状态", self.ch1_exec_var),
            ("Q2 指令", self.q2_cmd_var),
            ("Q2 设备参数换算值（非物理流量）", self.q2_actual_var),
            ("CH2执行状态", self.ch2_exec_var),
            ("原因", self.reason_var),
        ]
        for i, (name, var) in enumerate(rows):
            ttk.Label(ctrl_frame, text=f"{name}:").grid(row=i, column=0, padx=4, pady=2, sticky="w")
            ttk.Label(ctrl_frame, textvariable=var, wraplength=360).grid(row=i, column=1, padx=4, pady=2, sticky="w")
        ctrl_frame.columnconfigure(1, weight=1)

    def _toggle_sidebar(self) -> None:
        if self._sidebar_visible:
            self._main_pane.forget(self._sidebar)
            self._sidebar_visible = False
            self.sidebar_btn.configure(text="显示状态侧栏")
        else:
            self._main_pane.add(self._sidebar, weight=1)
            self._sidebar_visible = True
            self.sidebar_btn.configure(text="隐藏状态侧栏")

    def _auto_calibrate(self) -> None:
        result: dict[str, object] = {}
        self.calibration_btn.configure(state="disabled")
        self.calibration_var.set("正在采集实时液滴样本…")

        def task() -> None:
            result.update(self.app.orchestrator.auto_calibrate_detection(3.0))

        def done() -> None:
            self.calibration_btn.configure(state="normal")
            self.calibration_var.set(
                f"标定完成：{int(result.get('sample_count', 0))} 样本，"
                f"直径 {float(result.get('preferred_diameter_px', 0.0)):.1f}px，"
                f"{result.get('polarity', '--')}，噪声 {float(result.get('noise_sigma', 0.0)):.1f}"
            )

        def failed(exc: Exception) -> None:
            self.calibration_btn.configure(state="normal")
            self.calibration_var.set("标定失败")
            messagebox.showwarning("自动标定失败", str(exc))

        self.app.run_backend_task(task, on_success=done, on_error=failed)

    def _on_content_configure(self, _event=None) -> None:
        if self._canvas is None or self._scrollregion_pending:
            return
        self._scrollregion_pending = True
        self.after_idle(self._update_scrollregion)

    def _update_scrollregion(self) -> None:
        self._scrollregion_pending = False
        if self._canvas is not None:
            self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        if self._canvas is not None and self._content_window_id is not None:
            self._canvas.itemconfigure(self._content_window_id, width=event.width)

    def _bind_mousewheel(self) -> None:
        if self._canvas is None:
            return

        def _on_wheel(event):
            delta = 0
            if hasattr(event, "delta") and event.delta:
                delta = int(-event.delta / 120)
            elif getattr(event, "num", None) == 5:
                delta = 1
            elif getattr(event, "num", None) == 4:
                delta = -1
            if delta != 0 and self._canvas is not None:
                self._canvas.yview_scroll(delta, "units")

        self._canvas.bind_all("<MouseWheel>", _on_wheel)
        self._canvas.bind_all("<Button-4>", _on_wheel)
        self._canvas.bind_all("<Button-5>", _on_wheel)

    def on_show(self) -> None:
        self._start_poll()

    def on_hide(self) -> None:
        self._stop_poll()

    def _open_video_window(self) -> None:
        if self._video_process is not None and self._video_process.is_alive():
            return
        if self._video_process is not None:
            self._video_process.stop()
        self._video_process = VideoProcessController(self.app.orchestrator.get_video_frame_snapshot)
        self._video_process.start()

    def _close_video_window(self) -> None:
        controller = self._video_process
        self._video_process = None
        if controller is not None:
            controller.stop()

    def _start_poll(self) -> None:
        self._stop_poll()
        self._status_stop_event = threading.Event()
        self._status_worker = threading.Thread(
            target=self._status_snapshot_loop,
            args=(self._status_stop_event,),
            name="monitor-status-loop",
            daemon=True,
        )
        self._status_worker.start()
        self._open_video_window()
        self._poll_status_once()

    def _stop_poll(self) -> None:
        self._close_video_window()
        if self._status_poll_job is not None:
            self.after_cancel(self._status_poll_job)
            self._status_poll_job = None
        self._status_stop_event.set()
        self._status_worker = None

    def _status_snapshot_loop(self, stop_event: threading.Event) -> None:
        last_signature = None
        while not stop_event.is_set():
            try:
                snapshot = self.app.orchestrator.get_snapshot()
                control = getattr(snapshot, "control", None)
                signature = (
                    float(getattr(control, "timestamp", 0.0) or 0.0),
                    str(getattr(snapshot, "system_state", "")),
                    str(getattr(snapshot, "message", "") or ""),
                    str(getattr(snapshot, "error", "") or ""),
                )
                if signature != last_signature:
                    with self._status_snapshot_lock:
                        self._pending_status_snapshot = snapshot
                    last_signature = signature
            except Exception:
                pass
            # Lightweight background observation; publish only when a new PID
            # control-period result or system-state change appears.
            stop_event.wait(0.05)

    @staticmethod
    def _parse_channel_status(reason: str, channel: int) -> str:
        if not reason:
            return "--"
        text = reason.upper()
        tag = f"CH{channel}"
        if tag not in text:
            return "--"
        fail_keys = ("失败", "FAIL", "ERROR", "异常")
        if any(k in text for k in fail_keys):
            return "失败"
        return "已执行"

    @staticmethod
    def _channel_flow(snapshot, name: str) -> str:
        state = snapshot.pump_state if snapshot is not None else None
        if state is None or not state.channels:
            return "--"
        channel = state.channels.get(name)
        if channel is None or channel.actual_flow_rate is None:
            return "--"
        return f"{float(channel.actual_flow_rate):.6f}"

    def _poll_status_once(self) -> None:
        with self._status_snapshot_lock:
            snap = self._pending_status_snapshot
            self._pending_status_snapshot = None
        if snap is None:
            self._status_poll_job = self.after(self._status_refresh_interval_ms, self._poll_status_once)
            return
        self.recognition_panel.update_snapshot(snap)
        self.pump_panel.update_snapshot(snap)
        self.status_panel.update_snapshot(snap)

        set_var_if_changed(self.q1_actual_var, self._channel_flow(snap, "Q1"))
        set_var_if_changed(self.q2_actual_var, self._channel_flow(snap, "Q2"))

        rec = snap.recognition
        if rec is not None:
            set_var_if_changed(self.video_mode_var, rec.video_source_type or "--")
            set_var_if_changed(self.video_source_var, rec.video_source or "--")
            if rec.frame_width > 0 and rec.frame_height > 0:
                set_var_if_changed(self.video_res_var, f"{rec.frame_width} x {rec.frame_height}")
            else:
                set_var_if_changed(self.video_res_var, "--")

        ctrl = snap.control
        if ctrl is not None:
            set_var_if_changed(self.pid_mode_var, str(getattr(ctrl, "control_mode", "--") or "--"))
            set_var_if_changed(
                self.pid_gains_var,
                f"{float(getattr(ctrl, 'kp', 0.0)):.6f} / "
                f"{float(getattr(ctrl, 'ki', 0.0)):.6f} / "
                f"{float(getattr(ctrl, 'kd', 0.0)):.6f}",
            )
            adaptive_enabled = bool(getattr(ctrl, "adaptive_enabled", False))
            adaptive_active = bool(getattr(ctrl, "adaptive_active", False))
            adaptive_text = "enabled / tuning" if adaptive_active else "enabled / warming up" if adaptive_enabled else "disabled"
            set_var_if_changed(self.adaptive_var, adaptive_text)
            set_var_if_changed(self.adaptive_reason_var, str(getattr(ctrl, "adaptive_reason", "") or "--"))
            set_var_if_changed(
                self.output_gain_var,
                f"{float(getattr(ctrl, 'q1_output_gain', 1.0)):.2f} / "
                f"{float(getattr(ctrl, 'q2_output_gain', 1.0)):.2f}",
            )
            set_var_if_changed(self.feedforward_var, "yes" if bool(getattr(ctrl, "feedforward_active", False)) else "no")
            set_var_if_changed(self.err_var, f"{ctrl.diameter_error:.6f}")
            set_var_if_changed(self.adjust_var, f"{ctrl.adjustment:.6f}")
            set_var_if_changed(self.pid_output_var, f"{float(getattr(ctrl, 'pid_output', 0.0)):.6f}")
            set_var_if_changed(
                self.feedforward_output_var, f"{float(getattr(ctrl, 'feedforward_output', 0.0)):.6f}"
            )
            set_var_if_changed(self.final_output_var, f"{float(getattr(ctrl, 'final_output', ctrl.adjustment)):.6f}")
            set_var_if_changed(self.freeze_var, "是" if ctrl.freeze_feedback else "否")
            set_var_if_changed(self.stop_var, "是" if ctrl.suggested_stop else "否")
            set_var_if_changed(self.q1_cmd_var, f"{ctrl.q1_command:.6f}")
            set_var_if_changed(self.q2_cmd_var, f"{ctrl.q2_command:.6f}")
            set_var_if_changed(self.reason_var, ctrl.reason or "--")
            set_var_if_changed(self.ch1_exec_var, self._parse_channel_status(ctrl.reason or "", 1))
            set_var_if_changed(self.ch2_exec_var, self._parse_channel_status(ctrl.reason or "", 2))

        model_status = getattr(snap, "disturbance_model", None) or {}
        prediction = getattr(snap, "disturbance_prediction", None) or {}
        if model_status:
            set_var_if_changed(self.model_state_var, str(model_status.get("state", "--")))
            set_var_if_changed(self.model_confidence_var, f"{float(model_status.get('confidence', 0.0) or 0.0):.3f}")
            set_var_if_changed(self.model_version_var, str(model_status.get("model_version", "") or "--"))
            set_var_if_changed(self.model_sample_count_var, str(model_status.get("sample_count", "--")))
        if prediction:
            value = prediction.get("predicted_diameter_change_um")
            set_var_if_changed(self.predicted_change_var, "--" if value is None else f"{float(value):.3f} um")
        state_val = snap.system_state.value if hasattr(snap.system_state, "value") else str(snap.system_state)
        if state_val != self._last_button_state:
            self._last_button_state = state_val
            try:
                self.buttons.update_by_state(SystemState(state_val))
            except Exception:
                pass
        if self._start_pending:
            self.buttons.start_btn.configure(state="disabled")
        self._status_poll_job = self.after(self._status_refresh_interval_ms, self._poll_status_once)

    def _on_init(self) -> None:
        self.app.show_page("init")

    def _on_start(self) -> None:
        if self._start_pending:
            return
        self._start_pending = True
        self.buttons.start_btn.configure(state="disabled", text="启动中…")

        def finished() -> None:
            self._start_pending = False
            self.buttons.start_btn.configure(text="开始")
            self._last_button_state = None

        def failed(exc: Exception) -> None:
            finished()
            messagebox.showerror("启动失败", str(exc))

        self.app.run_backend_task(
            self.app.orchestrator.start,
            on_success=finished,
            on_error=failed,
        )

    def _on_pause(self) -> None:
        self.app.run_backend_task(
            self.app.orchestrator.pause,
            on_error=lambda e: messagebox.showerror("暂停失败", str(e)),
        )

    def _on_resume(self) -> None:
        self.app.run_backend_task(
            self.app.orchestrator.resume,
            on_error=lambda e: messagebox.showerror("继续失败", str(e)),
        )

    def _on_stop(self) -> None:
        self.app.run_backend_task(
            self.app.orchestrator.stop,
            on_error=lambda e: messagebox.showerror("停止失败", str(e)),
        )
