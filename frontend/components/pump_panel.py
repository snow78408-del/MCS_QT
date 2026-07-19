from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk

try:
    from backend.orchestrator.models import PumpChannelState, PumpRuntimeState, SystemSnapshot
except Exception:  # pragma: no cover
    from ...backend.orchestrator.models import PumpChannelState, PumpRuntimeState, SystemSnapshot

from .ui_update import set_var_if_changed


class PumpPanel(ttk.LabelFrame):
    PUMP_STATE_TIMEOUT_S = 3.0

    def __init__(self, parent):
        super().__init__(parent, text="Q1/Q2/Q3 注射泵状态")
        self.summary_var = tk.StringVar(value="--")
        self._channel_vars: dict[str, dict[str, tk.StringVar]] = {}
        self._build()

    def _build(self) -> None:
        ttk.Label(self, textvariable=self.summary_var, wraplength=520).pack(anchor="w", padx=6, pady=(6, 4))
        grid = ttk.Frame(self)
        grid.pack(fill="x", padx=4, pady=(0, 6))
        for col, name in enumerate(("Q1", "Q2", "Q3")):
            frame = ttk.LabelFrame(grid, text=name)
            frame.grid(row=0, column=col, sticky="nsew", padx=4, pady=4)
            grid.columnconfigure(col, weight=1)
            self._channel_vars[name] = self._build_channel(frame)

    def _build_channel(self, parent: ttk.LabelFrame) -> dict[str, tk.StringVar]:
        vars_map = {
            "physical_channel": tk.StringVar(value="--"),
            "enabled": tk.StringVar(value="--"),
            "running": tk.StringVar(value="--"),
            "actual_flow_rate": tk.StringVar(value="--"),
            "target_flow_rate": tk.StringVar(value="--"),
            "unit": tk.StringVar(value="--"),
            "communication": tk.StringVar(value="--"),
            "last_readback": tk.StringVar(value="--"),
            "error": tk.StringVar(value="--"),
        }
        rows = [
            ("物理通道", "physical_channel"),
            ("启用状态", "enabled"),
            ("运行状态", "running"),
            ("回读流速", "actual_flow_rate"),
            ("目标流速", "target_flow_rate"),
            ("流速单位", "unit"),
            ("通信状态", "communication"),
            ("最近回读", "last_readback"),
            ("异常信息", "error"),
        ]
        for i, (label, key) in enumerate(rows):
            ttk.Label(parent, text=f"{label}:").grid(row=i, column=0, sticky="w", padx=5, pady=2)
            ttk.Label(parent, textvariable=vars_map[key], wraplength=170).grid(
                row=i, column=1, sticky="w", padx=5, pady=2
            )
        parent.columnconfigure(1, weight=1)
        return vars_map

    @staticmethod
    def _format_flow(value: float | None) -> str:
        if value is None:
            return "--"
        return f"{float(value):.2f}"

    @staticmethod
    def _default_channel(name: str) -> PumpChannelState:
        return PumpChannelState(
            logical_name=name,
            physical_channel="未配置",
            enabled=False,
            running=False,
            communication_ok=False,
            target_flow_rate=None,
            actual_flow_rate=None,
            error="未配置",
        )

    def _is_timeout(self, channel: PumpChannelState) -> bool:
        if channel.last_readback_time is None:
            return False
        return (time.time() - float(channel.last_readback_time)) > self.PUMP_STATE_TIMEOUT_S

    def _update_channel(self, name: str, channel: PumpChannelState) -> None:
        vars_map = self._channel_vars[name]
        timeout = self._is_timeout(channel)
        physical_channel = channel.physical_channel or "未配置"
        error = channel.error or ""
        unconfigured = physical_channel in {"未配置", "unconfigured"} or error in {"未配置", "unconfigured"}
        communication_ok = bool(channel.communication_ok) and not timeout and not unconfigured

        set_var_if_changed(vars_map["physical_channel"], "未配置" if physical_channel == "unconfigured" else physical_channel)
        set_var_if_changed(vars_map["enabled"], "已启用" if channel.enabled else "未启用")
        set_var_if_changed(vars_map["running"], "灌注中" if channel.running else "未运行")
        set_var_if_changed(vars_map["actual_flow_rate"], self._format_flow(channel.actual_flow_rate if communication_ok else None))
        set_var_if_changed(vars_map["target_flow_rate"], self._format_flow(channel.target_flow_rate))
        set_var_if_changed(vars_map["unit"], channel.flow_rate_unit or "uL/min")
        if unconfigured:
            communication_text = "未配置"
        elif timeout:
            communication_text = "状态回读超时"
        elif channel.communication_ok:
            communication_text = "正常"
        else:
            communication_text = "异常"
        set_var_if_changed(vars_map["communication"], communication_text)
        set_var_if_changed(
            vars_map["last_readback"],
            "--" if channel.last_readback_time is None else f"{float(channel.last_readback_time):.3f}",
        )
        if error == "unconfigured":
            error = "未配置"
        set_var_if_changed(vars_map["error"], error or "--")

    def update_pump_state(self, state: PumpRuntimeState | None) -> None:
        if state is None:
            set_var_if_changed(self.summary_var, "暂无泵状态快照")
            for name in ("Q1", "Q2", "Q3"):
                self._update_channel(name, self._default_channel(name))
            return

        summary = [
            "串口已连接" if state.connected else "串口未连接",
            "通信正常" if state.comm_established else "通信未建立",
            "设备就绪" if state.fully_ready else "设备未就绪",
            "系统灌注中" if state.running else "系统未灌注",
        ]
        if state.last_update_reason:
            summary.append(f"最近下发: {'成功' if state.last_update_ok else '失败'} {state.last_update_reason}")
        if state.last_error:
            summary.append(f"异常: {state.last_error}")
        set_var_if_changed(self.summary_var, "；".join(summary))

        channels = state.channels or {}
        for name in ("Q1", "Q2", "Q3"):
            self._update_channel(name, channels.get(name) or self._default_channel(name))

    def update_snapshot(self, snapshot: SystemSnapshot | None) -> None:
        if snapshot is None:
            self.update_pump_state(None)
            return
        self.update_pump_state(snapshot.pump_state)
