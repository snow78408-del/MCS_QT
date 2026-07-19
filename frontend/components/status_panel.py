from __future__ import annotations

import tkinter as tk
from tkinter import ttk

try:
    from backend.orchestrator.models import SystemSnapshot
except Exception:  # pragma: no cover
    from ...backend.orchestrator.models import SystemSnapshot

from .ui_update import set_var_if_changed


class StatusPanel(ttk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text="系统状态")
        self.system_state_var = tk.StringVar(value="--")
        self.stage_var = tk.StringVar(value="--")
        self.message_var = tk.StringVar(value="--")
        self.error_var = tk.StringVar(value="--")
        self._build()

    def _build(self) -> None:
        rows = [
            ("状态", self.system_state_var),
            ("阶段", self.stage_var),
            ("提示", self.message_var),
            ("错误", self.error_var),
        ]
        for i, (name, var) in enumerate(rows):
            ttk.Label(self, text=f"{name}:").grid(row=i, column=0, sticky="w", padx=4, pady=2)
            ttk.Label(self, textvariable=var, wraplength=260).grid(row=i, column=1, sticky="w", padx=4, pady=2)
        self.columnconfigure(1, weight=1)

    def update_snapshot(self, snapshot: SystemSnapshot | None) -> None:
        if snapshot is None:
            return
        value = getattr(snapshot.system_state, "value", str(snapshot.system_state))
        friendly = _friendly_state(value)
        set_var_if_changed(self.system_state_var, friendly)
        set_var_if_changed(self.stage_var, friendly)
        set_var_if_changed(self.message_var, _clean_status_text(snapshot.message, "message"))
        set_var_if_changed(self.error_var, _clean_status_text(snapshot.error, "error"))


_STATE_LABELS = {
    "idle": "空闲",
    "configured": "参数已配置",
    "video_ready": "视频已就绪",
    "initializing": "初始化中",
    "initialized": "初始化完成",
    "running": "运行中",
    "paused": "已暂停",
    "stopping": "停止中",
    "stopped": "已停止",
    "error": "错误",
}

_MESSAGE_LABELS = {
    "": "--",
    "configured": "参数已配置",
    "video ready": "视频已就绪",
    "initializing": "正在初始化",
    "initialized": "初始化完成",
    "running": "系统运行中",
    "paused": "系统已暂停",
    "stopping": "正在停止",
    "stopped": "系统已停止",
    "local video mode: skip pump initialization and PID output": "本地视频模式：跳过泵初始化和 PID 输出",
}

_MOJIBAKE_MARKERS = set("闂閻濞缂婵濠鐎柛梺妞鈧瑜閸閹幋娴瀹绾椤")


def _friendly_state(value: object) -> str:
    text = str(value or "").strip()
    return _STATE_LABELS.get(text.lower(), text or "--")


def _clean_status_text(value: object, kind: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "--"
    mapped = _MESSAGE_LABELS.get(text.lower())
    if mapped is not None:
        return mapped
    if _looks_like_mojibake(text):
        return "发生错误，请查看运行日志" if kind == "error" else "状态信息异常，请查看运行日志"
    if len(text) > 260:
        return text[:240] + "..."
    return text


def _looks_like_mojibake(text: str) -> bool:
    if not text:
        return False
    marker_count = sum(1 for ch in text if ch in _MOJIBAKE_MARKERS)
    if marker_count >= 4:
        return True
    return marker_count > 0 and marker_count / max(1, len(text)) > 0.08
