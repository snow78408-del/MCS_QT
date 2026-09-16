from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class FeedbackHold:
    """Host-monotonic eligibility boundary; never blocks stop/fault handling."""

    command_id: int = 0
    completed_at: float = 0.0
    ready_after: float = 0.0
    wait_source: str = ""

    def record(self, completed_at: float, wait_s: float, source: str, *, command_id: int | None = None) -> None:
        if not math.isfinite(completed_at) or not math.isfinite(wait_s) or wait_s < 0:
            raise ValueError("feedback wait must have finite time and nonnegative duration")
        self.command_id = self.command_id + 1 if command_id is None else command_id
        self.completed_at = completed_at
        self.ready_after = completed_at + wait_s
        self.wait_source = source

    def rejection_reason(self, now: float, start: float | None, end: float | None) -> str:
        if not self.command_id:
            return ""
        if not math.isfinite(now) or now < self.ready_after:
            return f"等待上次事务后的响应期结束（命令 {self.command_id}）"
        if start is None or end is None or not all(math.isfinite(t) for t in (start, end)):
            return "等待带采集时间边界的调整后尺寸窗口"
        if end <= start or end > now:
            return "尺寸窗口尚未完成或时间边界无效"
        if start < self.ready_after:
            return "尺寸窗口包含调整前或等待期样本，保留观察但不用于反馈"
        return ""
