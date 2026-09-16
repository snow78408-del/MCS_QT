from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ResponseDecision:
    reason: str = ""
    stop: bool = False
    recovering: bool = False


class ResponseGuard:
    """Bounded observation history; only verified flow changes count as attempts.

    This detects unsuitable feedback conditions, not bubbles or a zero plant gain.
    Tolerances are policy settings, not constants identified from a failed fit.
    """

    def __init__(self) -> None:
        self.history: deque[tuple[float, float]] = deque(maxlen=3)
        self.last_end = -math.inf
        self.invalid_since: float | None = None
        self.recovery_after = -math.inf
        self.recovery_pending = False
        self.pending_diameter: float | None = None
        self.pending_evaluated = False
        self.unresponsive_commands = 0
        self.cumulative_change = 0.0

    def invalidate(self, now: float) -> None:
        if self.invalid_since is None:
            self.invalid_since = now
        self.recovery_after = max(self.recovery_after, now)
        self.history.clear()
        self.recovery_pending = True

    def timed_out(self, now: float, timeout_s: float) -> bool:
        return self.invalid_since is not None and now - self.invalid_since >= timeout_s

    def observe(
        self, *, now: float, start: float | None, end: float | None,
        diameter: float, std: float | None, tolerance: float,
        recovery_s: float, timeout_s: float, error: float, deadband: float,
        max_attempts: int,
    ) -> ResponseDecision:
        if self.timed_out(now, timeout_s):
            return ResponseDecision("工况长时间未恢复稳定，安全停止；原因待确认", stop=True)
        if (start is None or end is None or not math.isfinite(start)
                or not math.isfinite(end) or end <= start or end > now
                or not math.isfinite(diameter) or diameter <= 0
                or (std is not None and (not math.isfinite(std) or std < 0))):
            self.invalidate(now)
            return ResponseDecision("测量无效，暂停调整并等待全新稳定窗口")
        if end <= self.last_end:
            return ResponseDecision("等待新的工况观测窗口")
        self.last_end = end
        if start < self.recovery_after:
            return ResponseDecision("观测窗口包含不稳定阶段，等待全新样本")
        # A broad within-window distribution is distinct from a settled step.
        if std is not None and 2.0 * std > tolerance:
            self.invalidate(now)
            return ResponseDecision("工况不稳定：尺寸离散较大，暂停调整；原因待确认")
        self.history.append((end, diameter))
        if len(self.history) == 3:
            values = [value for _, value in self.history]
            if max(values) - min(values) > tolerance:
                self.invalidate(now)
                return ResponseDecision("工况不稳定：窗口间尺寸漂移，暂停调整；原因待确认")
        if self.recovery_pending:
            if len(self.history) < 3 or self.history[-1][0] - self.history[0][0] < recovery_s:
                return ResponseDecision("工况恢复观察中，等待连续三个稳定窗口")
            self.invalid_since = None
            # Remains pending until a valid control decision is committed.
        if self.pending_diameter is not None:
            if abs(diameter - self.pending_diameter) >= tolerance:
                self.pending_diameter = None
                self.unresponsive_commands = 0
                self.cumulative_change = 0.0
            elif not self.pending_evaluated:
                self.unresponsive_commands += 1
                self.pending_evaluated = True
        if abs(error) <= deadband:
            self.pending_diameter = None
            self.unresponsive_commands = 0
            self.cumulative_change = 0.0
            return ResponseDecision("尺寸在目标容差内，保持当前流量")
        if self.unresponsive_commands >= max_attempts:
            return ResponseDecision("稳定但持续未检测到响应，已暂停自动追调；请检查工况")
        return ResponseDecision(recovering=self.recovery_pending)

    def record_command(self, diameter: float, flow_change: float) -> None:
        if flow_change <= 1e-9:
            return
        self.pending_diameter = diameter
        self.pending_evaluated = False
        if self.unresponsive_commands:
            self.cumulative_change += flow_change
        # Don't interpret a deliberate step between settled windows as drift.
        self.history.clear()
        self.recovery_pending = False
