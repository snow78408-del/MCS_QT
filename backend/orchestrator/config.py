from __future__ import annotations

from dataclasses import dataclass
import math

# Only a fallback for configurations that omit the field. Normal runs use the
# value entered by the user; configure() validates it without rewriting it.
default_control_interval_ms = 7500
min_control_interval_ms = 7500
# Dense-droplet recognition and the physical transport delay can both exceed
# five seconds. The UI already permits longer periods, so do not silently
# clamp a requested 10–30 s control interval back to 5 s.
max_control_interval_ms = 30000
pump_command_retry = 2
init_timeout_s = 8.0
stop_timeout_s = 5.0
max_recognition_age_ms = 1500
pump_update_watchdog_timeout_s = 15.0


@dataclass(slots=True)
class OrchestratorConfig:
    default_control_interval_ms: int = default_control_interval_ms
    min_control_interval_ms: int = min_control_interval_ms
    max_control_interval_ms: int = max_control_interval_ms
    pump_command_retry: int = pump_command_retry
    init_timeout_s: float = init_timeout_s
    stop_timeout_s: float = stop_timeout_s
    max_recognition_age_ms: int = max_recognition_age_ms
    # A verified two-channel update at 1200 baud spans several request/readback
    # exchanges and can legitimately exceed the normal control-loop watchdog.
    # Keep it bounded, but give the in-flight hardware transaction enough time.
    pump_update_watchdog_timeout_s: float = pump_update_watchdog_timeout_s
    # Provisional policy limits, to be reviewed against stable-device recordings.
    response_guard_tolerance_um: float = 4.0
    response_guard_recovery_s: float = 15.0
    response_guard_timeout_s: float = 120.0
    response_guard_max_attempts: int = 3
    response_guard_retry_step: float = 1.0
    response_guard_retry_budget: float = 4.0
    calibration_unstable_timeout_s: float = 180.0

    def __post_init__(self) -> None:
        for name in (
            "response_guard_tolerance_um", "response_guard_recovery_s",
            "response_guard_timeout_s", "response_guard_retry_step",
            "response_guard_retry_budget", "calibration_unstable_timeout_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (isinstance(self.response_guard_max_attempts, bool)
                or not isinstance(self.response_guard_max_attempts, int)
                or self.response_guard_max_attempts < 1):
            raise ValueError("response_guard_max_attempts must be a positive integer")
        if self.response_guard_timeout_s <= self.response_guard_recovery_s:
            raise ValueError("response guard timeout must exceed recovery duration")
