from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PIDControlMode(StrEnum):
    CLASSIC_PID = "CLASSIC_PID"
    ADAPTIVE_PID = "ADAPTIVE_PID"
    ADAPTIVE_PID_WITH_FEEDFORWARD = "ADAPTIVE_PID_WITH_FEEDFORWARD"


@dataclass(slots=True)
class PIDConfig:
    control_mode: str = PIDControlMode.ADAPTIVE_PID_WITH_FEEDFORWARD.value

    base_kp: float = 0.08
    base_ki: float = 0.01
    base_kd: float = 0.0

    kp_min: float = 0.0
    kp_max: float = 2.0
    ki_min: float = 0.0
    ki_max: float = 1.0
    kd_min: float = 0.0
    kd_max: float = 1.0
    kp_step_limit: float = 0.02
    ki_step_limit: float = 0.005
    kd_step_limit: float = 0.005
    adaptive_update_interval: int = 5
    adaptive_min_sample_count: int = 12
    adaptive_confidence_threshold: float = 0.55

    feedforward_enabled: bool = True
    feedforward_min: float = -150.0
    feedforward_max: float = 150.0
    feedforward_rate_limit: float = 50.0
    feedforward_confidence_threshold: float = 0.65
    feedforward_timeout_ms: int = 2000
    feedforward_gain: float = 0.5

    output_min: float = -500.0
    output_max: float = 500.0
    output_rate_limit: float = 200.0
    integral_limit: float = 10000.0
    diameter_deadband: float = 1.0
    min_droplet_count_for_feedback: int = 1

    q1_min: float = 0.0
    q1_max: float = 5000.0
    q2_min: float = 0.0
    q2_max: float = 5000.0

    # Backward-compatible aliases used by older call sites.
    kp: float | None = None
    ki: float | None = None
    kd: float | None = None
    adjustment_min: float | None = None
    adjustment_max: float | None = None

    def __post_init__(self) -> None:
        if self.kp is not None:
            self.base_kp = float(self.kp)
        if self.ki is not None:
            self.base_ki = float(self.ki)
        if self.kd is not None:
            self.base_kd = float(self.kd)
        if self.adjustment_min is not None:
            self.output_min = float(self.adjustment_min)
        if self.adjustment_max is not None:
            self.output_max = float(self.adjustment_max)
        self.kp = self.base_kp
        self.ki = self.base_ki
        self.kd = self.base_kd
        self.adjustment_min = self.output_min
        self.adjustment_max = self.output_max
