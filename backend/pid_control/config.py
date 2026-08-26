from __future__ import annotations

from dataclasses import dataclass, fields
import math
try:
    from enum import StrEnum
except ImportError:  # Python 3.10 compatibility
    from enum import Enum

    class StrEnum(str, Enum):
        pass


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
    # With a 10 s control period, 12 samples / every 5 updates made the
    # adaptive path appear inactive for roughly 2--3 minutes. Three valid
    # periods provide a short warm-up while keeping adaptation responsive.
    adaptive_update_interval: int = 1
    adaptive_min_sample_count: int = 3
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
    integral_decay_in_deadband: float = 0.8
    min_droplet_count_for_feedback: int = 1

    # The TS pump requires a positive, representable flow. With the preserved
    # 1000 uL / 0.1 min parameter profile, 0.2 uL/min remains exactly
    # representable within the 16-bit infusion-time field.
    q1_min: float = 0.2
    q1_max: float = 5000.0
    q2_min: float = 0.2
    q2_max: float = 5000.0
    # Oil phase Q1 must remain strictly faster than aqueous phase Q2. A
    # positive gap makes the strict inequality robust to pump quantization.
    min_q1_q2_gap: float = 0.2
    max_flow_change_per_cycle: float = 200.0
    total_flow_max: float = 8000.0
    use_initial_flow_as_output_bias: bool = True
    # Identified plant direction: flow_delta = sign * PID output. Defaults
    # preserve the current Q1/Q2 differential-control convention.
    q1_control_sign: float = -1.0
    q2_control_sign: float = 1.0
    # Non-symmetric actuator allocation. Q1 (oil) receives twice the flow
    # change of Q2 (water) for the same PID output.
    q1_output_gain: float = 2.0
    q2_output_gain: float = 1.0

    # Backward-compatible aliases used by older call sites.
    kp: float | None = None
    ki: float | None = None
    kd: float | None = None
    adjustment_min: float | None = None
    adjustment_max: float | None = None

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(float(value)):
                raise ValueError(f"{item.name} must be finite")
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
        if self.q1_control_sign == 0.0 or self.q2_control_sign == 0.0:
            raise ValueError("pump control signs must be non-zero")
        if self.q1_output_gain <= 0.0 or self.q2_output_gain <= 0.0:
            raise ValueError("pump output gains must be greater than zero")
        if not self.min_q1_q2_gap > 0.0:
            raise ValueError("min_q1_q2_gap must be greater than zero")
        if not 0.0 < self.q1_min <= self.q1_max or not 0.0 < self.q2_min <= self.q2_max:
            raise ValueError("pump min/max limits must be positive and ordered")
        self.q1_control_sign = 1.0 if self.q1_control_sign > 0.0 else -1.0
        self.q2_control_sign = 1.0 if self.q2_control_sign > 0.0 else -1.0
