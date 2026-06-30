from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class VisionMetrics:
    avg_diameter: float
    droplet_count: int
    valid_for_control: bool
    frame_id: int = 0
    noise_estimate: float = 0.0


@dataclass(slots=True)
class TargetParams:
    target_diameter: float


@dataclass(slots=True)
class PumpState:
    q1: float
    q2: float
    communication_ok: bool = True


@dataclass(slots=True)
class PIDCommand:
    q1: float
    q2: float
    diameter_error: float
    adjustment: float
    freeze_feedback: bool
    suggested_stop: bool
    reason: str
    p_term: float = 0.0
    i_term: float = 0.0
    d_term: float = 0.0
    pid_output: float = 0.0
    feedforward_output: float = 0.0
    final_output: float = 0.0
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    adaptive_active: bool = False
    feedforward_active: bool = False
    control_mode: str = "CLASSIC_PID"
    frame_id: int = 0


@dataclass(slots=True)
class PIDInput:
    target_diameter_um: float
    current_diameter_um: float | None
    current_q1: float
    current_q2: float
    dt: float
    frame_id: int
    vision_valid: bool
    pump_communication_ok: bool
    droplet_count: int = 0
    disturbance_prediction: Any | None = None
    system_running: bool = True
    measurement_noise_est: float = 0.0
    control_jitter_ms: float = 0.0
    pump_response_delay_ms: float = 0.0


@dataclass(slots=True)
class AdaptivePIDState:
    kp: float
    ki: float
    kd: float
    base_kp: float
    base_ki: float
    base_kd: float
    update_count: int = 0
    sample_count: int = 0
    active: bool = False
    reason: str = ""


@dataclass(slots=True)
class FeedforwardResult:
    u_ff: float
    active: bool
    reason: str
    confidence: float = 0.0
