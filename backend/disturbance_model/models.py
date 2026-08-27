from __future__ import annotations

from dataclasses import asdict, dataclass, field
try:
    from enum import StrEnum
except ImportError:  # Python 3.10 compatibility
    from enum import Enum

    class StrEnum(str, Enum):
        pass
from typing import Any


class DisturbanceStage(StrEnum):
    BASELINE = "baseline"
    DISTURBED = "disturbed"
    RECOVERY = "recovery"


class ModelState(StrEnum):
    COLLECTING = "COLLECTING"
    TRAINING = "TRAINING"
    VALIDATING = "VALIDATING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"


class DisturbanceControlStage(StrEnum):
    COLLECT_ONLY = "COLLECT_ONLY"
    OFFLINE_TRAINING = "OFFLINE_TRAINING"
    SHADOW = "SHADOW"
    LOW_WEIGHT_FEEDFORWARD = "LOW_WEIGHT_FEEDFORWARD"
    FULL_FEEDFORWARD = "FULL_FEEDFORWARD"


@dataclass(slots=True)
class DisturbanceSample:
    timestamp: float
    experiment_id: str = ""
    chip_id: str = ""
    disturbance_name: str = ""
    disturbance_stage: str = DisturbanceStage.BASELINE.value
    disturbance_amplitude: float = 0.0
    run_state: str = ""
    video_source_type: str = ""
    q1_set: float = 0.0
    q2_set: float = 0.0
    q1_feedback: float = 0.0
    q2_feedback: float = 0.0
    q1_error: float = 0.0
    q2_error: float = 0.0
    pump_response_delay_ms: float = 0.0
    pump_comm_status: bool = False
    droplet_mean_diameter_um: float | None = None
    droplet_std_um: float | None = None
    droplet_cv: float | None = None
    droplet_frequency_hz: float = 0.0
    droplet_count_frame: int = 0
    droplet_count_total: int = 0
    valid_sample_count: int = 0
    single_cell_rate: float | None = None
    vision_valid: bool = False
    vision_invalid_reason: str = ""
    vision_latency_ms: float = 0.0
    measurement_noise_est: float = 0.0
    image_brightness_mean: float = 0.0
    focus_score: float = 0.0
    target_diameter_um: float = 0.0
    diameter_error_um: float | None = None
    pid_output: float = 0.0
    feedback_frozen: bool = False
    freeze_reason: str = ""
    control_cycle_ms: float = 0.0
    control_jitter_ms: float = 0.0
    temperature_c: float | None = None

    def __post_init__(self) -> None:
        if self.disturbance_stage not in {item.value for item in DisturbanceStage}:
            self.disturbance_stage = DisturbanceStage.BASELINE.value
        self.q1_error = float(self.q1_feedback) - float(self.q1_set)
        self.q2_error = float(self.q2_feedback) - float(self.q2_set)
        if self.droplet_mean_diameter_um is not None and self.target_diameter_um:
            self.diameter_error_um = float(self.target_diameter_um) - float(self.droplet_mean_diameter_um)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DisturbancePrediction:
    timestamp: float
    model_ready: bool = False
    model_valid: bool = False
    confidence: float = 0.0
    predicted_diameter_um: float | None = None
    predicted_diameter_change_um: float = 0.0
    predicted_response_delay_ms: float = 0.0
    predicted_cv: float | None = None
    disturbance_effect: str = "unknown"
    # None means that the predictive model has not supplied a physically
    # calibrated inverse-model command. PID may derive one only when an
    # explicit plant calibration is configured.
    recommended_feedforward: float | None = None
    feedforward_weight: float = 0.0
    control_stage: str = DisturbanceControlStage.COLLECT_ONLY.value
    shadow_error_um: float | None = None
    safety_fallback: bool = False
    model_version: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelMetrics:
    mae: float = 0.0
    rmse: float = 0.0
    r2: float = 0.0
    direction_accuracy: float = 0.0
    response_delay_error_ms: float = 0.0
    persistence_rmse: float = 0.0
    persistence_improvement: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelStatus:
    state: str = ModelState.COLLECTING.value
    sample_count: int = 0
    model_version: str = ""
    model_ready: bool = False
    model_valid: bool = False
    confidence: float = 0.0
    control_stage: str = DisturbanceControlStage.COLLECT_ONLY.value
    feedforward_weight: float = 0.0
    shadow_mae_um: float = 0.0
    shadow_change_mae_um: float = 0.0
    shadow_direction_accuracy: float = 0.0
    safety_fallback: bool = False
    last_error: str = ""
    metrics: ModelMetrics = field(default_factory=ModelMetrics)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metrics"] = self.metrics.to_dict()
        return data
