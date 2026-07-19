from __future__ import annotations

from .models import DisturbanceSample, DisturbanceStage


FEATURE_NAMES = [
    "stage_baseline",
    "stage_disturbed",
    "stage_recovery",
    "disturbance_amplitude",
    "q1_set",
    "q2_set",
    "q1_feedback",
    "q2_feedback",
    "q1_error",
    "q2_error",
    "pump_comm_status",
    "droplet_count_frame",
    "valid_sample_count",
    "measurement_noise_est",
    "diameter_error_um",
    "pid_output",
    "feedback_frozen",
    "control_cycle_ms",
    "control_jitter_ms",
]

_FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
_NONLINEAR_INTERACTION_PAIRS = [
    ("disturbance_amplitude", "pid_output"),
    ("disturbance_amplitude", "diameter_error_um"),
    ("q1_set", "q2_set"),
    ("q1_feedback", "q2_feedback"),
    ("q1_error", "q2_error"),
    ("diameter_error_um", "pid_output"),
    ("measurement_noise_est", "droplet_count_frame"),
    ("control_cycle_ms", "control_jitter_ms"),
]

NONLINEAR_FEATURE_NAMES = (
    list(FEATURE_NAMES)
    + [f"{name}^2" for name in FEATURE_NAMES]
    + [f"{left}*{right}" for left, right in _NONLINEAR_INTERACTION_PAIRS]
)


TARGET_NAMES = [
    "future_droplet_mean_diameter_um",
    "future_diameter_change_um",
    "future_droplet_std_um",
    "future_droplet_cv",
    "future_single_cell_rate",
    "response_delay_ms",
]


def build_features(sample: DisturbanceSample) -> list[float]:
    stage = str(sample.disturbance_stage or DisturbanceStage.BASELINE.value)
    return [
        1.0 if stage == DisturbanceStage.BASELINE.value else 0.0,
        1.0 if stage == DisturbanceStage.DISTURBED.value else 0.0,
        1.0 if stage == DisturbanceStage.RECOVERY.value else 0.0,
        float(sample.disturbance_amplitude or 0.0),
        float(sample.q1_set or 0.0),
        float(sample.q2_set or 0.0),
        float(sample.q1_feedback or 0.0),
        float(sample.q2_feedback or 0.0),
        float(sample.q1_error or 0.0),
        float(sample.q2_error or 0.0),
        1.0 if sample.pump_comm_status else 0.0,
        float(sample.droplet_count_frame or 0),
        float(sample.valid_sample_count or 0),
        float(sample.measurement_noise_est or 0.0),
        float(sample.diameter_error_um or 0.0),
        float(sample.pid_output or 0.0),
        1.0 if sample.feedback_frozen else 0.0,
        float(sample.control_cycle_ms or 0.0),
        float(sample.control_jitter_ms or 0.0),
    ]


def build_nonlinear_features(features: list[float]) -> list[float]:
    values = [float(value) for value in features]
    expanded = list(values)
    expanded.extend(value * value for value in values)
    for left, right in _NONLINEAR_INTERACTION_PAIRS:
        left_index = _FEATURE_INDEX[left]
        right_index = _FEATURE_INDEX[right]
        expanded.append(values[left_index] * values[right_index])
    return expanded


def sample_is_usable(sample: DisturbanceSample) -> bool:
    return bool(
        sample.vision_valid
        and sample.droplet_mean_diameter_um is not None
        and int(sample.droplet_count_frame or 0) > 0
        and int(sample.valid_sample_count or 0) > 0
    )


def build_targets(sample: DisturbanceSample, future_sample: DisturbanceSample | None = None) -> list[float] | None:
    future_sample = future_sample or sample
    if sample.droplet_mean_diameter_um is None or future_sample.droplet_mean_diameter_um is None:
        return None
    response_delay_ms = max(0.0, (float(future_sample.timestamp) - float(sample.timestamp)) * 1000.0)
    return [
        float(future_sample.droplet_mean_diameter_um),
        float(future_sample.droplet_mean_diameter_um) - float(sample.droplet_mean_diameter_um),
        float(future_sample.droplet_std_um or 0.0),
        float(future_sample.droplet_cv or 0.0),
        float(future_sample.single_cell_rate or 0.0),
        response_delay_ms,
    ]
