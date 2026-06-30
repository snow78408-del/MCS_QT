from __future__ import annotations

from .models import DisturbanceSample, DisturbanceStage


FEATURE_NAMES = [
    "stage_baseline",
    "stage_disturbed",
    "stage_recovery",
    "disturbance_amplitude",
    "temperature_c",
    "q1_set",
    "q2_set",
    "q1_feedback",
    "q2_feedback",
    "q1_error",
    "q2_error",
    "pump_response_delay_ms",
    "pump_comm_status",
    "vision_latency_ms",
    "measurement_noise_est",
    "image_brightness_mean",
    "focus_score",
    "control_cycle_ms",
    "control_jitter_ms",
    "pid_output",
]


TARGET_NAMES = [
    "droplet_mean_diameter_um",
    "droplet_std_um",
    "droplet_cv",
    "droplet_frequency_hz",
    "single_cell_rate",
    "pump_response_delay_ms",
]


def build_features(sample: DisturbanceSample) -> list[float]:
    stage = str(sample.disturbance_stage or DisturbanceStage.BASELINE.value)
    return [
        1.0 if stage == DisturbanceStage.BASELINE.value else 0.0,
        1.0 if stage == DisturbanceStage.DISTURBED.value else 0.0,
        1.0 if stage == DisturbanceStage.RECOVERY.value else 0.0,
        float(sample.disturbance_amplitude or 0.0),
        float(sample.temperature_c or 0.0),
        float(sample.q1_set or 0.0),
        float(sample.q2_set or 0.0),
        float(sample.q1_feedback or 0.0),
        float(sample.q2_feedback or 0.0),
        float(sample.q1_error or 0.0),
        float(sample.q2_error or 0.0),
        float(sample.pump_response_delay_ms or 0.0),
        1.0 if sample.pump_comm_status else 0.0,
        float(sample.vision_latency_ms or 0.0),
        float(sample.measurement_noise_est or 0.0),
        float(sample.image_brightness_mean or 0.0),
        float(sample.focus_score or 0.0),
        float(sample.control_cycle_ms or 0.0),
        float(sample.control_jitter_ms or 0.0),
        float(sample.pid_output or 0.0),
    ]


def build_targets(sample: DisturbanceSample) -> list[float] | None:
    if sample.droplet_mean_diameter_um is None:
        return None
    return [
        float(sample.droplet_mean_diameter_um),
        float(sample.droplet_std_um or 0.0),
        float(sample.droplet_cv or 0.0),
        float(sample.droplet_frequency_hz or 0.0),
        float(sample.single_cell_rate or 0.0),
        float(sample.pump_response_delay_ms or 0.0),
    ]
