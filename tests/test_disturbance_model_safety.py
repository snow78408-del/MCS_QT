from __future__ import annotations

import threading
from collections import deque

import pytest

from backend.disturbance_model.config import DisturbanceModelConfig
from backend.disturbance_model.evaluator import evaluate_predictions
from backend.disturbance_model.feature_builder import (
    FEATURE_NAMES,
    NONLINEAR_FEATURE_NAMES,
    TARGET_NAMES,
    build_targets,
)
from backend.disturbance_model.model import LinearDisturbanceModel
from backend.disturbance_model.models import (
    DisturbanceControlStage,
    DisturbancePrediction,
    DisturbanceSample,
    ModelStatus,
)
from backend.disturbance_model.service import DisturbanceModelService
from backend.disturbance_model.trainer import DisturbanceModelTrainer
from backend.pid_control.config import PIDConfig
from backend.pid_control.feedforward import FeedforwardCompensator
from backend.pid_control.models import PIDInput


def _sample(timestamp: float, diameter: float, **overrides) -> DisturbanceSample:
    values = {
        "timestamp": timestamp,
        "experiment_id": "experiment-1",
        "chip_id": "chip-1",
        "disturbance_name": "q1-step",
        "disturbance_stage": "disturbed",
        "disturbance_amplitude": 1.0,
        "droplet_mean_diameter_um": diameter,
        "droplet_count_frame": 5,
        "valid_sample_count": 5,
        "vision_valid": True,
        "target_diameter_um": 100.0,
        "control_cycle_ms": 1000.0,
    }
    values.update(overrides)
    return DisturbanceSample(**values)


def _prediction(**overrides) -> DisturbancePrediction:
    values = {
        "timestamp": __import__("time").time(),
        "model_ready": True,
        "model_valid": True,
        "confidence": 0.99,
        "predicted_diameter_um": 105.0,
        "predicted_diameter_change_um": 5.0,
        "feedforward_weight": 0.2,
        "control_stage": "LOW_WEIGHT_FEEDFORWARD",
        "leading_signal_available": True,
        "signal_lead_time_ms": 2000.0,
        "prediction_horizon_ms": 1000.0,
    }
    values.update(overrides)
    return DisturbancePrediction(**values)


def _pid_input(prediction: DisturbancePrediction) -> PIDInput:
    return PIDInput(
        target_diameter_um=100.0,
        current_diameter_um=100.0,
        current_q1=10.0,
        current_q2=5.0,
        dt=1.0,
        frame_id=1,
        vision_valid=True,
        pump_communication_ok=True,
        disturbance_prediction=prediction,
        pump_response_delay_ms=500.0,
    )


def test_response_delay_target_uses_measured_delay_not_pair_spacing() -> None:
    current = _sample(10.0, 100.0, pump_response_delay_ms=5800.0)
    future = _sample(20.0, 105.0)

    targets = build_targets(current, future)

    assert targets is not None
    assert targets[1] == 5.0
    assert targets[5] == 5800.0


def test_metrics_score_delta_direction_and_persistence_baseline() -> None:
    metrics = evaluate_predictions(
        [2.0, -4.0, 1.0],
        [1.0, -3.0, -1.0],
        response_delay_true=[100.0, 200.0],
        response_delay_pred=[110.0, 180.0],
    )

    assert metrics.direction_accuracy == pytest.approx(2 / 3)
    assert metrics.response_delay_error_ms == 15.0
    assert metrics.persistence_rmse > metrics.rmse
    assert metrics.persistence_improvement > 0.0


def test_training_rejects_baseline_only_data_even_when_sample_count_is_sufficient() -> None:
    config = DisturbanceModelConfig(minimum_training_samples=3)
    samples = [
        _sample(float(index), 100.0 + index, disturbance_stage="baseline", disturbance_amplitude=0.0)
        for index in range(8)
    ]

    model, _, reason = DisturbanceModelTrainer(config).train(samples)

    assert model is None
    assert "disturbance events" in reason


def test_training_rejects_missing_experiment_and_chip_metadata() -> None:
    config = DisturbanceModelConfig(minimum_training_samples=3)
    samples = [_sample(float(index), 100.0 + index, experiment_id="", chip_id="") for index in range(8)]

    model, _, reason = DisturbanceModelTrainer(config).train(samples)

    assert model is None
    assert "experiment_id and chip_id" in reason


def test_training_rejects_model_that_only_has_absolute_diameter_persistence() -> None:
    config = DisturbanceModelConfig(
        minimum_training_samples=20,
        prediction_horizon_ms=1000,
        prediction_horizon_tolerance_ms=1,
        align_horizon_to_control_cycle=False,
        minimum_r2=0.1,
    )
    samples: list[DisturbanceSample] = []
    timestamp = 0.0
    for group in range(3):
        diameter = 100.0 + group * 20.0
        for index in range(16):
            samples.append(
                _sample(
                    timestamp,
                    diameter,
                    experiment_id=f"experiment-{group}",
                    chip_id=f"chip-{group}",
                    disturbance_name=f"step-{group}",
                )
            )
            diameter += 10.0 if index % 2 == 0 else -10.0
            timestamp += 1.0
        timestamp += 10.0

    model, metrics, reason = DisturbanceModelTrainer(config).train(samples)

    assert model is None
    assert metrics.persistence_improvement <= config.minimum_persistence_improvement or metrics.r2 < config.minimum_r2
    assert "failed" in reason


def test_legacy_unscaled_model_is_not_loaded() -> None:
    legacy = {
        "version": "legacy",
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "coefficients": [[0.0] * len(FEATURE_NAMES) for _ in TARGET_NAMES],
        "intercepts": [0.0] * len(TARGET_NAMES),
        "confidence": 1.0,
        "model_type": "linear",
    }

    assert LinearDisturbanceModel.from_dict(legacy) is None


def test_model_does_not_invent_an_uncalibrated_feedforward_command() -> None:
    model = LinearDisturbanceModel(
        version="v2",
        feature_names=list(FEATURE_NAMES),
        target_names=list(TARGET_NAMES),
        coefficients=[[0.0] * len(NONLINEAR_FEATURE_NAMES) for _ in TARGET_NAMES],
        intercepts=[100.0, 5.0, 0.0, 0.0, 0.0, 5800.0],
        confidence=0.9,
        feature_means=[0.0] * len(NONLINEAR_FEATURE_NAMES),
        feature_scales=[1.0] * len(NONLINEAR_FEATURE_NAMES),
        training_data_hash="0" * 64,
    )

    prediction = model.predict(_sample(1.0, 100.0))

    assert prediction.predicted_diameter_change_um == 5.0
    assert prediction.recommended_feedforward is None


def test_model_inverse_uses_learned_local_actuator_sensitivity() -> None:
    coefficients = [[0.0] * len(FEATURE_NAMES) for _ in TARGET_NAMES]
    change_index = TARGET_NAMES.index("future_diameter_change_um")
    coefficients[change_index][FEATURE_NAMES.index("q2_feedback")] = 2.0
    intercepts = [0.0] * len(TARGET_NAMES)
    intercepts[change_index] = -36.0
    model = LinearDisturbanceModel(
        version="linear-test",
        feature_names=list(FEATURE_NAMES),
        target_names=list(TARGET_NAMES),
        coefficients=coefficients,
        intercepts=intercepts,
        confidence=0.9,
        model_type="linear",
        feature_means=[0.0] * len(FEATURE_NAMES),
        feature_scales=[1.0] * len(FEATURE_NAMES),
        training_data_hash="1" * 64,
    )
    sample = _sample(1.0, 100.0, q1_set=50.0, q2_set=20.0, q1_feedback=50.0, q2_feedback=20.0)

    recommendation = model.recommend_feedforward(
        sample,
        predicted_change_um=4.0,
        probe_output=1.0,
        min_sensitivity=0.02,
        max_output=50.0,
        q1_control_sign=-1.0,
        q2_control_sign=1.0,
        q1_output_gain=2.0,
        q2_output_gain=1.0,
    )

    assert recommendation == pytest.approx(-2.0)


def test_feedforward_is_fail_closed_until_plant_gain_is_calibrated() -> None:
    prediction = _prediction()
    uncalibrated = FeedforwardCompensator(PIDConfig(feedforward_enabled=True, feedforward_calibrated=False))
    calibrated = FeedforwardCompensator(
        PIDConfig(feedforward_enabled=True, feedforward_calibrated=True, feedforward_gain=2.0)
    )

    blocked = uncalibrated.compute(_pid_input(prediction))
    active = calibrated.compute(_pid_input(prediction))

    assert not blocked.active
    assert "not calibrated" in blocked.reason
    assert active.active
    assert active.u_ff == -2.0  # -gain * delta-D * stage weight


def test_validated_model_inverse_can_assist_without_static_gain_or_leading_signal() -> None:
    prediction = _prediction(
        recommended_feedforward=-1.5,
        leading_signal_available=False,
    )
    compensator = FeedforwardCompensator(
        PIDConfig(feedforward_enabled=True, feedforward_calibrated=False)
    )

    result = compensator.compute(_pid_input(prediction))

    assert result.active
    assert result.u_ff == pytest.approx(-1.5)


def test_feedforward_requires_a_signal_that_leads_the_measured_pump_delay() -> None:
    compensator = FeedforwardCompensator(
        PIDConfig(feedforward_enabled=True, feedforward_calibrated=True, feedforward_gain=2.0)
    )

    absent = compensator.compute(_pid_input(_prediction(leading_signal_available=False)))
    late = compensator.compute(_pid_input(_prediction(signal_lead_time_ms=550.0)))
    unmeasured_input = _pid_input(_prediction())
    unmeasured_input.pump_response_delay_ms = 0.0
    unmeasured = compensator.compute(unmeasured_input)

    assert not absent.active and "no causal leading" in absent.reason
    assert not late.active and "too late" in late.reason
    assert not unmeasured.active and "unmeasured" in unmeasured.reason


def test_shadow_does_not_match_a_late_observation_to_expired_predictions() -> None:
    config = DisturbanceModelConfig(
        deployment_stage="SHADOW",
        online_update_enabled=False,
        prediction_horizon_ms=200,
        prediction_horizon_tolerance_ms=100,
        align_horizon_to_control_cycle=False,
    )
    service = object.__new__(DisturbanceModelService)
    service.config = config
    service._lock = threading.RLock()
    service._log = lambda _message: None
    service._control_stage = DisturbanceControlStage.SHADOW
    service._pending_shadow = deque(maxlen=200)
    service._shadow_errors = deque(maxlen=40)
    service._shadow_change_errors = deque(maxlen=40)
    service._shadow_direction_matches = deque(maxlen=40)
    service._status = ModelStatus()
    service._safety_fallback = False

    origin = _sample(0.0, 100.0, control_cycle_ms=0.0)
    service._record_shadow_prediction_locked(origin, _prediction(predicted_diameter_um=105.0))
    service._update_shadow_metrics_locked(_sample(5.0, 110.0, control_cycle_ms=0.0))

    assert service.get_status().shadow_mae_um == 0.0
    assert len(service._shadow_errors) == 0


def test_feedforward_stage_requires_shadow_window_and_explicit_authorization() -> None:
    config = DisturbanceModelConfig(
        shadow_min_comparisons=1,
        allow_low_weight_feedforward=False,
    )
    service = object.__new__(DisturbanceModelService)
    service.config = config
    service._shadow_errors = deque(maxlen=40)
    service._status = ModelStatus(shadow_change_mae_um=0.0, shadow_direction_accuracy=1.0)

    assert service._feedforward_weight_for_stage(DisturbanceControlStage.LOW_WEIGHT_FEEDFORWARD) == 0.0

    service._shadow_errors.append(0.0)
    assert service._feedforward_weight_for_stage(DisturbanceControlStage.LOW_WEIGHT_FEEDFORWARD) == 0.0

    config.allow_low_weight_feedforward = True
    assert service._feedforward_weight_for_stage(DisturbanceControlStage.LOW_WEIGHT_FEEDFORWARD) == pytest.approx(0.2)
    assert service._feedforward_weight_for_stage(DisturbanceControlStage.FULL_FEEDFORWARD) == 0.0
