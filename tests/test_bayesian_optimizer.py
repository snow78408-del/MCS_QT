from __future__ import annotations

import pytest

from backend.optimization import (
    BayesianOptimizationConfig,
    OptimizationObservation,
    SafeBayesianOptimizer,
)


def _config(**overrides) -> BayesianOptimizationConfig:
    values = {
        "target_diameter_um": 50.0,
        "q1_min": 40.0,
        "q1_max": 80.0,
        "q2_min": 5.0,
        "q2_max": 35.0,
        "measured_response_delay_ms": 1200.0,
        "settling_time_ms": 3000.0,
        "response_delay_source": "visual_step_response",
        "initial_sample_count": 4,
        "maximum_observations": 10,
        "confirmation_count": 2,
        "minimum_valid_droplets": 3,
        "random_seed": 7,
    }
    values.update(overrides)
    return BayesianOptimizationConfig(**values)


def _observation(candidate, diameter: float, **overrides) -> OptimizationObservation:
    values = {
        "candidate_id": candidate.candidate_id,
        "q1": candidate.q1,
        "q2": candidate.q2,
        "diameter_um": diameter,
        "frequency_hz": 10.0,
        "diameter_cv_percent": 2.0,
        "valid_droplets": 10,
    }
    values.update(overrides)
    return OptimizationObservation(**values)


def test_optimizer_requires_independently_measured_response_delay() -> None:
    with pytest.raises(ValueError, match="independently measured"):
        _config(response_delay_source="serial_reply")
    with pytest.raises(ValueError, match="independently measured"):
        _config(response_delay_source="串口应答时间")
    with pytest.raises(ValueError, match="measured physical/visual"):
        _config(measured_response_delay_ms=0.0)
    with pytest.raises(ValueError, match="shorter"):
        _config(measured_response_delay_ms=4000.0, settling_time_ms=3000.0)


def test_every_candidate_obeys_joint_pump_constraints() -> None:
    optimizer = SafeBayesianOptimizer(_config())
    for index in range(6):
        candidate = optimizer.ask()
        assert optimizer.is_feasible(candidate.q1, candidate.q2)
        optimizer.tell(_observation(candidate, diameter=70.0 - index))


def test_optimizer_starts_from_current_verified_operating_point() -> None:
    optimizer = SafeBayesianOptimizer(_config(), initial_point=(50.0, 20.0))

    candidate = optimizer.ask()

    assert candidate.q1 == pytest.approx(50.0)
    assert candidate.q2 == pytest.approx(20.0)
    assert "verified operating-point seed" in candidate.reason


def test_default_total_flow_limit_matches_pid_envelope() -> None:
    assert _config().total_flow_max == 125.0


def test_q1_above_q2_invariant_survives_runtime_config_mutation() -> None:
    config = _config(q1_min=5.0)
    config.min_q1_q2_gap = -100.0
    optimizer = SafeBayesianOptimizer(config)

    assert not optimizer.is_feasible(20.0, 20.0)
    assert not optimizer.is_feasible(20.1, 20.0)
    assert optimizer.is_feasible(20.2, 20.0)


def test_successful_point_is_repeated_before_completion() -> None:
    optimizer = SafeBayesianOptimizer(_config())
    first = optimizer.ask()
    optimizer.tell(_observation(first, diameter=50.5))

    confirmation = optimizer.ask()
    assert confirmation.q1 == pytest.approx(first.q1)
    assert confirmation.q2 == pytest.approx(first.q2)
    optimizer.tell(_observation(confirmation, diameter=49.5))

    status = optimizer.status()
    assert status.completed
    assert status.best_operating_point is not None
    assert status.confirmation_count == 2


def test_invalid_measurements_retry_without_training_the_gp() -> None:
    optimizer = SafeBayesianOptimizer(_config(invalid_retry_limit=1))
    candidate = optimizer.ask()
    optimizer.tell(
        _observation(
            candidate,
            diameter=50.0,
            measurement_valid=False,
            invalid_reason="vision invalid",
        )
    )
    assert optimizer.ask() == candidate
    assert optimizer.status().observation_count == 0

    optimizer.tell(
        _observation(
            candidate,
            diameter=50.0,
            valid_droplets=0,
            invalid_reason="sample insufficient",
        )
    )
    assert optimizer.status().failed


def test_frequency_target_is_optional_but_enforced_when_configured() -> None:
    optimizer = SafeBayesianOptimizer(_config(target_frequency_hz=20.0))
    candidate = optimizer.ask()
    optimizer.tell(_observation(candidate, diameter=50.0, frequency_hz=10.0))
    assert optimizer.status().confirmation_count == 0
