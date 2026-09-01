from __future__ import annotations

from datetime import datetime, timezone
import threading

import pytest

from backend.disturbance_model.config import DisturbanceModelConfig
from backend.disturbance_model.models import DisturbanceControlStage
from backend.optimization.models import OptimizationObservation
from backend.orchestrator.drift import DriftSupervisor, DriftSupervisorConfig
from backend.orchestrator.service import OrchestratorService
from backend.pid_control.calibration import PlantCalibrationRecord
from backend.pid_control.config import PIDConfig, PIDControlMode
from backend.pid_control.diameter_pid import DiameterPIDController
from backend.pid_control.models import PIDInput


def _plant_record() -> PlantCalibrationRecord:
    return PlantCalibrationRecord(
        schema_version=1,
        calibration_id="plant-cal-001",
        created_at=datetime.now(timezone.utc).isoformat(),
        plant_id="rig-a",
        chip_id="chip-a",
        fluid_id="water-oil-a",
        pump_model="pump-a",
        syringe_profile="10-ml-glass",
        response_delay_median_ms=1200.0,
        response_delay_uncertainty_ms=250.0,
        diameter_sensitivity_um_per_output=0.5,
        q1_control_sign=-1.0,
        q2_control_sign=1.0,
        q1_output_gain=2.0,
        q2_output_gain=1.0,
        q1_min=20.0,
        q1_max=90.0,
        q2_min=6.0,
        q2_max=24.0,
        total_flow_max=110.0,
    )


def test_plant_record_supplies_conservative_delay_and_signed_gain() -> None:
    record = _plant_record()

    assert record.conservative_response_delay_ms == 1450.0
    assert record.feedforward_gain == 2.0


def test_disturbance_defaults_train_candidates_without_actuator_authority() -> None:
    config = DisturbanceModelConfig()

    assert config.deployment_stage == DisturbanceControlStage.COLLECT_ONLY.value
    assert not config.allow_low_weight_feedforward
    assert not config.allow_full_feedforward
    assert config.online_update_enabled


def test_bo_window_aggregation_uses_robust_period_statistics() -> None:
    observations = [
        OptimizationObservation(1, 50.0, 20.0, 60.0, 5.0, 2.0, 5),
        OptimizationObservation(1, 50.2, 20.1, 200.0, 50.0, 50.0, 6),
        OptimizationObservation(1, 49.9, 19.9, 61.0, 5.2, 2.2, 7),
    ]

    result = OrchestratorService._aggregate_optimization_window(1, observations)

    assert result.diameter_um == 61.0
    assert result.frequency_hz == 5.2
    assert result.period_count == 3
    assert result.valid_droplets == 18


def test_pid_after_bo_is_limited_to_local_trim_band() -> None:
    config = PIDConfig(
        control_mode=PIDControlMode.CLASSIC_PID.value,
        base_kp=10.0,
        base_ki=0.0,
        base_kd=0.0,
        output_rate_limit=1000.0,
        operating_point_local_span_fraction=0.20,
    )
    controller = DiameterPIDController(config)
    controller.set_operating_point(52.0, 18.0)

    command = controller.update_input(
        PIDInput(
            target_diameter_um=100.0,
            current_diameter_um=50.0,
            current_q1=52.0,
            current_q2=18.0,
            dt=1.0,
            frame_id=1,
            vision_valid=True,
            pump_communication_ok=True,
            droplet_count=5,
        )
    )

    assert command.actuator_saturated
    q1_radius = (config.q1_max - config.q1_min) * 0.20
    q2_radius = (config.q2_max - config.q2_min) * 0.20
    assert 52.0 - q1_radius <= command.q1 <= 52.0 + q1_radius
    assert 18.0 - q2_radius <= command.q2 <= 18.0 + q2_radius


def test_drift_supervisor_recommends_but_does_not_switch_control() -> None:
    supervisor = DriftSupervisor(
        DriftSupervisorConfig(consecutive_periods=3, healthy_clear_periods=2)
    )

    for _ in range(3):
        status = supervisor.observe(
            target_diameter_um=60.0,
            diameter_error_um=5.0,
            integral_state=0.0,
            integral_limit=100.0,
            actuator_saturated=False,
        )

    assert status.reoptimization_recommended
    assert "diameter error" in status.reason


def test_invalid_plant_record_cannot_authorize_zero_sensitivity() -> None:
    values = _plant_record().to_dict()
    values["diameter_sensitivity_um_per_output"] = 0.0

    with pytest.raises(ValueError, match="sensitivity"):
        PlantCalibrationRecord.from_mapping(values)


def test_disturbance_stage_update_preserves_active_experiment_group() -> None:
    service = object.__new__(OrchestratorService)
    service._lock = threading.RLock()
    service._disturbance_context = {
        "experiment_id": "run-1",
        "chip_id": "chip-1",
        "disturbance_name": "pressure-step",
        "disturbance_stage": "baseline",
    }

    service.set_disturbance_context(
        disturbance_stage="disturbed",
        disturbance_amplitude=2.0,
    )

    assert service._disturbance_context["experiment_id"] == "run-1"
    assert service._disturbance_context["chip_id"] == "chip-1"
    assert service._disturbance_context["disturbance_stage"] == "disturbed"
