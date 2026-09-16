from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from backend.pid_control.config import PIDConfig, PIDControlMode
from backend.pid_control.diameter_pid import DiameterPIDController
from backend.pid_control.models import PIDInput


def config(**changes):
    values = dict(
        control_mode=PIDControlMode.CLASSIC_PID.value, base_kp=0, base_ki=0, base_kd=0,
        target_feedforward_enabled=True, target_feedforward_calibrated=True,
        disturbance_feedforward_enabled=False, log_sensitivity_calibrated=True,
        q1_log_diameter_sensitivity=1.0, q2_log_diameter_sensitivity=0.0,
        feedforward_baseline_q1=50, feedforward_baseline_q2=10,
        feedforward_baseline_diameter_um=50,
        q1_min=40, q1_max=60, q2_min=5, q2_max=15, total_flow_max=75,
    )
    values.update(changes)
    return PIDConfig(**values)


def measurement(**changes):
    values = dict(target_diameter_um=55, current_diameter_um=50, current_q1=50, current_q2=10,
                  dt=7.5, frame_id=1, vision_valid=True, pump_communication_ok=True,
                  droplet_count=5)
    values.update(changes)
    return PIDInput(**values)


@pytest.mark.parametrize("g1,g2", [(1.0, 0.0), (-0.5, 0.5), (0.0, 1.0)])
def test_target_feedforward_inverts_calibrated_model_with_fixed_pi(g1, g2):
    controller = DiameterPIDController(config(q1_log_diameter_sensitivity=g1, q2_log_diameter_sensitivity=g2))
    result = controller.update_input(measurement())
    predicted = 50 * (result.q1 / 50) ** g1 * (result.q2 / 10) ** g2
    assert predicted == pytest.approx(55)
    assert result.target_feedforward_active
    assert not result.disturbance_feedforward_active
    assert not result.adaptive_enabled
    assert result.feedforward_output == result.target_feedforward_output


def test_unchanged_target_does_not_accumulate_flow_and_returning_target_restores_point():
    controller = DiameterPIDController(config())
    first = controller.update_input(measurement())
    second = controller.update_input(measurement(frame_id=2, current_q1=first.q1, current_diameter_um=55))
    assert first.q1 == second.q1 == pytest.approx(55)
    returned = controller.update_input(measurement(frame_id=3, target_diameter_um=50, current_q1=second.q1))
    assert returned.q1 == pytest.approx(50)


@pytest.mark.parametrize("changes", [
    {"target_feedforward_enabled": False}, {"target_feedforward_calibrated": False},
    {"feedforward_baseline_diameter_um": 0},
])
def test_target_feedforward_requires_selection_and_calibrated_baseline(changes):
    result = DiameterPIDController(config(**changes)).update_input(measurement())
    assert not result.target_feedforward_active
    assert result.target_feedforward_output == 0
    assert result.q1 == 50


def test_unreachable_target_does_not_extrapolate_and_pi_remains_available():
    result = DiameterPIDController(config(base_kp=0.1)).update_input(measurement(target_diameter_um=80))
    assert not result.target_feedforward_active
    assert "范围" in result.target_feedforward_reason
    assert result.pid_output > 0
    assert 40 <= result.q1 <= 60


def test_total_flow_limit_and_bo_local_span_restrict_target_feedforward():
    result = DiameterPIDController(config(total_flow_max=62)).update_input(measurement())
    assert not result.target_feedforward_active
    assert result.q1 + result.q2 <= 62
    controller = DiameterPIDController(config())
    controller.set_operating_point(50, 10)
    result = controller.update_input(measurement())
    assert not result.target_feedforward_active  # BO local Q1 span is 46..54.


def test_target_feedforward_uses_existing_flow_step_limit_and_invalid_data_freezes():
    controller = DiameterPIDController(config(max_flow_change_per_cycle=0.5))
    result = controller.update_input(measurement())
    assert result.q1 == pytest.approx(50.5)
    assert result.actuator_saturated
    frozen = controller.update_input(measurement(frame_id=2, vision_valid=False, current_q1=50.5))
    assert frozen.freeze_feedback
    assert frozen.q1 == 50.5
    assert not frozen.target_feedforward_active


def prediction(**changes):
    values = dict(timestamp=time.time(), model_ready=True, model_valid=True, confidence=0.99,
                  leading_signal_available=True, signal_lead_time_ms=3000,
                  prediction_horizon_ms=2000, feedforward_weight=0.2,
                  predicted_disturbance_residual_um=5)
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("mode", [PIDControlMode.CLASSIC_PID.value, PIDControlMode.ADAPTIVE_PID.value])
def test_explicit_disturbance_selection_is_independent_of_feedback_mode(mode):
    cfg = PIDConfig(control_mode=mode, disturbance_feedforward_enabled=True,
                    feedforward_calibrated=True, feedforward_gain=2,
                    base_kp=0, base_ki=0, base_kd=0)
    result = DiameterPIDController(cfg).update_input(measurement(
        disturbance_prediction=prediction(), pump_response_delay_ms=1000))
    assert result.disturbance_feedforward_active
    assert result.disturbance_feedforward_output == pytest.approx(-2)
    assert not result.target_feedforward_active


@pytest.mark.parametrize("bad_signal", [{"leading_signal_available": False},
                                      {"signal_lead_time_ms": 500}, {"timestamp": 1}])
def test_independent_switch_does_not_bypass_disturbance_signal_gates(bad_signal):
    cfg = PIDConfig(control_mode=PIDControlMode.CLASSIC_PID.value,
                    disturbance_feedforward_enabled=True, feedforward_calibrated=True)
    result = DiameterPIDController(cfg).update_input(measurement(
        disturbance_prediction=prediction(**bad_signal), pump_response_delay_ms=1000))
    assert not result.disturbance_feedforward_active
    assert result.disturbance_feedforward_output == 0


def test_generation_model_cannot_authorize_old_disturbance_gain_units():
    result = DiameterPIDController(config(
        disturbance_feedforward_enabled=True, feedforward_calibrated=True,
    )).update_input(measurement(disturbance_prediction=prediction(), pump_response_delay_ms=1000))
    assert result.target_feedforward_active
    assert not result.disturbance_feedforward_active
    assert "单位" in result.disturbance_feedforward_reason


def test_saved_switches_are_independent_in_system_config():
    from backend.orchestrator.models import SystemConfig
    cfg = SystemConfig(50, 1, "video", "sample.mp4", 50, 10, 7500,
                       target_feedforward_enabled=True, disturbance_feedforward_enabled=False)
    changed = replace(cfg, target_feedforward_enabled=False, disturbance_feedforward_enabled=True)
    assert not changed.target_feedforward_enabled
    assert changed.disturbance_feedforward_enabled


def test_frontend_restores_both_independent_switches():
    from frontend.qt_app import FrontendApp
    cfg = dict(target_diameter=50, pixel_to_micron=1, video_source_type="video",
               video_source="sample.mp4", initial_q1=50, initial_q2=10, control_interval_ms=7500,
               target_feedforward_enabled=True, disturbance_feedforward_enabled=False)
    restored = FrontendApp.build_system_config(SimpleNamespace(frontend_config=cfg))
    assert restored.target_feedforward_enabled
    assert not restored.disturbance_feedforward_enabled
