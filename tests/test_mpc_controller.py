from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from backend.pid_control.calibration import PlantCalibrationRecord
from backend.pid_control.config import PIDConfig
from backend.pid_control.models import PIDInput
from backend.pid_control.mpc import DiameterMPCController, MPCSettings
from backend.pid_control.service import build_controller


def record(**changes):
    values = dict(
        schema_version=3, calibration_id="mpc-test", created_at="2026-10-15T00:00:00+00:00",
        plant_id="simulation", chip_id="test", fluid_id="test", pump_model="mock", syringe_profile="mock",
        response_delay_median_ms=1000, response_delay_uncertainty_ms=100,
        diameter_sensitivity_um_per_output=2, q1_control_sign=-1, q2_control_sign=1,
        q1_output_gain=2, q2_output_gain=1, q1_min=40, q1_max=60, q2_min=10, q2_max=25,
        total_flow_max=80, measurement_region="generation", channel_height_um=100, channel_width_um=100,
        baseline_q1=50, baseline_q2=20, baseline_diameter_um=60,
        q1_log_diameter_sensitivity=-0.4, q2_log_diameter_sensitivity=0.2,
        response_time_constant_ms=1200, controller_kp=0.5, controller_ki=0.01,
        continuous_phase_oil="test-oil", model_fit_method="robust_fopdt_grid",
        validation_sample_count=20, validation_mae_um=0.1, validated_for_pi=True, validated_for_mpc=True,
    )
    values.update(changes)
    return PlantCalibrationRecord(**values)


def measurement(**changes):
    values = dict(target_diameter_um=62, current_diameter_um=60, current_q1=50, current_q2=20,
                  dt=7.5, frame_id=1, vision_valid=True, pump_communication_ok=True, droplet_count=5)
    values.update(changes)
    return PIDInput(**values)


def controller(**changes):
    return DiameterMPCController(PIDConfig(control_mode="MPC", **changes), record(), 7.5)


def test_model_validation_and_factory_are_required():
    with pytest.raises(ValueError, match="validation"):
        DiameterMPCController(PIDConfig(), record(validated_for_mpc=False), 7.5)
    with pytest.raises(ValueError, match="calibration"):
        build_controller(PIDConfig(control_mode="MPC"))
    assert isinstance(build_controller(PIDConfig(control_mode="MPC"), calibration=record()), DiameterMPCController)
    with pytest.raises(ValueError, match="validation"):
        DiameterMPCController(PIDConfig(), record(validated_for_mpc="false"), 7.5)


def test_first_move_and_entire_future_plan_obey_constraints():
    ctrl = controller(max_flow_change_per_cycle=0.5, total_flow_max=70)
    output = ctrl.update_input(measurement(target_diameter_um=100))
    assert not output.freeze_feedback, output.reason
    assert output.control_owner == output.control_mode == "MPC"
    plan = np.asarray(ctrl.last_plan)
    assert len(plan) == 10
    assert plan[0] == pytest.approx([output.q1, output.q2])
    assert np.all(plan >= [46 - 1e-7, 17 - 1e-7])
    assert np.all(plan <= [54 + 1e-7, 23 + 1e-7])
    assert np.all(plan.sum(axis=1) <= 70 + 1e-7)
    assert np.all(plan[:, 0] - plan[:, 1] >= ctrl.config.min_q1_q2_gap - 1e-7)
    assert np.max(abs(np.diff(np.vstack([[50, 20], plan]), axis=0))) <= 0.5 + 1e-7
    assert output.q1 < 50 and output.q2 > 20
    assert output.actuator_saturated
    assert output.mpc_prediction_um[-1] > 60


def test_delay_and_lag_determine_prediction_spacing():
    ctrl = DiameterMPCController(PIDConfig(), record(response_delay_median_ms=20000,
                               response_time_constant_ms=10000), 7.5)
    output = ctrl.update_input(measurement())
    assert not output.freeze_feedback
    assert ctrl.sample_period_s == pytest.approx(50.1)
    assert output.mpc_horizon_s == pytest.approx(501)


@pytest.mark.parametrize("changes", [
    {"vision_valid": False}, {"pump_communication_ok": False}, {"droplet_count": 0},
    {"frame_id": 0}, {"dt": float("nan")}, {"system_running": False},
    {"current_diameter_um": None}, {"target_diameter_um": float("inf")},
    {"current_q1": 100},
])
def test_invalid_measurement_freezes_without_advancing_state(changes):
    ctrl = controller()
    output = ctrl.update_input(measurement(**changes))
    assert output.freeze_feedback
    assert ctrl.offset_um == 0 and ctrl.last_plan == () and ctrl._last_frame_id is None


@pytest.mark.parametrize("q1,q2", [(0, 20), (50, float("nan")), (-1, 20)])
def test_critical_flows_request_stop(q1, q2):
    output = controller().update_input(measurement(current_q1=q1, current_q2=q2))
    assert output.suggested_stop and output.freeze_feedback


def test_duplicate_frame_and_solver_failure_do_not_advance_observer(monkeypatch):
    ctrl = controller()
    assert not ctrl.update_input(measurement()).freeze_feedback
    before = (ctrl.offset_um, ctrl.offset_variance, ctrl.last_plan)
    assert ctrl.update_input(measurement()).freeze_feedback
    monkeypatch.setattr("backend.pid_control.mpc.minimize", lambda *a, **kw:
                        SimpleNamespace(success=False, message="iteration limit"))
    output = ctrl.update_input(measurement(frame_id=2))
    assert output.freeze_feedback and "iteration limit" in output.reason
    assert (ctrl.offset_um, ctrl.offset_variance, ctrl.last_plan) == before


def test_solver_claiming_success_with_unsafe_plan_is_rejected(monkeypatch):
    monkeypatch.setattr("backend.pid_control.mpc.minimize", lambda *a, **kw:
                        SimpleNamespace(success=True, x=np.ones(20) * 1000, message="success"))
    output = controller().update_input(measurement())
    assert output.freeze_feedback and "infeasible" in output.reason


def test_quadratic_model_uses_local_derivative_and_baseline_difference():
    model = dict(kind="quadratic_flow_response", center=[50, 20], scale=[10, 5],
                 coefficients=[-8, 5, 1, 0.5, 0.2])
    ctrl = DiameterMPCController(PIDConfig(), record(model_fit_method="quadratic_response_fopdt",
                                                  nonlinear_model=model), 7.5)
    q = np.array([52., 21.])
    assert ctrl.steady_diameter(np.array([50., 20.])) == 60
    finite_difference = np.array([(ctrl.steady_diameter(q + np.eye(2)[j] * 1e-5)
                                  - ctrl.steady_diameter(q - np.eye(2)[j] * 1e-5)) / 2e-5 for j in range(2)])
    assert ctrl._gradient(q) == pytest.approx(finite_difference)
    assert not ctrl.update_input(measurement()).freeze_feedback


def test_closed_loop_tracks_step_and_constant_model_bias_without_hardware():
    ctrl = controller(max_flow_change_per_cycle=1)
    flows = np.array([50., 20.])
    diameter = 61.0  # one micrometre of unmodeled constant disturbance
    errors = []
    for frame in range(1, 21):
        target = 62 if frame < 11 else 59
        output = ctrl.update_input(measurement(frame_id=frame, target_diameter_um=target,
                                  current_q1=flows[0], current_q2=flows[1], current_diameter_um=diameter))
        assert not output.freeze_feedback, output.reason
        flows = np.array([output.q1, output.q2])
        equilibrium = 60 * (flows[0] / 50) ** -0.4 * (flows[1] / 20) ** 0.2 + 1.0
        diameter = equilibrium + (diameter - equilibrium) * np.exp(-(7.5 - 1.0) / 1.2)
        errors.append(abs(diameter - target))
    assert errors[9] < 0.15
    assert errors[-1] < 0.15
    assert ctrl.offset_um == pytest.approx(1.0, abs=0.1)


def test_reset_and_bo_handoff_clear_observer_and_restrict_authority():
    ctrl = controller(operating_point_local_span_fraction=0.05)
    ctrl.update_input(measurement())
    ctrl.set_operating_point(50, 20)
    assert ctrl.offset_um == 0 and ctrl.last_plan == ()
    output = ctrl.update_input(measurement(target_diameter_um=100))
    assert 49 <= output.q1 <= 51
    assert 19.25 <= output.q2 <= 20.75


def test_speculative_copy_does_not_change_live_observer():
    ctrl = controller()
    candidate = copy.deepcopy(ctrl)
    assert not candidate.update_input(measurement(current_diameter_um=61)).freeze_feedback
    assert candidate.offset_um > 0
    assert ctrl.offset_um == 0 and ctrl.last_plan == ()


@pytest.mark.parametrize("settings", [{"horizon": 0}, {"move_weight": float("nan")},
                                     {"local_span_fraction": 1}, {"max_iterations": 0}])
def test_invalid_tuning_is_rejected(settings):
    with pytest.raises(ValueError):
        MPCSettings(**settings)
