from __future__ import annotations

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from backend.pid_control.calibration_experiment import (
    CalibrationIdentificationError, PlantCalibrationObservation,
    build_plant_calibration_result, save_failed_calibration, save_plant_calibration_result,
)
from backend.pid_control.nonlinear_calibration import fit_response_surface, predict_change
from backend.pid_control.calibration import load_plant_calibration
from backend.orchestrator.service import OrchestratorService
from test_plant_calibration_experiment import _config, _measurement


def nonlinear_measurements(coefficients=(-1.0, 1.5, -2.0, 0.4, -0.2)):
    items = []
    for repeat in range(2):
        for channel in ("q1", "q2", "combined"):
            for direction in (1, -1):
                item = _measurement(f"{channel}-{repeat}-{direction}", channel, direction, 0)
                items.append(item)
    items += [_measurement(f"validation-{direction}", "validation", direction, 0) for direction in (1, -1)]
    model = {"center": [50.0, 20.0], "scale": [2.0, 1.0], "coefficients": coefficients}
    result = []
    for item in items:
        change = predict_change(model, item)
        observations = tuple(
            PlantCalibrationObservation(
                frame_id=index, capture_monotonic=10 + elapsed, observed_monotonic=10 + elapsed,
                diameter_um=60 + change * (1 - math.exp(-max(0, elapsed - 1.2))), droplet_count=1,
            )
            for index, elapsed in enumerate((0.2, 0.7, 1.2, 1.6, 2.2, 3.2, 4.5, 7.0))
        )
        result.append(replace(item, steady_diameter_um=60 + change, diameter_change_um=change,
                              response_observations=observations, response_detected=False,
                              response_classification="below_detection_threshold"))
    return result


def build(items):
    return build_plant_calibration_result(
        config=replace(_config(), identification_model="quadratic_response"), measurements=items,
        session_id="nonlinear-test", started_at="2026-09-18T00:00:00Z",
        q1_min=20, q1_max=200, q2_min=5, q2_max=25, total_flow_max=225, min_q1_q2_gap=1,
    )


def test_opposite_steps_with_same_output_direction_fit_curvature_and_export(tmp_path):
    items = nonlinear_measurements()
    q1 = [item for item in items if item.channel == "q1"]
    assert all(item.diameter_change_um < 0 for item in q1)
    result = build(items)
    assert result.accepted
    assert result.record.nonlinear_model["coefficients"] == pytest.approx([-1, 1.5, -2, .4, -.2])
    assert result.q1_sensitivity_um_per_flow == pytest.approx(-.5)
    assert result.q2_sensitivity_um_per_flow == pytest.approx(1.5)
    assert result.record.controller_kp > 0
    assert result.record.q1_min > 48
    saved = save_plant_calibration_result(result, tmp_path / "calibration.json")
    training = json.loads((tmp_path / "calibration.mpc-training.json").read_text(encoding="utf-8"))
    assert saved["mpc_training_path"]
    assert len(training["training_trials"]) == 12
    assert len(training["validation_trials"]) == 2
    assert training["record"]["nonlinear_model"]
    assert result.diagnostics["trials"][-1]["predicted_change_um"] == pytest.approx(items[-1].diameter_change_um)
    loaded = load_plant_calibration(saved["path"])
    service = OrchestratorService(
        vision_service=SimpleNamespace(), vision_adapter=SimpleNamespace(),
        pump_service=SimpleNamespace(runtime_config=SimpleNamespace(min_q1_q2_gap=1.0)),
    )
    try:
        service._apply_plant_calibration(loaded)
        assert service.pid_config.log_sensitivity_calibrated
        assert service.pid_config.base_kp == loaded.controller_kp
        assert service.pid_config.base_ki == loaded.controller_ki
        assert service.pid_config.q1_log_diameter_sensitivity == loaded.q1_log_diameter_sensitivity
        assert service.pid_config.q1_min == loaded.q1_min
    finally:
        service.close_background_services()
        service._safety.shutdown()


def test_validation_is_not_used_to_fit_surface_or_pid_parameters():
    items = nonlinear_measurements()
    original = build(items)
    changed = [replace(item, steady_diameter_um=item.steady_diameter_um + 20,
                       response_observations=tuple(replace(obs, diameter_um=obs.diameter_um + 20)
                                                   for obs in item.response_observations))
               if item.channel == "validation" else item for item in items]
    result = build(changed)
    assert result.record.nonlinear_model == original.record.nonlinear_model
    assert result.record.controller_kp == original.record.controller_kp
    assert not result.accepted


def test_flat_working_point_keeps_model_data_without_inventing_pid_gain(tmp_path):
    items = nonlinear_measurements((0, 0, -2, .4, -.2))
    with pytest.raises(CalibrationIdentificationError, match="局部斜率"):
        build(items)
    path = tmp_path / "diagnostics.json"
    save_failed_calibration(path, config=replace(_config(), identification_model="quadratic_response"),
                            measurements=items, reason="flat", session_id="test", started_at="test")
    training = json.loads((tmp_path / "diagnostics.mpc-training.json").read_text(encoding="utf-8"))
    assert training["complete_collection"]
    assert training["nonlinear_model"]
    assert not training["loadable_calibration"]


def test_rank_deficient_experiment_does_not_invent_cross_coupling():
    items = [item for item in nonlinear_measurements() if item.channel in {"q1", "q2"}]
    with pytest.raises(CalibrationIdentificationError, match="交互项"):
        fit_response_surface(items)
