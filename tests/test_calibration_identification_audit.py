from __future__ import annotations

from dataclasses import replace
import json
import math

import pytest

from backend.pid_control.calibration_experiment import (
    CalibrationIdentificationError, PlantCalibrationObservation,
    _fit_fopdt, build_plant_calibration_result, calibration_diagnostics,
    identify_channel_sensitivities, identify_channel_log_sensitivities,
    save_failed_calibration,
)
from backend.pid_control.calibration import load_plant_calibration
from test_plant_calibration_experiment import _config, _measurements, _measurement


def build(items):
    return build_plant_calibration_result(
        config=_config(), measurements=items, session_id="test", started_at="2026-09-14T00:00:00Z",
        q1_min=15, q1_max=100, q2_min=5, q2_max=25, total_flow_max=125, min_q1_q2_gap=1,
    )


def test_between_trial_baseline_offset_does_not_become_linear_gain():
    original = _measurements()
    shifted = [replace(item,
                       baseline_diameter_um=item.baseline_diameter_um + (20 if item.direction > 0 else -10),
                       steady_diameter_um=item.steady_diameter_um + (20 if item.direction > 0 else -10))
               for item in original]
    assert identify_channel_sensitivities(shifted) == pytest.approx(identify_channel_sensitivities(original))
    assert build(shifted).record.diameter_sensitivity_um_per_output == pytest.approx(
        build(original).record.diameter_sensitivity_um_per_output)


def test_log_gain_uses_each_trials_own_baseline():
    original = _measurements()
    shifted = [replace(item, baseline_diameter_um=item.baseline_diameter_um * (2 if item.direction > 0 else 1.5),
                       steady_diameter_um=item.steady_diameter_um * (2 if item.direction > 0 else 1.5),
                       diameter_change_um=item.diameter_change_um * (2 if item.direction > 0 else 1.5))
               for item in original]
    assert identify_channel_log_sensitivities(shifted) == pytest.approx(identify_channel_log_sensitivities(original))


def test_validation_cannot_move_model_operating_point_or_bounds():
    items = _measurements()
    original = build(items).record
    held_out = [replace(_measurement(f"validation-{i}", "validation", 1, 30),
                        baseline_diameter_um=100, steady_diameter_um=130,
                        baseline_q1=60, actual_q1=62) for i in range(20)]
    result = build(items + held_out)
    for name in ("baseline_diameter_um", "baseline_q1", "q1_max", "q1_output_gain", "controller_kp"):
        assert getattr(result.record, name) == pytest.approx(getattr(original, name))
    assert not result.record.validated_for_pi
    assert len([item for item in result.diagnostics["trials"] if item["channel"] == "validation"]) == 20


def test_no_response_is_explicit_and_failed_audit_is_not_loadable(tmp_path):
    items = [replace(item, response_detected=False, response_classification="below_detection_threshold")
             if item.channel == "combined" else item for item in _measurements()]
    with pytest.raises(CalibrationIdentificationError, match="本次响应不足") as exc:
        build(items)
    path = tmp_path / "incomplete.measurements.json"
    save_failed_calibration(path, config=_config(), measurements=items, reason=str(exc.value),
                            session_id="test", started_at="2026-09-14T00:00:00Z",
                            partial_trial={"trial_id": "next", "response_observations": []})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["measurements"] == json.loads(json.dumps([item.to_dict() for item in items]))
    assert data["partial_trial"]["trial_id"] == "next"
    assert data["diagnostics"]["channels"]["combined"]["detected_count"] == 0
    assert all(item["response_delay_ms"] is None for item in data["diagnostics"]["trials"] if item["channel"] == "combined")
    with pytest.raises(ValueError):
        load_plant_calibration(path)


def test_validation_diagnostics_keep_even_undetected_large_residuals():
    validation = replace(_measurement("validation", "validation", 1, 15),
                         response_detected=False, response_classification="below_detection_threshold")
    result = build(_measurements() + [validation])
    entry = result.diagnostics["trials"][-1]
    assert entry["predicted_change_um"] == pytest.approx(2.1)
    assert entry["observed_change_um"] == 15
    assert entry["mae_um"] == pytest.approx(12.9)
    assert not result.record.validated_for_pi


def test_dense_sampling_does_not_overweight_a_trial():
    def curve(item, tau):
        return replace(item, response_observations=tuple(
            PlantCalibrationObservation(
                frame_id=i, capture_monotonic=10+t, observed_monotonic=10+t,
                diameter_um=60+item.diameter_change_um*(1-math.exp(-max(0,t-1.2)/tau)), droplet_count=1,
            ) for i,t in enumerate((0.2, 0.7, 1.2, 1.6, 2.2, 3.2, 4.5))
        ))
    first = curve(_measurement("plus", "combined", 1, 3.5), 1)
    second = curve(_measurement("minus", "combined", -1, -3.5), 2)
    fit = _fit_fopdt([first, second])
    dense = replace(first, response_observations=first.response_observations * 5)
    repeated = _fit_fopdt([dense, second])
    assert repeated.delay_ms == fit.delay_ms
    assert repeated.time_constant_ms == fit.time_constant_ms
