from __future__ import annotations

from dataclasses import replace
import json
import math

import pytest

from backend.pid_control.calibration_experiment import (
    CalibrationValidationError, PlantCalibrationObservation, save_plant_calibration_result,
)
from test_calibration_identification_audit import build
from test_plant_calibration_experiment import _measurements, _measurement, _config


def qualified_result():
    def curve(item):
        return replace(item, response_observations=tuple(
            PlantCalibrationObservation(
                frame_id=i, capture_monotonic=10+t, observed_monotonic=10+t,
                diameter_um=60+item.diameter_change_um*(1-math.exp(-max(0,t-item.response_delay_ms/1000))),
                droplet_count=1,
            ) for i,t in enumerate((0.2, 0.7, 1.2, 1.6, 2.2, 3.2, 4.5))
        ))
    items = [curve(item) for item in _measurements()]
    items += [curve(_measurement(f"validation-{direction}", "validation", direction, direction*2.1,
                                delay_ms=1260)) for direction in (1,-1)]
    result = build(items)
    return replace(result, config=replace(result.config, require_mpc_validation=True))


def test_failed_validation_cannot_publish_any_formal_file(tmp_path):
    result = build(_measurements())
    with pytest.raises(CalibrationValidationError):
        save_plant_calibration_result(result, tmp_path / "formal.json")
    assert not list(tmp_path.iterdir())


def test_pid_only_result_cannot_complete_default_dual_requirement(tmp_path):
    result = build(_measurements()+[_measurement("validation", "validation", 1, 2.1)])
    assert result.record.validated_for_pi and not result.record.validated_for_mpc
    result = replace(result, config=replace(result.config, require_mpc_validation=True))
    with pytest.raises(CalibrationValidationError):
        save_plant_calibration_result(result, tmp_path / "formal.json")


def test_qualified_result_delivers_calibration_and_separate_training_validation(tmp_path):
    result = qualified_result()
    assert result.accepted and result.record.validated_for_mpc
    saved = save_plant_calibration_result(result, tmp_path / "formal.json")
    record = json.loads((tmp_path / "formal.json").read_text(encoding="utf-8"))
    training = json.loads((tmp_path / "formal.mpc-training.json").read_text(encoding="utf-8"))
    assert record["validated_for_pi"] and record["validated_for_mpc"]
    assert saved["mpc_training_path"]
    assert all(item["channel"] != "validation" for item in training["training_trials"])
    assert len(training["validation_trials"]) == 2
    assert all(item["channel"] == "validation" for item in training["validation_trials"])
    assert training["training_trials"][0]["response_observations"]


@pytest.mark.parametrize("attempts", [0, 6, True, 1.5])
def test_retry_count_is_bounded(attempts):
    with pytest.raises(ValueError):
        replace(_config(), maximum_attempts=attempts)
