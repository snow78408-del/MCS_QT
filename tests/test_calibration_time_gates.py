from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from backend.orchestrator.service import OrchestratorService
from backend.pid_control import PlantCalibrationExperimentConfig, PlantCalibrationObservation


def observations(start: float, spacing: float, diameter: float, count: int = 40):
    return [PlantCalibrationObservation(
        frame_id=i, capture_monotonic=start+i*spacing, observed_monotonic=1000.0,
        diameter_um=diameter, droplet_count=1, droplet_id=i,
    ) for i in range(count)]


@pytest.mark.parametrize("start,spacing,diameter,expected", [
    (0.0, 0.1, 60.0, None),  # Many processed droplets do not advance capture time.
    (30.0, 0.001, 65.0, None),
    (30.0, 0.1, 65.0, "detected_stable"),
    (30.0, 0.1, 60.0, None),
    (59.0, 0.1, 60.0, "below_detection_threshold"),
    (60.0, 0.1, 60.0, "below_detection_threshold"),
])
def test_response_requires_elapsed_capture_time(start, spacing, diameter, expected):
    samples = observations(start, spacing, diameter)
    decision, tail = OrchestratorService._calibration_response_decision(
        list(reversed(samples)), baseline_diameter_um=60.0, response_threshold_um=0.5,
        stable_sample_count=5, minimum_observation_count=30, stability_tolerance_um=0.2,
        pixel_to_micron=0.1, noise_reference=tuple(observations(0, 1, 60, 5)),
        eligible_after=30.0, low_response_after=60.0, stability_duration_s=3.0,
    )
    assert decision == expected
    if decision:
        assert tail[-1].capture_monotonic-tail[0].capture_monotonic >= 3.0
        assert len(tail) > 5


@pytest.mark.parametrize("values", [
    {"minimum_response_wait_s": 0}, {"low_response_wait_s": 20},
    {"stability_duration_s": -1}, {"minimum_response_wait_s": float("nan")},
    {"baseline_wait_s": -1}, {"baseline_wait_s": float("nan")},
    {"baseline_wait_s": float("inf")},
])
def test_invalid_calibration_horizons_rejected(values):
    config = PlantCalibrationExperimentConfig(
        plant_id="rig", chip_id="chip", fluid_id="fluid", pump_model="pump",
        syringe_profile="syringe", q1_step=2, q2_step=1,
    )
    with pytest.raises(ValueError):
        replace(config, **values)


def test_baseline_wait_is_independent_and_legacy_config_keeps_its_horizon():
    config = PlantCalibrationExperimentConfig(
        plant_id="rig", chip_id="chip", fluid_id="fluid", pump_model="pump",
        syringe_profile="syringe", q1_step=2, q2_step=1,
        minimum_response_wait_s=20,
    )
    assert config.effective_baseline_wait_s == 20
    updated = replace(config, baseline_wait_s=0)
    assert updated.effective_baseline_wait_s == 0
    assert updated.minimum_response_wait_s == 20
    assert updated.to_dict()["baseline_wait_s"] == 0


@pytest.mark.parametrize("onset,expected", [(None, None), (5.0, "detected_stable"), (8.0, None)])
def test_confirmed_onset_allows_early_completion_but_needs_full_stable_window(onset, expected):
    decision, _ = OrchestratorService._calibration_response_decision(
        observations(5.0, 0.1, 65.0), baseline_diameter_um=60.0,
        response_threshold_um=0.5, stable_sample_count=5, minimum_observation_count=30,
        stability_tolerance_um=0.2, pixel_to_micron=0.1,
        noise_reference=tuple(observations(0, 1, 60, 5)),
        eligible_after=30.0, low_response_after=60.0, stability_duration_s=3.0,
        confirmed_response_after=onset,
    )
    assert decision == expected


@pytest.mark.parametrize("case,writes", [
    ("verified", 0), ("mismatch", 1), ("old_session", 1), ("old_generation", 1),
    ("read_failed", 0), ("stopped", 0), ("cancelled_during_read", 0),
])
def test_baseline_reuse_needs_fresh_readback_and_current_lifecycle(case, writes):
    calls = []
    cancelled = False
    def read_flows():
        nonlocal cancelled
        calls.append("read")
        cancelled = case == "cancelled_during_read"
        return (49.0, 20.0) if case == "mismatch" else (50.0, 20.0)

    def require(*_args):
        if cancelled:
            raise RuntimeError("cancelled")

    def apply(*_args):
        calls.append("write")
        return 50.0, 20.0, 200.0, 201.0

    receipt = (50.0, 20.0, 100.0, 8 if case == "old_generation" else 7,
               "old" if case == "old_session" else "current")
    service = SimpleNamespace(
        _calibration_last_write=receipt, _require_calibration_lifecycle=require,
        pump_service=SimpleNamespace(
            read_run_state=lambda: SimpleNamespace(ok=case != "read_failed", parsed_reply={}),
            are_required_channels_running=lambda *_args: (case != "stopped", "stopped"),
            get_current_q_state=read_flows,
        ),
        _flow_matches=OrchestratorService._flow_matches,
        _log=lambda _message: None, _update_plant_calibration_experiment=lambda **_kw: None,
        _apply_calibration_flow=apply,
    )
    kwargs = dict(baseline_q1=50.0, baseline_q2=20.0, generation=7,
                  token=SimpleNamespace(session_id="current"), next_trial_id="q1-plus")
    if case in {"read_failed", "stopped", "cancelled_during_read"}:
        with pytest.raises(RuntimeError):
            OrchestratorService._restore_plant_calibration_baseline(service, **kwargs)
    else:
        assert OrchestratorService._restore_plant_calibration_baseline(service, **kwargs) == (50.0, 20.0)
    assert calls.count("write") == writes
    if case == "verified":
        assert calls == ["read"]
        assert service._calibration_last_write == receipt  # Do not restart physical waiting.


@pytest.mark.parametrize("start,source_start,expected", [(5, 5, "detected_stable"), (8, 4, None), (8, 8, "detected_stable")])
def test_early_response_uses_post_response_size_sources_without_extra_wait(start, source_start, expected):
    samples = [replace(item, sample_start_monotonic=source_start) for item in observations(start, .1, 65)]
    decision, _ = OrchestratorService._calibration_response_decision(
        samples, baseline_diameter_um=60, response_threshold_um=.5,
        stable_sample_count=5, minimum_observation_count=30, stability_tolerance_um=.2,
        pixel_to_micron=.1, noise_reference=tuple(observations(0, 1, 60, 5)),
        eligible_after=30, confirmed_response_after=5,
        low_response_after=60, stability_duration_s=3,
    )
    assert decision == expected
