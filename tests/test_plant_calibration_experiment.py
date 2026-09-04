from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

from backend.pid_control import (
    PIDConfig,
    PlantCalibrationExperimentConfig,
    PlantCalibrationMeasurement,
    PlantCalibrationObservation,
    build_plant_calibration_result,
    identify_channel_sensitivities,
    identify_channel_log_sensitivities,
    save_plant_calibration_result,
)
from backend.pid_control.calibration import load_plant_calibration
from backend.orchestrator.models import SystemConfig
from backend.orchestrator.models import RecognitionSnapshot
from backend.orchestrator.service import OrchestratorService
from backend.orchestrator.state import SystemState
from backend.pump_hardware.models import PumpOperationResult


def _config() -> PlantCalibrationExperimentConfig:
    return PlantCalibrationExperimentConfig(
        plant_id="rig-a",
        chip_id="chip-a",
        fluid_id="water-oil-a",
        pump_model="pump-a",
        syringe_profile="10ml-glass",
        q1_step=2.0,
        q2_step=1.0,
        repetitions=2,
    )


def _measurement(
    trial_id: str,
    channel: str,
    direction: int,
    diameter_change: float,
    *,
    delay_ms: float = 1200.0,
) -> PlantCalibrationMeasurement:
    q1_change = direction * 2.0 if channel == "q1" else 0.0
    q2_change = direction * 1.0 if channel == "q2" else 0.0
    actuator_step = None
    if channel in {"combined", "validation"}:
        fraction = 1.0 if channel == "combined" else 0.6
        q1_change = -direction * 2.0 * fraction
        q2_change = direction * 1.0 * fraction
        actuator_step = float(direction) * fraction
    command_started = 10.0
    return PlantCalibrationMeasurement(
        trial_id=trial_id,
        channel=channel,
        direction=direction,
        baseline_q1=50.0,
        baseline_q2=20.0,
        commanded_q1=50.0 + q1_change,
        commanded_q2=20.0 + q2_change,
        actual_q1=50.0 + q1_change,
        actual_q2=20.0 + q2_change,
        actuator_step=actuator_step,
        command_started_monotonic=command_started,
        readback_completed_monotonic=10.2,
        response_started_monotonic=command_started + delay_ms / 1000.0,
        response_stable_monotonic=13.0,
        baseline_diameter_um=60.0,
        steady_diameter_um=60.0 + diameter_change,
        diameter_change_um=diameter_change,
        response_delay_ms=delay_ms,
    )


def _measurements() -> list[PlantCalibrationMeasurement]:
    items: list[PlantCalibrationMeasurement] = []
    for repeat in range(2):
        for direction in (1, -1):
            items.append(
                _measurement(
                    f"q1-{repeat}-{direction}",
                    "q1",
                    direction,
                    -direction * 2.0,
                )
            )
            items.append(
                _measurement(
                    f"q2-{repeat}-{direction}",
                    "q2",
                    direction,
                    direction * 1.5,
                )
            )
            items.append(
                _measurement(
                    f"combined-{repeat}-{direction}",
                    "combined",
                    direction,
                    direction * 3.5,
                    delay_ms=1200.0 + repeat * 100.0 + (25.0 if direction < 0 else 0.0),
                )
            )
    return items


def test_step_measurements_identify_channel_signs_and_combined_gain() -> None:
    measurements = _measurements()

    q1_sensitivity, q2_sensitivity = identify_channel_sensitivities(measurements)
    result = build_plant_calibration_result(
        config=_config(),
        measurements=measurements,
        session_id="session-12345678",
        started_at="2026-08-30T10:00:00+00:00",
        q1_min=15.0,
        q1_max=100.0,
        q2_min=5.0,
        q2_max=25.0,
        total_flow_max=125.0,
        min_q1_q2_gap=1.0,
    )

    assert q1_sensitivity == pytest.approx(-1.0)
    assert q2_sensitivity == pytest.approx(1.5)
    assert result.record.q1_control_sign == -1.0
    assert result.record.q2_control_sign == 1.0
    log_q1,log_q2=identify_channel_log_sensitivities(measurements)
    assert result.record.schema_version == 3
    assert result.record.measurement_region == "generation"
    assert result.record.q1_log_diameter_sensitivity == pytest.approx(log_q1)
    assert result.record.q2_log_diameter_sensitivity == pytest.approx(log_q2)
    assert result.record.q1_output_gain > 0.0
    assert result.record.q2_output_gain > 0.0
    assert result.record.q1_output_gain / result.record.q2_output_gain != pytest.approx(2.0)
    assert result.record.controller_kp > 0.0
    assert result.record.controller_ki >= 0.0
    assert result.record.diameter_sensitivity_um_per_output == pytest.approx(3.5)
    assert result.record.response_delay_median_ms > 0.0
    assert result.record.response_delay_uncertainty_ms >= 0.0
    assert result.record.q1_min == pytest.approx(48.0)
    assert result.record.q1_max == pytest.approx(52.0)
    assert result.record.q2_min == pytest.approx(19.0)
    assert result.record.q2_max == pytest.approx(21.0)
    assert result.record.total_flow_max == pytest.approx(72.0)


def test_full_response_fit_and_independent_validation_authorize_predictive_model() -> None:
    time_constant_ms = 1000.0

    def with_curve(item: PlantCalibrationMeasurement) -> PlantCalibrationMeasurement:
        observations = []
        for index, elapsed_ms in enumerate((200.0, 700.0, 1200.0, 1600.0, 2200.0, 3200.0, 4500.0), start=1):
            active_ms = max(0.0, elapsed_ms - float(item.response_delay_ms))
            fraction = 1.0 - math.exp(-active_ms / time_constant_ms)
            observations.append(
                PlantCalibrationObservation(
                    frame_id=index,
                    capture_monotonic=item.command_started_monotonic + elapsed_ms / 1000.0,
                    observed_monotonic=item.command_started_monotonic + elapsed_ms / 1000.0,
                    diameter_um=item.baseline_diameter_um + item.diameter_change_um * fraction,
                    droplet_count=1,
                    droplet_id=index,
                    generation_frequency_hz=25.0,
                    diameter_cv=0.02,
                )
            )
        return replace(item, response_observations=tuple(observations))

    measurements = [
        with_curve(item) if item.channel == "combined" else item
        for item in _measurements()
    ]
    for direction in (1, -1):
        validation = _measurement(
            f"validation-{direction}",
            "validation",
            direction,
            direction * 2.1,
            delay_ms=1260.0,
        )
        measurements.append(with_curve(validation))

    result = build_plant_calibration_result(
        config=_config(),
        measurements=measurements,
        session_id="session-12345678",
        started_at="2026-08-30T10:00:00+00:00",
        q1_min=15.0,
        q1_max=100.0,
        q2_min=5.0,
        q2_max=25.0,
        total_flow_max=125.0,
        min_q1_q2_gap=1.0,
    )

    assert result.record.model_fit_method == "robust_fopdt_grid"
    assert result.record.response_time_constant_ms == pytest.approx(time_constant_ms, rel=0.35)
    assert result.record.model_fit_nrmse < 0.1
    assert result.record.validation_sample_count == 14
    assert result.record.validation_nrmse < 0.1
    assert result.record.validated_for_pi
    assert result.record.validated_for_mpc
    assert result.record.authorized_for_pi
    assert result.record.continuous_phase_oil == "fluorinated oil"
    assert result.record.surfactant_concentration_percent == pytest.approx(2.0)
    assert result.record.surfactant_concentration_basis == "unspecified"
    assert result.record.aqueous_phase == "water"
    assert result.record.flow_measurement_kind == "device_parameter_readback"
    assert result.record.baseline_generation_frequency_hz == 0.0


def test_paired_steps_cancel_baseline_drift_that_flips_individual_responses() -> None:
    measurements = _measurements()
    q1_changes = iter((-2.0, -1.0, 1.0, 2.0))
    drifted = []
    for item in measurements:
        if item.channel != "q1":
            drifted.append(item)
            continue
        diameter_change = next(q1_changes)
        drifted.append(
            replace(
                item,
                diameter_change_um=diameter_change,
                steady_diameter_um=item.baseline_diameter_um + diameter_change,
            )
        )

    q1_sensitivity, q2_sensitivity = identify_channel_sensitivities(drifted)

    assert q1_sensitivity == pytest.approx(-0.25)
    assert q2_sensitivity == pytest.approx(1.5)


def test_calibration_save_keeps_loadable_record_and_separate_raw_audit() -> None:
    result = build_plant_calibration_result(
        config=_config(),
        measurements=_measurements(),
        session_id="session-12345678",
        started_at="2026-08-30T10:00:00+00:00",
        q1_min=15.0,
        q1_max=100.0,
        q2_min=5.0,
        q2_max=25.0,
        total_flow_max=125.0,
        min_q1_q2_gap=1.0,
    )

    stem = f".plant-calibration-test-{uuid.uuid4().hex}"
    record_path = Path.cwd() / f"{stem}.json"
    audit_path = Path.cwd() / f"{stem}.measurements.json"
    try:
        saved = save_plant_calibration_result(result, record_path)

        loaded = load_plant_calibration(saved["path"])
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        assert loaded.calibration_id == result.record.calibration_id
        assert saved["measurement_count"] == 12
        assert len(audit["measurements"]) == 12
        assert audit["record"]["measurement_source"] == "generation_zone_volume_step_response"
    finally:
        record_path.unlink(missing_ok=True)
        audit_path.unlink(missing_ok=True)


def test_calibration_config_rejects_missing_physical_identity() -> None:
    with pytest.raises(ValueError, match="chip_id"):
        PlantCalibrationExperimentConfig(
            plant_id="rig-a",
            chip_id="",
            fluid_id="water-oil-a",
            pump_model="pump-a",
            syringe_profile="10ml-glass",
            q1_step=2.0,
            q2_step=1.0,
        )


def test_calibration_response_limit_must_cover_stable_window() -> None:
    with pytest.raises(ValueError,match="response_observation_limit"):
        PlantCalibrationExperimentConfig(
            plant_id="rig-a",chip_id="chip-a",fluid_id="water-oil-a",
            pump_model="pump-a",syringe_profile="10ml-glass",
            q1_step=2.0,q2_step=1.0,stable_sample_count=10,
            response_observation_limit=9,
        )


def test_low_response_measurement_is_retained_but_not_used_as_delay_onset() -> None:
    measurements=_measurements()
    measurements[-1]=replace(
        measurements[-1],diameter_change_um=0.1,steady_diameter_um=60.1,
        response_delay_ms=9999.0,response_detected=False,
        response_classification="below_detection_threshold",
    )

    result=build_plant_calibration_result(
        config=_config(),measurements=measurements,session_id="session-12345678",
        started_at="2026-08-30T10:00:00+00:00",q1_min=15.0,q1_max=100.0,
        q2_min=5.0,q2_max=25.0,total_flow_max=125.0,min_q1_q2_gap=1.0,
    )

    assert result.measurements[-1].response_classification=="below_detection_threshold"
    assert result.record.response_delay_median_ms < 9999.0


def test_channel_with_only_stable_low_responses_is_disabled_not_given_a_noise_direction() -> None:
    measurements=[]
    for item in _measurements():
        if item.channel != "q2":
            measurements.append(item)
            continue
        measurements.append(replace(
            item,
            steady_diameter_um=60.05,
            diameter_change_um=0.05,
            response_detected=False,
            response_classification="below_detection_threshold",
        ))

    q1_sensitivity,q2_sensitivity=identify_channel_sensitivities(measurements)
    result=build_plant_calibration_result(
        config=_config(),measurements=measurements,session_id="session-12345678",
        started_at="2026-08-30T10:00:00+00:00",q1_min=15.0,q1_max=100.0,
        q2_min=5.0,q2_max=25.0,total_flow_max=125.0,min_q1_q2_gap=1.0,
    )

    assert q1_sensitivity == pytest.approx(-1.0)
    assert q2_sensitivity == 0.0
    assert result.record.q2_control_sign == 0.0
    assert result.record.q2_output_gain == 0.0
    assert result.record.q1_output_gain > 0.0


def test_calibration_uses_unique_crossing_droplets_not_repeated_frame_average() -> None:
    snapshot=RecognitionSnapshot(
        frame_droplet_count=2,total_droplet_count=2,new_crossing_count=2,
        avg_diameter=61.0,single_cell_rate=100.0,valid_for_control=True,
        timestamp=1.0,reason="ok",droplet_count=2,active_droplet_count=2,
        has_droplet=True,control_reason="ok",frame_id=18,capture_monotonic=12.0,
        crossed_track_diameters={7:60.0,8:62.0},
        crossed_track_capture_monotonic={7:11.8,8:11.9},
        crossed_track_frame_ids={7:16,8:17},
    )

    observations=OrchestratorService._plant_calibration_observations(snapshot)

    assert [item.droplet_id for item in observations]==[7,8]
    assert [item.diameter_um for item in observations]==[60.0,62.0]
    assert [item.capture_monotonic for item in observations]==[11.8,11.9]
    assert all(item.droplet_count==1 for item in observations)


def test_calibration_observation_keeps_the_live_scale_provenance() -> None:
    snapshot=RecognitionSnapshot(
        frame_droplet_count=1,total_droplet_count=1,new_crossing_count=1,
        avg_diameter=61.0,single_cell_rate=100.0,valid_for_control=True,
        timestamp=2.0,reason="ok",droplet_count=1,active_droplet_count=1,
        has_droplet=True,control_reason="ok",
        frame_id=2,capture_monotonic=2.0,pixel_to_micron=0.69,
        scale_source="channel_430um",crossed_track_diameters={9:61.0},
    )

    observation=OrchestratorService._plant_calibration_observations(snapshot)[0]

    assert observation.pixel_to_micron == pytest.approx(0.69)
    assert observation.scale_source == "channel_430um"


def test_calibration_stability_uses_droplet_median_drift_not_raw_range() -> None:
    values=[100.0,110.0,104.0,106.0,110.0,100.0]
    observations=[
        OrchestratorService._plant_calibration_observations(
            RecognitionSnapshot(
                frame_droplet_count=1,total_droplet_count=index,new_crossing_count=1,
                avg_diameter=value,single_cell_rate=100.0,valid_for_control=True,
                timestamp=float(index),reason="ok",droplet_count=index,active_droplet_count=1,
                has_droplet=True,control_reason="ok",frame_id=index,capture_monotonic=float(index),
                crossed_track_diameters={index:value},
            )
        )[0]
        for index,value in enumerate(values,start=1)
    ]

    assert max(values)-min(values)>1.0
    assert OrchestratorService._calibration_droplet_window_is_stable(observations,1.0)


def test_calibration_stability_tolerance_respects_resolution_and_measured_noise() -> None:
    values=[100.0,100.5,100.0,100.5,100.0]
    observations=[
        PlantCalibrationObservation(
            frame_id=index,capture_monotonic=float(index),observed_monotonic=float(index),
            diameter_um=value,droplet_count=1,droplet_id=index,
        )
        for index,value in enumerate(values,start=1)
    ]

    effective=OrchestratorService._calibration_effective_stability_tolerance_um(
        observations,
        configured_tolerance_um=0.05,
        pixel_to_micron=0.69,
    )

    assert effective == pytest.approx(1.38)
    assert OrchestratorService._calibration_droplet_window_is_stable(
        observations,
        0.05,
        0.69,
    )


def test_response_threshold_cannot_be_lower_than_live_pixel_resolution() -> None:
    observations=tuple(
        PlantCalibrationObservation(
            frame_id=index,capture_monotonic=float(index),observed_monotonic=float(index),
            diameter_um=60.0,droplet_count=1,droplet_id=index,pixel_to_micron=0.69,
        )
        for index in range(1,6)
    )

    threshold=OrchestratorService._calibration_response_threshold_um(
        observations,minimum_response_um=0.5,pixel_to_micron=0.69,
    )

    assert threshold == pytest.approx(1.38)


def test_response_minimum_count_is_not_a_hard_stop_when_tail_is_unstable() -> None:
    baseline=tuple(
        PlantCalibrationObservation(
            frame_id=index,capture_monotonic=float(index),observed_monotonic=float(index),
            diameter_um=60.0,droplet_count=1,droplet_id=index,pixel_to_micron=0.1,
        )
        for index in range(1,6)
    )
    first_thirty=[
        PlantCalibrationObservation(
            frame_id=index,capture_monotonic=float(index),observed_monotonic=float(index),
            diameter_um=(60.0 if index <= 25 else 60.0 + index - 25),
            droplet_count=1,droplet_id=index,pixel_to_micron=0.1,
        )
        for index in range(1,31)
    ]

    decision,_tail=OrchestratorService._calibration_response_decision(
        first_thirty,baseline_diameter_um=60.0,response_threshold_um=0.5,
        stable_sample_count=5,minimum_observation_count=30,stability_tolerance_um=0.2,
        pixel_to_micron=0.1,noise_reference=baseline,
    )
    continued=first_thirty+[
        PlantCalibrationObservation(
            frame_id=index,capture_monotonic=float(index),observed_monotonic=float(index),
            diameter_um=65.0,droplet_count=1,droplet_id=index,pixel_to_micron=0.1,
        )
        for index in range(31,36)
    ]
    final_decision,tail=OrchestratorService._calibration_response_decision(
        continued,baseline_diameter_um=60.0,response_threshold_um=0.5,
        stable_sample_count=5,minimum_observation_count=30,stability_tolerance_um=0.2,
        pixel_to_micron=0.1,noise_reference=baseline,
    )

    assert decision is None
    assert final_decision == "detected_stable"
    assert len(tail) == 5


def test_stable_sub_threshold_tail_is_a_valid_low_response() -> None:
    baseline=tuple(
        PlantCalibrationObservation(
            frame_id=index,capture_monotonic=float(index),observed_monotonic=float(index),
            diameter_um=60.0,droplet_count=1,droplet_id=index,pixel_to_micron=0.1,
        )
        for index in range(1,6)
    )
    observations=[
        PlantCalibrationObservation(
            frame_id=index,capture_monotonic=float(index),observed_monotonic=float(index),
            diameter_um=60.1,droplet_count=1,droplet_id=index,pixel_to_micron=0.1,
        )
        for index in range(1,31)
    ]

    decision,tail=OrchestratorService._calibration_response_decision(
        observations,baseline_diameter_um=60.0,response_threshold_um=0.5,
        stable_sample_count=5,minimum_observation_count=30,stability_tolerance_um=0.2,
        pixel_to_micron=0.1,noise_reference=baseline,
    )

    assert decision == "below_detection_threshold"
    assert len(tail) == 5


def test_orchestrator_runs_calibration_under_exclusive_state_and_stops_pump() -> None:
    class PumpStub:
        def __init__(self) -> None:
            self.runtime_config = SimpleNamespace(min_q1_q2_gap=1.0)
            self.calls: list[str] = []

        def start_infusion_and_verify(self, channels) -> PumpOperationResult:
            self.calls.append(f"start:{list(channels)}")
            return PumpOperationResult(ok=True, verified=True)

        def stop_system_and_verify(self) -> PumpOperationResult:
            self.calls.append("stop")
            return PumpOperationResult(ok=True, verified=True)

    class VisionStub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def set_run_context(self, session_id: str, generation: int) -> None:
            self.calls.append(f"context:{session_id}:{generation}")

        def start(self) -> None:
            self.calls.append("start")

        def stop(self) -> None:
            self.calls.append("stop")

    pump = PumpStub()
    vision = VisionStub()
    service = OrchestratorService(
        vision_service=SimpleNamespace(),
        vision_adapter=vision,
        pump_service=pump,
    )
    service._cfg = SystemConfig(
        target_diameter=60.0,
        pixel_to_micron=1.0,
        video_source_type="camera",
        video_source="camera-a",
        initial_q1=100.0,
        initial_q2=10.0,
        control_interval_ms=7500,
        pump_port="COM3",
    )
    service._state = SystemState.INITIALIZED
    service._pump_control_enabled = True
    service._pump_state.connected = True
    service._pump_state.comm_established = True
    service._pump_state.q1 = 100.0
    service._pump_state.q2 = 10.0
    service._pump_state.q1_actual = 100.0
    service._pump_state.q2_actual = 10.0
    service.run_preflight_check = lambda: {"ok": True, "issues": []}
    service._sync_pump_flow_readback = lambda _source: True
    calibration_sequence: list[tuple[str, float, float] | tuple[str, str]] = []

    def fake_apply(q1, q2, _generation, _token):
        calibration_sequence.append(("baseline", float(q1), float(q2)))
        return float(q1), float(q2), 1.0, 1.1

    service._apply_calibration_flow = fake_apply

    def fake_trial(**kwargs):
        calibration_sequence.append(("trial", str(kwargs["trial_id"])))
        channel = kwargs["channel"]
        direction = int(kwargs["direction"])
        if channel == "q1":
            change = -direction * 2.0
        elif channel == "q2":
            change = direction * 1.5
        else:
            change = float(kwargs["actuator_step"]) * 3.5
        return _measurement(
            kwargs["trial_id"],
            channel,
            direction,
            change,
        )

    service._run_plant_calibration_trial = fake_trial
    try:
        result = service.run_plant_calibration_experiment(_config())

        assert result.record.diameter_sensitivity_um_per_output == pytest.approx(3.5)
        assert service._state == SystemState.STOPPED
        assert service.get_snapshot().plant_calibration_experiment["status"] == "completed"
        assert len(calibration_sequence) == 28
        assert result.record.validated_for_pi
        for index in range(0, len(calibration_sequence), 2):
            baseline_entry = calibration_sequence[index]
            assert baseline_entry[0] == "baseline"
            assert baseline_entry[1:] == pytest.approx((100.0, 10.0))
            assert calibration_sequence[index + 1][0] == "trial"
        assert pump.calls[0] == "start:[1, 2]"
        assert "stop" in pump.calls
        assert vision.calls[-1] == "stop"
    finally:
        service.close_background_services()
        service._safety.shutdown()


def test_calibration_moves_boundary_flow_to_an_interior_step_baseline() -> None:
    service = object.__new__(OrchestratorService)
    service.pid_config = PIDConfig()

    baseline = service._plant_calibration_baseline_with_step_margin(
        _config(),
        20.0,
        5.0,
    )

    assert baseline == pytest.approx((22.0,6.0))


def test_calibration_start_failure_still_issues_verified_stop() -> None:
    class PumpStub:
        runtime_config = SimpleNamespace(min_q1_q2_gap=1.0)

        def __init__(self) -> None:
            self.stop_calls = 0

        @staticmethod
        def start_infusion_and_verify(_channels) -> PumpOperationResult:
            return PumpOperationResult(ok=False, reason="start readback lost")

        def stop_system_and_verify(self) -> PumpOperationResult:
            self.stop_calls += 1
            return PumpOperationResult(ok=True, verified=True)

    pump = PumpStub()
    vision = SimpleNamespace(
        set_run_context=lambda _session, _generation: None,
        start=lambda: None,
        stop=lambda: None,
    )
    service = OrchestratorService(
        vision_service=SimpleNamespace(),
        vision_adapter=vision,
        pump_service=pump,
    )
    service._cfg = SystemConfig(
        target_diameter=60.0,
        pixel_to_micron=1.0,
        video_source_type="camera",
        video_source="camera-a",
        initial_q1=50.0,
        initial_q2=20.0,
        control_interval_ms=7500,
        pump_port="COM3",
    )
    service._state = SystemState.INITIALIZED
    service._pump_control_enabled = True
    service._pump_state.connected = True
    service._pump_state.comm_established = True
    service.run_preflight_check = lambda: {"ok": True, "issues": []}
    try:
        with pytest.raises(RuntimeError, match="start readback lost"):
            service.run_plant_calibration_experiment(_config())

        assert pump.stop_calls >= 1
        assert service._state == SystemState.ERROR
        assert service._safety.snapshot().stop_verified
    finally:
        service.close_background_services()
        service._safety.shutdown()
