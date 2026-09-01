from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable

from .calibration import PlantCalibrationRecord


@dataclass(frozen=True, slots=True)
class PlantCalibrationExperimentConfig:
    """Operator-approved settings for a bounded visual step experiment."""

    plant_id: str
    chip_id: str
    fluid_id: str
    pump_model: str
    syringe_profile: str
    q1_step: float
    q2_step: float
    repetitions: int = 2
    baseline_sample_count: int = 5
    stable_sample_count: int = 5
    # A step with no detectable single-droplet onset is still a usable
    # low-response observation.  This legacy-named field is the minimum
    # number of valid droplets before a stable response may be classified;
    # an unstable response keeps collecting beyond this count.
    response_observation_limit: int = 30
    # Kept for schema compatibility. Zero means there is no experiment
    # duration limit; progress is driven by valid droplet events.
    maximum_step_duration_s: float = 0.0
    # This is a sensor-liveness safety watchdog, not an experiment deadline.
    vision_liveness_timeout_s: float = 120.0
    minimum_response_um: float = 0.5
    stability_tolerance_um: float = 1.0
    sample_poll_interval_s: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "plant_id",
            "chip_id",
            "fluid_id",
            "pump_model",
            "syringe_profile",
        ):
            if not str(getattr(self, name) or "").strip():
                raise ValueError(f"calibration experiment {name} is required")
        for name in (
            "q1_step",
            "q2_step",
            "vision_liveness_timeout_s",
            "minimum_response_um",
            "stability_tolerance_um",
            "sample_poll_interval_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"calibration experiment {name} must be finite and positive")
        if not math.isfinite(float(self.maximum_step_duration_s)) or float(self.maximum_step_duration_s) < 0.0:
            raise ValueError("calibration experiment maximum_step_duration_s must be finite and nonnegative")
        if int(self.repetitions) < 1:
            raise ValueError("calibration experiment repetitions must be positive")
        if int(self.baseline_sample_count) < 3 or int(self.stable_sample_count) < 3:
            raise ValueError("calibration experiment sample counts must be at least 3")
        if int(self.response_observation_limit) < int(self.stable_sample_count):
            raise ValueError("response_observation_limit must cover the stable sample window")
        gain_ratio = max(float(self.q1_step), float(self.q2_step)) / min(
            float(self.q1_step), float(self.q2_step)
        )
        if gain_ratio > 10.0:
            raise ValueError("Q1/Q2 calibration step ratio must not exceed 10")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PlantCalibrationObservation:
    frame_id: int
    capture_monotonic: float
    observed_monotonic: float
    diameter_um: float
    droplet_count: int
    droplet_id: int = 0
    control_period_id: int = 0
    pixel_to_micron: float = 0.0
    scale_source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PlantCalibrationMeasurement:
    trial_id: str
    channel: str
    direction: int
    baseline_q1: float
    baseline_q2: float
    commanded_q1: float
    commanded_q2: float
    actual_q1: float
    actual_q2: float
    actuator_step: float | None
    command_started_monotonic: float
    readback_completed_monotonic: float
    response_started_monotonic: float
    response_stable_monotonic: float
    baseline_diameter_um: float
    steady_diameter_um: float
    diameter_change_um: float
    response_delay_ms: float
    baseline_observations: tuple[PlantCalibrationObservation, ...] = field(default_factory=tuple)
    response_observations: tuple[PlantCalibrationObservation, ...] = field(default_factory=tuple)
    response_detected: bool = True
    response_classification: str = "detected_stable"

    def __post_init__(self) -> None:
        if self.channel not in {"q1", "q2", "combined"}:
            raise ValueError("calibration measurement channel is invalid")
        if int(self.direction) not in {-1, 1}:
            raise ValueError("calibration measurement direction must be -1 or +1")
        if not math.isfinite(float(self.response_delay_ms)) or self.response_delay_ms <= 0.0:
            raise ValueError("calibration response delay must be finite and positive")
        if self.response_classification not in {"detected_stable","below_detection_threshold"}:
            raise ValueError("calibration response classification is invalid")
        if self.response_detected != (self.response_classification == "detected_stable"):
            raise ValueError("calibration response detection fields disagree")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PlantCalibrationExperimentResult:
    record: PlantCalibrationRecord
    config: PlantCalibrationExperimentConfig
    measurements: tuple[PlantCalibrationMeasurement, ...]
    q1_sensitivity_um_per_flow: float
    q2_sensitivity_um_per_flow: float
    started_at: str
    completed_at: str
    session_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "record": self.record.to_dict(),
            "config": self.config.to_dict(),
            "measurements": [item.to_dict() for item in self.measurements],
            "q1_sensitivity_um_per_flow": self.q1_sensitivity_um_per_flow,
            "q2_sensitivity_um_per_flow": self.q2_sensitivity_um_per_flow,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "session_id": self.session_id,
        }


def _finite_median(values: Iterable[float], *, name: str) -> float:
    usable = [float(value) for value in values if math.isfinite(float(value))]
    if not usable:
        raise ValueError(f"calibration has no finite {name} values")
    return float(median(usable))


def _direction_consistency(values: list[float], expected_sign: float) -> float:
    if not values:
        return 0.0
    total_weight = sum(abs(float(value)) for value in values)
    if total_weight <= 1e-12:
        return 0.0
    matching_weight = sum(
        abs(float(value))
        for value in values
        if float(value) * float(expected_sign) > 0.0
    )
    return matching_weight / total_weight


def _paired_central_sensitivities(
    measurements: Iterable[PlantCalibrationMeasurement],
    *,
    input_value: Callable[[PlantCalibrationMeasurement], float],
) -> list[float]:
    """Cancel baseline drift by comparing the +/- trials of each repetition."""
    items = list(measurements)
    positive = [item for item in items if int(item.direction) > 0]
    negative = [item for item in items if int(item.direction) < 0]
    values: list[float] = []
    for plus, minus in zip(positive, negative):
        input_span = float(input_value(plus)) - float(input_value(minus))
        if abs(input_span) <= 1e-9:
            continue
        output_span = float(plus.steady_diameter_um) - float(minus.steady_diameter_um)
        values.append(output_span / input_span)
    return values


def identify_channel_sensitivities(
    measurements: Iterable[PlantCalibrationMeasurement],
) -> tuple[float, float]:
    items = list(measurements)
    individual: dict[str, list[float]] = {"q1": [], "q2": []}
    for item in items:
        if item.channel == "q1":
            flow_change = float(item.actual_q1) - float(item.baseline_q1)
        elif item.channel == "q2":
            flow_change = float(item.actual_q2) - float(item.baseline_q2)
        else:
            continue
        if abs(flow_change) <= 1e-9:
            raise ValueError(f"{item.channel} calibration step has zero verified flow change")
        individual[item.channel].append(float(item.diameter_change_um) / flow_change)

    q1_items = [item for item in items if item.channel == "q1"]
    q2_items = [item for item in items if item.channel == "q2"]
    paired = {
        "q1": _paired_central_sensitivities(q1_items, input_value=lambda item: item.actual_q1),
        "q2": _paired_central_sensitivities(q2_items, input_value=lambda item: item.actual_q2),
    }
    channel_items = {"q1": q1_items, "q2": q2_items}
    sensitivities: dict[str, list[float]] = {}
    identified: dict[str, float] = {}
    for name in ("q1", "q2"):
        values = paired[name] if paired[name] else individual[name]
        sensitivities[name] = values
        # A stable change below the visual threshold is a valid negative
        # observation.  It must not be turned into an actuator direction by
        # taking the sign of sub-resolution noise.
        if not any(item.response_detected for item in channel_items[name]):
            identified[name] = 0.0
        else:
            identified[name] = _finite_median(values, name=f"{name.upper()} sensitivity")
    q1 = identified["q1"]
    q2 = identified["q2"]
    if abs(q1) > 1e-9 and _direction_consistency(sensitivities["q1"], q1) < 0.75:
        raise ValueError("Q1 step responses do not have a consistent direction")
    if abs(q2) > 1e-9 and _direction_consistency(sensitivities["q2"], q2) < 0.75:
        raise ValueError("Q2 step responses do not have a consistent direction")
    if abs(q1) <= 1e-9 and abs(q2) <= 1e-9:
        raise ValueError("Q1 and Q2 responses are both below the visual detection threshold")
    return q1, q2


def build_plant_calibration_result(
    *,
    config: PlantCalibrationExperimentConfig,
    measurements: Iterable[PlantCalibrationMeasurement],
    session_id: str,
    started_at: str,
    q1_min: float,
    q1_max: float,
    q2_min: float,
    q2_max: float,
    total_flow_max: float,
    min_q1_q2_gap: float,
) -> PlantCalibrationExperimentResult:
    items = tuple(measurements)
    q1_sensitivity, q2_sensitivity = identify_channel_sensitivities(items)
    combined = [item for item in items if item.channel == "combined"]
    if len(combined) < int(config.repetitions) * 2:
        raise ValueError("combined calibration step responses are incomplete")

    combined_sensitivities = _paired_central_sensitivities(
        combined,
        input_value=lambda item: float(item.actuator_step or 0.0),
    )
    individual_combined_sensitivities: list[float] = []
    for item in combined:
        actuator_step = float(item.actuator_step or 0.0)
        if abs(actuator_step) <= 1e-9:
            raise ValueError("combined calibration actuator step is zero")
        individual_combined_sensitivities.append(float(item.diameter_change_um) / actuator_step)
    if not combined_sensitivities:
        combined_sensitivities = individual_combined_sensitivities
    plant_sensitivity = _finite_median(
        combined_sensitivities,
        name="combined actuator sensitivity",
    )
    if plant_sensitivity <= 1e-9:
        raise ValueError("combined actuator direction is inconsistent with the identified pump signs")
    if _direction_consistency(combined_sensitivities, plant_sensitivity) < 0.75:
        raise ValueError("combined step responses do not have a consistent direction")

    delays = [float(item.response_delay_ms) for item in combined if item.response_detected]
    if not delays:
        raise ValueError("combined calibration produced no detectable response onset for delay identification")
    delay_median = _finite_median(delays, name="response delay")
    absolute_deviations = [abs(value - delay_median) for value in delays]
    delay_uncertainty = max(
        1.4826 * _finite_median(absolute_deviations, name="delay deviation"),
        max(delays) - delay_median,
        0.0,
    )
    base_step = min(float(config.q1_step), float(config.q2_step))
    q1_gain = float(config.q1_step) / base_step if abs(q1_sensitivity) > 1e-9 else 0.0
    q2_gain = float(config.q2_step) / base_step if abs(q2_sensitivity) > 1e-9 else 0.0
    observed_q1 = [value for item in items for value in (item.baseline_q1, item.actual_q1)]
    observed_q2 = [value for item in items for value in (item.baseline_q2, item.actual_q2)]
    identified_q1_min = max(float(q1_min), min(observed_q1))
    identified_q1_max = min(float(q1_max), max(observed_q1))
    identified_q2_min = max(float(q2_min), min(observed_q2))
    identified_q2_max = min(float(q2_max), max(observed_q2))
    identified_total_flow_max = min(
        float(total_flow_max),
        max(
            float(item.actual_q1) + float(item.actual_q2)
            for item in items
        ),
    )
    completed_at = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    calibration_id = f"plant-cal-{stamp}-{str(session_id)[:8] or uuid.uuid4().hex[:8]}"
    record = PlantCalibrationRecord(
        schema_version=1,
        calibration_id=calibration_id,
        created_at=completed_at,
        plant_id=str(config.plant_id).strip(),
        chip_id=str(config.chip_id).strip(),
        fluid_id=str(config.fluid_id).strip(),
        pump_model=str(config.pump_model).strip(),
        syringe_profile=str(config.syringe_profile).strip(),
        response_delay_median_ms=delay_median,
        response_delay_uncertainty_ms=delay_uncertainty,
        diameter_sensitivity_um_per_output=plant_sensitivity,
        q1_control_sign=(1.0 if q1_sensitivity > 0.0 else -1.0 if q1_sensitivity < 0.0 else 0.0),
        q2_control_sign=(1.0 if q2_sensitivity > 0.0 else -1.0 if q2_sensitivity < 0.0 else 0.0),
        q1_output_gain=q1_gain,
        q2_output_gain=q2_gain,
        q1_min=identified_q1_min,
        q1_max=identified_q1_max,
        q2_min=identified_q2_min,
        q2_max=identified_q2_max,
        total_flow_max=identified_total_flow_max,
        min_q1_q2_gap=float(min_q1_q2_gap),
        measurement_source="automated_visual_step_response",
    )
    return PlantCalibrationExperimentResult(
        record=record,
        config=config,
        measurements=items,
        q1_sensitivity_um_per_flow=q1_sensitivity,
        q2_sensitivity_um_per_flow=q2_sensitivity,
        started_at=str(started_at),
        completed_at=completed_at,
        session_id=str(session_id),
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def save_plant_calibration_result(
    result: PlantCalibrationExperimentResult,
    calibration_path: str | Path,
) -> dict[str, Any]:
    path = Path(calibration_path).expanduser()
    if not path.suffix:
        path = path.with_suffix(".json")
    audit_path = path.with_name(f"{path.stem}.measurements.json")
    # Publish the loadable record last. If the audit write fails, callers never
    # see a calibration file that has lost its supporting measurements.
    _write_json_atomic(audit_path, result.to_dict())
    _write_json_atomic(path, result.record.to_dict())
    return {
        "path": str(path.resolve()),
        "measurements_path": str(audit_path.resolve()),
        "record": result.record.to_dict(),
        "measurement_count": len(result.measurements),
    }
