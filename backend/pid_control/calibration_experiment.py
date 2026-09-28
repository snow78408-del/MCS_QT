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

import numpy as np

from .calibration import PlantCalibrationRecord, build_calibration_record


class CalibrationIdentificationError(ValueError):
    """Completed observations do not identify usable dynamics; not a pump fault."""


class CalibrationValidationError(CalibrationIdentificationError):
    """A candidate exists but is not a deployable calibration."""


def require_combined_response(measurements: Iterable[PlantCalibrationMeasurement]) -> None:
    combined = [item for item in measurements if item.channel == "combined"]
    if not any(item.response_detected for item in combined):
        changes = "、".join(f"{item.diameter_change_um:+.3f}" for item in combined)
        raise CalibrationIdentificationError(
            "本次响应不足，无法完成动力学辨识：组合阶跃均未检测到可信响应起点，"
            f"不能估计响应延迟。直径变化：{changes or '无组合观测'} μm。"
            "稳定无响应不等于故障或零增益；请检查观察时长、测量分辨率及工况后重新标定。"
        )


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
    repetitions: int = 1
    baseline_sample_count: int = 5
    stable_sample_count: int = 5
    # A step with no detectable single-droplet onset is still a usable
    # low-response observation.  This legacy-named field is the minimum
    # number of valid droplets before a stable response may be classified;
    # an unstable response keeps collecting beyond this count.
    response_observation_limit: int = 30
    # Zero uses the orchestrator's finite observation + instability deadline.
    # A positive value explicitly bounds the post-transaction response phase.
    maximum_step_duration_s: float = 0.0
    # This is a sensor-liveness safety watchdog, not an experiment deadline.
    vision_liveness_timeout_s: float = 120.0
    minimum_response_um: float = 0.5
    stability_tolerance_um: float = 1.0
    sample_poll_interval_s: float = 0.05
    channel_height_um: float = 50.0
    channel_width_um: float = 50.0
    volume_correction_factor: float = 1.0
    sensitivity_allocation_regularization: float = 0.01
    closed_loop_time_constant_ratio: float = 2.5
    continuous_phase_oil: str = "fluorinated oil"
    surfactant_name: str = ""
    surfactant_concentration_percent: float = 2.0
    surfactant_concentration_basis: str = "unspecified"
    aqueous_phase: str = "water"
    temperature_c: float = 25.0
    validation_repetitions: int = 1
    validation_step_fraction: float = 0.6
    validation_mae_limit_um: float = 2.0
    validation_nrmse_limit: float = 0.25
    # Operator-visible observation horizons, not measured plant constants.
    minimum_response_wait_s: float = 30.0
    low_response_wait_s: float = 60.0
    stability_duration_s: float = 3.0
    baseline_wait_s: float | None = None
    maximum_attempts: int = 1
    require_mpc_validation: bool = False
    identification_model: str = "quadratic_response"

    @property
    def effective_baseline_wait_s(self) -> float:
        # Older callers used the response horizon for both phases.
        return self.minimum_response_wait_s if self.baseline_wait_s is None else self.baseline_wait_s

    def __post_init__(self) -> None:
        if self.identification_model not in {"quadratic_response", "legacy_local_linear"}:
            raise ValueError("unsupported calibration identification_model")
        if isinstance(self.maximum_attempts, bool) or not isinstance(self.maximum_attempts, int) or not 1 <= self.maximum_attempts <= 5:
            raise ValueError("maximum_attempts must be an integer in [1, 5]")
        if not isinstance(self.require_mpc_validation, bool):
            raise ValueError("require_mpc_validation must be boolean")
        if self.baseline_wait_s is not None and (
            not math.isfinite(float(self.baseline_wait_s)) or self.baseline_wait_s < 0.0
        ):
            raise ValueError("baseline_wait_s must be finite and nonnegative")
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
            "channel_height_um",
            "channel_width_um",
            "volume_correction_factor",
            "sensitivity_allocation_regularization",
            "closed_loop_time_constant_ratio",
            "validation_step_fraction",
            "validation_mae_limit_um",
            "validation_nrmse_limit",
            "minimum_response_wait_s",
            "low_response_wait_s",
            "stability_duration_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"calibration experiment {name} must be finite and positive")
        if not math.isfinite(float(self.maximum_step_duration_s)) or float(self.maximum_step_duration_s) < 0.0:
            raise ValueError("calibration experiment maximum_step_duration_s must be finite and nonnegative")
        if int(self.repetitions) < 1:
            raise ValueError("calibration experiment repetitions must be positive")
        if self.low_response_wait_s < self.minimum_response_wait_s:
            raise ValueError("low-response observation horizon must cover the minimum response wait")
        if int(self.validation_repetitions) < 1:
            raise ValueError("calibration experiment needs at least one validation repetition")
        if not 0.1 <= float(self.validation_step_fraction) <= 1.0:
            raise ValueError("validation_step_fraction must be in [0.1, 1.0]")
        if not 0.0 < float(self.validation_nrmse_limit) <= 1.0:
            raise ValueError("validation_nrmse_limit must be in (0, 1]")
        if not str(self.continuous_phase_oil or "").strip():
            raise ValueError("continuous-phase oil identity is required")
        if not str(self.aqueous_phase or "").strip():
            raise ValueError("aqueous-phase identity is required")
        if (
            not math.isfinite(float(self.surfactant_concentration_percent))
            or float(self.surfactant_concentration_percent) < 0.0
        ):
            raise ValueError("surfactant concentration must be non-negative")
        if self.surfactant_concentration_basis not in {"w/w", "v/v", "w/v", "unspecified"}:
            raise ValueError("surfactant concentration basis is invalid")
        if not math.isfinite(float(self.temperature_c)) or not -50.0 <= float(self.temperature_c) <= 150.0:
            raise ValueError("calibration temperature is outside the supported range")
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
    measurement_region: str = "generation"
    volume_correction_factor: float = 1.0
    generation_frequency_hz: float = 0.0
    diameter_cv: float = 0.0
    sample_start_monotonic: float | None = None

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
        if self.channel not in {"q1", "q2", "combined", "validation"}:
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
    attempt_history: tuple[dict[str, Any], ...] = ()

    @property
    def accepted(self) -> bool:
        return bool(self.record.validated_for_pi
                    and self.record.validation_sample_count > 0
                    and self.record.validation_mae_um <= self.config.validation_mae_limit_um
                    and self.record.validation_nrmse <= self.config.validation_nrmse_limit and (
            self.record.validated_for_mpc or not self.config.require_mpc_validation
        ))

    def require_accepted(self) -> None:
        PlantCalibrationRecord.from_mapping(self.record.to_dict())
        if not self.accepted:
            fit_detail = ""
            if self.record.nonlinear_model is not None:
                fit_detail = (
                    f"非线性训练 NRMSE={self.record.nonlinear_model['training_nrmse']:.3f}，"
                    f"动态拟合 NRMSE={self.record.model_fit_nrmse:.3f}，"
                    f"各项上限={self.config.validation_nrmse_limit:g}。"
                )
            raise CalibrationValidationError(
                f"标定尚未成功：PI {'通过' if self.record.validated_for_pi else '未通过'}，"
                f"MPC {'通过' if self.record.validated_for_mpc else '未通过'}；"
                f"验证 MAE={self.record.validation_mae_um:.3f}/{self.config.validation_mae_limit_um:g} μm，"
                f"NRMSE={self.record.validation_nrmse:.3f}/{self.config.validation_nrmse_limit:g}。"
                f"{fit_detail}保留 MPC 建模数据，但不能应用正式 PID 标定。"
            )

    @property
    def diagnostics(self) -> dict[str, Any]:
        return calibration_diagnostics(self.measurements, self.record)

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
            "diagnostics": self.diagnostics,
            "attempt_history": list(self.attempt_history),
        }


def _finite_median(values: Iterable[float], *, name: str) -> float:
    usable = [float(value) for value in values if math.isfinite(float(value))]
    if not usable:
        raise ValueError(f"calibration has no finite {name} values")
    return float(median(usable))


def _robust_uncertainty(values: Iterable[float]) -> float:
    usable = [float(value) for value in values if math.isfinite(float(value))]
    if len(usable) < 2:
        return 0.0
    center = float(median(usable))
    return 1.4826 * float(median(abs(value - center) for value in usable))


@dataclass(frozen=True, slots=True)
class _FOPDTFit:
    delay_ms: float
    time_constant_ms: float
    time_constant_uncertainty_ms: float
    mae_um: float
    nrmse: float
    sample_count: int
    method: str


def _fit_fopdt(measurements: Iterable[PlantCalibrationMeasurement], *, nonlinear: bool = False) -> _FOPDTFit:
    """Fit one FOPDT shape to all identified-direction response curves.

    Each trial retains its own measured baseline and steady-state amplitude;
    the shared delay and time constant are selected with a robust normalized
    absolute-error grid search. This avoids deriving dynamics from a single
    threshold-crossing droplet.
    """
    measurements = tuple(measurements)
    trials = [
        item
        for item in measurements
        if (item.channel != "validation" if nonlinear else item.channel == "combined" and item.response_detected)
        and abs(float(item.diameter_change_um)) > 1e-9
        and len(item.response_observations) >= 3
    ]
    if not trials:
        if nonlinear:
            raise CalibrationIdentificationError("已完成采集，但动态曲线不足；保留 MPC 建模数据，不能据此整定 PID")
        items = list(measurements)
        delays = [item.response_delay_ms for item in items if item.channel == "combined" and item.response_detected]
        return _FOPDTFit(
            delay_ms=_finite_median(delays, name="response delay"),
            time_constant_ms=_estimate_response_time_constant_ms(items),
            time_constant_uncertainty_ms=0.0,
            mae_um=0.0,
            nrmse=0.0,
            sample_count=0,
            method="legacy_threshold",
        )

    points: list[tuple[float, float, float, float]] = []
    trial_slices: list[slice] = []
    positive_gaps_ms: list[float] = []
    trial_tau_estimates: list[float] = []
    for item in trials:
        trial_start = len(points)
        ordered = sorted(item.response_observations, key=lambda obs: obs.capture_monotonic)
        previous_time: float | None = None
        for observation in ordered:
            elapsed_ms = max(
                0.0,
                (float(observation.capture_monotonic) - float(item.command_started_monotonic))
                * 1000.0,
            )
            points.append(
                (
                    elapsed_ms,
                    float(observation.diameter_um),
                    float(item.baseline_diameter_um),
                    float(item.diameter_change_um),
                )
            )
            if previous_time is not None and elapsed_ms > previous_time:
                positive_gaps_ms.append(elapsed_ms - previous_time)
            previous_time = elapsed_ms
        trial_slices.append(slice(trial_start, len(points)))
        target = float(item.baseline_diameter_um) + 0.6321205588 * float(item.diameter_change_um)
        direction = 1.0 if item.diameter_change_um >= 0.0 else -1.0
        reached = next(
            (
                obs
                for obs in ordered
                if direction * (float(obs.diameter_um) - target) >= 0.0
            ),
            None,
        )
        if reached is not None:
            estimate = (
                float(reached.capture_monotonic)
                - float(item.command_started_monotonic)
            ) * 1000.0 - float(item.response_delay_ms)
            if estimate > 0.0:
                trial_tau_estimates.append(estimate)

    elapsed = np.asarray([point[0] for point in points], dtype=float)
    observed = np.asarray([point[1] for point in points], dtype=float)
    baselines = np.asarray([point[2] for point in points], dtype=float)
    amplitudes = np.asarray([point[3] for point in points], dtype=float)
    scales = np.maximum(np.abs(amplitudes), 0.25)
    maximum_elapsed = max(1.0, float(np.max(elapsed)))
    observed_delay = _finite_median(
        (item.response_delay_ms for item in trials),
        name="response delay",
    )
    delay_high = maximum_elapsed * 0.8 if nonlinear else min(maximum_elapsed * 0.8, max(observed_delay * 2.0, 1.0))
    delay_candidates = np.linspace(0.0, max(1.0, delay_high), 61)
    sampling_ms = float(median(positive_gaps_ms)) if positive_gaps_ms else maximum_elapsed / 20.0
    tau_low = max(1.0, sampling_ms * 0.25)
    tau_high = max(tau_low * 2.0, maximum_elapsed * 2.0)
    tau_candidates = np.geomspace(tau_low, tau_high, 81)

    best_loss = math.inf
    best_delay = observed_delay
    best_tau = max(1.0, maximum_elapsed / 3.0)
    for delay in delay_candidates:
        active_time = np.maximum(0.0, elapsed - float(delay))
        for time_constant in tau_candidates:
            response_fraction = 1.0 - np.exp(-active_time / float(time_constant))
            predicted = baselines + amplitudes * response_fraction
            normalized = np.abs(predicted - observed) / scales
            # Robust within a trial, equal weight across trials. A densely
            # sampled long curve must not drown out a shorter contrary trial.
            loss = float(np.mean([np.median(normalized[part]) for part in trial_slices]))
            if loss < best_loss:
                best_loss = loss
                best_delay = float(delay)
                best_tau = float(time_constant)

    active_time = np.maximum(0.0, elapsed - best_delay)
    predicted = baselines + amplitudes * (1.0 - np.exp(-active_time / best_tau))
    residual = predicted - observed
    mae = float(np.mean(np.abs(residual)))
    rms = float(np.sqrt(np.mean(residual * residual)))
    normalization = max(1e-9, float(np.median(np.abs(amplitudes))))
    return _FOPDTFit(
        delay_ms=best_delay,
        time_constant_ms=best_tau,
        time_constant_uncertainty_ms=_robust_uncertainty(trial_tau_estimates),
        mae_um=mae,
        nrmse=rms / normalization,
        sample_count=len(points),
        method="robust_fopdt_grid",
    )


def _validation_metrics(
    measurements: Iterable[PlantCalibrationMeasurement],
    *,
    delay_ms: float,
    time_constant_ms: float,
    steady_gain_um_per_output: float,
    predict_change: Callable[[PlantCalibrationMeasurement], float] | None = None,
) -> tuple[float, float, int]:
    residuals: list[float] = []
    scales: list[float] = []
    for item in measurements:
        if item.channel != "validation":
            continue
        predicted_change = (predict_change(item) if predict_change is not None
                            else steady_gain_um_per_output * float(item.actuator_step or 0.0))
        observations = tuple(item.response_observations)
        if observations:
            for observation in observations:
                elapsed_ms = max(
                    0.0,
                    (float(observation.capture_monotonic) - float(item.command_started_monotonic))
                    * 1000.0,
                )
                active_ms = max(0.0, elapsed_ms - float(delay_ms))
                fraction = 1.0 - math.exp(-active_ms / max(1.0, float(time_constant_ms)))
                prediction = float(item.baseline_diameter_um) + predicted_change * fraction
                residuals.append(prediction - float(observation.diameter_um))
                scales.append(max(abs(predicted_change), 0.25))
        else:
            prediction = float(item.baseline_diameter_um) + predicted_change
            residuals.append(prediction - float(item.steady_diameter_um))
            scales.append(max(abs(predicted_change), 0.25))
    if not residuals:
        return 0.0, 0.0, 0
    values = np.asarray(residuals, dtype=float)
    scale = max(1e-9, float(np.median(np.asarray(scales, dtype=float))))
    return (
        float(np.mean(np.abs(values))),
        float(np.sqrt(np.mean(values * values))) / scale,
        len(residuals),
    )


def calibration_diagnostics(
    measurements: Iterable[PlantCalibrationMeasurement],
    record: PlantCalibrationRecord | None = None,
) -> dict[str, Any]:
    """Audit every trial; never filter held-out data according to fit residuals."""
    items = tuple(measurements)
    channels: dict[str, Any] = {}
    for channel in ("q1", "q2", "combined", "validation"):
        selected = [item for item in items if item.channel == channel]
        baselines = [item.baseline_diameter_um for item in selected]
        channels[channel] = {
            "trial_count": len(selected),
            "detected_count": sum(item.response_detected for item in selected),
            "baseline_span_um": max(baselines) - min(baselines) if baselines else None,
            "repeated_direction_changes_um": {
                str(direction): [item.diameter_change_um for item in selected if item.direction == direction]
                for direction in (1, -1)
            },
        }
    trials = []
    for item in items:
        entry: dict[str, Any] = {
            "trial_id": item.trial_id, "channel": item.channel,
            "response_classification": item.response_classification,
            "baseline_diameter_um": item.baseline_diameter_um,
            "observed_change_um": item.diameter_change_um,
            "response_delay_ms": item.response_delay_ms if item.response_detected else None,
            "observation_duration_s": item.response_stable_monotonic - item.command_started_monotonic,
        }
        if item.channel == "validation" and record is not None:
            predictor = None
            if record.nonlinear_model is not None:
                from .nonlinear_calibration import predict_change

                predictor = lambda trial: predict_change(record.nonlinear_model, trial)
            mae, nrmse, count = _validation_metrics(
                [item], delay_ms=record.response_delay_median_ms,
                time_constant_ms=record.response_time_constant_ms,
                steady_gain_um_per_output=record.diameter_sensitivity_um_per_output,
                predict_change=predictor,
            )
            entry.update(predicted_change_um=(predictor(item) if predictor else record.diameter_sensitivity_um_per_output * float(item.actuator_step or 0)),
                         mae_um=mae, nrmse=nrmse, sample_count=count)
        trials.append(entry)
    return {"channels": channels, "trials": trials,
            "note": "未响应不等于零增益；基线跨度是跨轮变化，不能单独判定气泡或测量错误。NRMSE按预测响应幅度归一化。"}


def save_failed_calibration(
    path: str | Path, *, config: PlantCalibrationExperimentConfig,
    measurements: Iterable[PlantCalibrationMeasurement], reason: str,
    session_id: str, started_at: str, partial_trial: dict[str, Any] | None = None,
    attempt_history: Iterable[dict[str, Any]] = (),
) -> str:
    items = tuple(measurements)
    training_path = Path(path).with_name(f"{Path(path).stem}.mpc-training.json")
    model = None
    model_reason = ""
    if config.identification_model == "quadratic_response":
        from .nonlinear_calibration import fit_response_surface

        try:
            model = fit_response_surface(items)
        except (ValueError, np.linalg.LinAlgError) as exc:
            model_reason = str(exc)
    _write_json_atomic(training_path, {
        "schema_version": 1, "session_id": session_id,
        "purpose": "MPC model building data; incomplete or unqualified experiments do not authorize control",
        "loadable_calibration": False, "validated_for_mpc": False,
        "complete_collection": len(items) == config.repetitions * 6 + config.validation_repetitions * 2 and partial_trial is None,
        "reason": reason, "experiment_config": config.to_dict(),
        "units": {"flow": "uL/min", "diameter": "um", "time": "host monotonic seconds"},
        "flow_measurement_kind": "device_parameter_readback",
        "nonlinear_model": model, "model_reason": model_reason,
        "training_trials": [item.to_dict() for item in items if item.channel != "validation"],
        "validation_trials": [item.to_dict() for item in items if item.channel == "validation"],
        "partial_trial": partial_trial,
    })
    _write_json_atomic(Path(path), {
        "status": "incomplete", "loadable_calibration": False,
        "reason": reason, "session_id": session_id, "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(), "measurements": [item.to_dict() for item in items],
        "partial_trial": partial_trial, "diagnostics": calibration_diagnostics(items),
        "attempt_history": list(attempt_history),
        "mpc_training_path": str(training_path.resolve()),
    })
    return str(Path(path).resolve())


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
    """Compare local changes, so between-trial baseline shifts are not gain."""
    items = list(measurements)
    positive = [item for item in items if int(item.direction) > 0]
    negative = [item for item in items if int(item.direction) < 0]
    values: list[float] = []
    for plus, minus in zip(positive, negative):
        input_span = float(input_value(plus)) - float(input_value(minus))
        if abs(input_span) <= 1e-9:
            continue
        output_span = float(plus.diameter_change_um) - float(minus.diameter_change_um)
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
        "q1": _paired_central_sensitivities(q1_items, input_value=lambda item: item.actual_q1 - item.baseline_q1),
        "q2": _paired_central_sensitivities(q2_items, input_value=lambda item: item.actual_q2 - item.baseline_q2),
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
            # Zero disables allocation for this record; it is not evidence of
            # zero physical gain. Preserve every low-response measurement.
            identified[name] = 0.0
        else:
            identified[name] = _finite_median(values, name=f"{name.upper()} sensitivity")
    q1 = identified["q1"]
    q2 = identified["q2"]
    if abs(q1) > 1e-9 and _direction_consistency(sensitivities["q1"], q1) < 0.75:
        raise CalibrationIdentificationError("Q1 step responses do not have a consistent direction")
    if abs(q2) > 1e-9 and _direction_consistency(sensitivities["q2"], q2) < 0.75:
        raise CalibrationIdentificationError("Q2 step responses do not have a consistent direction")
    if abs(q1) <= 1e-9 and abs(q2) <= 1e-9:
        raise CalibrationIdentificationError("本次 Q1、Q2 均未辨识出有效响应；保留低响应观测，不能据此计算控制方向和增益")
    return q1, q2


def identify_channel_log_sensitivities(
    measurements: Iterable[PlantCalibrationMeasurement],
) -> tuple[float, float]:
    """Identify d(log diameter)/d(log flow) using paired +/- trials."""
    items = list(measurements)
    identified: dict[str, float] = {}
    all_values: dict[str, list[float]] = {}
    for channel in ("q1", "q2"):
        channel_items = [item for item in items if item.channel == channel]
        positives = [item for item in channel_items if item.direction > 0]
        negatives = [item for item in channel_items if item.direction < 0]
        values: list[float] = []
        for plus, minus in zip(positives, negatives):
            q_plus = plus.actual_q1 if channel == "q1" else plus.actual_q2
            q_minus = minus.actual_q1 if channel == "q1" else minus.actual_q2
            if min(q_plus, q_minus, plus.steady_diameter_um, minus.steady_diameter_um) <= 0.0:
                raise ValueError(f"{channel.upper()} log-sensitivity inputs must be positive")
            base_plus = plus.baseline_q1 if channel == "q1" else plus.baseline_q2
            base_minus = minus.baseline_q1 if channel == "q1" else minus.baseline_q2
            input_span = math.log(float(q_plus) / float(base_plus)) - math.log(float(q_minus) / float(base_minus))
            if abs(input_span) <= 1e-12:
                raise ValueError(f"{channel.upper()} calibration has zero log-flow span")
            values.append(
                (math.log(float(plus.steady_diameter_um) / float(plus.baseline_diameter_um))
                 - math.log(float(minus.steady_diameter_um) / float(minus.baseline_diameter_um)))
                / input_span
            )
        all_values[channel] = values
        if not any(item.response_detected for item in channel_items):
            identified[channel] = 0.0
        else:
            identified[channel] = _finite_median(values, name=f"{channel.upper()} log sensitivity")
        if (
            abs(identified[channel]) > 1e-12
            and _direction_consistency(values, identified[channel]) < 0.75
        ):
            raise CalibrationIdentificationError(f"{channel.upper()} log step responses do not have a consistent direction")
    if abs(identified["q1"]) <= 1e-12 and abs(identified["q2"]) <= 1e-12:
        raise ValueError("Q1 and Q2 log responses are both below the visual detection threshold")
    return identified["q1"], identified["q2"]


def _channel_log_sensitivity_uncertainty(
    measurements: Iterable[PlantCalibrationMeasurement],
    channel: str,
) -> float:
    channel_items = [item for item in measurements if item.channel == channel]
    positives = [item for item in channel_items if item.direction > 0]
    negatives = [item for item in channel_items if item.direction < 0]
    values: list[float] = []
    for plus, minus in zip(positives, negatives):
        q_plus = plus.actual_q1 if channel == "q1" else plus.actual_q2
        q_minus = minus.actual_q1 if channel == "q1" else minus.actual_q2
        base_plus = plus.baseline_q1 if channel == "q1" else plus.baseline_q2
        base_minus = minus.baseline_q1 if channel == "q1" else minus.baseline_q2
        input_span = math.log(float(q_plus) / float(base_plus)) - math.log(float(q_minus) / float(base_minus))
        if abs(input_span) > 1e-12:
            values.append(
                (math.log(float(plus.steady_diameter_um) / float(plus.baseline_diameter_um))
                 - math.log(float(minus.steady_diameter_um) / float(minus.baseline_diameter_um)))
                / input_span
            )
    return _robust_uncertainty(values)


def _estimate_response_time_constant_ms(
    measurements: Iterable[PlantCalibrationMeasurement],
) -> float:
    estimates: list[float] = []
    for item in measurements:
        if item.channel != "combined" or not item.response_detected:
            continue
        baseline = float(item.baseline_diameter_um)
        final = float(item.steady_diameter_um)
        target = baseline + 0.6321205588 * (final - baseline)
        direction = 1.0 if final >= baseline else -1.0
        reached = next(
            (
                obs
                for obs in item.response_observations
                if obs.capture_monotonic >= item.response_started_monotonic
                and direction * (obs.diameter_um - target) >= 0.0
            ),
            None,
        )
        if reached is not None:
            estimates.append(
                max(1.0, (reached.capture_monotonic - item.response_started_monotonic) * 1000.0)
            )
        else:
            estimates.append(
                max(1.0, item.response_stable_monotonic * 1000.0 - item.response_started_monotonic * 1000.0)
            )
    return _finite_median(estimates, name="response time constant")


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
    if config.identification_model == "quadratic_response":
        from .nonlinear_calibration import build_nonlinear_calibration

        return build_nonlinear_calibration(
            config=config, measurements=measurements, session_id=session_id, started_at=started_at,
            q1_min=q1_min, q1_max=q1_max, q2_min=q2_min, q2_max=q2_max,
            total_flow_max=total_flow_max, min_q1_q2_gap=min_q1_q2_gap,
        )
    items = tuple(measurements)
    model_items = tuple(item for item in items if item.channel != "validation")
    require_combined_response(model_items)
    q1_sensitivity, q2_sensitivity = identify_channel_sensitivities(items)
    q1_log_sensitivity, q2_log_sensitivity = identify_channel_log_sensitivities(items)
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
        raise CalibrationIdentificationError("combined actuator direction is inconsistent with the identified pump signs")
    if _direction_consistency(combined_sensitivities, plant_sensitivity) < 0.75:
        raise CalibrationIdentificationError("combined step responses do not have a consistent direction")

    delays = [float(item.response_delay_ms) for item in combined if item.response_detected]
    if not delays:
        require_combined_response(items)
    onset_delay_median = _finite_median(delays, name="response delay")
    fopdt_fit = _fit_fopdt(items)
    delay_median = max(1.0, (
        float(fopdt_fit.delay_ms)
        if fopdt_fit.method == "robust_fopdt_grid"
        else onset_delay_median
    ))
    absolute_deviations = [abs(value - delay_median) for value in delays]
    delay_uncertainty = max(
        1.4826 * _finite_median(absolute_deviations, name="delay deviation"),
        max(delays) - onset_delay_median,
        abs(delay_median - onset_delay_median),
        0.0,
    )
    baseline_q1 = _finite_median((item.baseline_q1 for item in model_items), name="baseline Q1")
    baseline_q2 = _finite_median((item.baseline_q2 for item in model_items), name="baseline Q2")
    baseline_diameter = _finite_median(
        (item.baseline_diameter_um for item in model_items), name="baseline diameter"
    )
    allocation_denominator = (
        q1_log_sensitivity * q1_log_sensitivity
        + q2_log_sensitivity * q2_log_sensitivity
        + float(config.sensitivity_allocation_regularization)
    )
    q1_coefficient = baseline_q1 * q1_log_sensitivity / allocation_denominator
    q2_coefficient = baseline_q2 * q2_log_sensitivity / allocation_denominator
    response_time_constant_ms = float(fopdt_fit.time_constant_ms)
    closed_loop_ms = max(
        float(config.closed_loop_time_constant_ratio) * delay_median,
        2.0 * delay_median,
    )
    controller_kp = response_time_constant_ms / (closed_loop_ms + delay_median)
    integral_time_ms = min(
        response_time_constant_ms,
        4.0 * (closed_loop_ms + delay_median),
    )
    controller_ki = controller_kp / max(0.001, integral_time_ms / 1000.0)
    observed_q1 = [value for item in model_items for value in (item.baseline_q1, item.actual_q1)]
    observed_q2 = [value for item in model_items for value in (item.baseline_q2, item.actual_q2)]
    identified_q1_min = max(float(q1_min), min(observed_q1))
    identified_q1_max = min(float(q1_max), max(observed_q1))
    identified_q2_min = max(float(q2_min), min(observed_q2))
    identified_q2_max = min(float(q2_max), max(observed_q2))
    identified_total_flow_max = min(
        float(total_flow_max),
        max(
            float(item.actual_q1) + float(item.actual_q2)
            for item in model_items
        ),
    )
    validation_mae, validation_nrmse, validation_sample_count = _validation_metrics(
        items,
        delay_ms=delay_median,
        time_constant_ms=response_time_constant_ms,
        steady_gain_um_per_output=plant_sensitivity,
    )
    validated_for_pi = bool(
        validation_sample_count > 0
        and validation_mae <= float(config.validation_mae_limit_um)
        and validation_nrmse <= float(config.validation_nrmse_limit)
    )
    validated_for_mpc = bool(
        validated_for_pi
        and fopdt_fit.method == "robust_fopdt_grid"
        and fopdt_fit.sample_count >= 12
        and fopdt_fit.nrmse <= float(config.validation_nrmse_limit)
    )
    q1_delays = [
        float(item.response_delay_ms)
        for item in items
        if item.channel == "q1" and item.response_detected
    ]
    q2_delays = [
        float(item.response_delay_ms)
        for item in items
        if item.channel == "q2" and item.response_detected
    ]
    baseline_observations = [
        observation
        for item in model_items
        for observation in item.baseline_observations
    ]
    baseline_frequency_values = [
        float(observation.generation_frequency_hz)
        for observation in baseline_observations
        if float(observation.generation_frequency_hz) > 0.0
    ]
    baseline_cv_values = [
        float(observation.diameter_cv)
        for observation in baseline_observations
        if float(observation.diameter_cv) >= 0.0
    ]
    completed_at_dt = datetime.now(timezone.utc)
    record = build_calibration_record(
        config=config,
        session_id=str(session_id),
        completed_at=completed_at_dt,
        measurement_source="generation_zone_volume_step_response",
        response_delay_median_ms=delay_median,
        response_delay_uncertainty_ms=delay_uncertainty,
        diameter_sensitivity_um_per_output=plant_sensitivity,
        q1_control_sign=(1.0 if q1_sensitivity > 0.0 else -1.0 if q1_sensitivity < 0.0 else 0.0),
        q2_control_sign=(1.0 if q2_sensitivity > 0.0 else -1.0 if q2_sensitivity < 0.0 else 0.0),
        q1_output_gain=abs(q1_coefficient),
        q2_output_gain=abs(q2_coefficient),
        q1_min=identified_q1_min,
        q1_max=identified_q1_max,
        q2_min=identified_q2_min,
        q2_max=identified_q2_max,
        total_flow_max=identified_total_flow_max,
        min_q1_q2_gap=float(min_q1_q2_gap),
        baseline_q1=baseline_q1,
        baseline_q2=baseline_q2,
        baseline_diameter_um=baseline_diameter,
        q1_log_diameter_sensitivity=q1_log_sensitivity,
        q2_log_diameter_sensitivity=q2_log_sensitivity,
        response_time_constant_ms=response_time_constant_ms,
        controller_kp=controller_kp,
        controller_ki=controller_ki,
        response_time_constant_uncertainty_ms=float(
            fopdt_fit.time_constant_uncertainty_ms
        ),
        q1_log_sensitivity_uncertainty=_channel_log_sensitivity_uncertainty(items, "q1"),
        q2_log_sensitivity_uncertainty=_channel_log_sensitivity_uncertainty(items, "q2"),
        model_fit_method=str(fopdt_fit.method),
        model_fit_mae_um=float(fopdt_fit.mae_um),
        model_fit_nrmse=float(fopdt_fit.nrmse),
        validation_mae_um=validation_mae,
        validation_nrmse=validation_nrmse,
        validation_sample_count=validation_sample_count,
        validated_for_pi=validated_for_pi,
        validated_for_mpc=validated_for_mpc,
        q1_response_delay_ms=(
            0.0 if not q1_delays else _finite_median(q1_delays, name="Q1 response delay")
        ),
        q2_response_delay_ms=(
            0.0 if not q2_delays else _finite_median(q2_delays, name="Q2 response delay")
        ),
        baseline_generation_frequency_hz=(
            0.0
            if not baseline_frequency_values
            else _finite_median(baseline_frequency_values, name="generation frequency")
        ),
        baseline_diameter_cv=(
            0.0
            if not baseline_cv_values
            else _finite_median(baseline_cv_values, name="diameter CV")
        ),
    )
    completed_at = completed_at_dt.isoformat()
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
    result.require_accepted()
    path = Path(calibration_path).expanduser()
    if not path.suffix:
        path = path.with_suffix(".json")
    audit_path = path.with_name(f"{path.stem}.measurements.json")
    training_path = path.with_name(f"{path.stem}.mpc-training.json")
    # Publish the loadable record last. If the audit write fails, callers never
    # see a calibration file that has lost its supporting measurements.
    _write_json_atomic(audit_path, result.to_dict())
    _write_json_atomic(training_path, {
        "schema_version": 1, "calibration_id": result.record.calibration_id,
        "purpose": "MPC model training and held-out validation; not controller deployment authorization",
        "validated_for_mpc": result.record.validated_for_mpc,
        "units": {"flow": "uL/min", "diameter": "um", "time": "host monotonic seconds"},
        "flow_measurement_kind": result.record.flow_measurement_kind,
        "record": result.record.to_dict(), "experiment_config": result.config.to_dict(),
        "training_trials": [item.to_dict() for item in result.measurements if item.channel != "validation"],
        "validation_trials": [item.to_dict() for item in result.measurements if item.channel == "validation"],
    })
    _write_json_atomic(path, result.record.to_dict())
    return {
        "path": str(path.resolve()),
        "measurements_path": str(audit_path.resolve()),
        "record": result.record.to_dict(),
        "measurement_count": len(result.measurements),
        "mpc_training_path": str(training_path.resolve()),
    }
