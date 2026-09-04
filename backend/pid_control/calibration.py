from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from ..pump_hardware.invariants import STRICT_Q1_Q2_GAP_UL_MIN


@dataclass(frozen=True, slots=True)
class PlantCalibrationRecord:
    """Versioned plant identification used to authorize predictive control."""

    schema_version: int
    calibration_id: str
    created_at: str
    plant_id: str
    chip_id: str
    fluid_id: str
    pump_model: str
    syringe_profile: str
    response_delay_median_ms: float
    response_delay_uncertainty_ms: float
    diameter_sensitivity_um_per_output: float
    q1_control_sign: float
    q2_control_sign: float
    q1_output_gain: float
    q2_output_gain: float
    q1_min: float
    q1_max: float
    q2_min: float
    q2_max: float
    total_flow_max: float
    min_q1_q2_gap: float = STRICT_Q1_Q2_GAP_UL_MIN
    measurement_source: str = "visual_step_response"
    measurement_region: str = "observation"
    channel_height_um: float = 0.0
    channel_width_um: float = 0.0
    volume_correction_factor: float = 1.0
    baseline_q1: float = 0.0
    baseline_q2: float = 0.0
    baseline_diameter_um: float = 0.0
    q1_log_diameter_sensitivity: float = 0.0
    q2_log_diameter_sensitivity: float = 0.0
    sensitivity_allocation_regularization: float = 0.01
    response_time_constant_ms: float = 0.0
    controller_kp: float = 0.0
    controller_ki: float = 0.0
    controller_kd: float = 0.0
    continuous_phase_oil: str = ""
    surfactant_name: str = ""
    surfactant_concentration_percent: float = 0.0
    surfactant_concentration_basis: str = "unspecified"
    aqueous_phase: str = "water"
    temperature_c: float = 25.0
    q1_response_delay_ms: float = 0.0
    q2_response_delay_ms: float = 0.0
    response_time_constant_uncertainty_ms: float = 0.0
    q1_log_sensitivity_uncertainty: float = 0.0
    q2_log_sensitivity_uncertainty: float = 0.0
    model_fit_method: str = "legacy_threshold"
    model_fit_mae_um: float = 0.0
    model_fit_nrmse: float = 0.0
    validation_mae_um: float = 0.0
    validation_nrmse: float = 0.0
    validation_sample_count: int = 0
    validated_for_pi: bool = False
    validated_for_mpc: bool = False
    baseline_generation_frequency_hz: float = 0.0
    baseline_diameter_cv: float = 0.0
    flow_measurement_kind: str = "device_parameter_readback"

    def __post_init__(self) -> None:
        if int(self.schema_version) not in {1, 2, 3}:
            raise ValueError("unsupported plant calibration schema_version")
        for name in (
            "calibration_id",
            "created_at",
            "plant_id",
            "chip_id",
            "fluid_id",
            "pump_model",
            "syringe_profile",
            "measurement_source",
        ):
            if not str(getattr(self, name) or "").strip():
                raise ValueError(f"plant calibration {name} is required")
        try:
            datetime.fromisoformat(str(self.created_at).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("plant calibration created_at must be ISO-8601") from exc
        numeric_names = (
            "response_delay_median_ms",
            "response_delay_uncertainty_ms",
            "diameter_sensitivity_um_per_output",
            "q1_control_sign",
            "q2_control_sign",
            "q1_output_gain",
            "q2_output_gain",
            "q1_min",
            "q1_max",
            "q2_min",
            "q2_max",
            "total_flow_max",
            "min_q1_q2_gap",
            "channel_height_um",
            "channel_width_um",
            "volume_correction_factor",
            "baseline_q1",
            "baseline_q2",
            "baseline_diameter_um",
            "q1_log_diameter_sensitivity",
            "q2_log_diameter_sensitivity",
            "sensitivity_allocation_regularization",
            "response_time_constant_ms",
            "controller_kp",
            "controller_ki",
            "controller_kd",
            "surfactant_concentration_percent",
            "temperature_c",
            "q1_response_delay_ms",
            "q2_response_delay_ms",
            "response_time_constant_uncertainty_ms",
            "q1_log_sensitivity_uncertainty",
            "q2_log_sensitivity_uncertainty",
            "model_fit_mae_um",
            "model_fit_nrmse",
            "validation_mae_um",
            "validation_nrmse",
            "baseline_generation_frequency_hz",
            "baseline_diameter_cv",
        )
        for name in numeric_names:
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"plant calibration {name} must be finite")
        if self.response_delay_median_ms <= 0.0:
            raise ValueError("plant response delay must be measured and positive")
        if self.response_delay_uncertainty_ms < 0.0:
            raise ValueError("plant response delay uncertainty must be non-negative")
        if abs(self.diameter_sensitivity_um_per_output) < 1e-9:
            raise ValueError("plant diameter sensitivity must be non-zero")
        for channel, sign, gain in (
            ("Q1", self.q1_control_sign, self.q1_output_gain),
            ("Q2", self.q2_control_sign, self.q2_output_gain),
        ):
            if float(sign) not in {-1.0, 0.0, 1.0}:
                raise ValueError(f"plant {channel} control sign must be -1, 0, or +1")
            if float(gain) < 0.0:
                raise ValueError(f"plant {channel} output gain must be non-negative")
            if (float(sign) == 0.0) != (float(gain) == 0.0):
                raise ValueError(
                    f"plant {channel} direction and gain must both be zero when the channel is inactive"
                )
        if self.q1_output_gain == 0.0 and self.q2_output_gain == 0.0:
            raise ValueError("plant calibration must retain at least one active control channel")
        if not 0.0 < self.q1_min < self.q1_max or not 0.0 < self.q2_min < self.q2_max:
            raise ValueError("plant flow bounds must be positive and ordered")
        if self.total_flow_max <= 0.0:
            raise ValueError("plant total-flow limit must be positive")
        if self.min_q1_q2_gap < STRICT_Q1_Q2_GAP_UL_MIN:
            raise ValueError("plant phase gap is weaker than the hardware invariant")
        if self.q1_max < self.q2_min + self.min_q1_q2_gap:
            raise ValueError("plant calibration contains no feasible phase-flow region")
        if int(self.schema_version) >= 2:
            if self.measurement_region != "generation":
                raise ValueError("schema v2+ calibration must measure the generation region")
            if self.channel_height_um <= 0.0 or self.channel_width_um <= 0.0:
                raise ValueError("generation channel dimensions must be positive")
            if self.volume_correction_factor <= 0.0:
                raise ValueError("volume correction factor must be positive")
            if self.baseline_q1 <= 0.0 or self.baseline_q2 <= 0.0 or self.baseline_diameter_um <= 0.0:
                raise ValueError("generation calibration baseline must be positive")
            if (
                abs(self.q1_log_diameter_sensitivity) <= 1e-12
                and abs(self.q2_log_diameter_sensitivity) <= 1e-12
            ):
                raise ValueError("generation calibration has no active diameter response")
            if self.sensitivity_allocation_regularization <= 0.0:
                raise ValueError("sensitivity allocation regularization must be positive")
            if self.response_time_constant_ms <= 0.0:
                raise ValueError("response time constant must be identified")
            if self.controller_kp <= 0.0 or self.controller_ki < 0.0 or self.controller_kd != 0.0:
                raise ValueError("generation controller must contain a conservative PI tuning")
        if int(self.schema_version) >= 3:
            if not str(self.continuous_phase_oil or "").strip():
                raise ValueError("continuous-phase oil identity is required")
            if not str(self.aqueous_phase or "").strip():
                raise ValueError("aqueous-phase identity is required")
            if self.surfactant_concentration_percent < 0.0:
                raise ValueError("surfactant concentration must be non-negative")
            if self.surfactant_concentration_basis not in {
                "w/w",
                "v/v",
                "w/v",
                "unspecified",
            }:
                raise ValueError("surfactant concentration basis is invalid")
            if not -50.0 <= self.temperature_c <= 150.0:
                raise ValueError("calibration temperature is outside the supported range")
            for name in (
                "q1_response_delay_ms",
                "q2_response_delay_ms",
                "response_time_constant_uncertainty_ms",
                "q1_log_sensitivity_uncertainty",
                "q2_log_sensitivity_uncertainty",
                "model_fit_mae_um",
                "model_fit_nrmse",
                "validation_mae_um",
                "validation_nrmse",
                "baseline_generation_frequency_hz",
                "baseline_diameter_cv",
            ):
                if float(getattr(self, name)) < 0.0:
                    raise ValueError(f"plant calibration {name} must be non-negative")
            if int(self.validation_sample_count) < 0:
                raise ValueError("validation_sample_count must be non-negative")
            if self.validated_for_pi and int(self.validation_sample_count) <= 0:
                raise ValueError("PI authorization requires independent validation samples")
            if self.validated_for_mpc and (
                not self.validated_for_pi
                or self.model_fit_method != "robust_fopdt_grid"
            ):
                raise ValueError("MPC authorization requires validated robust FOPDT fitting")
            if self.model_fit_method not in {"legacy_threshold", "robust_fopdt_grid"}:
                raise ValueError("model_fit_method is invalid")
            if self.flow_measurement_kind not in {
                "device_parameter_readback",
                "physical_flow_sensor",
            }:
                raise ValueError("flow_measurement_kind is invalid")

    @property
    def conservative_response_delay_ms(self) -> float:
        return float(self.response_delay_median_ms + self.response_delay_uncertainty_ms)

    @property
    def feedforward_gain(self) -> float:
        return 1.0 / float(self.diameter_sensitivity_um_per_output)

    @property
    def has_generation_control_mapping(self) -> bool:
        return bool(
            int(self.schema_version) >= 2
            and self.measurement_region == "generation"
            and (
                abs(float(self.q1_log_diameter_sensitivity)) > 1e-12
                or abs(float(self.q2_log_diameter_sensitivity)) > 1e-12
            )
        )

    @property
    def authorized_for_pi(self) -> bool:
        return bool(int(self.schema_version) >= 3 and self.validated_for_pi)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PlantCalibrationRecord":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown plant calibration fields: {', '.join(unknown)}")
        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_plant_calibration(path: str | Path) -> PlantCalibrationRecord:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("plant calibration JSON root must be an object")
    return PlantCalibrationRecord.from_mapping(payload)
