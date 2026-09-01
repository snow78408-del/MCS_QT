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

    def __post_init__(self) -> None:
        if int(self.schema_version) != 1:
            raise ValueError("unsupported plant calibration schema_version")
        for name in (
            "calibration_id",
            "created_at",
            "plant_id",
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

    @property
    def conservative_response_delay_ms(self) -> float:
        return float(self.response_delay_median_ms + self.response_delay_uncertainty_ms)

    @property
    def feedforward_gain(self) -> float:
        return 1.0 / float(self.diameter_sensitivity_um_per_output)

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
