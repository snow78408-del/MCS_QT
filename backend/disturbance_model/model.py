from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .feature_builder import FEATURE_NAMES, TARGET_NAMES, build_features
from .models import DisturbancePrediction, DisturbanceSample


@dataclass(slots=True)
class LinearDisturbanceModel:
    version: str
    feature_names: list[str]
    target_names: list[str]
    coefficients: list[list[float]]
    intercepts: list[float]
    confidence: float

    def predict_values(self, features: list[float]) -> list[float]:
        values: list[float] = []
        for coeffs, intercept in zip(self.coefficients, self.intercepts):
            values.append(float(intercept) + sum(float(a) * float(b) for a, b in zip(coeffs, features)))
        return values

    def predict(self, sample: DisturbanceSample, previous_diameter: float | None = None) -> DisturbancePrediction:
        values = self.predict_values(build_features(sample))
        mapping = dict(zip(self.target_names, values))
        predicted_diameter = mapping.get("droplet_mean_diameter_um")
        change = 0.0
        if predicted_diameter is not None and previous_diameter is not None:
            change = float(predicted_diameter) - float(previous_diameter)
        elif predicted_diameter is not None and sample.droplet_mean_diameter_um is not None:
            change = float(predicted_diameter) - float(sample.droplet_mean_diameter_um)
        recommended = -0.5 * change
        return DisturbancePrediction(
            timestamp=time.time(),
            model_ready=True,
            model_valid=True,
            confidence=float(self.confidence),
            predicted_diameter_um=predicted_diameter,
            predicted_diameter_change_um=change,
            predicted_response_delay_ms=float(mapping.get("pump_response_delay_ms", sample.pump_response_delay_ms or 0.0)),
            predicted_cv=mapping.get("droplet_cv"),
            disturbance_effect="increase" if change > 0 else "decrease" if change < 0 else "neutral",
            recommended_feedforward=recommended,
            model_version=self.version,
            reason="model prediction",
        )

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "LinearDisturbanceModel | None":
        p = Path(path)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            version=str(data["version"]),
            feature_names=list(data.get("feature_names", FEATURE_NAMES)),
            target_names=list(data.get("target_names", TARGET_NAMES)),
            coefficients=[list(map(float, row)) for row in data["coefficients"]],
            intercepts=list(map(float, data["intercepts"])),
            confidence=float(data.get("confidence", 0.0)),
        )
