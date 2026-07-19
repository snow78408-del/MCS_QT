from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .feature_builder import FEATURE_NAMES, NONLINEAR_FEATURE_NAMES, TARGET_NAMES, build_features, build_nonlinear_features
from .models import DisturbancePrediction, DisturbanceSample


@dataclass(slots=True)
class LinearDisturbanceModel:
    version: str
    feature_names: list[str]
    target_names: list[str]
    coefficients: list[list[float]]
    intercepts: list[float]
    confidence: float
    model_type: str = "quadratic_ridge"

    def predict_values(self, features: list[float]) -> list[float]:
        model_features = build_nonlinear_features(features) if self.model_type == "quadratic_ridge" else features
        values: list[float] = []
        for coeffs, intercept in zip(self.coefficients, self.intercepts):
            values.append(float(intercept) + sum(float(a) * float(b) for a, b in zip(coeffs, model_features)))
        return values

    def predict(self, sample: DisturbanceSample, previous_diameter: float | None = None) -> DisturbancePrediction:
        values = self.predict_values(build_features(sample))
        mapping = dict(zip(self.target_names, values))
        predicted_diameter = mapping.get("future_droplet_mean_diameter_um", mapping.get("droplet_mean_diameter_um"))
        predicted_change = mapping.get("future_diameter_change_um")
        if predicted_change is not None:
            change = float(predicted_change)
        elif predicted_diameter is not None and previous_diameter is not None:
            change = float(predicted_diameter) - float(previous_diameter)
        elif predicted_diameter is not None and sample.droplet_mean_diameter_um is not None:
            change = float(predicted_diameter) - float(sample.droplet_mean_diameter_um)
        else:
            change = 0.0
        recommended = -0.5 * change
        return DisturbancePrediction(
            timestamp=time.time(),
            model_ready=True,
            model_valid=True,
            confidence=float(self.confidence),
            predicted_diameter_um=predicted_diameter,
            predicted_diameter_change_um=change,
            predicted_response_delay_ms=float(
                mapping.get("response_delay_ms", mapping.get("pump_response_delay_ms", sample.pump_response_delay_ms or 0.0))
            ),
            predicted_cv=mapping.get("future_droplet_cv", mapping.get("droplet_cv")),
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
        feature_names = list(data.get("feature_names", FEATURE_NAMES))
        target_names = list(data.get("target_names", TARGET_NAMES))
        model_type = str(data.get("model_type", "linear"))
        expected_coeff_len = len(NONLINEAR_FEATURE_NAMES) if model_type == "quadratic_ridge" else len(FEATURE_NAMES)
        if feature_names != list(FEATURE_NAMES) or target_names != list(TARGET_NAMES):
            return None
        coefficients = [list(map(float, row)) for row in data["coefficients"]]
        if any(len(row) != expected_coeff_len for row in coefficients):
            return None
        return cls(
            version=str(data["version"]),
            feature_names=feature_names,
            target_names=target_names,
            coefficients=coefficients,
            intercepts=list(map(float, data["intercepts"])),
            confidence=float(data.get("confidence", 0.0)),
            model_type=model_type,
        )
