from __future__ import annotations

import time

from .config import DisturbanceModelConfig
from .model import LinearDisturbanceModel
from .models import DisturbancePrediction, DisturbanceSample


class DisturbancePredictor:
    def __init__(self, config: DisturbanceModelConfig) -> None:
        self.config = config
        self._model: LinearDisturbanceModel | None = LinearDisturbanceModel.load(config.model_path)
        self._previous_model: LinearDisturbanceModel | None = None
        self._last_prediction = DisturbancePrediction(timestamp=0.0, reason="model not ready")

    @property
    def model(self) -> LinearDisturbanceModel | None:
        return self._model

    @property
    def model_version(self) -> str:
        return self._model.version if self._model is not None else ""

    def set_model(self, model: LinearDisturbanceModel) -> None:
        self._previous_model = self._model
        self._model = model
        model.save(self.config.model_path)

    def predict(self, sample: DisturbanceSample | None) -> DisturbancePrediction:
        if sample is None:
            return DisturbancePrediction(timestamp=time.time(), reason="no sample")
        if self._model is None:
            return DisturbancePrediction(timestamp=time.time(), reason="model not ready")
        try:
            self._last_prediction = self._model.predict(sample, previous_diameter=sample.droplet_mean_diameter_um)
            return self._last_prediction
        except Exception as exc:
            if self._previous_model is not None:
                self._model = self._previous_model
                self._last_prediction = self._model.predict(sample, previous_diameter=sample.droplet_mean_diameter_um)
                self._last_prediction.reason = f"rolled back after prediction error: {exc}"
                return self._last_prediction
            return DisturbancePrediction(timestamp=time.time(), reason=f"prediction error: {exc}")

    def get_last_prediction(self) -> DisturbancePrediction:
        age_ms = (time.time() - self._last_prediction.timestamp) * 1000.0 if self._last_prediction.timestamp else float("inf")
        if age_ms > float(self.config.prediction_timeout_ms):
            return DisturbancePrediction(timestamp=time.time(), reason="prediction stale")
        return self._last_prediction
