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
        self._candidate_model: LinearDisturbanceModel | None = None
        self._last_prediction = DisturbancePrediction(timestamp=0.0, reason="model not ready")

    @property
    def model(self) -> LinearDisturbanceModel | None:
        return self._model

    @property
    def model_version(self) -> str:
        return self._model.version if self._model is not None else ""

    @property
    def candidate_model(self) -> LinearDisturbanceModel | None:
        return self._candidate_model

    @property
    def candidate_model_version(self) -> str:
        return self._candidate_model.version if self._candidate_model is not None else ""

    def set_model(self, model: LinearDisturbanceModel) -> None:
        model.save(self.config.model_path)
        self._previous_model = self._model
        self._model = model

    def set_candidate_model(self, model: LinearDisturbanceModel) -> None:
        self._candidate_model = model

    def discard_candidate_model(self) -> None:
        self._candidate_model = None

    def promote_candidate_model(self) -> LinearDisturbanceModel:
        candidate = self._candidate_model
        if candidate is None:
            raise RuntimeError("no candidate disturbance model to promote")
        candidate.save(self.config.model_path)
        self._previous_model = self._model
        self._model = candidate
        self._candidate_model = None
        return candidate

    def predict(self, sample: DisturbanceSample | None) -> DisturbancePrediction:
        if sample is None:
            return DisturbancePrediction(timestamp=time.time(), reason="no sample")
        if self._model is None:
            return DisturbancePrediction(timestamp=time.time(), reason="model not ready")
        try:
            self._last_prediction = self._model.predict(sample, previous_diameter=sample.droplet_mean_diameter_um)
            self._last_prediction.prediction_horizon_ms = float(self.config.prediction_horizon_ms)
            return self._last_prediction
        except Exception as exc:
            if self._previous_model is not None:
                rollback_model = self._previous_model
                self._model = rollback_model
                self._previous_model = None
                persistence_note = ""
                try:
                    rollback_model.save(self.config.model_path)
                except Exception as save_exc:
                    persistence_note = f"; rollback persistence failed: {save_exc}"
                self._last_prediction = self._model.predict(sample, previous_diameter=sample.droplet_mean_diameter_um)
                self._last_prediction.reason = (
                    f"rolled back after prediction error: {exc}{persistence_note}"
                )
                return self._last_prediction
            return DisturbancePrediction(timestamp=time.time(), reason=f"prediction error: {exc}")

    def get_last_prediction(self) -> DisturbancePrediction:
        age_ms = (time.time() - self._last_prediction.timestamp) * 1000.0 if self._last_prediction.timestamp else float("inf")
        if age_ms > float(self.config.prediction_timeout_ms):
            return DisturbancePrediction(timestamp=time.time(), reason="prediction stale")
        return self._last_prediction

    def predict_candidate(self, sample: DisturbanceSample | None) -> DisturbancePrediction:
        if sample is None:
            return DisturbancePrediction(timestamp=time.time(), reason="no sample")
        candidate = self._candidate_model
        if candidate is None:
            return DisturbancePrediction(timestamp=time.time(), reason="candidate model not ready")
        try:
            prediction = candidate.predict(
                sample,
                previous_diameter=sample.droplet_mean_diameter_um,
            )
            prediction.prediction_horizon_ms = float(self.config.prediction_horizon_ms)
            prediction.reason = "candidate shadow prediction"
            return prediction
        except Exception as exc:
            return DisturbancePrediction(
                timestamp=time.time(),
                model_version=candidate.version,
                reason=f"candidate prediction error: {exc}",
            )
