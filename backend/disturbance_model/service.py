from __future__ import annotations

import threading
import time
from typing import Any

from .collector import DisturbanceSampleCollector
from .config import DisturbanceModelConfig
from .models import DisturbancePrediction, DisturbanceSample, ModelState, ModelStatus
from .predictor import DisturbancePredictor
from .storage import DisturbanceStorage
from .trainer import DisturbanceModelTrainer


class DisturbanceModelService:
    def __init__(self, config: DisturbanceModelConfig | None = None, logger=None) -> None:
        self.config = config or DisturbanceModelConfig()
        self._log = logger or (lambda _msg: None)
        self.collector = DisturbanceSampleCollector()
        self.storage = DisturbanceStorage(self.config, logger=self._log)
        self.trainer = DisturbanceModelTrainer(self.config)
        self.predictor = DisturbancePredictor(self.config)
        self._lock = threading.RLock()
        self._status = ModelStatus()
        self._last_sample: DisturbanceSample | None = None
        self._last_train_count = 0
        self._training_thread: threading.Thread | None = None
        self.storage.start()
        if self.predictor.model is not None:
            self._status.state = ModelState.READY.value
            self._status.model_ready = True
            self._status.model_valid = True
            self._status.model_version = self.predictor.model.version
            self._status.confidence = self.predictor.model.confidence

    def close(self) -> None:
        self.storage.stop()

    def build_and_submit_sample(self, **kwargs: Any) -> DisturbanceSample:
        sample = self.collector.build_sample(**kwargs)
        self.submit_sample(sample)
        return sample

    def submit_sample(self, sample: DisturbanceSample) -> None:
        with self._lock:
            self._last_sample = sample
            self._status.sample_count += 1
        self.storage.submit(sample)
        self._maybe_schedule_training()

    def predict(self, sample: DisturbanceSample | None = None) -> DisturbancePrediction:
        with self._lock:
            target = sample or self._last_sample
        prediction = self.predictor.predict(target)
        with self._lock:
            self._status.model_ready = bool(prediction.model_ready)
            self._status.model_valid = bool(prediction.model_valid)
            self._status.confidence = float(prediction.confidence)
            if prediction.model_version:
                self._status.model_version = prediction.model_version
        return prediction

    def get_status(self) -> ModelStatus:
        with self._lock:
            return ModelStatus(
                state=self._status.state,
                sample_count=self._status.sample_count,
                model_version=self._status.model_version,
                model_ready=self._status.model_ready,
                model_valid=self._status.model_valid,
                confidence=self._status.confidence,
                last_error=self._status.last_error,
                metrics=self._status.metrics,
            )

    def _maybe_schedule_training(self) -> None:
        if not self.config.online_update_enabled:
            return
        count = self.storage.count_samples()
        if count < int(self.config.minimum_training_samples):
            with self._lock:
                self._status.state = ModelState.COLLECTING.value
            return
        if count - self._last_train_count < int(self.config.model_update_interval):
            return
        if self._training_thread and self._training_thread.is_alive():
            return
        self._last_train_count = count
        self._training_thread = threading.Thread(target=self._train_background, name="disturbance-trainer", daemon=True)
        self._training_thread.start()

    def _train_background(self) -> None:
        try:
            with self._lock:
                self._status.state = ModelState.TRAINING.value
            samples = self.storage.load_recent_samples(int(self.config.training_window_size))
            model, metrics, reason = self.trainer.train(samples)
            if model is None:
                with self._lock:
                    self._status.state = ModelState.COLLECTING.value
                    self._status.last_error = reason
                return
            with self._lock:
                self._status.state = ModelState.VALIDATING.value
            valid = metrics.r2 >= float(self.config.minimum_r2) and metrics.rmse <= float(self.config.maximum_rmse)
            if valid:
                self.predictor.set_model(model)
                self.storage.record_model_version(model.version, {"confidence": model.confidence})
                self.storage.record_metrics(model.version, metrics)
                with self._lock:
                    self._status.state = ModelState.READY.value
                    self._status.model_ready = True
                    self._status.model_valid = True
                    self._status.model_version = model.version
                    self._status.confidence = model.confidence
                    self._status.metrics = metrics
                    self._status.last_error = ""
                self._log(f"[DISTURBANCE][MODEL][READY] version={model.version} r2={metrics.r2:.3f}")
            else:
                with self._lock:
                    self._status.state = ModelState.DEGRADED.value if self.predictor.model else ModelState.COLLECTING.value
                    self._status.model_ready = self.predictor.model is not None
                    self._status.model_valid = self.predictor.model is not None
                    self._status.metrics = metrics
                    self._status.last_error = "validation failed"
        except Exception as exc:
            with self._lock:
                self._status.state = ModelState.ERROR.value if self.predictor.model is None else ModelState.DEGRADED.value
                self._status.last_error = str(exc)
            self._log(f"[DISTURBANCE][MODEL][ERROR] {exc}")
