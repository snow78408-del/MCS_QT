from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from .collector import DisturbanceSampleCollector
from .config import DisturbanceModelConfig
from .models import DisturbanceControlStage, DisturbancePrediction, DisturbanceSample, ModelState, ModelStatus
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
        self._control_stage = self._normalize_stage(self.config.deployment_stage)
        self._safety_fallback = False
        self._consecutive_prediction_errors = 0
        self._pending_shadow: deque[tuple[float, float]] = deque(maxlen=200)
        self._shadow_errors: deque[float] = deque(maxlen=max(1, int(self.config.shadow_validation_window)))
        self.storage.start()
        self._status.control_stage = self._control_stage.value
        self._status.feedforward_weight = self._feedforward_weight_for_stage(self._control_stage)
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
            self._update_shadow_metrics_locked(sample)
        self.storage.submit(sample)
        self._maybe_schedule_training()

    def predict(self, sample: DisturbanceSample | None = None) -> DisturbancePrediction:
        with self._lock:
            target = sample or self._last_sample
            stage = self._control_stage
            fallback = self._safety_fallback
        prediction = self.predictor.predict(target)
        prediction = self._apply_control_stage(prediction, target, stage, fallback)
        with self._lock:
            self._status.model_ready = bool(prediction.model_ready)
            self._status.model_valid = bool(prediction.model_valid)
            self._status.confidence = float(prediction.confidence)
            self._status.control_stage = prediction.control_stage
            self._status.feedforward_weight = float(prediction.feedforward_weight)
            self._status.safety_fallback = bool(prediction.safety_fallback)
            if prediction.model_version:
                self._status.model_version = prediction.model_version
            if target is not None and prediction.predicted_diameter_um is not None:
                self._record_shadow_prediction_locked(target, prediction)
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
                control_stage=self._status.control_stage,
                feedforward_weight=self._status.feedforward_weight,
                shadow_mae_um=self._status.shadow_mae_um,
                safety_fallback=self._status.safety_fallback,
                last_error=self._status.last_error,
                metrics=self._status.metrics,
            )

    def set_control_stage(self, stage: str | DisturbanceControlStage) -> None:
        with self._lock:
            self._control_stage = self._normalize_stage(stage)
            self.config.deployment_stage = self._control_stage.value
            self._safety_fallback = False
            self._consecutive_prediction_errors = 0
            self._status.control_stage = self._control_stage.value
            self._status.feedforward_weight = self._feedforward_weight_for_stage(self._control_stage)
            self._status.safety_fallback = False
            self._status.last_error = ""
        self._log(f"[DISTURBANCE][STAGE] {self._control_stage.value}")

    def _maybe_schedule_training(self) -> None:
        if not self.config.online_update_enabled:
            return
        if self._control_stage == DisturbanceControlStage.COLLECT_ONLY:
            with self._lock:
                self._status.state = ModelState.COLLECTING.value
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

    def _apply_control_stage(
        self,
        prediction: DisturbancePrediction,
        sample: DisturbanceSample | None,
        stage: DisturbanceControlStage,
        fallback: bool,
    ) -> DisturbancePrediction:
        prediction.control_stage = stage.value
        prediction.shadow_error_um = self._shadow_errors[-1] if self._shadow_errors else None
        prediction.safety_fallback = bool(fallback)
        if fallback:
            prediction.feedforward_weight = 0.0
            prediction.recommended_feedforward = 0.0
            prediction.model_valid = False
            prediction.reason = f"safety fallback; no feedforward PID ({prediction.reason})"
            return prediction

        if not prediction.model_ready or not prediction.model_valid:
            if stage in {
                DisturbanceControlStage.SHADOW,
                DisturbanceControlStage.LOW_WEIGHT_FEEDFORWARD,
                DisturbanceControlStage.FULL_FEEDFORWARD,
            }:
                self._note_prediction_error("model not ready or invalid")
            prediction.feedforward_weight = 0.0
            prediction.recommended_feedforward = 0.0
            return prediction

        self._clear_prediction_error()
        weight = self._feedforward_weight_for_stage(stage)
        prediction.feedforward_weight = weight
        prediction.recommended_feedforward = float(prediction.recommended_feedforward) * weight
        if weight <= 0.0:
            prediction.reason = f"{stage.value}: prediction display only"
        elif stage == DisturbanceControlStage.LOW_WEIGHT_FEEDFORWARD:
            prediction.reason = "low-weight feedforward enabled"
        elif stage == DisturbanceControlStage.FULL_FEEDFORWARD:
            prediction.reason = "full feedforward enabled"
        return prediction

    def _record_shadow_prediction_locked(self, sample: DisturbanceSample, prediction: DisturbancePrediction) -> None:
        if self._control_stage not in {
            DisturbanceControlStage.SHADOW,
            DisturbanceControlStage.LOW_WEIGHT_FEEDFORWARD,
            DisturbanceControlStage.FULL_FEEDFORWARD,
        }:
            return
        target_ts = float(sample.timestamp) + max(0.0, float(self.config.prediction_horizon_ms) / 1000.0)
        self._pending_shadow.append((target_ts, float(prediction.predicted_diameter_um)))

    def _update_shadow_metrics_locked(self, sample: DisturbanceSample) -> None:
        if sample.droplet_mean_diameter_um is None:
            return
        if not sample.vision_valid:
            return
        current_ts = float(sample.timestamp)
        actual = float(sample.droplet_mean_diameter_um)
        while self._pending_shadow and current_ts >= self._pending_shadow[0][0]:
            _, predicted = self._pending_shadow.popleft()
            self._shadow_errors.append(abs(predicted - actual))
        if not self._shadow_errors:
            return
        shadow_mae = sum(self._shadow_errors) / len(self._shadow_errors)
        self._status.shadow_mae_um = shadow_mae
        if len(self._shadow_errors) >= int(self.config.shadow_min_comparisons) and shadow_mae > float(self.config.shadow_max_mae_um):
            self._trip_safety_fallback_locked(f"shadow MAE too high: {shadow_mae:.3f} um")

    def _note_prediction_error(self, reason: str) -> None:
        with self._lock:
            self._consecutive_prediction_errors += 1
            if self._consecutive_prediction_errors >= int(self.config.max_consecutive_prediction_errors):
                self._trip_safety_fallback_locked(reason)

    def _clear_prediction_error(self) -> None:
        with self._lock:
            self._consecutive_prediction_errors = 0

    def _trip_safety_fallback_locked(self, reason: str) -> None:
        self._safety_fallback = True
        self._status.safety_fallback = True
        self._status.feedforward_weight = 0.0
        self._status.model_valid = False
        self._status.state = ModelState.DEGRADED.value
        self._status.last_error = reason
        self._log(f"[DISTURBANCE][FALLBACK] {reason}; feedforward disabled")

    @staticmethod
    def _normalize_stage(stage: str | DisturbanceControlStage) -> DisturbanceControlStage:
        if isinstance(stage, DisturbanceControlStage):
            return stage
        text = str(stage or "").strip().upper()
        for item in DisturbanceControlStage:
            if text in {item.value, item.name}:
                return item
        return DisturbanceControlStage.COLLECT_ONLY

    def _feedforward_weight_for_stage(self, stage: DisturbanceControlStage) -> float:
        if stage == DisturbanceControlStage.LOW_WEIGHT_FEEDFORWARD:
            return max(0.0, min(1.0, float(self.config.low_weight_feedforward_weight)))
        if stage == DisturbanceControlStage.FULL_FEEDFORWARD:
            return max(0.0, min(1.0, float(self.config.full_feedforward_weight)))
        return 0.0

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
