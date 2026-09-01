from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from .collector import DisturbanceSampleCollector
from .config import DisturbanceModelConfig
from .models import (
    DisturbanceControlStage,
    DisturbancePrediction,
    DisturbanceSample,
    ModelMetrics,
    ModelState,
    ModelStatus,
)
from .predictor import DisturbancePredictor
from .storage import DisturbanceStorage
from .trainer import DisturbanceModelTrainer


@dataclass(frozen=True, slots=True)
class _CandidateShadowPrediction:
    target_timestamp: float
    candidate_version: str
    candidate_diameter_um: float
    candidate_residual_um: float
    candidate_nominal_change_um: float
    origin_diameter_um: float
    origin_timestamp: float
    active_diameter_um: float | None = None
    active_residual_um: float | None = None
    active_nominal_change_um: float | None = None


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
        self._closed = False
        self._control_stage = self._normalize_stage(self.config.deployment_stage)
        self._safety_fallback = False
        self._consecutive_prediction_errors = 0
        self._pending_shadow: deque[tuple[float, float, float, float, float, float]] = deque(maxlen=200)
        self._shadow_errors: deque[float] = deque(maxlen=max(1, int(self.config.shadow_validation_window)))
        self._shadow_change_errors: deque[float] = deque(maxlen=max(1, int(self.config.shadow_validation_window)))
        self._shadow_direction_matches: deque[bool] = deque(maxlen=max(1, int(self.config.shadow_validation_window)))
        self._pending_candidate_shadow: deque[_CandidateShadowPrediction] = deque(maxlen=200)
        self._candidate_shadow_errors: deque[float] = deque(
            maxlen=max(1, int(self.config.shadow_validation_window))
        )
        self._candidate_shadow_change_errors: deque[float] = deque(
            maxlen=max(1, int(self.config.shadow_validation_window))
        )
        self._candidate_shadow_direction_matches: deque[bool] = deque(
            maxlen=max(1, int(self.config.shadow_validation_window))
        )
        self._candidate_active_change_errors: deque[float] = deque(
            maxlen=max(1, int(self.config.shadow_validation_window))
        )
        self._candidate_metrics = ModelMetrics()
        self._candidate_baseline_active_version = ""
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
        with self._lock:
            self._closed = True
        self.storage.stop()
        thread = self._training_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
            if thread.is_alive():
                self._log(
                    "[DISTURBANCE][CANDIDATE][WARN] training thread did not finish before close timeout"
                )

    def build_and_submit_sample(self, **kwargs: Any) -> DisturbanceSample:
        sample = self.collector.build_sample(**kwargs)
        self.submit_sample(sample)
        return sample

    def submit_sample(self, sample: DisturbanceSample) -> None:
        with self._lock:
            self._last_sample = sample
            self._status.sample_count += 1
            self._update_shadow_metrics_locked(sample)
            self._update_candidate_shadow_metrics_locked(sample)
        self.storage.submit(sample)
        self._maybe_schedule_training()

    def predict(self, sample: DisturbanceSample | None = None) -> DisturbancePrediction:
        with self._lock:
            target = sample or self._last_sample
            stage = self._control_stage
            fallback = self._safety_fallback
        active_prediction = self.predictor.predict(target)
        candidate_prediction = self.predictor.predict_candidate(target)
        active_available = self.predictor.model is not None
        candidate_available = self.predictor.candidate_model is not None
        if active_available:
            prediction = self._apply_control_stage(
                active_prediction,
                target,
                stage,
                fallback,
            )
        elif candidate_available:
            # Candidate output is visible for observability but cannot enter
            # the actuator allocation before explicit promotion.
            prediction = candidate_prediction
            prediction.control_stage = stage.value
            prediction.feedforward_weight = 0.0
            prediction.recommended_feedforward = 0.0
            prediction.reason = "candidate shadow prediction; not promoted"
        else:
            prediction = self._apply_control_stage(
                active_prediction,
                target,
                stage,
                fallback,
            )
        with self._lock:
            self._status.model_ready = bool(active_available and active_prediction.model_ready)
            self._status.model_valid = bool(active_available and active_prediction.model_valid)
            self._status.confidence = (
                float(active_prediction.confidence) if active_available else 0.0
            )
            self._status.control_stage = prediction.control_stage
            self._status.feedforward_weight = float(prediction.feedforward_weight)
            self._status.safety_fallback = bool(prediction.safety_fallback)
            self._status.model_version = self.predictor.model_version
            if (
                active_available
                and target is not None
                and active_prediction.predicted_diameter_um is not None
            ):
                self._record_shadow_prediction_locked(target, active_prediction)
            if candidate_available and target is not None:
                self._record_candidate_shadow_prediction_locked(
                    target,
                    candidate_prediction,
                    active_prediction if active_available else None,
                )
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
                shadow_change_mae_um=self._status.shadow_change_mae_um,
                shadow_direction_accuracy=self._status.shadow_direction_accuracy,
                candidate_model_version=self._status.candidate_model_version,
                candidate_ready=self._status.candidate_ready,
                candidate_comparisons=self._status.candidate_comparisons,
                candidate_shadow_mae_um=self._status.candidate_shadow_mae_um,
                candidate_shadow_change_mae_um=self._status.candidate_shadow_change_mae_um,
                candidate_shadow_direction_accuracy=self._status.candidate_shadow_direction_accuracy,
                candidate_active_change_mae_um=self._status.candidate_active_change_mae_um,
                candidate_relative_improvement=self._status.candidate_relative_improvement,
                candidate_promotion_ready=self._status.candidate_promotion_ready,
                candidate_promotion_reason=self._status.candidate_promotion_reason,
                candidate_metrics=ModelMetrics(
                    **self._status.candidate_metrics.to_dict()
                ),
                safety_fallback=self._status.safety_fallback,
                last_error=self._status.last_error,
                metrics=ModelMetrics(**self._status.metrics.to_dict()),
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
        if self._closed or not self.config.online_update_enabled:
            return
        count = self.storage.count_samples()
        if count < int(self.config.minimum_training_samples):
            with self._lock:
                if self.predictor.model is None and self.predictor.candidate_model is None:
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
        if prediction.recommended_feedforward is not None:
            prediction.recommended_feedforward = float(prediction.recommended_feedforward) * weight
        if weight <= 0.0:
            prediction.reason = self._feedforward_gate_reason(stage)
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
        horizon_s, _ = self._pairing_window(sample)
        target_ts = float(sample.timestamp) + horizon_s
        if prediction.predicted_disturbance_residual_um is None:
            return
        if self._pending_shadow and self._pending_shadow[-1][5] == float(sample.timestamp):
            return
        self._pending_shadow.append(
            (
                target_ts,
                float(prediction.predicted_diameter_um),
                float(prediction.predicted_disturbance_residual_um),
                float(prediction.predicted_nominal_change_um),
                float(sample.droplet_mean_diameter_um or 0.0),
                float(sample.timestamp),
            )
        )

    def _update_shadow_metrics_locked(self, sample: DisturbanceSample) -> None:
        if sample.droplet_mean_diameter_um is None:
            return
        if not sample.vision_valid:
            return
        current_ts = float(sample.timestamp)
        actual = float(sample.droplet_mean_diameter_um)
        _, tolerance_s = self._pairing_window(sample)
        while self._pending_shadow and self._pending_shadow[0][0] < current_ts - tolerance_s:
            self._pending_shadow.popleft()
        candidates = [item for item in self._pending_shadow if abs(item[0] - current_ts) <= tolerance_s]
        if candidates:
            matched = min(candidates, key=lambda item: abs(item[0] - current_ts))
            self._pending_shadow.remove(matched)
            _, predicted, predicted_residual, predicted_nominal, origin_diameter, _ = matched
            actual_change = actual - origin_diameter
            actual_residual = actual_change - predicted_nominal
            self._shadow_errors.append(abs(predicted - actual))
            self._shadow_change_errors.append(abs(predicted_residual - actual_residual))
            self._shadow_direction_matches.append(
                (actual_residual == 0.0 and predicted_residual == 0.0)
                or actual_residual * predicted_residual > 0.0
            )
        if not self._shadow_errors:
            return
        shadow_mae = sum(self._shadow_errors) / len(self._shadow_errors)
        shadow_change_mae = sum(self._shadow_change_errors) / len(self._shadow_change_errors)
        shadow_direction_accuracy = sum(self._shadow_direction_matches) / len(self._shadow_direction_matches)
        self._status.shadow_mae_um = shadow_mae
        self._status.shadow_change_mae_um = shadow_change_mae
        self._status.shadow_direction_accuracy = shadow_direction_accuracy
        if len(self._shadow_errors) >= int(self.config.shadow_min_comparisons):
            if shadow_mae > float(self.config.shadow_max_mae_um):
                self._trip_safety_fallback_locked(f"shadow absolute-diameter MAE too high: {shadow_mae:.3f} um")
            elif shadow_change_mae > float(self.config.shadow_max_change_mae_um):
                self._trip_safety_fallback_locked(f"shadow delta-D MAE too high: {shadow_change_mae:.3f} um")
            elif shadow_direction_accuracy < float(self.config.shadow_min_direction_accuracy):
                self._trip_safety_fallback_locked(
                    f"shadow delta-D direction accuracy too low: {shadow_direction_accuracy:.3f}"
                )

    def _record_candidate_shadow_prediction_locked(
        self,
        sample: DisturbanceSample,
        candidate: DisturbancePrediction,
        active: DisturbancePrediction | None,
    ) -> None:
        if (
            not candidate.model_ready
            or not candidate.model_valid
            or candidate.predicted_diameter_um is None
            or candidate.predicted_disturbance_residual_um is None
        ):
            return
        active_version = self.predictor.model_version
        if self._candidate_baseline_active_version != active_version:
            self._reset_candidate_shadow_locked()
            self._candidate_baseline_active_version = active_version
        horizon_s, _ = self._pairing_window(sample)
        origin_timestamp = float(sample.timestamp)
        if (
            self._pending_candidate_shadow
            and self._pending_candidate_shadow[-1].origin_timestamp == origin_timestamp
        ):
            return
        active_valid = bool(
            active is not None
            and active.model_ready
            and active.model_valid
            and active.predicted_diameter_um is not None
            and active.predicted_disturbance_residual_um is not None
        )
        self._pending_candidate_shadow.append(
            _CandidateShadowPrediction(
                target_timestamp=origin_timestamp + horizon_s,
                candidate_version=str(candidate.model_version or ""),
                candidate_diameter_um=float(candidate.predicted_diameter_um),
                candidate_residual_um=float(candidate.predicted_disturbance_residual_um),
                candidate_nominal_change_um=float(candidate.predicted_nominal_change_um),
                origin_diameter_um=float(sample.droplet_mean_diameter_um or 0.0),
                origin_timestamp=origin_timestamp,
                active_diameter_um=(
                    float(active.predicted_diameter_um) if active_valid else None
                ),
                active_residual_um=(
                    float(active.predicted_disturbance_residual_um) if active_valid else None
                ),
                active_nominal_change_um=(
                    float(active.predicted_nominal_change_um) if active_valid else None
                ),
            )
        )

    def _update_candidate_shadow_metrics_locked(
        self,
        sample: DisturbanceSample,
    ) -> None:
        if sample.droplet_mean_diameter_um is None or not sample.vision_valid:
            return
        current_timestamp = float(sample.timestamp)
        actual_diameter = float(sample.droplet_mean_diameter_um)
        _, tolerance_s = self._pairing_window(sample)
        while (
            self._pending_candidate_shadow
            and self._pending_candidate_shadow[0].target_timestamp
            < current_timestamp - tolerance_s
        ):
            self._pending_candidate_shadow.popleft()
        matches = [
            item
            for item in self._pending_candidate_shadow
            if abs(item.target_timestamp - current_timestamp) <= tolerance_s
        ]
        if not matches:
            return
        matched = min(
            matches,
            key=lambda item: abs(item.target_timestamp - current_timestamp),
        )
        self._pending_candidate_shadow.remove(matched)
        if matched.candidate_version != self.predictor.candidate_model_version:
            return

        actual_change = actual_diameter - matched.origin_diameter_um
        candidate_actual_residual = (
            actual_change - matched.candidate_nominal_change_um
        )
        self._candidate_shadow_errors.append(
            abs(matched.candidate_diameter_um - actual_diameter)
        )
        self._candidate_shadow_change_errors.append(
            abs(matched.candidate_residual_um - candidate_actual_residual)
        )
        self._candidate_shadow_direction_matches.append(
            (
                candidate_actual_residual == 0.0
                and matched.candidate_residual_um == 0.0
            )
            or candidate_actual_residual * matched.candidate_residual_um > 0.0
        )
        if (
            matched.active_residual_um is not None
            and matched.active_nominal_change_um is not None
        ):
            active_actual_residual = (
                actual_change - matched.active_nominal_change_um
            )
            self._candidate_active_change_errors.append(
                abs(matched.active_residual_um - active_actual_residual)
            )
        self._refresh_candidate_status_locked()

    def _refresh_candidate_status_locked(self) -> None:
        candidate = self.predictor.candidate_model
        status = self._status
        if candidate is None:
            status.candidate_model_version = ""
            status.candidate_ready = False
            status.candidate_comparisons = 0
            status.candidate_shadow_mae_um = 0.0
            status.candidate_shadow_change_mae_um = 0.0
            status.candidate_shadow_direction_accuracy = 0.0
            status.candidate_active_change_mae_um = None
            status.candidate_relative_improvement = None
            status.candidate_promotion_ready = False
            status.candidate_promotion_reason = "no candidate model"
            status.candidate_metrics = ModelMetrics()
            return

        comparisons = len(self._candidate_shadow_errors)
        candidate_mae = (
            sum(self._candidate_shadow_errors) / comparisons
            if comparisons
            else 0.0
        )
        candidate_change_mae = (
            sum(self._candidate_shadow_change_errors)
            / len(self._candidate_shadow_change_errors)
            if self._candidate_shadow_change_errors
            else 0.0
        )
        candidate_direction = (
            sum(self._candidate_shadow_direction_matches)
            / len(self._candidate_shadow_direction_matches)
            if self._candidate_shadow_direction_matches
            else 0.0
        )
        active_change_mae = (
            sum(self._candidate_active_change_errors)
            / len(self._candidate_active_change_errors)
            if self._candidate_active_change_errors
            else None
        )
        relative_improvement = None
        if active_change_mae is not None:
            relative_improvement = (
                active_change_mae - candidate_change_mae
            ) / max(active_change_mae, 1e-9)

        ready = False
        minimum = int(self.config.shadow_min_comparisons)
        if comparisons < minimum:
            reason = f"candidate shadow validation incomplete ({comparisons}/{minimum})"
        elif candidate_mae > float(self.config.shadow_max_mae_um):
            reason = "candidate absolute-diameter MAE exceeds threshold"
        elif candidate_change_mae > float(self.config.shadow_max_change_mae_um):
            reason = "candidate disturbance-residual MAE exceeds threshold"
        elif candidate_direction < float(self.config.shadow_min_direction_accuracy):
            reason = "candidate disturbance direction accuracy is below threshold"
        elif self.predictor.model is not None and len(
            self._candidate_active_change_errors
        ) < minimum:
            reason = "paired active-model comparison is incomplete"
        elif (
            self.predictor.model is not None
            and (
                relative_improvement is None
                or relative_improvement
                < float(self.config.candidate_min_relative_improvement)
            )
        ):
            reason = "candidate does not improve enough over the active model"
        else:
            ready = True
            reason = "candidate passed offline and paired online validation; explicit promotion required"

        status.candidate_model_version = candidate.version
        status.candidate_ready = True
        status.candidate_comparisons = comparisons
        status.candidate_shadow_mae_um = candidate_mae
        status.candidate_shadow_change_mae_um = candidate_change_mae
        status.candidate_shadow_direction_accuracy = candidate_direction
        status.candidate_active_change_mae_um = active_change_mae
        status.candidate_relative_improvement = relative_improvement
        status.candidate_promotion_ready = ready
        status.candidate_promotion_reason = reason
        status.candidate_metrics = self._candidate_metrics

    def _reset_candidate_shadow_locked(self) -> None:
        self._pending_candidate_shadow.clear()
        self._candidate_shadow_errors.clear()
        self._candidate_shadow_change_errors.clear()
        self._candidate_shadow_direction_matches.clear()
        self._candidate_active_change_errors.clear()

    def promote_candidate_model(self) -> dict[str, Any]:
        """Explicitly promote a candidate that passed paired shadow validation."""
        with self._lock:
            self._refresh_candidate_status_locked()
            if not self._status.candidate_promotion_ready:
                raise RuntimeError(self._status.candidate_promotion_reason)
            promoted_metrics = self._candidate_metrics
            promoted = self.predictor.promote_candidate_model()
            self._shadow_errors = deque(
                self._candidate_shadow_errors,
                maxlen=max(1, int(self.config.shadow_validation_window)),
            )
            self._shadow_change_errors = deque(
                self._candidate_shadow_change_errors,
                maxlen=max(1, int(self.config.shadow_validation_window)),
            )
            self._shadow_direction_matches = deque(
                self._candidate_shadow_direction_matches,
                maxlen=max(1, int(self.config.shadow_validation_window)),
            )
            self._pending_shadow.clear()
            self._reset_candidate_shadow_locked()
            self._candidate_metrics = ModelMetrics()
            self._candidate_baseline_active_version = promoted.version
            self._status.state = ModelState.READY.value
            self._status.model_ready = True
            self._status.model_valid = True
            self._status.model_version = promoted.version
            self._status.confidence = promoted.confidence
            self._status.metrics = promoted_metrics
            if self._shadow_errors:
                self._status.shadow_mae_um = sum(self._shadow_errors) / len(self._shadow_errors)
                self._status.shadow_change_mae_um = (
                    sum(self._shadow_change_errors) / len(self._shadow_change_errors)
                )
                self._status.shadow_direction_accuracy = (
                    sum(self._shadow_direction_matches)
                    / len(self._shadow_direction_matches)
                )
            self._refresh_candidate_status_locked()
        self._log(
            f"[DISTURBANCE][PROMOTED] version={promoted.version}; "
            "feedforward authorization remains unchanged"
        )
        try:
            self.storage.mark_model_promoted(promoted.version)
        except Exception as exc:
            self._log(
                f"[DISTURBANCE][PROMOTED][AUDIT][WARN] version={promoted.version} error={exc}"
            )
        return {
            "model_version": promoted.version,
            "metrics": promoted_metrics.to_dict(),
            "control_stage": self._control_stage.value,
            "feedforward_authorized": bool(
                self.config.allow_low_weight_feedforward
                or self.config.allow_full_feedforward
            ),
        }

    def discard_candidate_model(self) -> None:
        with self._lock:
            version = self.predictor.candidate_model_version
            self.predictor.discard_candidate_model()
            self._reset_candidate_shadow_locked()
            self._candidate_metrics = ModelMetrics()
            self._candidate_baseline_active_version = self.predictor.model_version
            self._refresh_candidate_status_locked()
            if self.predictor.model is None:
                self._status.state = ModelState.COLLECTING.value
        if version:
            self._log(f"[DISTURBANCE][CANDIDATE][DISCARDED] version={version}")

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
        if stage in {
            DisturbanceControlStage.LOW_WEIGHT_FEEDFORWARD,
            DisturbanceControlStage.FULL_FEEDFORWARD,
        }:
            if len(self._shadow_errors) < int(self.config.shadow_min_comparisons):
                return 0.0
            if self._status.shadow_change_mae_um > float(self.config.shadow_max_change_mae_um):
                return 0.0
            if self._status.shadow_direction_accuracy < float(self.config.shadow_min_direction_accuracy):
                return 0.0
        if stage == DisturbanceControlStage.LOW_WEIGHT_FEEDFORWARD:
            if not self.config.allow_low_weight_feedforward:
                return 0.0
            return max(0.0, min(1.0, float(self.config.low_weight_feedforward_weight)))
        if stage == DisturbanceControlStage.FULL_FEEDFORWARD:
            if not self.config.allow_full_feedforward:
                return 0.0
            return max(0.0, min(1.0, float(self.config.full_feedforward_weight)))
        return 0.0

    def _feedforward_gate_reason(self, stage: DisturbanceControlStage) -> str:
        if stage in {
            DisturbanceControlStage.LOW_WEIGHT_FEEDFORWARD,
            DisturbanceControlStage.FULL_FEEDFORWARD,
        } and len(self._shadow_errors) < int(self.config.shadow_min_comparisons):
            return f"{stage.value}: shadow validation incomplete"
        if stage == DisturbanceControlStage.LOW_WEIGHT_FEEDFORWARD and not self.config.allow_low_weight_feedforward:
            return "LOW_WEIGHT_FEEDFORWARD: explicit authorization disabled"
        if stage == DisturbanceControlStage.FULL_FEEDFORWARD and not self.config.allow_full_feedforward:
            return "FULL_FEEDFORWARD: explicit authorization disabled"
        return f"{stage.value}: prediction display only"

    def _pairing_window(self, sample: DisturbanceSample) -> tuple[float, float]:
        horizon_ms = max(0.0, float(self.config.prediction_horizon_ms))
        if self.config.align_horizon_to_control_cycle:
            horizon_ms = max(horizon_ms, float(sample.control_cycle_ms or 0.0))
        tolerance_ms = max(
            float(self.config.prediction_horizon_tolerance_ms),
            horizon_ms * max(0.0, float(self.config.horizon_tolerance_fraction)),
        )
        return horizon_ms / 1000.0, tolerance_ms / 1000.0

    def _train_background(self) -> None:
        try:
            with self._lock:
                self._status.state = ModelState.TRAINING.value
            samples = self.storage.load_recent_samples(int(self.config.training_window_size))
            model, metrics, reason = self.trainer.train(samples)
            if model is None:
                with self._lock:
                    self._status.state = (
                        ModelState.READY.value
                        if self.predictor.model is not None
                        else ModelState.COLLECTING.value
                    )
                    self._status.last_error = reason
                return
            with self._lock:
                self._status.state = ModelState.VALIDATING.value
            valid = (
                metrics.r2 >= float(self.config.minimum_r2)
                and metrics.rmse <= float(self.config.maximum_rmse)
                and metrics.direction_accuracy >= float(self.config.minimum_direction_accuracy)
                and metrics.persistence_improvement >= float(self.config.minimum_persistence_improvement)
            )
            if valid:
                with self._lock:
                    if self._closed:
                        return
                self.storage.record_model_version(
                    model.version,
                    {
                        "role": "candidate",
                        "confidence": model.confidence,
                        "training_data_hash": model.training_data_hash,
                        "feature_version": model.feature_version,
                        "metrics": metrics.to_dict(),
                    },
                )
                self.storage.record_metrics(model.version, metrics)
                with self._lock:
                    self.predictor.set_candidate_model(model)
                    self._reset_candidate_shadow_locked()
                    self._candidate_metrics = metrics
                    self._candidate_baseline_active_version = self.predictor.model_version
                    self._status.state = (
                        ModelState.READY.value
                        if self.predictor.model is not None
                        else ModelState.VALIDATING.value
                    )
                    self._refresh_candidate_status_locked()
                    self._status.last_error = ""
                self._log(
                    f"[DISTURBANCE][CANDIDATE][READY] version={model.version} "
                    f"r2={metrics.r2:.3f}; awaiting paired shadow validation"
                )
            else:
                with self._lock:
                    self._status.state = (
                        ModelState.READY.value
                        if self.predictor.model is not None
                        else ModelState.COLLECTING.value
                    )
                    self._status.model_ready = self.predictor.model is not None
                    self._status.model_valid = self.predictor.model is not None
                    self._status.metrics = metrics
                    self._status.last_error = "validation failed"
        except Exception as exc:
            with self._lock:
                self._status.state = (
                    ModelState.ERROR.value
                    if self.predictor.model is None
                    else ModelState.READY.value
                )
                self._status.last_error = str(exc)
            self._log(f"[DISTURBANCE][CANDIDATE][ERROR] {exc}")
