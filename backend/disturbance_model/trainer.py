from __future__ import annotations

import hashlib
import json
import time

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from .config import DisturbanceModelConfig
from .evaluator import evaluate_predictions
from .feature_builder import (
    FEATURE_NAMES,
    NONLINEAR_FEATURE_NAMES,
    TARGET_NAMES,
    build_features,
    build_nonlinear_features,
    build_targets,
    sample_is_usable,
)
from .model import LinearDisturbanceModel
from .models import DisturbanceSample, DisturbanceStage, ModelMetrics


class DisturbanceModelTrainer:
    def __init__(self, config: DisturbanceModelConfig) -> None:
        self.config = config

    def train(self, samples: list[DisturbanceSample]) -> tuple[LinearDisturbanceModel | None, ModelMetrics, str]:
        usable: list[tuple[list[float], list[float], tuple[str, str], bool]] = []
        window = samples[-int(self.config.training_window_size) :]
        disturbance_events = self._count_disturbance_events(window)
        if disturbance_events < int(self.config.minimum_disturbance_events):
            return None, ModelMetrics(), (
                f"insufficient disturbance events: {disturbance_events} < {int(self.config.minimum_disturbance_events)}"
            )
        min_droplets = max(1, int(self.config.minimum_valid_droplets))
        for index, sample in enumerate(window):
            if not sample_is_usable(sample) or int(sample.droplet_count_frame or 0) < min_droplets:
                continue
            if self.config.require_group_metadata and (not sample.experiment_id or not sample.chip_id):
                continue
            horizon_s, tolerance_s = self._pairing_window(sample)
            future_sample = self._find_future_sample(window, index, horizon_s, tolerance_s, min_droplets)
            if future_sample is None:
                continue
            targets = build_targets(sample, future_sample)
            if targets is not None:
                usable.append(
                    (
                        build_features(sample),
                        targets,
                        (sample.experiment_id, sample.chip_id),
                        bool(
                            sample.disturbance_stage == DisturbanceStage.BASELINE.value
                            and abs(float(sample.disturbance_amplitude or 0.0)) <= 1e-12
                        ),
                    )
                )
        if len(usable) < int(self.config.minimum_training_samples):
            suffix = " with experiment_id and chip_id" if self.config.require_group_metadata else ""
            return None, ModelMetrics(), f"insufficient paired samples{suffix}"

        x = [build_nonlinear_features(item[0]) for item in usable]
        y = [item[1] for item in usable]
        if np is None:
            return None, ModelMetrics(), "numpy unavailable; validated training disabled"

        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        groups = [item[2] for item in usable]
        ordered_groups = list(dict.fromkeys(groups))
        minimum_groups = max(3, int(self.config.minimum_evaluation_groups))
        if len(ordered_groups) < minimum_groups:
            return None, ModelMetrics(), f"insufficient independent experiment/chip groups: {len(ordered_groups)} < {minimum_groups}"
        test_count = max(1, int(round(len(ordered_groups) * float(self.config.test_ratio))))
        validation_count = max(1, int(round(len(ordered_groups) * float(self.config.validation_ratio))))
        if test_count + validation_count >= len(ordered_groups):
            test_count = 1
            validation_count = 1
        train_groups = set(ordered_groups[: -(validation_count + test_count)])
        validation_groups = set(ordered_groups[-(validation_count + test_count) : -test_count])
        test_groups = set(ordered_groups[-test_count:])
        train_indices = [index for index, group in enumerate(groups) if group in train_groups]
        validation_indices = [index for index, group in enumerate(groups) if group in validation_groups]
        test_indices = [index for index, group in enumerate(groups) if group in test_groups]
        if min(len(train_indices), len(validation_indices), len(test_indices)) < 1:
            return None, ModelMetrics(), "empty grouped train/validation/test split"

        x_train = x_arr[train_indices]
        feature_means = x_train.mean(axis=0)
        feature_scales = x_train.std(axis=0)
        feature_scales[feature_scales < 1e-12] = 1.0
        x_all_scaled = (x_arr - feature_means) / feature_scales
        x_train_scaled = x_all_scaled[train_indices]
        baseline_train_positions = [
            position
            for position, source_index in enumerate(train_indices)
            if usable[source_index][3]
        ]
        minimum_baseline_pairs = max(5, int(self.config.minimum_training_samples) // 5)
        if len(baseline_train_positions) < minimum_baseline_pairs:
            return None, ModelMetrics(), (
                "nominal baseline fit failed: insufficient baseline pairs "
                f"({len(baseline_train_positions)} < {minimum_baseline_pairs})"
            )
        baseline_x = x_train_scaled[baseline_train_positions]
        baseline_y = y_arr[train_indices, 1][baseline_train_positions]
        baseline_aug = np.column_stack([np.ones(len(baseline_x)), baseline_x])
        baseline_penalty = np.eye(baseline_aug.shape[1], dtype=float)
        baseline_penalty[0, 0] = 0.0
        reg = max(0.0, float(self.config.nonlinear_l2_regularization))
        try:
            nominal_coefficients = np.linalg.solve(
                baseline_aug.T @ baseline_aug + reg * baseline_penalty,
                baseline_aug.T @ baseline_y,
            )
        except Exception:
            nominal_coefficients, *_ = np.linalg.lstsq(
                baseline_aug,
                baseline_y,
                rcond=None,
            )
        nominal_change_all = (
            np.column_stack([np.ones(len(x_all_scaled)), x_all_scaled])
            @ nominal_coefficients
        )
        # The second target is deliberately the disturbance-only residual.
        # This prevents the predictive path from relearning nominal PID/plant
        # dynamics and then compensating them a second time.
        y_arr[:, 1] = y_arr[:, 1] - nominal_change_all
        y_train = y_arr[train_indices]
        x_aug = np.column_stack([np.ones(len(x_train_scaled)), x_train_scaled])
        penalty = np.eye(x_aug.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        try:
            coeff_matrix = np.linalg.solve(x_aug.T @ x_aug + reg * penalty, x_aug.T @ y_train)
        except Exception:
            coeff_matrix, *_ = np.linalg.lstsq(x_aug, y_train, rcond=None)
        intercepts = coeff_matrix[0, :].tolist()
        coefficients = coeff_matrix[1:, :].T.tolist()
        validation_metrics = self._evaluate_split(
            x_arr[validation_indices], y_arr[validation_indices], feature_means, feature_scales, coeff_matrix
        )
        test_metrics = self._evaluate_split(
            x_arr[test_indices], y_arr[test_indices], feature_means, feature_scales, coeff_matrix
        )
        for name, metrics in (("validation", validation_metrics), ("test", test_metrics)):
            failure = self._validation_failure(metrics)
            if failure:
                return None, test_metrics, f"{name} failed: {failure}"
        confidence = min(
            validation_metrics.direction_accuracy,
            test_metrics.direction_accuracy,
            max(0.0, validation_metrics.persistence_improvement),
            max(0.0, test_metrics.persistence_improvement),
        )
        version = f"disturbance-{int(time.time())}"
        model = LinearDisturbanceModel(
            version=version,
            feature_names=list(FEATURE_NAMES),
            target_names=list(TARGET_NAMES),
            coefficients=coefficients,
            intercepts=intercepts,
            confidence=confidence,
            model_type="quadratic_ridge",
            schema_version=3,
            feature_means=feature_means.tolist(),
            feature_scales=feature_scales.tolist(),
            training_data_hash=hashlib.sha256(
                json.dumps(usable, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            feature_version="causal-residual-v3",
            nominal_change_coefficients=nominal_coefficients[1:].tolist(),
            nominal_change_intercept=float(nominal_coefficients[0]),
        )
        return model, test_metrics, (
            "trained with independent grouped validation/test "
            f"({len(train_groups)}/{len(validation_groups)}/{len(test_groups)} groups)"
        )

    def _evaluate_split(self, x, y, means, scales, coefficients) -> ModelMetrics:
        x_scaled = (x - means) / scales
        predictions = np.column_stack([np.ones(len(x_scaled)), x_scaled]) @ coefficients
        return evaluate_predictions(
            y[:, 1].tolist(),
            predictions[:, 1].tolist(),
            response_delay_true=y[:, 5].tolist(),
            response_delay_pred=predictions[:, 5].tolist(),
        )

    def _validation_failure(self, metrics: ModelMetrics) -> str:
        if metrics.r2 < float(self.config.minimum_r2):
            return f"delta-D R2 {metrics.r2:.3f} below {float(self.config.minimum_r2):.3f}"
        if metrics.rmse > float(self.config.maximum_rmse):
            return f"delta-D RMSE {metrics.rmse:.3f} above {float(self.config.maximum_rmse):.3f}"
        if metrics.direction_accuracy < float(self.config.minimum_direction_accuracy):
            return "delta-D direction accuracy below threshold"
        if metrics.persistence_improvement < float(self.config.minimum_persistence_improvement):
            return "does not beat persistence baseline"
        return ""

    @staticmethod
    def _count_disturbance_events(samples: list[DisturbanceSample]) -> int:
        count = 0
        active_key: tuple[str, str, str] | None = None
        for sample in samples:
            disturbed = (
                sample.disturbance_stage == DisturbanceStage.DISTURBED.value
                or abs(float(sample.disturbance_amplitude or 0.0)) > 0.0
            )
            key = (sample.experiment_id, sample.chip_id, sample.disturbance_name)
            if disturbed and active_key is None:
                count += 1
                active_key = key
            elif disturbed and key != active_key:
                count += 1
                active_key = key
            elif not disturbed:
                active_key = None
        return count

    def _pairing_window(self, sample: DisturbanceSample) -> tuple[float, float]:
        horizon_ms = max(0.0, float(self.config.prediction_horizon_ms))
        if self.config.align_horizon_to_control_cycle:
            horizon_ms = max(horizon_ms, float(sample.control_cycle_ms or 0.0))
        tolerance_ms = max(
            float(self.config.prediction_horizon_tolerance_ms),
            horizon_ms * max(0.0, float(self.config.horizon_tolerance_fraction)),
        )
        return horizon_ms / 1000.0, tolerance_ms / 1000.0

    def _find_future_sample(
        self,
        samples: list[DisturbanceSample],
        current_index: int,
        horizon_s: float,
        tolerance_s: float,
        min_droplets: int,
    ) -> DisturbanceSample | None:
        if horizon_s <= 0.0:
            current = samples[current_index]
            return current if sample_is_usable(current) and int(current.droplet_count_frame or 0) >= min_droplets else None
        current_ts = float(samples[current_index].timestamp)
        current_group = (samples[current_index].experiment_id, samples[current_index].chip_id)
        target_ts = current_ts + horizon_s
        best_sample: DisturbanceSample | None = None
        best_error = float("inf")
        for future_sample in samples[current_index + 1 :]:
            if (future_sample.experiment_id, future_sample.chip_id) != current_group:
                continue
            future_ts = float(future_sample.timestamp)
            if future_ts <= current_ts:
                continue
            if sample_is_usable(future_sample) and int(future_sample.droplet_count_frame or 0) >= min_droplets:
                error = abs(future_ts - target_ts)
                if error < best_error:
                    best_sample = future_sample
                    best_error = error
                if future_ts >= target_ts and error <= tolerance_s:
                    return future_sample
            if future_ts > target_ts + tolerance_s and best_sample is not None:
                break
        if best_sample is not None and best_error <= tolerance_s:
            return best_sample
        return None
