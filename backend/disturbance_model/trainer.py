from __future__ import annotations

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
from .models import DisturbanceSample, ModelMetrics


class DisturbanceModelTrainer:
    def __init__(self, config: DisturbanceModelConfig) -> None:
        self.config = config

    def train(self, samples: list[DisturbanceSample]) -> tuple[LinearDisturbanceModel | None, ModelMetrics, str]:
        usable: list[tuple[list[float], list[float]]] = []
        window = samples[-int(self.config.training_window_size) :]
        horizon_s = max(0.0, float(self.config.prediction_horizon_ms) / 1000.0)
        tolerance_s = max(0.0, float(self.config.prediction_horizon_tolerance_ms) / 1000.0)
        min_droplets = max(1, int(self.config.minimum_valid_droplets))
        for index, sample in enumerate(window):
            if not sample_is_usable(sample) or int(sample.droplet_count_frame or 0) < min_droplets:
                continue
            future_sample = self._find_future_sample(window, index, horizon_s, tolerance_s, min_droplets)
            if future_sample is None:
                continue
            targets = build_targets(sample, future_sample)
            if targets is not None:
                usable.append((build_features(sample), targets))
        if len(usable) < int(self.config.minimum_training_samples):
            return None, ModelMetrics(), "insufficient samples"

        x = [build_nonlinear_features(item[0]) for item in usable]
        y = [item[1] for item in usable]
        if np is None:
            return self._train_mean_model(y), ModelMetrics(), "numpy unavailable; mean model"

        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        x_aug = np.column_stack([np.ones(len(x_arr)), x_arr])
        penalty = np.eye(x_aug.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        reg = max(0.0, float(self.config.nonlinear_l2_regularization))
        try:
            coeff_matrix = np.linalg.solve(x_aug.T @ x_aug + reg * penalty, x_aug.T @ y_arr)
        except Exception:
            coeff_matrix, *_ = np.linalg.lstsq(x_aug, y_arr, rcond=None)
        intercepts = coeff_matrix[0, :].tolist()
        coefficients = coeff_matrix[1:, :].T.tolist()
        pred = x_aug @ coeff_matrix
        metrics = evaluate_predictions(y_arr[:, 0].tolist(), pred[:, 0].tolist())
        confidence = max(0.0, min(1.0, (metrics.r2 + 1.0) / 2.0))
        version = f"disturbance-{int(time.time())}"
        model = LinearDisturbanceModel(
            version=version,
            feature_names=list(FEATURE_NAMES),
            target_names=list(TARGET_NAMES),
            coefficients=coefficients,
            intercepts=intercepts,
            confidence=confidence,
            model_type="quadratic_ridge",
        )
        return model, metrics, "trained"

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
        target_ts = current_ts + horizon_s
        best_sample: DisturbanceSample | None = None
        best_error = float("inf")
        for future_sample in samples[current_index + 1 :]:
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

    def _train_mean_model(self, y: list[list[float]]) -> LinearDisturbanceModel:
        cols = list(zip(*y))
        intercepts = [sum(col) / len(col) for col in cols]
        coefficients = [[0.0 for _ in NONLINEAR_FEATURE_NAMES] for _ in TARGET_NAMES]
        return LinearDisturbanceModel(
            version=f"disturbance-{int(time.time())}",
            feature_names=list(FEATURE_NAMES),
            target_names=list(TARGET_NAMES),
            coefficients=coefficients,
            intercepts=intercepts,
            confidence=0.25,
            model_type="quadratic_ridge",
        )
