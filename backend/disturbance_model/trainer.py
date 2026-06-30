from __future__ import annotations

import time

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from .config import DisturbanceModelConfig
from .evaluator import evaluate_predictions
from .feature_builder import FEATURE_NAMES, TARGET_NAMES, build_features, build_targets
from .model import LinearDisturbanceModel
from .models import DisturbanceSample, ModelMetrics


class DisturbanceModelTrainer:
    def __init__(self, config: DisturbanceModelConfig) -> None:
        self.config = config

    def train(self, samples: list[DisturbanceSample]) -> tuple[LinearDisturbanceModel | None, ModelMetrics, str]:
        usable: list[tuple[list[float], list[float]]] = []
        for sample in samples[-int(self.config.training_window_size) :]:
            targets = build_targets(sample)
            if targets is not None and sample.vision_valid:
                usable.append((build_features(sample), targets))
        if len(usable) < int(self.config.minimum_training_samples):
            return None, ModelMetrics(), "insufficient samples"

        x = [item[0] for item in usable]
        y = [item[1] for item in usable]
        if np is None:
            return self._train_mean_model(y), ModelMetrics(), "numpy unavailable; mean model"

        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        x_aug = np.column_stack([np.ones(len(x_arr)), x_arr])
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
        )
        return model, metrics, "trained"

    def _train_mean_model(self, y: list[list[float]]) -> LinearDisturbanceModel:
        cols = list(zip(*y))
        intercepts = [sum(col) / len(col) for col in cols]
        coefficients = [[0.0 for _ in FEATURE_NAMES] for _ in TARGET_NAMES]
        return LinearDisturbanceModel(
            version=f"disturbance-{int(time.time())}",
            feature_names=list(FEATURE_NAMES),
            target_names=list(TARGET_NAMES),
            coefficients=coefficients,
            intercepts=intercepts,
            confidence=0.25,
        )
