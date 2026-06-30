from __future__ import annotations

import math

from .models import ModelMetrics


def evaluate_predictions(y_true: list[float], y_pred: list[float]) -> ModelMetrics:
    if not y_true:
        return ModelMetrics()
    n = len(y_true)
    errors = [float(p) - float(t) for t, p in zip(y_true, y_pred)]
    abs_errors = [abs(v) for v in errors]
    mse = sum(v * v for v in errors) / n
    mean_true = sum(y_true) / n
    ss_tot = sum((v - mean_true) ** 2 for v in y_true)
    ss_res = sum(v * v for v in errors)
    r2 = 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    direction_ok = 0
    direction_total = 0
    for idx in range(1, n):
        true_delta = y_true[idx] - y_true[idx - 1]
        pred_delta = y_pred[idx] - y_pred[idx - 1]
        if true_delta == 0 and pred_delta == 0:
            direction_ok += 1
        elif true_delta * pred_delta > 0:
            direction_ok += 1
        direction_total += 1
    return ModelMetrics(
        mae=sum(abs_errors) / n,
        rmse=math.sqrt(mse),
        r2=r2,
        direction_accuracy=(direction_ok / direction_total) if direction_total else 0.0,
        response_delay_error_ms=0.0,
    )
