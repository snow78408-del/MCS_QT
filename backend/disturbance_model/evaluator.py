from __future__ import annotations

import math

from .models import ModelMetrics


def evaluate_predictions(
    y_true: list[float],
    y_pred: list[float],
    *,
    response_delay_true: list[float] | None = None,
    response_delay_pred: list[float] | None = None,
) -> ModelMetrics:
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
    direction_ok = sum(
        1
        for true_delta, pred_delta in zip(y_true, y_pred)
        if (true_delta == 0.0 and pred_delta == 0.0) or true_delta * pred_delta > 0.0
    )
    persistence_rmse = math.sqrt(sum(float(value) ** 2 for value in y_true) / n)
    persistence_improvement = (
        (persistence_rmse - math.sqrt(mse)) / persistence_rmse if persistence_rmse > 0.0 else 0.0
    )
    delay_error = 0.0
    if response_delay_true and response_delay_pred:
        delay_pairs = list(zip(response_delay_true, response_delay_pred))
        if delay_pairs:
            delay_error = sum(abs(float(pred) - float(true)) for true, pred in delay_pairs) / len(delay_pairs)
    return ModelMetrics(
        mae=sum(abs_errors) / n,
        rmse=math.sqrt(mse),
        r2=r2,
        direction_accuracy=direction_ok / n,
        response_delay_error_ms=delay_error,
        persistence_rmse=persistence_rmse,
        persistence_improvement=persistence_improvement,
    )
