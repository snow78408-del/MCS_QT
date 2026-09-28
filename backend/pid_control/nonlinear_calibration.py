from __future__ import annotations

import math
from datetime import datetime, timezone
from statistics import median
from typing import Any, Iterable

import numpy as np

from .calibration import build_calibration_record


def validate_model(model: dict[str, Any]) -> None:
    if model.get("kind") != "quadratic_flow_response":
        raise ValueError("unsupported nonlinear model")
    for key, length in (("center", 2), ("scale", 2), ("coefficients", 5)):
        values = model.get(key, [])
        if len(values) != length or not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"invalid nonlinear model {key}")
    if min(*model["center"], *model["scale"]) <= 0:
        raise ValueError("nonlinear model center and scale must be positive")


def features(model: dict[str, Any], q1: float, q2: float) -> np.ndarray:
    first = (q1 - model["center"][0]) / model["scale"][0]
    second = (q2 - model["center"][1]) / model["scale"][1]
    return np.asarray([first, second, first * first, first * second, second * second])


def predict_change(model: dict[str, Any], item: Any) -> float:
    difference = features(model, item.actual_q1, item.actual_q2) - features(
        model, item.baseline_q1, item.baseline_q2,
    )
    return float(difference @ np.asarray(model["coefficients"]))


def fit_response_surface(measurements: Iterable[Any]) -> dict[str, Any]:
    from .calibration_experiment import CalibrationIdentificationError

    items = [item for item in measurements if item.channel != "validation"]
    if not items:
        raise CalibrationIdentificationError("没有可用于非线性拟合的建模试验")
    center = [float(median(getattr(item, name) for item in items))
              for name in ("baseline_q1", "baseline_q2")]
    scale = [max(abs(getattr(item, actual) - center[index]) for item in items)
             for index, actual in enumerate(("actual_q1", "actual_q2"))]
    if min(scale) <= 1e-9:
        raise CalibrationIdentificationError("流量激励不足，无法辨识两输入非线性模型")
    model: dict[str, Any] = {
        "kind": "quadratic_flow_response", "center": center, "scale": scale,
        "coefficients": [0.0] * 5,
        "terms": ["x1", "x2", "x1^2", "x1*x2", "x2^2"],
        "output": "diameter_change_um",
        "baseline_handling": "per_trial_feature_difference",
        "reference_diameter_um": float(median(item.baseline_diameter_um for item in items)),
    }
    matrix = np.asarray([features(model, item.actual_q1, item.actual_q2)
                         - features(model, item.baseline_q1, item.baseline_q2) for item in items])
    observed = np.asarray([item.diameter_change_um for item in items])
    coefficients, _, rank, singular = np.linalg.lstsq(matrix, observed, rcond=None)
    if rank < 5 or singular[0] / singular[-1] > 1e6:
        raise CalibrationIdentificationError("试验点不能独立辨识二次项与交互项；数据已保留")
    residual = matrix @ coefficients - observed
    dof = len(items) - 5
    variance = float(residual @ residual) / max(1, dof)
    covariance = variance * np.linalg.inv(matrix.T @ matrix)
    model.update(
        coefficients=coefficients.tolist(), rank=int(rank),
        condition_number=float(singular[0] / singular[-1]),
        coefficient_standard_errors=np.sqrt(np.maximum(0, np.diag(covariance))).tolist(),
        training_mae_um=float(np.mean(np.abs(residual))),
        training_nrmse=float(np.sqrt(np.mean(residual ** 2))) / max(0.25, float(np.median(np.abs(observed)))),
        training_trial_ids=[item.trial_id for item in items],
        training_bounds=[[min(getattr(item, actual) for item in items),
                          max(getattr(item, actual) for item in items)]
                         for actual in ("actual_q1", "actual_q2")],
    )
    validate_model(model)
    return model


def build_nonlinear_calibration(
    *, config: Any, measurements: Iterable[Any], session_id: str, started_at: str,
    q1_min: float, q1_max: float, q2_min: float, q2_max: float,
    total_flow_max: float, min_q1_q2_gap: float,
) -> Any:
    from .calibration_experiment import (
        CalibrationIdentificationError, PlantCalibrationExperimentResult,
        _fit_fopdt, _validation_metrics,
    )

    items = tuple(measurements)
    model_items = tuple(item for item in items if item.channel != "validation")
    model = fit_response_surface(model_items)
    dynamic = _fit_fopdt(model_items, nonlinear=True)
    gaps = [later.capture_monotonic - earlier.capture_monotonic
            for item in model_items
            for earlier, later in zip(item.response_observations, item.response_observations[1:])
            if later.capture_monotonic > earlier.capture_monotonic]
    delay_resolution_ms = float(median(gaps)) * 1000 if gaps else dynamic.time_constant_ms
    model["dynamics"] = {
        "kind": "first_order_plus_delay", "delay_ms": dynamic.delay_ms,
        "time_constant_ms": dynamic.time_constant_ms, "delay_resolution_ms": delay_resolution_ms,
        "mae_um": dynamic.mae_um, "nrmse": dynamic.nrmse, "sample_count": dynamic.sample_count,
    }
    baseline_diameter = float(median(item.baseline_diameter_um for item in model_items))
    gradients = [model["coefficients"][index] / model["scale"][index] for index in (0, 1)]
    errors = model["coefficient_standard_errors"]
    active = [abs(model["coefficients"][index]) > max(1e-8, 2 * errors[index]) for index in (0, 1)]
    if not any(active):
        raise CalibrationIdentificationError("非线性模型已拟合，但工作点局部斜率接近零或不确定；数据可建模，当前点不能整定固定参数 PID")
    gradients = [value if active[index] else 0.0 for index, value in enumerate(gradients)]
    log_gains = [gradients[index] * model["center"][index] / baseline_diameter for index in (0, 1)]
    denominator = sum(value * value for value in log_gains) + config.sensitivity_allocation_regularization
    allocations = [model["center"][index] * log_gains[index] / denominator for index in (0, 1)]
    mae, nrmse, count = _validation_metrics(
        items, delay_ms=dynamic.delay_ms, time_constant_ms=dynamic.time_constant_ms,
        steady_gain_um_per_output=1.0, predict_change=lambda item: predict_change(model, item),
    )
    bounds = [[max(q1_min, model["training_bounds"][0][0]), min(q1_max, model["training_bounds"][0][1])],
              [max(q2_min, model["training_bounds"][1][0]), min(q2_max, model["training_bounds"][1][1])]]
    for index in (0, 1):
        curvature = abs(2 * model["coefficients"][2 if index == 0 else 4])
        coupling = abs(model["coefficients"][3])
        radius = min(1.0, 0.4 * abs(model["coefficients"][index]) / max(1e-12, curvature + coupling))
        if active[index]:
            bounds[index] = [max(bounds[index][0], model["center"][index] - radius * model["scale"][index]),
                             min(bounds[index][1], model["center"][index] + radius * model["scale"][index])]
    model["pid_local_bounds"] = bounds
    qualified = (count > 0 and mae <= config.validation_mae_limit_um
                 and nrmse <= config.validation_nrmse_limit
                 and dynamic.nrmse <= config.validation_nrmse_limit
                 and model["training_nrmse"] <= config.validation_nrmse_limit)
    delay = max(1.0, dynamic.delay_ms)
    closed_loop = max(config.closed_loop_time_constant_ratio * delay, 2 * delay)
    kp = dynamic.time_constant_ms / (closed_loop + delay)
    ki = kp / max(0.001, min(dynamic.time_constant_ms, 4 * (closed_loop + delay)) / 1000)
    completed_dt = datetime.now(timezone.utc)
    # 末尾四个字段由 build_calibration_record 要求显式传入。二次响应曲面分支目前不产出
    # 它们（per-channel 延迟、基线生成频率与 CV 都不参与该模型），故传 0.0——与改动前
    # 落到 dataclass 默认值的行为一致。是否应按同一份 measurements 实测补建，取决于
    # 响应曲面模型是否需要这些量，待定，故此处显式写明而非靠默认值隐式继承。
    record = build_calibration_record(
        config=config,
        session_id=str(session_id),
        completed_at=completed_dt,
        measurement_source="nonlinear_generation_response",
        response_delay_median_ms=delay,
        response_delay_uncertainty_ms=delay_resolution_ms,
        diameter_sensitivity_um_per_output=sum(gradients[index] * allocations[index] for index in (0, 1)),
        q1_control_sign=float(np.sign(gradients[0])),
        q2_control_sign=float(np.sign(gradients[1])),
        q1_output_gain=abs(allocations[0]),
        q2_output_gain=abs(allocations[1]),
        q1_min=bounds[0][0],
        q1_max=bounds[0][1],
        q2_min=bounds[1][0],
        q2_max=bounds[1][1],
        total_flow_max=min(total_flow_max, max(item.actual_q1 + item.actual_q2 for item in model_items)),
        min_q1_q2_gap=min_q1_q2_gap,
        baseline_q1=model["center"][0],
        baseline_q2=model["center"][1],
        baseline_diameter_um=baseline_diameter,
        q1_log_diameter_sensitivity=log_gains[0],
        q2_log_diameter_sensitivity=log_gains[1],
        response_time_constant_ms=dynamic.time_constant_ms,
        controller_kp=kp,
        controller_ki=ki,
        response_time_constant_uncertainty_ms=dynamic.time_constant_uncertainty_ms,
        q1_log_sensitivity_uncertainty=errors[0] / model["scale"][0] * model["center"][0] / baseline_diameter,
        q2_log_sensitivity_uncertainty=errors[1] / model["scale"][1] * model["center"][1] / baseline_diameter,
        model_fit_method="quadratic_response_fopdt",
        model_fit_mae_um=dynamic.mae_um,
        model_fit_nrmse=dynamic.nrmse,
        validation_mae_um=mae,
        validation_nrmse=nrmse,
        validation_sample_count=count,
        validated_for_pi=qualified,
        validated_for_mpc=qualified and dynamic.sample_count >= 12,
        q1_response_delay_ms=0.0,
        q2_response_delay_ms=0.0,
        baseline_generation_frequency_hz=0.0,
        baseline_diameter_cv=0.0,
        nonlinear_model=model,
    )
    completed = completed_dt.isoformat()
    return PlantCalibrationExperimentResult(
        record=record, config=config, measurements=items, q1_sensitivity_um_per_flow=gradients[0],
        q2_sensitivity_um_per_flow=gradients[1], started_at=started_at, completed_at=completed, session_id=session_id,
    )
