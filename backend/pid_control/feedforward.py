from __future__ import annotations

import time
import math

from .config import PIDConfig, PIDControlMode
from .models import FeedforwardResult, PIDInput
from .safety import clamp, is_finite, rate_limit


class FeedforwardCompensator:
    def __init__(self, config: PIDConfig) -> None:
        self.config = config
        self._last_u_ff = 0.0

    def reset(self) -> None:
        self._last_u_ff = 0.0

    def compute(self, pid_input: PIDInput) -> FeedforwardResult:
        selected = self.config.disturbance_feedforward_enabled
        if selected is None:
            selected = (self.config.feedforward_enabled
                        and self.config.control_mode == PIDControlMode.ADAPTIVE_PID_WITH_FEEDFORWARD.value)
        if not selected:
            self._last_u_ff = 0.0
            return FeedforwardResult(0.0, False, "feedforward disabled")
        if self.config.log_sensitivity_calibrated:
            # The legacy disturbance gain is output/um, not log-diameter/um.
            # A target/plant step calibration does not identify that gain.
            self._last_u_ff = 0.0
            return FeedforwardResult(0.0, False, "扰动前馈缺少对数尺寸输出单位的独立标定")
        if not pid_input.vision_valid:
            self._last_u_ff = 0.0
            return FeedforwardResult(0.0, False, "vision invalid")
        if not pid_input.pump_communication_ok:
            self._last_u_ff = 0.0
            return FeedforwardResult(0.0, False, "pump communication abnormal")
        if not pid_input.system_running:
            self._last_u_ff = 0.0
            return FeedforwardResult(0.0, False, "system not running")

        prediction = pid_input.disturbance_prediction
        if prediction is None:
            self._last_u_ff = 0.0
            return FeedforwardResult(0.0, False, "no disturbance prediction")

        ready = bool(getattr(prediction, "model_ready", False))
        valid = bool(getattr(prediction, "model_valid", False))
        confidence = float(getattr(prediction, "confidence", 0.0) or 0.0)
        ts = float(getattr(prediction, "timestamp", 0.0) or 0.0)
        age_ms = (time.time() - ts) * 1000.0 if ts > 0 else float("inf")
        if not ready or not valid:
            self._last_u_ff = 0.0
            return FeedforwardResult(0.0, False, "model not ready or invalid", confidence)
        if confidence < float(self.config.feedforward_confidence_threshold):
            self._last_u_ff = 0.0
            return FeedforwardResult(0.0, False, "confidence below threshold", confidence)
        if age_ms > float(self.config.feedforward_timeout_ms):
            self._last_u_ff = 0.0
            return FeedforwardResult(0.0, False, "prediction stale", confidence)

        if not self.config.feedforward_calibrated:
            self._last_u_ff = 0.0
            return FeedforwardResult(0.0, False, "feedforward plant gain not calibrated", confidence)

        # Predictive compensation is only feedforward when an exogenous event
        # is observed before its diameter effect. This gate is intentionally
        # unconditional; a model forecast alone cannot manufacture causality.
        leading_available = bool(getattr(prediction, "leading_signal_available", False))
        lead_ms = max(0.0, float(getattr(prediction, "signal_lead_time_ms", 0.0) or 0.0))
        measured_delay_ms = max(0.0, float(pid_input.pump_response_delay_ms or 0.0))
        required_lead_ms = measured_delay_ms + max(0.0, float(self.config.feedforward_min_lead_margin_ms))
        if measured_delay_ms <= 0.0:
            self._last_u_ff = 0.0
            return FeedforwardResult(0.0, False, "physical pump response delay is unmeasured", confidence)
        prediction_horizon_ms = max(
            0.0,
            float(getattr(prediction, "prediction_horizon_ms", 0.0) or 0.0),
        )
        if prediction_horizon_ms < measured_delay_ms:
            self._last_u_ff = 0.0
            return FeedforwardResult(
                0.0,
                False,
                f"prediction horizon is shorter than pump delay ({prediction_horizon_ms:.0f} < {measured_delay_ms:.0f} ms)",
                confidence,
            )
        if not leading_available:
            self._last_u_ff = 0.0
            return FeedforwardResult(0.0, False, "no causal leading disturbance signal", confidence)
        if lead_ms < required_lead_ms:
            self._last_u_ff = 0.0
            return FeedforwardResult(
                0.0,
                False,
                f"leading signal is too late ({lead_ms:.0f} < {required_lead_ms:.0f} ms)",
                confidence,
            )

        weight = float(getattr(prediction, "feedforward_weight", 1.0) or 0.0)
        if weight <= 0.0:
            self._last_u_ff = 0.0
            stage = str(getattr(prediction, "control_stage", "") or "")
            return FeedforwardResult(0.0, False, f"feedforward gated by stage {stage}".strip(), confidence)

        residual_value = getattr(prediction, "predicted_disturbance_residual_um", None)
        if residual_value is None:
            self._last_u_ff = 0.0
            return FeedforwardResult(
                0.0,
                False,
                "prediction does not provide a disturbance residual",
                confidence,
            )
        residual = float(residual_value or 0.0)
        recommended = -float(self.config.feedforward_gain) * residual * weight
        if not is_finite(recommended):
            self._last_u_ff = 0.0
            return FeedforwardResult(0.0, False, "invalid feedforward value", confidence)

        authority = max(0.0, float(self.config.feedforward_max_output_fraction)) * min(
            abs(float(self.config.output_min)),
            abs(float(self.config.output_max)),
        )
        lower = max(float(self.config.feedforward_min), -authority)
        upper = min(float(self.config.feedforward_max), authority)
        limited = clamp(float(recommended), lower, upper)
        limited = rate_limit(limited, self._last_u_ff, self.config.feedforward_rate_limit)
        self._last_u_ff = limited
        return FeedforwardResult(limited, True, "feedforward active", confidence)


class TargetFeedforward:
    """Invert the calibrated local log model along the existing pump allocator."""

    @staticmethod
    def compute(config: PIDConfig, target: float, *, q1_base: float, q2_base: float,
                c1: float, c2: float, output_low: float, output_high: float,
                q1_bounds: tuple[float, float], q2_bounds: tuple[float, float]) -> FeedforwardResult:
        if not config.target_feedforward_enabled:
            return FeedforwardResult(0.0, False, "目标前馈未启用")
        if not config.target_feedforward_calibrated or not config.log_sensitivity_calibrated:
            return FeedforwardResult(0.0, False, "目标前馈需要通过独立验证的生成区标定")
        baselines = (config.feedforward_baseline_q1, config.feedforward_baseline_q2,
                     config.feedforward_baseline_diameter_um, target)
        if not all(math.isfinite(value) and value > 0 for value in baselines):
            return FeedforwardResult(0.0, False, "目标前馈标定基准无效")
        if not (q1_bounds[0] <= q1_base <= q1_bounds[1] and q2_bounds[0] <= q2_base <= q2_bounds[1]):
            return FeedforwardResult(0.0, False, "当前工作点超出局部标定范围")
        low, high = output_low, output_high
        total_slope = c1 + c2
        if total_slope > 0:
            high = min(high, (config.total_flow_max - q1_base - q2_base) / total_slope)
        elif total_slope < 0:
            low = max(low, (config.total_flow_max - q1_base - q2_base) / total_slope)
        elif q1_base + q2_base > config.total_flow_max:
            return FeedforwardResult(0.0, False, "工作点超过总流量上限")
        if low > high:
            return FeedforwardResult(0.0, False, "没有可用的目标前馈流量范围")

        def log_diameter(output: float) -> float:
            q1, q2 = q1_base + c1 * output, q2_base + c2 * output
            return (math.log(config.feedforward_baseline_diameter_um)
                    + config.q1_log_diameter_sensitivity * math.log(q1 / config.feedforward_baseline_q1)
                    + config.q2_log_diameter_sensitivity * math.log(q2 / config.feedforward_baseline_q2))

        desired = math.log(target)
        # This allocator's derivative sum(g_j*c_j/Q_j) is nonnegative.
        if not log_diameter(low) - 1e-10 <= desired <= log_diameter(high) + 1e-10:
            return FeedforwardResult(0.0, False, "目标超出局部模型可达范围，保留 PI 调节")
        for _ in range(50):
            middle = (low + high) * 0.5
            if log_diameter(middle) < desired:
                low = middle
            else:
                high = middle
        output = (low + high) * 0.5
        if abs(output) < 1e-10:
            output = 0.0
        return FeedforwardResult(output, True, "局部标定模型目标前馈；与 PI 共用限幅和调流事务")
