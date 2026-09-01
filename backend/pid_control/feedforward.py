from __future__ import annotations

import time

from .config import PIDConfig
from .models import FeedforwardResult, PIDInput
from .safety import clamp, is_finite, rate_limit


class FeedforwardCompensator:
    def __init__(self, config: PIDConfig) -> None:
        self.config = config
        self._last_u_ff = 0.0

    def reset(self) -> None:
        self._last_u_ff = 0.0

    def compute(self, pid_input: PIDInput) -> FeedforwardResult:
        if not self.config.feedforward_enabled:
            self._last_u_ff = 0.0
            return FeedforwardResult(0.0, False, "feedforward disabled")
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
