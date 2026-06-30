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

        recommended = getattr(prediction, "recommended_feedforward", None)
        if recommended is None:
            change = float(getattr(prediction, "predicted_diameter_change_um", 0.0) or 0.0)
            recommended = -float(self.config.feedforward_gain) * change
        if not is_finite(recommended):
            self._last_u_ff = 0.0
            return FeedforwardResult(0.0, False, "invalid feedforward value", confidence)

        limited = clamp(float(recommended), self.config.feedforward_min, self.config.feedforward_max)
        limited = rate_limit(limited, self._last_u_ff, self.config.feedforward_rate_limit)
        self._last_u_ff = limited
        return FeedforwardResult(limited, True, "feedforward active", confidence)
