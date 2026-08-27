from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .config import PIDConfig
from .models import AdaptivePIDState, PIDInput
from .safety import clamp, rate_limit


@dataclass(slots=True)
class AdaptivePIDManager:
    config: PIDConfig
    _errors: deque[float] = field(default_factory=lambda: deque(maxlen=80))
    _state: AdaptivePIDState | None = None

    def __post_init__(self) -> None:
        self._state = AdaptivePIDState(
            kp=float(self.config.base_kp),
            ki=float(self.config.base_ki),
            kd=float(self.config.base_kd),
            base_kp=float(self.config.base_kp),
            base_ki=float(self.config.base_ki),
            base_kd=float(self.config.base_kd),
            reason="base parameters",
        )

    @property
    def state(self) -> AdaptivePIDState:
        assert self._state is not None
        return self._state

    def reset(self) -> None:
        self._errors.clear()
        self.__post_init__()

    def update(self, pid_input: PIDInput, error: float) -> AdaptivePIDState:
        state = self.state
        self._errors.append(float(error))
        state.sample_count = len(self._errors)
        state.update_count += 1

        if len(self._errors) < int(self.config.adaptive_min_sample_count):
            state.active = False
            state.reason = "insufficient samples; keep base PID"
            state.kp = self._step(state.kp, state.base_kp, self.config.kp_step_limit)
            state.ki = self._step(state.ki, state.base_ki, self.config.ki_step_limit)
            state.kd = self._step(state.kd, state.base_kd, self.config.kd_step_limit)
            return state

        if state.update_count % max(1, int(self.config.adaptive_update_interval)) != 0:
            state.reason = "waiting adaptive update interval"
            return state

        # Feedback adaptation must remain available before the optional
        # disturbance model has collected and validated enough data. Model
        # confidence gates only model-derived hints, never the PID tuner itself.
        prediction = pid_input.disturbance_prediction
        confidence = float(getattr(prediction, "confidence", 0.0) or 0.0)
        model_hint_active = confidence >= float(self.config.adaptive_confidence_threshold)
        mean_abs = sum(abs(v) for v in self._errors) / len(self._errors)
        recent_slope = self._errors[-1] - self._errors[-min(5, len(self._errors))]
        predicted = (
            abs(float(getattr(prediction, "predicted_diameter_change_um", 0.0) or 0.0))
            if model_hint_active else 0.0
        )
        # Only an observed/identified actuator delay may tune derivative action.
        # A model's horizon or inferred delay is not a physical delay estimate.
        delay_ms = float(pid_input.pump_response_delay_ms or 0.0)

        recent = list(self._errors)[-min(12, len(self._errors)):]
        sign_changes = sum(
            1 for left, right in zip(recent, recent[1:])
            if left * right < 0.0
        )
        oscillating = sign_changes >= max(2, len(recent) // 3)
        signed_mean = sum(recent) / len(recent)

        kp_scale = 1.0 + min(0.75, mean_abs / 80.0 + predicted / 120.0)
        if oscillating:
            kp_scale *= 0.85
        kp_target = state.base_kp * kp_scale
        # Increase integral action only for a persistent one-sided offset;
        # suppress it when errors alternate to avoid exciting oscillation.
        persistent_offset = abs(signed_mean) > max(1.0, mean_abs * 0.55)
        ki_target = state.base_ki * (1.2 if persistent_offset and not oscillating else 0.7 if oscillating else 1.0)
        kd_target = state.base_kd + min(
            0.2,
            delay_ms / 5000.0 + abs(recent_slope) / 200.0 + (0.01 if oscillating else 0.0),
        )

        state.kp = clamp(
            rate_limit(kp_target, state.kp, self.config.kp_step_limit),
            self.config.kp_min,
            self.config.kp_max,
        )
        state.ki = clamp(
            rate_limit(ki_target, state.ki, self.config.ki_step_limit),
            self.config.ki_min,
            self.config.ki_max,
        )
        state.kd = clamp(
            rate_limit(kd_target, state.kd, self.config.kd_step_limit),
            self.config.kd_min,
            self.config.kd_max,
        )
        state.active = True
        state.reason = (
            "adaptive PID updated with model hint"
            if model_hint_active else "adaptive PID updated from feedback history"
        )
        return state

    @staticmethod
    def _step(current: float, target: float, limit: float) -> float:
        return rate_limit(float(target), float(current), float(limit))
