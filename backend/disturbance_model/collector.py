from __future__ import annotations

import time
from typing import Any

from .models import DisturbanceSample


class DisturbanceSampleCollector:
    def build_sample(
        self,
        *,
        recognition: Any,
        pump_state: Any,
        control: Any,
        config: Any,
        system_state: Any,
        dt: float,
        jitter_ms: float,
        disturbance: dict[str, Any] | None = None,
    ) -> DisturbanceSample:
        disturbance = disturbance or {}
        q1_set = float(getattr(pump_state, "q1", 0.0) or 0.0)
        q2_set = float(getattr(pump_state, "q2", 0.0) or 0.0)
        channels = getattr(pump_state, "channels", {}) or {}
        q1_feedback = self._channel_value(channels, "Q1", q1_set)
        q2_feedback = self._channel_value(channels, "Q2", q2_set)
        mean = getattr(recognition, "frame_avg_diameter", None)
        if mean is None:
            mean = getattr(recognition, "avg_diameter", None)
        target = float(getattr(config, "target_diameter", 0.0) or 0.0)
        return DisturbanceSample(
            timestamp=time.time(),
            experiment_id=str(disturbance.get("experiment_id", "")),
            chip_id=str(disturbance.get("chip_id", "")),
            disturbance_name=str(disturbance.get("disturbance_name", "")),
            disturbance_stage=str(disturbance.get("disturbance_stage", "baseline")),
            disturbance_amplitude=float(disturbance.get("disturbance_amplitude", 0.0) or 0.0),
            run_state=str(getattr(system_state, "value", system_state)),
            video_source_type=str(getattr(recognition, "video_source_type", "") or getattr(config, "video_source_type", "")),
            q1_set=q1_set,
            q2_set=q2_set,
            q1_feedback=q1_feedback,
            q2_feedback=q2_feedback,
            pump_response_delay_ms=float(getattr(pump_state, "pump_response_delay_ms", 0.0) or 0.0),
            pump_comm_status=bool(getattr(pump_state, "comm_established", False) and not getattr(pump_state, "last_error", "")),
            droplet_mean_diameter_um=None if mean is None else float(mean),
            droplet_std_um=self._maybe_float(getattr(recognition, "frame_diameter_std", None)),
            droplet_cv=self._maybe_float(getattr(recognition, "frame_diameter_cv", None)),
            droplet_frequency_hz=float(getattr(recognition, "droplet_frequency_hz", 0.0) or 0.0),
            droplet_count_frame=int(getattr(recognition, "frame_droplet_count", 0) or 0),
            droplet_count_total=int(getattr(recognition, "total_droplet_count", 0) or 0),
            valid_sample_count=int(getattr(recognition, "frame_droplet_count", 0) or 0),
            single_cell_rate=self._maybe_float(getattr(recognition, "frame_single_cell_rate", None)),
            vision_valid=bool(getattr(recognition, "valid_for_control", False)),
            vision_invalid_reason=str(getattr(recognition, "reason", "") or getattr(recognition, "control_reason", "")),
            vision_latency_ms=max(0.0, (time.time() - float(getattr(recognition, "timestamp", time.time()))) * 1000.0),
            measurement_noise_est=float(getattr(recognition, "frame_diameter_cv", 0.0) or 0.0),
            image_brightness_mean=float(getattr(recognition, "image_brightness_mean", 0.0) or 0.0),
            focus_score=float(getattr(recognition, "focus_score", 0.0) or 0.0),
            target_diameter_um=target,
            pid_output=float(getattr(control, "final_output", getattr(control, "adjustment", 0.0)) or 0.0),
            feedback_frozen=bool(getattr(control, "freeze_feedback", False)),
            freeze_reason=str(getattr(control, "reason", "") or ""),
            control_cycle_ms=max(0.0, float(dt) * 1000.0),
            control_jitter_ms=max(0.0, float(jitter_ms)),
            temperature_c=self._maybe_float(disturbance.get("temperature_c")),
        )

    @staticmethod
    def _channel_value(channels: dict[str, Any], name: str, fallback: float) -> float:
        channel = channels.get(name)
        if channel is None:
            return float(fallback)
        value = getattr(channel, "actual_flow_rate", None)
        return float(fallback if value is None else value)

    @staticmethod
    def _maybe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None
