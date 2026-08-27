from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class BayesianOptimizationConfig:
    """Configuration for one safe, finite operating-point search.

    The bounds and response timing are deliberately mandatory.  A search is
    allowed to move real pumps, so silently deriving either from hardware
    limits or serial reply latency would be unsafe.
    """

    target_diameter_um: float
    q1_min: float
    q1_max: float
    q2_min: float
    q2_max: float
    measured_response_delay_ms: float
    settling_time_ms: float
    response_delay_source: str
    response_delay_uncertainty_ms: float = 0.0
    target_frequency_hz: float | None = None
    diameter_relative_tolerance: float = 0.05
    frequency_relative_tolerance: float = 0.10
    initial_sample_count: int = 8
    maximum_observations: int = 24
    confirmation_count: int = 2
    minimum_valid_droplets: int = 5
    candidate_timeout_ms: float = 120_000.0
    invalid_retry_limit: int = 3
    min_q1_q2_gap: float = 0.2
    total_flow_max: float = 8000.0
    cv_weight: float = 0.02
    invalid_fraction_weight: float = 1.0
    movement_weight: float = 0.01
    gp_noise: float = 1e-5
    acquisition_candidates: int = 2048
    random_seed: int = 20250827

    def __post_init__(self) -> None:
        numeric = {
            "target_diameter_um": self.target_diameter_um,
            "q1_min": self.q1_min,
            "q1_max": self.q1_max,
            "q2_min": self.q2_min,
            "q2_max": self.q2_max,
            "measured_response_delay_ms": self.measured_response_delay_ms,
            "settling_time_ms": self.settling_time_ms,
            "response_delay_uncertainty_ms": self.response_delay_uncertainty_ms,
            "candidate_timeout_ms": self.candidate_timeout_ms,
            "min_q1_q2_gap": self.min_q1_q2_gap,
        }
        for name, value in numeric.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.target_diameter_um <= 0.0:
            raise ValueError("target_diameter_um must be positive")
        if self.target_frequency_hz is not None and (
            not math.isfinite(float(self.target_frequency_hz)) or float(self.target_frequency_hz) <= 0.0
        ):
            raise ValueError("target_frequency_hz must be finite and positive")
        if not (0.0 < self.q1_min < self.q1_max):
            raise ValueError("q1 bounds must satisfy 0 < q1_min < q1_max")
        if not (0.0 < self.q2_min < self.q2_max):
            raise ValueError("q2 bounds must satisfy 0 < q2_min < q2_max")
        if self.measured_response_delay_ms <= 0.0:
            raise ValueError("a measured physical/visual response delay is required")
        if self.response_delay_uncertainty_ms < 0.0:
            raise ValueError("response_delay_uncertainty_ms must be non-negative")
        conservative_delay_ms = self.measured_response_delay_ms + self.response_delay_uncertainty_ms
        if self.settling_time_ms < conservative_delay_ms:
            raise ValueError(
                "settling_time_ms must not be shorter than measured response delay plus uncertainty"
            )
        if self.candidate_timeout_ms <= self.settling_time_ms:
            raise ValueError("candidate_timeout_ms must be longer than settling_time_ms")
        source = str(self.response_delay_source or "").strip().lower()
        invalid_source_markers = {
            "unknown", "unmeasured", "serial", "reply", "readback", "device_readback",
            "串口", "应答", "回读", "未测",
        }
        if not source or any(marker in source for marker in invalid_source_markers):
            raise ValueError("response_delay_source must identify an independently measured response")
        if self.initial_sample_count < 2:
            raise ValueError("initial_sample_count must be at least 2")
        if self.maximum_observations < self.initial_sample_count:
            raise ValueError("maximum_observations must cover the initial samples")
        if self.confirmation_count < 1 or self.minimum_valid_droplets < 1:
            raise ValueError("confirmation_count and minimum_valid_droplets must be positive")
        if self.invalid_retry_limit < 0 or self.acquisition_candidates < 32:
            raise ValueError("invalid retry/acquisition settings are not usable")
        if self.min_q1_q2_gap <= 0.0:
            raise ValueError("min_q1_q2_gap must be positive")
        if self.total_flow_max <= 0.0:
            raise ValueError("total_flow_max must be positive")
        if not self._has_feasible_region():
            raise ValueError("configured Q1/Q2 bounds contain no safe feasible point")

    def _has_feasible_region(self) -> bool:
        return (
            self.q1_max >= self.q2_min + self.min_q1_q2_gap
            and self.q1_min + self.q2_min <= self.total_flow_max
        )
