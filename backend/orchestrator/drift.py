from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(slots=True)
class DriftSupervisorConfig:
    error_threshold_fraction: float = 0.05
    minimum_error_um: float = 2.0
    integral_threshold_fraction: float = 0.70
    consecutive_periods: int = 5
    healthy_clear_periods: int = 3


@dataclass(slots=True)
class DriftStatus:
    reoptimization_recommended: bool = False
    consecutive_drift_periods: int = 0
    consecutive_healthy_periods: int = 0
    diameter_error_um: float = 0.0
    error_threshold_um: float = 0.0
    integral_fraction: float = 0.0
    actuator_saturated: bool = False
    reason: str = "waiting for PID observations"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DriftSupervisor:
    """Advisory-only detector for loss of local PID authority."""

    def __init__(self, config: DriftSupervisorConfig | None = None) -> None:
        self.config = config or DriftSupervisorConfig()
        self._status = DriftStatus()

    def reset(self) -> None:
        self._status = DriftStatus()

    def observe(
        self,
        *,
        target_diameter_um: float,
        diameter_error_um: float,
        integral_state: float,
        integral_limit: float,
        actuator_saturated: bool,
    ) -> DriftStatus:
        threshold = max(
            float(self.config.minimum_error_um),
            abs(float(target_diameter_um)) * float(self.config.error_threshold_fraction),
        )
        integral_fraction = (
            0.0
            if abs(float(integral_limit)) <= 1e-12
            else min(1.0, abs(float(integral_state)) / abs(float(integral_limit)))
        )
        values_finite = all(
            math.isfinite(value)
            for value in (
                float(target_diameter_um),
                float(diameter_error_um),
                float(integral_state),
                float(integral_limit),
            )
        )
        error_high = values_finite and abs(float(diameter_error_um)) > threshold
        integral_high = values_finite and integral_fraction >= float(
            self.config.integral_threshold_fraction
        )
        drifting = bool(error_high or integral_high or actuator_saturated)
        status = self._status
        if drifting:
            status.consecutive_drift_periods += 1
            status.consecutive_healthy_periods = 0
        else:
            status.consecutive_healthy_periods += 1
            status.consecutive_drift_periods = 0

        if status.consecutive_drift_periods >= int(self.config.consecutive_periods):
            status.reoptimization_recommended = True
        elif (
            status.reoptimization_recommended
            and status.consecutive_healthy_periods >= int(self.config.healthy_clear_periods)
        ):
            status.reoptimization_recommended = False

        reasons: list[str] = []
        if error_high:
            reasons.append("diameter error persistently outside local band")
        if integral_high:
            reasons.append("PID integral is consuming local authority")
        if actuator_saturated:
            reasons.append("local actuator allocation is saturated")
        status.reason = (
            "; ".join(reasons)
            if reasons
            else "local PID operating point remains healthy"
        )
        status.diameter_error_um = float(diameter_error_um)
        status.error_threshold_um = threshold
        status.integral_fraction = integral_fraction
        status.actuator_saturated = bool(actuator_saturated)
        return self.status()

    def status(self) -> DriftStatus:
        return DriftStatus(**asdict(self._status))
