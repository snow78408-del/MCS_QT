from __future__ import annotations

from dataclasses import dataclass

from .config import PIDConfig
from .safety import clamp


@dataclass(slots=True)
class PIDParameterManager:
    config: PIDConfig

    def clamp_gains(self, kp: float, ki: float, kd: float) -> tuple[float, float, float]:
        return (
            clamp(kp, self.config.kp_min, self.config.kp_max),
            clamp(ki, self.config.ki_min, self.config.ki_max),
            clamp(kd, self.config.kd_min, self.config.kd_max),
        )

    def base_gains(self) -> tuple[float, float, float]:
        return (
            float(self.config.base_kp),
            float(self.config.base_ki),
            float(self.config.base_kd),
        )
