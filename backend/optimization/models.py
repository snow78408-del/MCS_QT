from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class OptimizationPhase(str, Enum):
    IDLE = "IDLE"
    INITIAL_SAMPLING = "INITIAL_SAMPLING"
    BAYESIAN_SEARCH = "BAYESIAN_SEARCH"
    CONFIRMING = "CONFIRMING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class OptimizationCandidate:
    candidate_id: int
    q1: float
    q2: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OptimizationObservation:
    candidate_id: int
    q1: float
    q2: float
    diameter_um: float | None
    frequency_hz: float | None
    diameter_cv_percent: float | None
    valid_droplets: int
    invalid_fraction: float = 0.0
    measurement_valid: bool = True
    invalid_reason: str = ""


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    q1: float
    q2: float
    diameter_um: float
    frequency_hz: float | None
    diameter_cv_percent: float | None
    objective: float
    observation_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OptimizationStatus:
    phase: str = OptimizationPhase.IDLE.value
    observation_count: int = 0
    invalid_observation_count: int = 0
    confirmation_count: int = 0
    current_candidate: OptimizationCandidate | None = None
    best_operating_point: OperatingPoint | None = None
    completed: bool = False
    failed: bool = False
    reason: str = ""
    objective_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["current_candidate"] = (
            None if self.current_candidate is None else self.current_candidate.to_dict()
        )
        data["best_operating_point"] = (
            None if self.best_operating_point is None else self.best_operating_point.to_dict()
        )
        return data
