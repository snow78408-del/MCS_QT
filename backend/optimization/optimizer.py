from __future__ import annotations

import math

import numpy as np

from .config import BayesianOptimizationConfig
from ..pump_hardware.invariants import q1_is_strictly_above_q2
from .models import (
    OperatingPoint,
    OptimizationCandidate,
    OptimizationObservation,
    OptimizationPhase,
    OptimizationStatus,
)


class SafeBayesianOptimizer:
    """Sequential two-dimensional BO with fail-closed experiment handling."""

    def __init__(
        self,
        config: BayesianOptimizationConfig,
        *,
        initial_point: tuple[float, float] | None = None,
    ) -> None:
        self.config = config
        self._rng = np.random.default_rng(int(config.random_seed))
        self._seed = None
        if initial_point is not None and self.is_feasible(*initial_point):
            self._seed = (float(initial_point[0]), float(initial_point[1]))
        lhs_count = max(0, int(config.initial_sample_count) - (1 if self._seed is not None else 0))
        self._initial = self._latin_hypercube_candidates(lhs_count) if lhs_count else []
        self._x: list[tuple[float, float]] = []
        self._y: list[float] = []
        self._observations: list[OptimizationObservation] = []
        self._candidate_counter = 0
        self._current: OptimizationCandidate | None = None
        self._best: OperatingPoint | None = None
        self._confirmation_point: OperatingPoint | None = None
        self._confirmed_point: OperatingPoint | None = None
        self._confirmations = 0
        self._invalid_for_current = 0
        self._invalid_total = 0
        self._phase = OptimizationPhase.INITIAL_SAMPLING
        self._reason = "waiting for first safe candidate"
        self._failure_kind = ""

    def ask(self) -> OptimizationCandidate:
        if self._phase in {OptimizationPhase.COMPLETED, OptimizationPhase.FAILED}:
            raise RuntimeError(f"optimizer is terminal: {self._phase.value}")
        if self._current is not None:
            return self._current

        if self._confirmations > 0 and self._confirmation_point is not None:
            q1, q2 = self._confirmation_point.q1, self._confirmation_point.q2
            reason = "repeat target-feasible point for independent confirmation"
            self._phase = OptimizationPhase.CONFIRMING
        elif self._seed is not None and not self._x:
            q1, q2 = self._seed
            reason = "current verified operating-point seed"
            self._phase = OptimizationPhase.INITIAL_SAMPLING
        elif len(self._x) < len(self._initial) + (1 if self._seed is not None else 0):
            index = len(self._x) - (1 if self._seed is not None else 0)
            q1, q2 = self._initial[index]
            reason = "Latin-hypercube initialization"
            self._phase = OptimizationPhase.INITIAL_SAMPLING
        else:
            q1, q2 = self._acquire_expected_improvement()
            reason = "Gaussian-process expected improvement"
            self._phase = OptimizationPhase.BAYESIAN_SEARCH

        self._candidate_counter += 1
        self._current = OptimizationCandidate(self._candidate_counter, float(q1), float(q2), reason)
        self._invalid_for_current = 0
        return self._current

    def reject_current(self, reason: str) -> None:
        if self._current is None:
            return
        self._invalid_for_current += 1
        self._invalid_total += 1
        self._reason = str(reason or "invalid experimental observation")
        if self._invalid_for_current > int(self.config.invalid_retry_limit):
            self._phase = OptimizationPhase.FAILED
            self._failure_kind = "measurement_quality"
            self._reason = (
                f"candidate {self._current.candidate_id} exceeded invalid retry limit: {self._reason}"
            )
            self._current = None

    def tell(self, observation: OptimizationObservation) -> None:
        current = self._current
        if current is None:
            raise RuntimeError("tell() requires a current candidate from ask()")
        if int(observation.candidate_id) != int(current.candidate_id):
            raise ValueError("observation does not belong to the active candidate")
        if not self._observation_is_usable(observation):
            self.reject_current(observation.invalid_reason or "measurement quality gate failed")
            return

        objective = self.objective(observation)
        self._observations.append(observation)
        self._x.append((float(observation.q1), float(observation.q2)))
        self._y.append(float(objective))
        point = OperatingPoint(
            q1=float(observation.q1),
            q2=float(observation.q2),
            diameter_um=float(observation.diameter_um),
            frequency_hz=(None if observation.frequency_hz is None else float(observation.frequency_hz)),
            diameter_cv_percent=(
                None if observation.diameter_cv_percent is None else float(observation.diameter_cv_percent)
            ),
            objective=float(objective),
            observation_count=len(self._observations),
        )
        if self._best is None or point.objective < self._best.objective:
            self._best = point

        success = self._meets_target(observation)
        if success:
            same_point = (
                self._confirmation_point is not None
                and abs(point.q1 - self._confirmation_point.q1) <= 1e-9
                and abs(point.q2 - self._confirmation_point.q2) <= 1e-9
            )
            if self._confirmation_point is None or not same_point:
                self._confirmation_point = point
                self._confirmations = 1
            else:
                self._confirmations += 1
        else:
            self._confirmations = 0
            self._confirmation_point = None
        self._current = None
        self._invalid_for_current = 0

        if self._confirmations >= int(self.config.confirmation_count):
            self._phase = OptimizationPhase.COMPLETED
            self._confirmed_point = self._confirmation_point
            self._reason = "target confirmed in independent settled windows"
            return
        if len(self._observations) >= int(self.config.maximum_observations):
            self._phase = OptimizationPhase.FAILED
            self._failure_kind = "target_not_found"
            self._reason = "maximum valid observations reached before target confirmation"
            return
        self._reason = "target met; confirming" if success else "search continues"

    def objective(self, observation: OptimizationObservation) -> float:
        diameter = float(observation.diameter_um)
        size_error = abs(diameter - float(self.config.target_diameter_um)) / float(self.config.target_diameter_um)
        value = max(0.0, size_error - float(self.config.diameter_relative_tolerance))
        if self.config.target_frequency_hz is not None:
            frequency = max(0.0, float(observation.frequency_hz or 0.0))
            frequency_error = abs(frequency - float(self.config.target_frequency_hz)) / float(
                self.config.target_frequency_hz
            )
            value += max(0.0, frequency_error - float(self.config.frequency_relative_tolerance))
        if observation.diameter_cv_percent is not None:
            value += float(self.config.cv_weight) * max(0.0, float(observation.diameter_cv_percent)) / 100.0
        value += float(self.config.invalid_fraction_weight) * max(0.0, float(observation.invalid_fraction))
        q1_span = max(1e-12, float(self.config.q1_max - self.config.q1_min))
        q2_span = max(1e-12, float(self.config.q2_max - self.config.q2_min))
        if self._observations:
            previous = self._observations[-1]
            movement = abs(float(observation.q1) - float(previous.q1)) / q1_span
            movement += abs(float(observation.q2) - float(previous.q2)) / q2_span
            value += float(self.config.movement_weight) * movement
        return float(value)

    def status(self) -> OptimizationStatus:
        return OptimizationStatus(
            phase=self._phase.value,
            observation_count=len(self._observations),
            invalid_observation_count=self._invalid_total,
            confirmation_count=self._confirmations,
            current_candidate=self._current,
            best_operating_point=self._best,
            confirmed_operating_point=self._confirmed_point,
            completed=self._phase == OptimizationPhase.COMPLETED,
            failed=self._phase == OptimizationPhase.FAILED,
            reason=self._reason,
            failure_kind=self._failure_kind,
            objective_history=list(self._y),
        )

    def is_feasible(self, q1: float, q2: float) -> bool:
        cfg = self.config
        return bool(
            cfg.q1_min <= q1 <= cfg.q1_max
            and cfg.q2_min <= q2 <= cfg.q2_max
            and q1_is_strictly_above_q2(q1, q2, cfg.min_q1_q2_gap)
            and q1 + q2 <= cfg.total_flow_max
        )

    def _observation_is_usable(self, observation: OptimizationObservation) -> bool:
        return bool(
            observation.measurement_valid
            and observation.diameter_um is not None
            and math.isfinite(float(observation.diameter_um))
            and float(observation.diameter_um) > 0.0
            and int(observation.valid_droplets) >= int(self.config.minimum_valid_droplets)
            and self.is_feasible(float(observation.q1), float(observation.q2))
        )

    def _meets_target(self, observation: OptimizationObservation) -> bool:
        size_error = abs(float(observation.diameter_um) - self.config.target_diameter_um) / self.config.target_diameter_um
        if size_error > self.config.diameter_relative_tolerance:
            return False
        if self.config.target_frequency_hz is not None:
            if observation.frequency_hz is None:
                return False
            frequency_error = abs(float(observation.frequency_hz) - self.config.target_frequency_hz) / self.config.target_frequency_hz
            if frequency_error > self.config.frequency_relative_tolerance:
                return False
        return True

    def _latin_hypercube_candidates(self, count: int) -> list[tuple[float, float]]:
        for _ in range(200):
            q1_bins = (np.arange(count) + self._rng.random(count)) / count
            q2_bins = (np.arange(count) + self._rng.random(count)) / count
            self._rng.shuffle(q2_bins)
            q1s = self.config.q1_min + q1_bins * (self.config.q1_max - self.config.q1_min)
            q2s = self.config.q2_min + q2_bins * (self.config.q2_max - self.config.q2_min)
            candidates = [(float(q1), float(q2)) for q1, q2 in zip(q1s, q2s) if self.is_feasible(q1, q2)]
            if len(candidates) >= count:
                return candidates[:count]
        candidates = self._random_feasible(max(count * 100, 1000))
        if len(candidates) < count:
            raise ValueError("unable to sample enough feasible Q1/Q2 points")
        return candidates[:count]

    def _random_feasible(self, count: int) -> list[tuple[float, float]]:
        q1 = self._rng.uniform(self.config.q1_min, self.config.q1_max, count)
        q2 = self._rng.uniform(self.config.q2_min, self.config.q2_max, count)
        return [(float(a), float(b)) for a, b in zip(q1, q2) if self.is_feasible(float(a), float(b))]

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        mins = np.asarray([self.config.q1_min, self.config.q2_min], dtype=float)
        spans = np.asarray(
            [self.config.q1_max - self.config.q1_min, self.config.q2_max - self.config.q2_min],
            dtype=float,
        )
        return (values - mins) / spans

    @staticmethod
    def _matern52(left: np.ndarray, right: np.ndarray, length_scale: float = 0.35) -> np.ndarray:
        delta = left[:, None, :] - right[None, :, :]
        distance = np.sqrt(np.sum(delta * delta, axis=2)) / max(1e-9, float(length_scale))
        root5 = math.sqrt(5.0)
        return (1.0 + root5 * distance + 5.0 * distance * distance / 3.0) * np.exp(-root5 * distance)

    def _acquire_expected_improvement(self) -> tuple[float, float]:
        feasible = self._random_feasible(int(self.config.acquisition_candidates) * 3)
        if len(feasible) < 32:
            raise RuntimeError("safe acquisition candidate generation failed")
        candidates = np.asarray(feasible[: int(self.config.acquisition_candidates)], dtype=float)
        x_train = self._normalize(np.asarray(self._x, dtype=float))
        x_test = self._normalize(candidates)
        y = np.asarray(self._y, dtype=float)
        y_mean = float(np.mean(y))
        y_scale = max(1e-9, float(np.std(y)))
        y_scaled = (y - y_mean) / y_scale
        kernel = self._matern52(x_train, x_train)
        kernel += np.eye(len(kernel)) * max(1e-10, float(self.config.gp_noise))
        try:
            chol = np.linalg.cholesky(kernel)
            alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_scaled))
            cross = self._matern52(x_train, x_test)
            mean = cross.T @ alpha
            solved = np.linalg.solve(chol, cross)
            variance = np.maximum(1e-12, 1.0 - np.sum(solved * solved, axis=0))
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(kernel)
            cross = self._matern52(x_train, x_test)
            mean = cross.T @ inverse @ y_scaled
            variance = np.maximum(1e-12, 1.0 - np.sum(cross * (inverse @ cross), axis=0))
        sigma = np.sqrt(variance)
        best = float(np.min(y_scaled))
        improvement = best - mean - 0.01
        z = improvement / sigma
        cdf = np.asarray([0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))) for value in z])
        pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        expected = improvement * cdf + sigma * pdf
        expected[sigma <= 1e-12] = 0.0
        # Avoid exact repeats except when explicitly confirming a successful point.
        for previous in np.asarray(self._x, dtype=float):
            distance = np.linalg.norm(self._normalize(candidates) - self._normalize(previous[None, :]), axis=1)
            expected[distance < 1e-4] = -np.inf
        index = int(np.argmax(expected))
        return float(candidates[index, 0]), float(candidates[index, 1])
