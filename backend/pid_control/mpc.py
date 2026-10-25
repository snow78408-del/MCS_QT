from __future__ import annotations

from dataclasses import dataclass
import copy
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from .calibration import PlantCalibrationRecord
from .config import PIDConfig, PIDControlMode
from .diameter_pid import DiameterPIDController
from .models import PIDCommand, PIDInput
from .nonlinear_calibration import features
from ..pump_hardware.invariants import effective_q1_q2_gap


@dataclass(frozen=True)
class MPCSettings:
    horizon: int = 10
    move_weight: float = 0.1
    local_span_fraction: float = 0.2
    max_iterations: int = 100

    def __post_init__(self) -> None:
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or not 2 <= self.horizon <= 30:
            raise ValueError("MPC horizon must be an integer in [2, 30]")
        if not math.isfinite(self.move_weight) or self.move_weight <= 0:
            raise ValueError("MPC move_weight must be finite and positive")
        if not math.isfinite(self.local_span_fraction) or not 0 < self.local_span_fraction <= 0.2:
            raise ValueError("MPC local span must be in (0, 0.2]")
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int) or not 1 <= self.max_iterations <= 500:
            raise ValueError("MPC iteration budget must be in [1, 500]")


class DiameterMPCController(DiameterPIDController):
    """Locally linear, constrained diameter MPC on qualified FOPDT calibration.

    The shared controller contract preserves speculative transaction commits.
    Measurements must pass orchestration's complete post-response-window gate.
    No device communication or clock/scheduler runs inside this controller.
    """

    def __init__(
        self, config: PIDConfig, record: PlantCalibrationRecord,
        sample_period_s: float, settings: MPCSettings | None = None,
    ) -> None:
        if not record.authorized_for_mpc:
            raise ValueError("MPC requires a schema-v3 model with independent MPC validation")
        if not math.isfinite(sample_period_s) or sample_period_s <= 0:
            raise ValueError("MPC sample period must be finite and positive")
        self.record = copy.deepcopy(record)
        self.settings = settings or MPCSettings()
        # Commands are gated until a post-response window exists. Predict at
        # this conservative decision spacing, never at the paper's 100 ms rate.
        self.sample_period_s = max(sample_period_s, (
            record.conservative_response_delay_ms + 3 * (
                record.response_time_constant_ms + record.response_time_constant_uncertainty_ms
            )
        ) / 1000)
        super().__init__(config)
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.offset_um = 0.0
        self.offset_variance = 1.0
        self.last_plan: tuple[tuple[float, float], ...] = ()

    def steady_diameter(self, flows: np.ndarray) -> float:
        record = self.record
        if record.nonlinear_model is not None:
            model = record.nonlinear_model
            delta = features(model, *flows) - features(model, record.baseline_q1, record.baseline_q2)
            return record.baseline_diameter_um + float(delta @ np.asarray(model["coefficients"]))
        return record.baseline_diameter_um * math.exp(
            record.q1_log_diameter_sensitivity * math.log(flows[0] / record.baseline_q1)
            + record.q2_log_diameter_sensitivity * math.log(flows[1] / record.baseline_q2)
        )

    def _gradient(self, flows: np.ndarray) -> np.ndarray:
        model = self.record.nonlinear_model
        if model is None:
            return self.steady_diameter(flows) * np.array([
                self.record.q1_log_diameter_sensitivity, self.record.q2_log_diameter_sensitivity,
            ]) / flows
        x1, x2 = (flows - np.asarray(model["center"])) / np.asarray(model["scale"])
        a, b, c, d, e = model["coefficients"]
        return np.array([a + 2 * c * x1 + d * x2, b + d * x1 + 2 * e * x2]) / model["scale"]

    def _flow_bounds(self, flows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cfg, record = self.config, self.record
        lower = np.maximum([cfg.q1_min, cfg.q2_min], [record.q1_min, record.q2_min])
        upper = np.minimum([cfg.q1_max, cfg.q2_max], [record.q1_max, record.q2_max])
        radius = (upper - lower) * self.settings.local_span_fraction
        if self._local_authority_enabled:
            point = np.asarray(self.operating_point)
            bo_radius = (upper - lower) * cfg.operating_point_local_span_fraction
            lower, upper = np.maximum(lower, point - bo_radius), np.minimum(upper, point + bo_radius)
        # Constrain the entire plan to the domain where this linearization is local.
        return np.maximum(lower, flows - radius), np.minimum(upper, flows + radius)

    def _frozen(self, pid_input: PIDInput, reason: str, mode: str = "MPC") -> PIDCommand:
        command = super()._frozen(pid_input, reason, PIDControlMode.MPC.value)
        command.control_owner = "MPC"
        return command

    def update_input(self, pid_input: PIDInput) -> PIDCommand:
        reason = self._validate_input(pid_input)
        if reason:
            return self._frozen(pid_input, reason)
        if not pid_input.system_running or not math.isfinite(pid_input.dt):
            return self._frozen(pid_input, "MPC requires a running system and finite dt")
        if pid_input.frame_id <= 0 or (self._last_frame_id is not None and pid_input.frame_id <= self._last_frame_id):
            return self._frozen(pid_input, "stale or regressed frame_id")
        flows = np.array([pid_input.current_q1, pid_input.current_q2], dtype=float)
        lower, upper = self._flow_bounds(flows)
        cfg = self.config
        gap = effective_q1_q2_gap(max(cfg.min_q1_q2_gap, self.record.min_q1_q2_gap))
        total = min(cfg.total_flow_max, self.record.total_flow_max)
        if np.any(flows < lower) or np.any(flows > upper) or flows[0] - flows[1] < gap or sum(flows) > total:
            return self._frozen(pid_input, "current flows outside MPC calibrated constraints")
        step = float(cfg.max_flow_change_per_cycle)
        if not math.isfinite(step) or step <= 0:
            return self._frozen(pid_input, "invalid MPC flow step limit")
        current, target = float(pid_input.current_diameter_um), float(pid_input.target_diameter_um)
        try:
            nominal = self.steady_diameter(flows)
            gradient = self._gradient(flows)
            if not math.isfinite(nominal) or nominal <= 0 or not np.all(np.isfinite(gradient)):
                return self._frozen(pid_input, "invalid MPC model prediction")
            if np.linalg.norm(gradient) < 1e-9:
                return self._frozen(pid_input, "MPC local model has no control authority")
            # Scalar random-walk disturbance observer, corrected only with an
            # eligible post-response measurement. Not a flow sensor estimate.
            variance = self.offset_variance + 0.1
            noise = max(0.25, self.record.validation_mae_um) ** 2
            gain = variance / (variance + noise)
            offset = self.offset_um + gain * (current - nominal - self.offset_um)
            if abs(offset) > 0.25 * self.record.baseline_diameter_um:
                return self._frozen(pid_input, "MPC model mismatch exceeds local diameter range; recalibrate")
            return self._optimize(pid_input, flows, lower, upper, gap, total, step,
                                  current, target, nominal, gradient, offset, (1 - gain) * variance)
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            return self._frozen(pid_input, f"MPC numerical failure: {exc}")

    def _optimize(
        self, item: PIDInput, flows: np.ndarray, lower: np.ndarray, upper: np.ndarray,
        gap: float, total: float, step: float, current: float, target: float,
        nominal: float, gradient: np.ndarray, offset: float, variance: float,
    ) -> PIDCommand:
        n, period = self.settings.horizon, self.sample_period_s
        tau = self.record.response_time_constant_ms / 1000
        delay = self.record.conservative_response_delay_ms / 1000
        times = np.arange(1, n + 1) * period
        free = nominal + offset + (current - nominal - offset) * np.exp(-times / tau)
        scales = np.maximum(upper - lower, 1e-6)
        cumulative = np.kron(np.tril(np.ones((n, n))), np.diag(scales))
        response = np.zeros((n, 2 * n))
        for i, future in enumerate(times):
            for j in range(i + 1):
                elapsed = max(0.0, future - j * period - delay)
                response[i, 2*j:2*j+2] = (1 - math.exp(-elapsed / tau)) * gradient * scales
        output_scale = max(1.0, 0.05 * self.record.baseline_diameter_um)
        matrix, error = response / output_scale, (free - target) / output_scale
        hessian = matrix.T @ matrix + self.settings.move_weight * np.eye(2 * n)
        linear = matrix.T @ error
        # All moves, not just the first, obey absolute, slew, phase and sum limits.
        difference = np.kron(np.eye(n), [[1.0, -1.0]]) @ cumulative
        summed = np.kron(np.eye(n), [[1.0, 1.0]]) @ cumulative
        constraints = LinearConstraint(
            np.vstack([cumulative, difference, summed]),
            np.concatenate([np.tile(lower - flows, n), np.full(n, gap - flows[0] + flows[1]), np.full(n, -np.inf)]),
            np.concatenate([np.tile(upper - flows, n), np.full(n, np.inf), np.full(n, total - sum(flows))]),
        )
        limits = np.tile(step / scales, n)
        result = minimize(
            lambda x: float(x @ hessian @ x + 2 * linear @ x), np.zeros(2 * n),
            jac=lambda x: 2 * (hessian @ x + linear), method="SLSQP",
            bounds=Bounds(-limits, limits), constraints=[constraints],
            options={"maxiter": self.settings.max_iterations, "ftol": 1e-9},
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            return self._frozen(item, f"MPC optimization failed: {result.message}")
        plan = flows + (cumulative @ result.x).reshape(n, 2)
        moves = np.diff(np.vstack([flows, plan]), axis=0)
        if (np.any(plan < lower - 1e-7) or np.any(plan > upper + 1e-7)
                or np.any(plan[:, 0] - plan[:, 1] < gap - 1e-7)
                or np.any(plan.sum(axis=1) > total + 1e-7) or np.any(abs(moves) > step + 1e-7)):
            return self._frozen(item, "MPC solver returned an infeasible plan")
        # Scale a tiny solver tolerance violation toward the feasible hold point;
        # this preserves direction and all convex constraints without pump clipping.
        first = plan[0] - flows
        fraction = 1.0
        for a, b in zip(np.vstack([np.eye(2), -np.eye(2), [-1, 1], [1, 1]]),
                        np.r_[upper - flows, flows - lower, flows[0] - flows[1] - gap, total - sum(flows)]):
            if a @ first > 0:
                fraction = min(fraction, max(0.0, float(b / (a @ first))))
        if np.max(abs(first)) > step:
            fraction = min(fraction, step / np.max(abs(first)))
        command = flows + first * fraction
        prediction = free + response @ result.x
        self._last_frame_id = item.frame_id
        self.offset_um, self.offset_variance = offset, variance
        self.last_plan = tuple(tuple(float(v) for v in row) for row in plan)
        saturated = bool(np.any(np.isclose(plan, lower, atol=1e-6)) or np.any(np.isclose(plan, upper, atol=1e-6))
                         or np.any(np.isclose(abs(moves), step, atol=1e-6))
                         or np.any(np.isclose(plan.sum(axis=1), total, atol=1e-6))
                         or np.any(np.isclose(plan[:, 0] - plan[:, 1], gap, atol=1e-6)))
        return PIDCommand(
            q1=float(command[0]), q2=float(command[1]), diameter_error=target - current,
            adjustment=float(np.linalg.norm(command - flows)), freeze_feedback=False, suggested_stop=False,
            reason="MPC rolling optimization completed", control_mode="MPC", control_owner="MPC",
            frame_id=item.frame_id, actuator_saturated=saturated,
            mpc_prediction_um=tuple(float(v) for v in prediction), mpc_horizon_s=n * period,
            mpc_offset_um=offset,
        )
