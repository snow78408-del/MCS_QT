from __future__ import annotations

from .adaptive import AdaptivePIDManager
from .base import BaseDiameterController
from .config import PIDConfig, PIDControlMode
from .feedforward import FeedforwardCompensator
from .models import PIDCommand, PIDInput, PumpState, TargetParams, VisionMetrics
from .parameter_manager import PIDParameterManager
from .safety import clamp, is_finite, rate_limit


class DiameterPIDController(BaseDiameterController):
    def __init__(self, config: PIDConfig | None = None) -> None:
        self.config = config or PIDConfig()
        self.parameter_manager = PIDParameterManager(self.config)
        self.adaptive = AdaptivePIDManager(self.config)
        self.feedforward = FeedforwardCompensator(self.config)
        self.kp = float(self.config.base_kp)
        self.ki = float(self.config.base_ki)
        self.kd = float(self.config.base_kd)
        self.integral = 0.0
        self.previous_error = 0.0
        self._has_previous = False
        self._last_frame_id: int | None = None
        self._last_output = 0.0
        self._q1_bias: float | None = None
        self._q2_bias: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = 0.0
        self._has_previous = False
        self._last_frame_id = None
        self._last_output = 0.0
        self._q1_bias = None
        self._q2_bias = None
        self.adaptive.reset()
        self.feedforward.reset()
        self.kp = float(self.config.base_kp)
        self.ki = float(self.config.base_ki)
        self.kd = float(self.config.base_kd)

    def update(
        self,
        vision_metrics: VisionMetrics,
        target_params: TargetParams,
        pump_state: PumpState,
        dt: float,
    ) -> PIDCommand:
        return self.update_input(
            PIDInput(
                target_diameter_um=float(target_params.target_diameter),
                current_diameter_um=float(vision_metrics.avg_diameter) if vision_metrics.avg_diameter is not None else None,
                current_q1=float(pump_state.q1),
                current_q2=float(pump_state.q2),
                dt=float(dt),
                frame_id=int(vision_metrics.frame_id),
                vision_valid=bool(vision_metrics.valid_for_control),
                pump_communication_ok=bool(pump_state.communication_ok),
                droplet_count=int(vision_metrics.droplet_count),
                measurement_noise_est=float(vision_metrics.noise_estimate),
            )
        )

    def update_input(self, pid_input: PIDInput) -> PIDCommand:
        q1_current = float(pid_input.current_q1)
        q2_current = float(pid_input.current_q2)
        if self._q1_bias is None:
            self._q1_bias = q1_current
        if self._q2_bias is None:
            self._q2_bias = q2_current
        mode = str(self.config.control_mode or PIDControlMode.CLASSIC_PID.value)

        freeze = self._validate_input(pid_input)
        if freeze:
            return self._frozen(pid_input, freeze, mode)

        if pid_input.frame_id > 0 and self._last_frame_id == int(pid_input.frame_id):
            return self._frozen(pid_input, "same frame_id already controlled", mode)

        target = float(pid_input.target_diameter_um)
        current = float(pid_input.current_diameter_um)
        error = target - current
        if abs(error) <= float(self.config.diameter_deadband):
            decay = min(1.0, max(0.0, float(self.config.integral_decay_in_deadband)))
            self.integral *= decay
            self.previous_error = error
            self._has_previous = True
            if pid_input.frame_id > 0:
                self._last_frame_id = int(pid_input.frame_id)
            return PIDCommand(
                q1=q1_current,
                q2=q2_current,
                diameter_error=error,
                adjustment=0.0,
                freeze_feedback=False,
                suggested_stop=False,
                reason="diameter error inside deadband",
                kp=self.kp,
                ki=self.ki,
                kd=self.kd,
                control_mode=mode,
                frame_id=int(pid_input.frame_id),
            )

        adaptive_active = False
        if mode in {PIDControlMode.ADAPTIVE_PID.value, PIDControlMode.ADAPTIVE_PID_WITH_FEEDFORWARD.value}:
            state = self.adaptive.update(pid_input, error)
            self.kp, self.ki, self.kd = self.parameter_manager.clamp_gains(state.kp, state.ki, state.kd)
            adaptive_active = bool(state.active)
        else:
            self.kp, self.ki, self.kd = self.parameter_manager.base_gains()

        previous_integral = self.integral
        candidate_integral = self.integral + error * float(pid_input.dt)
        candidate_integral = clamp(
            candidate_integral,
            -abs(self.config.integral_limit),
            abs(self.config.integral_limit),
        )
        derivative = 0.0 if not self._has_previous else (error - self.previous_error) / float(pid_input.dt)
        p_term = self.kp * error
        candidate_i_term = self.ki * candidate_integral
        d_term = self.kd * derivative
        unsaturated_pid = p_term + candidate_i_term + d_term
        saturating_high = unsaturated_pid > float(self.config.output_max) and error > 0.0
        saturating_low = unsaturated_pid < float(self.config.output_min) and error < 0.0
        self.integral = previous_integral if (saturating_high or saturating_low) else candidate_integral
        i_term = self.ki * self.integral
        u_pid = clamp(p_term + i_term + d_term, self.config.output_min, self.config.output_max)

        ff = self.feedforward.compute(pid_input) if mode == PIDControlMode.ADAPTIVE_PID_WITH_FEEDFORWARD.value else None
        u_ff = float(ff.u_ff) if ff is not None else 0.0
        feedforward_active = bool(ff.active) if ff is not None else False

        u_final = clamp(u_pid + u_ff, self.config.output_min, self.config.output_max)
        u_final = rate_limit(u_final, self._last_output, self.config.output_rate_limit)

        q1_base = float(self._q1_bias) if self.config.use_initial_flow_as_output_bias else q1_current
        q2_base = float(self._q2_bias) if self.config.use_initial_flow_as_output_bias else q2_current
        # Differential control keeps the two phases from being accelerated or
        # decelerated together. error = target - measured:
        #   droplet too large (error < 0): Q1 up, Q2 down
        #   droplet too small (error > 0): Q1 down, Q2 up
        q1_raw = q1_base - u_final
        q2_raw = q2_base + u_final
        max_step = max(0.0, float(self.config.max_flow_change_per_cycle))
        if max_step > 0.0:
            q1_raw = clamp(q1_raw, q1_current - max_step, q1_current + max_step)
            q2_raw = clamp(q2_raw, q2_current - max_step, q2_current + max_step)
        total_max = max(0.0, float(self.config.total_flow_max))
        if total_max > 0.0 and q1_raw + q2_raw > total_max:
            scale = total_max / (q1_raw + q2_raw)
            q1_raw *= scale
            q2_raw *= scale
        self.previous_error = error
        self._has_previous = True
        self._last_output = u_final
        if pid_input.frame_id > 0:
            self._last_frame_id = int(pid_input.frame_id)

        common = dict(
            diameter_error=error,
            adjustment=u_final,
            p_term=p_term,
            i_term=i_term,
            d_term=d_term,
            pid_output=u_pid,
            feedforward_output=u_ff,
            final_output=u_final,
            kp=self.kp,
            ki=self.ki,
            kd=self.kd,
            adaptive_active=adaptive_active,
            feedforward_active=feedforward_active,
            control_mode=mode,
            frame_id=int(pid_input.frame_id),
        )
        if q1_raw <= 0 or q2_raw <= 0:
            return PIDCommand(
                q1=q1_current,
                q2=q2_current,
                freeze_feedback=False,
                suggested_stop=True,
                reason=f"computed non-positive flow; q1_raw={q1_raw:.6f}, q2_raw={q2_raw:.6f}",
                **common,
            )

        return PIDCommand(
            q1=clamp(q1_raw, self.config.q1_min, self.config.q1_max),
            q2=clamp(q2_raw, self.config.q2_min, self.config.q2_max),
            freeze_feedback=False,
            suggested_stop=False,
            reason="PID update completed",
            **common,
        )

    def _validate_input(self, pid_input: PIDInput) -> str:
        if float(pid_input.dt) <= 0:
            return "dt <= 0"
        if not bool(pid_input.vision_valid):
            return "vision invalid"
        if not bool(pid_input.pump_communication_ok):
            return "pump communication abnormal"
        if int(pid_input.droplet_count) < int(self.config.min_droplet_count_for_feedback):
            return f"droplet_count below threshold ({pid_input.droplet_count} < {self.config.min_droplet_count_for_feedback})"
        if not is_finite(pid_input.current_diameter_um) or float(pid_input.current_diameter_um or 0.0) <= 0:
            return "current diameter invalid"
        if not is_finite(pid_input.target_diameter_um) or float(pid_input.target_diameter_um) <= 0:
            return "target diameter invalid"
        return ""

    def _frozen(self, pid_input: PIDInput, reason: str, mode: str) -> PIDCommand:
        return PIDCommand(
            q1=float(pid_input.current_q1),
            q2=float(pid_input.current_q2),
            diameter_error=0.0,
            adjustment=0.0,
            freeze_feedback=True,
            suggested_stop=False,
            reason=f"feedback frozen: {reason}",
            kp=self.kp,
            ki=self.ki,
            kd=self.kd,
            control_mode=mode,
            frame_id=int(pid_input.frame_id),
        )
