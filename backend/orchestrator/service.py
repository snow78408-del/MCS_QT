from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from statistics import median
from typing import Any, Callable

from ..disturbance_model import DisturbanceModelConfig, DisturbanceModelService
from ..optimization import BayesianOptimizationConfig, OptimizationObservation, SafeBayesianOptimizer
from ..pid_control import (
    DiameterPIDController,
    PIDConfig,
    PIDControlMode,
    PIDInput,
    PlantCalibrationExperimentConfig,
    PlantCalibrationExperimentResult,
    PlantCalibrationMeasurement,
    PlantCalibrationObservation,
    PlantCalibrationRecord,
    PumpState,
    TargetParams,
    VisionMetrics,
    build_plant_calibration_result,
    identify_channel_sensitivities,
    save_plant_calibration_result,
)
from ..pump_hardware import ChannelParams, PumpHardwareService
from ..pump_hardware.invariants import effective_q1_q2_gap, q1_is_strictly_above_q2
from .config import OrchestratorConfig
from .drift import DriftSupervisor
from .models import (
    ControlSnapshot,
    FrameSnapshot,
    PumpChannelState,
    PumpRuntimeState,
    RecognitionSnapshot,
    SystemConfig,
    SystemSnapshot,
)
from .pid_database import PIDReplayData, PIDSessionRecorder, load_pid_replay
from .safety import RunToken, SafetyState, SafetySupervisor
from .state import SystemState
from .vision_adapter import GenericVisionAdapter, PipelineVisionService, VisionAdapterProtocol


class OrchestratorService:
    def __init__(
        self,
        vision_service: Any = None,
        vision_adapter: VisionAdapterProtocol | None = None,
        pump_service: PumpHardwareService | None = None,
        logger: Callable[[str], None] | None = None,
        orchestrator_config: OrchestratorConfig | None = None,
        pid_config: PIDConfig | None = None,
        disturbance_service: DisturbanceModelService | None = None,
        disturbance_config: DisturbanceModelConfig | None = None,
    ) -> None:
        self._log = logger or (lambda _msg: None)

        if vision_adapter is not None:
            self.vision_service = vision_service
            self.vision_adapter = vision_adapter
        elif vision_service is not None:
            self.vision_service = vision_service
            self.vision_adapter = GenericVisionAdapter(vision_service)
        else:
            self.vision_service = PipelineVisionService(logger=self._log)
            self.vision_adapter = GenericVisionAdapter(self.vision_service)

        self.pump_service = pump_service or PumpHardwareService(logger=logger)
        self.runtime = orchestrator_config or OrchestratorConfig()
        self.pid_config = pid_config or PIDConfig()
        self._pid_calibration_defaults = {
            name: copy.deepcopy(getattr(self.pid_config, name))
            for name in (
                "feedforward_gain",
                "feedforward_enabled",
                "feedforward_calibrated",
                "control_mode",
                "base_kp",
                "base_ki",
                "base_kd",
                "q1_control_sign",
                "q2_control_sign",
                "q1_output_gain",
                "q2_output_gain",
                "q1_log_diameter_sensitivity",
                "q2_log_diameter_sensitivity",
                "sensitivity_allocation_regularization",
                "log_sensitivity_calibrated",
                "q1_min",
                "q1_max",
                "q2_min",
                "q2_max",
                "total_flow_max",
                "min_q1_q2_gap",
            )
        }
        self._pid_controller = DiameterPIDController(self.pid_config)
        self._plant_calibration: PlantCalibrationRecord | None = None
        self._plant_calibration_experiment: dict[str, Any] = {
            "status": "idle",
            "phase": "",
            "reason": "",
            "completed_trials": 0,
            "total_trials": 0,
        }
        self._plant_calibration_in_progress = False
        self._drift_supervisor = DriftSupervisor()
        pump_runtime = getattr(self.pump_service, "runtime_config", None)
        if pump_runtime is not None and hasattr(pump_runtime, "min_q1_q2_gap"):
            pump_runtime.min_q1_q2_gap = float(self.pid_config.min_q1_q2_gap)
        self.disturbance_service = disturbance_service or DisturbanceModelService(
            config=disturbance_config,
            logger=self._log,
        )

        self._state = SystemState.IDLE
        self._cfg: SystemConfig | None = None
        self._recognition: RecognitionSnapshot | None = None
        self._pump_control_enabled = False
        self._pump_state = PumpRuntimeState(
            connected=False,
            comm_established=False,
            fully_ready=False,
            q1=0.0,
            q2=0.0,
            running=False,
            last_error="",
            last_update_ok=False,
            last_update_reason="",
        )
        self._control: ControlSnapshot | None = None
        self._pid_data_recorder = PIDSessionRecorder()
        self._message = ""
        self._error = ""

        self._lock = threading.RLock()
        # Hardware starts run outside _lock so stop/pause can invalidate the
        # attempt while the pump is waiting on serial I/O. The generation and
        # this narrow gate prevent two recovery callers from starting at once.
        self._pump_lifecycle_lock = threading.Lock()
        self._lifecycle_generation = 0
        self._start_in_progress = False
        self._control_condition = threading.Condition(self._lock)
        self._loop_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._last_control_ts: float | None = None
        self._last_control_frame_id: int | None = None
        self._last_control_period_id: int | None = None
        # Newest vision period observed, including rejected/stale periods.
        # This is intentionally separate from the last period applied to PID.
        self._last_seen_vision_period_id: int | None = None
        self._last_disturbance_prediction = None
        self._disturbance_context: dict[str, Any] = {}
        self._auto_disturbance_experiment_id = ""
        self._target_revision = 0
        self._optimizer: SafeBayesianOptimizer | None = None
        self._optimization_candidate = None
        self._optimization_candidate_applied_monotonic = 0.0
        self._optimization_candidate_period_id = 0
        self._optimization_window_observations: list[OptimizationObservation] = []
        self._optimization_review_required = False
        self._stabilizing_until_monotonic = 0.0
        self._run_token: RunToken | None = None
        self._safety = SafetySupervisor(self._safety_stop_pump, logger=self._log)
        self._refresh_pump_channels(communication_ok=False, error="not connected")

    def _safety_stop_pump(self) -> bool:
        if not self._is_realtime_mode():
            return True
        try:
            result = self.pump_service.stop_system_and_verify()
        except Exception as exc:
            with self._lock:
                self._pump_state.last_error = str(exc) or "pump stop raised an exception"
                self._refresh_pump_channels(
                    communication_ok=False,
                    error=self._pump_state.last_error,
                )
            self._log(f"[SAFETY][STOP][ERROR] {exc}")
            return False
        stopped = bool(result.ok)
        with self._lock:
            self._pump_state.running = False if stopped else self._pump_state.running
            self._pump_state.last_error = "" if stopped else str(result.reason or result.error or "stop failed")
            self._refresh_pump_channels(
                communication_ok=stopped,
                error="" if stopped else self._pump_state.last_error,
            )
        return stopped

    def _try_resume_infusion(self, source: str) -> tuple[bool, str]:
        """Legacy recovery hook that is deliberately fail-closed.

        Infusion may only start through an explicit start() call, which creates
        a fresh safety token and binds a new vision run context.
        """
        reason = f"automatic pump recovery is disabled ({source})"
        self._log(f"[PUMP][RECOVERY][BLOCKED] {reason}")
        return False, reason

    def emergency_stop(self, reason: str = "operator emergency stop") -> bool:
        self._stop_event.set()
        self._pause_event.set()
        self._run_token = None
        self._safety.trip(reason, emergency=True)
        stopped = self._safety_stop_pump()
        if stopped:
            self._safety.confirm_stopped()
        self._set_state(
            SystemState.ERROR,
            error=(reason if stopped else f"{reason}; pump stop is not verified"),
        )
        return stopped

    def reset_safety_latch(self) -> None:
        self._safety.reset_latch()

    def _refresh_pump_channels(
        self,
        *,
        channel_running: list[bool] | None = None,
        communication_ok: bool | None = None,
        error: str | None = None,
    ) -> None:
        now = time.time()
        if communication_ok is not None:
            self._pump_state.last_readback_time = now

        comm_ok = (
            bool(communication_ok)
            if communication_ok is not None
            else bool(self._pump_state.comm_established and not self._pump_state.last_error)
        )
        err = str(error if error is not None else self._pump_state.last_error or "")

        def _running(index: int) -> bool:
            if channel_running is not None and index < len(channel_running):
                return bool(channel_running[index])
            return bool(self._pump_state.running)

        enabled_q12 = bool(self._pump_control_enabled and self._pump_state.comm_established)
        ts = self._pump_state.last_readback_time
        q1_actual = self._pump_state.q1_actual if self._pump_state.q1_actual is not None else self._pump_state.q1
        q2_actual = self._pump_state.q2_actual if self._pump_state.q2_actual is not None else self._pump_state.q2
        self._pump_state.channels = {
            "Q1": PumpChannelState(
                logical_name="Q1",
                physical_channel="CH1",
                enabled=enabled_q12,
                running=enabled_q12 and _running(0),
                communication_ok=comm_ok,
                target_flow_rate=float(self._pump_state.q1),
                actual_flow_rate=float(q1_actual) if comm_ok else None,
                last_readback_time=ts,
                error=err,
            ),
            "Q2": PumpChannelState(
                logical_name="Q2",
                physical_channel="CH2",
                enabled=enabled_q12,
                running=enabled_q12 and _running(1),
                communication_ok=comm_ok,
                target_flow_rate=float(self._pump_state.q2),
                actual_flow_rate=float(q2_actual) if comm_ok else None,
                last_readback_time=ts,
                error=err,
            ),
            "Q3": PumpChannelState(
                logical_name="Q3",
                physical_channel="unconfigured",
                enabled=False,
                running=False,
                communication_ok=False,
                target_flow_rate=None,
                actual_flow_rate=None,
                last_readback_time=None,
                error="unconfigured",
            ),
        }

    @staticmethod
    def _flow_from_channel_params(params: ChannelParams | None) -> float | None:
        return PumpHardwareService.flow_from_channel_params(params)

    def _sync_pump_flow_readback(self, source: str, *, update_command: bool = True) -> bool:
        try:
            q1, q2 = self.pump_service.get_current_q_state()
        except Exception as exc:
            self._pump_state.q1_actual = None
            self._pump_state.q2_actual = None
            self._pump_state.comm_established = False
            self._pump_state.last_error = str(exc)
            self._pump_control_enabled = False
            self._pump_state.running = False
            self._refresh_pump_channels(communication_ok=False, error=str(exc))
            self._log(f"[PUMP][READBACK][FAIL] source={source} error={exc}")
            return False

        self._pump_state.q1_actual = float(q1)
        self._pump_state.q2_actual = float(q2)
        if update_command:
            self._pump_state.q1 = float(q1)
            self._pump_state.q2 = float(q2)
        self._pump_state.last_error = ""
        self._refresh_pump_channels(communication_ok=True, error="")
        self._log(f"[PUMP][READBACK][OK] source={source} q1={q1:.6f} q2={q2:.6f}")
        return True

    def set_disturbance_context(
        self,
        *,
        experiment_id: str = "",
        chip_id: str = "",
        disturbance_name: str = "",
        disturbance_stage: str = "baseline",
        disturbance_amplitude: float = 0.0,
        temperature_c: float | None = None,
        leading_signal_available: bool = False,
        signal_lead_time_ms: float = 0.0,
        leading_signal_name: str = "",
    ) -> None:
        with self._lock:
            existing = dict(self._disturbance_context)
        context = {
            "experiment_id": str(experiment_id or existing.get("experiment_id", "")),
            "chip_id": str(chip_id or existing.get("chip_id", "")),
            "disturbance_name": str(
                disturbance_name or existing.get("disturbance_name", "")
            ),
            "disturbance_stage": disturbance_stage,
            "disturbance_amplitude": disturbance_amplitude,
            "temperature_c": (
                temperature_c
                if temperature_c is not None
                else existing.get("temperature_c")
            ),
            "leading_signal_available": bool(leading_signal_available),
            "signal_lead_time_ms": max(0.0, float(signal_lead_time_ms)),
            "leading_signal_name": str(leading_signal_name or ""),
            "leading_signal_observed_monotonic": (
                time.monotonic() if leading_signal_available else 0.0
            ),
        }
        with self._lock:
            self._disturbance_context = context

    def promote_disturbance_candidate_model(self) -> dict[str, Any]:
        """Explicitly promote a shadow-validated model while control is quiescent."""
        with self._lock:
            state = self._state
        allowed = {
            SystemState.CONFIGURED,
            SystemState.VIDEO_READY,
            SystemState.INITIALIZED,
            SystemState.PAUSED,
            SystemState.STOPPED,
        }
        if state not in allowed:
            raise RuntimeError(
                "candidate model promotion requires configured, initialized, paused, or stopped state"
            )
        result = self.disturbance_service.promote_candidate_model()
        self._log(
            f"[DISTURBANCE][PROMOTION][OPERATOR] version={result['model_version']} "
            f"state={state.value}"
        )
        return result

    def discard_disturbance_candidate_model(self) -> None:
        self.disturbance_service.discard_candidate_model()

    def close_background_services(self) -> None:
        """Flush persistent queues and stop bounded background workers."""
        self.disturbance_service.close()

    def _is_realtime_mode(self) -> bool:
        if self._cfg is None:
            return False
        mode = str(self._cfg.video_source_type or "").strip().lower()
        return mode in {
            "camera",
            "realtime",
            "real_time",
            "live",
            "rtsp",
            "usb",
            "opencv",
            "hikrobot",
            "hikrobot_industrial_camera",
            "usb_camera",
        }

    def _set_state(self, state: SystemState, message: str = "", error: str = "") -> None:
        with self._lock:
            self._state = state
            if message:
                self._message = message
            if error:
                self._error = error
        if message:
            self._log(f"[ORCH][{state.value}] {message}")
        if error:
            self._log(f"[ORCH][ERROR] {error}")

    def _get_drift_supervisor(self) -> DriftSupervisor:
        supervisor = getattr(self, "_drift_supervisor", None)
        if supervisor is None:
            supervisor = DriftSupervisor()
            self._drift_supervisor = supervisor
        return supervisor

    def _apply_plant_calibration(
        self,
        record: PlantCalibrationRecord | None,
    ) -> None:
        defaults = self._pid_calibration_defaults
        if record is not None:
            if record.q1_min < float(defaults["q1_min"]) or record.q1_max > float(defaults["q1_max"]):
                raise ValueError("plant calibration Q1 bounds exceed controller safety limits")
            if record.q2_min < float(defaults["q2_min"]) or record.q2_max > float(defaults["q2_max"]):
                raise ValueError("plant calibration Q2 bounds exceed controller safety limits")
            if record.total_flow_max > float(defaults["total_flow_max"]):
                raise ValueError("plant calibration total-flow limit exceeds controller safety limit")
            if record.min_q1_q2_gap < float(defaults["min_q1_q2_gap"]):
                raise ValueError("plant calibration phase gap is weaker than controller safety limit")

        for name, value in defaults.items():
            setattr(self.pid_config, name, copy.deepcopy(value))
        # An ad-hoc constructor flag is not sufficient authority for real
        # feedforward. Every configure call must present the versioned record.
        self.pid_config.feedforward_calibrated = False
        self.pid_config.log_sensitivity_calibrated = False
        self.pid_config.q1_log_diameter_sensitivity = 0.0
        self.pid_config.q2_log_diameter_sensitivity = 0.0
        if record is not None:
            if not record.has_generation_control_mapping:
                raise ValueError(
                    "旧观察区标定不包含生成区 Q1/Q2 对数灵敏度，不能用于闭环；请重新执行生成区标定"
                )
            if not record.authorized_for_pi:
                raise ValueError(
                    "该动力学标定未通过独立验证，不能授权实时闭环；请使用新版自动标定重新采集"
                )
            # The new controller is an identified generation-zone PI.  The
            # former prediction feedforward and heuristic adaptive gains use
            # incompatible units and remain disabled.
            self.pid_config.control_mode = PIDControlMode.CLASSIC_PID.value
            self.pid_config.feedforward_enabled = False
            self.pid_config.q1_control_sign = float(record.q1_control_sign)
            self.pid_config.q2_control_sign = float(record.q2_control_sign)
            self.pid_config.q1_output_gain = float(record.q1_output_gain)
            self.pid_config.q2_output_gain = float(record.q2_output_gain)
            self.pid_config.q1_log_diameter_sensitivity = float(
                record.q1_log_diameter_sensitivity
            )
            self.pid_config.q2_log_diameter_sensitivity = float(
                record.q2_log_diameter_sensitivity
            )
            self.pid_config.sensitivity_allocation_regularization = float(
                record.sensitivity_allocation_regularization
            )
            self.pid_config.log_sensitivity_calibrated = True
            self.pid_config.base_kp = float(record.controller_kp)
            self.pid_config.base_ki = float(record.controller_ki)
            self.pid_config.base_kd = 0.0
            self.pid_config.q1_min = float(record.q1_min)
            self.pid_config.q1_max = float(record.q1_max)
            self.pid_config.q2_min = float(record.q2_min)
            self.pid_config.q2_max = float(record.q2_max)
            self.pid_config.total_flow_max = float(record.total_flow_max)
            self.pid_config.min_q1_q2_gap = float(record.min_q1_q2_gap)
        pump_runtime = getattr(self.pump_service, "runtime_config", None)
        if pump_runtime is not None and hasattr(pump_runtime, "min_q1_q2_gap"):
            pump_runtime.min_q1_q2_gap = float(self.pid_config.min_q1_q2_gap)
        self._pid_controller.reset()

    def _update_plant_calibration_experiment(self, **values: Any) -> None:
        with self._lock:
            current = dict(self._plant_calibration_experiment)
            current.update(values)
            self._plant_calibration_experiment = current

    def _require_calibration_lifecycle(
        self,
        generation: int,
        token: RunToken,
    ) -> None:
        with self._lock:
            valid = (
                generation == self._lifecycle_generation
                and self._state == SystemState.CALIBRATING
                and not self._stop_event.is_set()
                and not self._pause_event.is_set()
            )
        if not valid or not self._safety.permits(token):
            raise RuntimeError("plant calibration was cancelled by a lifecycle transition")

    @staticmethod
    def _plant_calibration_observations(
        rec: RecognitionSnapshot,
    ) -> tuple[PlantCalibrationObservation, ...]:
        crossed = {
            int(track_id): float(value)
            for track_id, value in dict(getattr(rec, "crossed_track_diameters", {}) or {}).items()
            if math.isfinite(float(value)) and float(value) > 0.0
        }
        capture_times = {
            int(track_id): float(value)
            for track_id, value in dict(getattr(rec,"crossed_track_capture_monotonic",{}) or {}).items()
        }
        frame_ids = {
            int(track_id): int(value)
            for track_id, value in dict(getattr(rec,"crossed_track_frame_ids",{}) or {}).items()
        }
        current_frame_id = int(getattr(rec, "frame_id", 0) or 0)
        if current_frame_id <= 0 or not crossed:
            return ()
        observed = time.monotonic()
        fallback_capture = float(getattr(rec, "capture_monotonic", 0.0) or observed)
        period_id = int(getattr(rec, "control_period_id", 0) or 0)
        return tuple(
            PlantCalibrationObservation(
                frame_id=int(frame_ids.get(track_id,current_frame_id)),
                capture_monotonic=float(capture_times.get(track_id,fallback_capture)),
                observed_monotonic=observed,
                diameter_um=diameter,
                droplet_count=1,
                droplet_id=track_id,
                control_period_id=period_id,
                pixel_to_micron=float(getattr(rec, "pixel_to_micron", 0.0) or 0.0),
                scale_source=str(getattr(rec, "scale_source", "unknown") or "unknown"),
                measurement_region=str(
                    getattr(rec, "measurement_region", "generation") or "generation"
                ),
                volume_correction_factor=float(
                    getattr(rec, "generation_volume_correction", 1.0) or 1.0
                ),
                generation_frequency_hz=max(
                    0.0,
                    float(getattr(rec, "droplet_generation_rate_hz", 0.0) or 0.0),
                ),
                diameter_cv=max(
                    0.0,
                    float(getattr(rec, "frame_diameter_cv", 0.0) or 0.0),
                ),
            )
            for track_id, diameter in sorted(crossed.items())
        )

    @staticmethod
    def _calibration_median_drift_um(
        observations: list[PlantCalibrationObservation] | tuple[PlantCalibrationObservation, ...],
    ) -> float | None:
        if len(observations) < 3:
            return None
        edge_count = max(1, min(len(observations) // 2, max(2, len(observations) // 3)))
        early = median(item.diameter_um for item in observations[:edge_count])
        late = median(item.diameter_um for item in observations[-edge_count:])
        return abs(float(late) - float(early))

    @staticmethod
    def _calibration_effective_stability_tolerance_um(
        observations: list[PlantCalibrationObservation] | tuple[PlantCalibrationObservation, ...],
        configured_tolerance_um: float,
        pixel_to_micron: float = 0.0,
    ) -> float:
        values = [float(item.diameter_um) for item in observations]
        noise_sigma_um = 0.0
        if len(values) >= 3:
            differences = [current - previous for previous,current in zip(values,values[1:])]
            difference_center = float(median(differences))
            difference_mad = float(median(abs(value-difference_center) for value in differences))
            # First differences contain sqrt(2) times the independent sample
            # noise while largely removing slow drift.
            noise_sigma_um = 1.4826 * difference_mad / math.sqrt(2.0)
        return max(
            float(configured_tolerance_um),
            2.0 * max(0.0,float(pixel_to_micron)),
            2.0 * noise_sigma_um,
        )

    @staticmethod
    def _calibration_response_threshold_um(
        observations: list[PlantCalibrationObservation] | tuple[PlantCalibrationObservation, ...],
        minimum_response_um: float,
        pixel_to_micron: float,
    ) -> float:
        baseline_diameter = float(median(item.diameter_um for item in observations))
        deviations = [abs(item.diameter_um - baseline_diameter) for item in observations]
        robust_noise_threshold = (
            3.0 * 1.4826 * float(median(deviations)) if deviations else 0.0
        )
        return max(
            float(minimum_response_um),
            2.0 * max(0.0, float(pixel_to_micron)),
            robust_noise_threshold,
        )

    @staticmethod
    def _calibration_droplet_window_is_stable(
        observations: list[PlantCalibrationObservation],
        tolerance_um: float,
        pixel_to_micron: float = 0.0,
        noise_reference: list[PlantCalibrationObservation] | tuple[PlantCalibrationObservation, ...] | None = None,
    ) -> bool:
        """Detect drift using early/late droplet medians, not frame extremes."""
        drift_um = OrchestratorService._calibration_median_drift_um(observations)
        if drift_um is None:
            return False
        effective_tolerance_um = OrchestratorService._calibration_effective_stability_tolerance_um(
            noise_reference if noise_reference is not None else observations,
            tolerance_um,
            pixel_to_micron,
        )
        return float(drift_um) <= float(effective_tolerance_um)

    @staticmethod
    def _calibration_response_decision(
        observations: list[PlantCalibrationObservation],
        *,
        baseline_diameter_um: float,
        response_threshold_um: float,
        stable_sample_count: int,
        minimum_observation_count: int,
        stability_tolerance_um: float,
        pixel_to_micron: float,
        noise_reference: tuple[PlantCalibrationObservation, ...],
    ) -> tuple[str | None, list[PlantCalibrationObservation]]:
        """Classify only a stable tail after the minimum droplet count.

        A stable sub-threshold tail is a real low-response observation.  An
        unstable tail is not a failure at the minimum count; callers keep
        collecting and reconsider after every new valid crossing droplet.
        """
        if len(observations) < int(minimum_observation_count):
            return None, []
        count = int(stable_sample_count)
        tail = observations[-count:]
        if len(tail) < count or not OrchestratorService._calibration_droplet_window_is_stable(
            tail,
            float(stability_tolerance_um),
            float(pixel_to_micron),
            noise_reference=noise_reference,
        ):
            return None, []
        steady_diameter = float(median(item.diameter_um for item in tail))
        if abs(steady_diameter - float(baseline_diameter_um)) >= float(response_threshold_um):
            return "detected_stable", tail
        return "below_detection_threshold", tail

    def _collect_stable_calibration_baseline(
        self,
        *,
        config: PlantCalibrationExperimentConfig,
        generation: int,
        token: RunToken,
        minimum_observation_count: int | None = None,
    ) -> tuple[PlantCalibrationObservation, ...]:
        observations: list[PlantCalibrationObservation] = []
        initial_rec = self._read_recognition()
        last_frame_id = int(getattr(initial_rec,"frame_id",0) or 0)
        seen_droplet_ids = set(
            int(track_id)
            for track_id in dict(getattr(initial_rec,"crossed_track_diameters",{}) or {})
        )
        last_valid_droplet = time.monotonic()
        required = int(config.baseline_sample_count)
        minimum_collected = max(required, int(minimum_observation_count or required))
        collected = 0
        while True:
            self._require_calibration_lifecycle(generation, token)
            self._safety.heartbeat(token, timeout_s=5.0)
            rec = self._read_recognition()
            if (
                str(rec.session_id or "") == token.session_id
                and int(rec.run_generation or 0) == token.generation
            ):
                frame_id = int(getattr(rec, "frame_id", 0) or 0)
                if frame_id > last_frame_id:
                    last_frame_id = frame_id
                    available = self._plant_calibration_observations(rec)
                    new_droplets = tuple(item for item in available if item.droplet_id not in seen_droplet_ids)
                    if new_droplets:
                        seen_droplet_ids.update(item.droplet_id for item in new_droplets)
                        last_valid_droplet = time.monotonic()
                        collected += len(new_droplets)
                        observations.extend(new_droplets)
                        observations = observations[-required:]
                        diameters = [item.diameter_um for item in observations]
                        pixel_to_micron = float(
                            getattr(rec,"pixel_to_micron",0.0)
                            or getattr(self._cfg,"pixel_to_micron",0.0)
                            or 0.0
                        )
                        drift = self._calibration_median_drift_um(observations)
                        effective_tolerance = self._calibration_effective_stability_tolerance_um(
                            observations,
                            float(config.stability_tolerance_um),
                            pixel_to_micron,
                        )
                        self._update_plant_calibration_experiment(
                            sample_mode="valid_droplet_count",
                            baseline_valid_droplets=collected,
                            baseline_required_droplets=minimum_collected,
                            baseline_recent_range_um=(max(diameters)-min(diameters) if diameters else None),
                            baseline_median_drift_um=drift,
                            baseline_effective_tolerance_um=effective_tolerance,
                        )
                        if collected >= minimum_collected and self._calibration_droplet_window_is_stable(
                            observations,
                            float(config.stability_tolerance_um),
                            pixel_to_micron,
                        ):
                            return tuple(observations)
            if time.monotonic() - last_valid_droplet >= float(config.vision_liveness_timeout_s):
                raise RuntimeError(
                    f"连续 {float(config.vision_liveness_timeout_s):g} 秒没有新的有效过线液滴；"
                    "已安全终止标定，请检查成滴、计数线和视觉识别"
                )
            if self._stop_event.wait(float(config.sample_poll_interval_s)):
                break
        raise RuntimeError("泵动力学标定已由停止请求中止")

    def _restore_plant_calibration_baseline(
        self,
        *,
        baseline_q1: float,
        baseline_q2: float,
        generation: int,
        token: RunToken,
        next_trial_id: str,
    ) -> tuple[float, float]:
        self._update_plant_calibration_experiment(
            phase=f"{next_trial_id}: returning to the common baseline",
            current_trial=next_trial_id,
            reason=(
                f"restoring Q1={float(baseline_q1):.3f}, "
                f"Q2={float(baseline_q2):.3f} uL/min before the trial"
            ),
        )
        actual_q1, actual_q2, _command_started, _readback_completed = (
            self._apply_calibration_flow(
                float(baseline_q1),
                float(baseline_q2),
                generation,
                token,
            )
        )
        self._log(
            "[PLANT_CAL][BASELINE_RESET] "
            f"next_trial={next_trial_id} q1={actual_q1:.6f} q2={actual_q2:.6f}"
        )
        return float(actual_q1), float(actual_q2)

    def _apply_calibration_flow(
        self,
        q1: float,
        q2: float,
        generation: int,
        token: RunToken,
    ) -> tuple[float, float, float, float]:
        self._require_calibration_lifecycle(generation, token)
        result = self._update_flow_with_lifecycle_guard(float(q1), float(q2), generation)
        if result is None:
            raise RuntimeError("plant calibration flow update was cancelled")
        if not result.ok:
            raise RuntimeError(
                f"plant calibration flow update failed: {result.reason or result.rollback_error or 'unknown error'}"
            )
        q1_actual = self._flow_from_channel_params(result.verified_q1) or float(q1)
        q2_actual = self._flow_from_channel_params(result.verified_q2) or float(q2)
        ok1, reason1 = self._flow_matches("Q1", q1, q1_actual)
        ok2, reason2 = self._flow_matches("Q2", q2, q2_actual)
        if not (ok1 and ok2):
            raise RuntimeError(f"plant calibration readback mismatch: {reason1 or reason2}")
        with self._lock:
            self._pump_state.q1 = float(q1)
            self._pump_state.q2 = float(q2)
            self._pump_state.q1_actual = float(q1_actual)
            self._pump_state.q2_actual = float(q2_actual)
            self._pump_state.last_update_ok = True
            self._pump_state.last_update_reason = "plant calibration flow update succeeded"
            self._pump_state.last_error = ""
            self._refresh_pump_channels(communication_ok=True, error="")
        return (
            float(q1_actual),
            float(q2_actual),
            float(getattr(result, "command_started_monotonic", 0.0) or time.monotonic()),
            float(getattr(result, "readback_completed_monotonic", 0.0) or time.monotonic()),
        )

    def _run_plant_calibration_trial(
        self,
        *,
        config: PlantCalibrationExperimentConfig,
        generation: int,
        token: RunToken,
        trial_id: str,
        channel: str,
        direction: int,
        baseline_q1: float,
        baseline_q2: float,
        commanded_q1: float,
        commanded_q2: float,
        actuator_step: float | None = None,
    ) -> PlantCalibrationMeasurement:
        self._update_plant_calibration_experiment(
            phase=f"{trial_id}：等待成滴及基线稳定",
            sample_mode="valid_droplet_count",
            baseline_valid_droplets=0,
            baseline_required_droplets=max(
                int(config.baseline_sample_count),
                int(config.response_observation_limit),
            ),
            baseline_median_drift_um=None,
            baseline_effective_tolerance_um=None,
            response_valid_droplets=0,
            response_required_droplets=int(config.stable_sample_count),
            response_observation_limit=int(config.response_observation_limit),
            response_detected=False,
            response_classification="waiting",
            response_median_drift_um=None,
            response_effective_tolerance_um=None,
        )
        baseline = self._collect_stable_calibration_baseline(
            config=config,
            generation=generation,
            token=token,
            minimum_observation_count=int(config.response_observation_limit),
        )
        baseline_diameter = float(median(item.diameter_um for item in baseline))
        if any(item.measurement_region != "generation" for item in baseline):
            raise RuntimeError("动力学标定必须使用生成区体积换算后的等效直径")
        observed_volume_factors = [item.volume_correction_factor for item in baseline]
        live_volume_factor = float(median(observed_volume_factors))
        if abs(live_volume_factor - float(config.volume_correction_factor)) > 1e-6:
            raise RuntimeError(
                "标定配置的体积修正系数与实时生成区识别配置不一致"
            )
        live_scales = [
            float(item.pixel_to_micron)
            for item in baseline
            if math.isfinite(float(item.pixel_to_micron)) and float(item.pixel_to_micron) > 0.0
        ]
        pixel_to_micron = (
            float(median(live_scales))
            if live_scales
            else float(getattr(self._cfg, "pixel_to_micron", 0.0) or 0.0)
        )
        effective_stability_tolerance = self._calibration_effective_stability_tolerance_um(
            baseline,
            float(config.stability_tolerance_um),
            pixel_to_micron,
        )
        response_threshold = self._calibration_response_threshold_um(
            baseline,
            float(config.minimum_response_um),
            pixel_to_micron,
        )
        self._update_plant_calibration_experiment(
            response_pixel_to_micron=pixel_to_micron,
            response_scale_source=str(baseline[-1].scale_source or "unknown"),
            response_detection_threshold_um=response_threshold,
        )

        samples: list[PlantCalibrationObservation] = []
        samples_lock = threading.Lock()
        collector_stop = threading.Event()

        def collect_frames() -> None:
            initial_rec = self._read_recognition()
            last_frame_id = int(getattr(initial_rec,"frame_id",0) or 0)
            seen_droplet_ids = set(
                int(track_id)
                for track_id in dict(getattr(initial_rec,"crossed_track_diameters",{}) or {})
            )
            while not collector_stop.is_set():
                try:
                    rec = self._read_recognition()
                    if (
                        str(rec.session_id or "") == token.session_id
                        and int(rec.run_generation or 0) == token.generation
                    ):
                        frame_id = int(getattr(rec,"frame_id",0) or 0)
                        if frame_id > last_frame_id:
                            last_frame_id = frame_id
                            available = self._plant_calibration_observations(rec)
                            observations = tuple(item for item in available if item.droplet_id not in seen_droplet_ids)
                            seen_droplet_ids.update(item.droplet_id for item in observations)
                            with samples_lock:
                                samples.extend(observations)
                except Exception as exc:
                    self._log(f"[PLANT_CAL][VISION][WARN] {exc}")
                collector_stop.wait(float(config.sample_poll_interval_s))

        collector = threading.Thread(
            target=collect_frames,
            name=f"plant-calibration-vision-{trial_id}",
            daemon=True,
        )
        collector.start()
        update_completed = False
        try:
            self._update_plant_calibration_experiment(
                phase=f"{trial_id}：执行阶跃并等待有效液滴响应",
            )
            actual_q1, actual_q2, command_started, readback_completed = (
                self._apply_calibration_flow(
                    commanded_q1,
                    commanded_q2,
                    generation,
                    token,
                )
            )
            update_completed = True
            onset: PlantCalibrationObservation | None = None
            stable_tail: list[PlantCalibrationObservation] = []
            response_classification = "detected_stable"
            while True:
                self._require_calibration_lifecycle(generation, token)
                self._safety.heartbeat(token, timeout_s=5.0)
                with samples_lock:
                    post_command = [
                        item for item in samples if item.capture_monotonic >= command_started
                    ]
                self._update_plant_calibration_experiment(
                    response_valid_droplets=len(post_command),
                    response_required_droplets=int(config.stable_sample_count),
                    response_detected=onset is not None,
                    response_effective_tolerance_um=effective_stability_tolerance,
                )
                if onset is None and len(post_command) >= 2:
                    for previous, current in zip(post_command, post_command[1:]):
                        first = previous.diameter_um - baseline_diameter
                        second = current.diameter_um - baseline_diameter
                        if (
                            abs(first) >= response_threshold
                            and abs(second) >= response_threshold
                            and first * second > 0.0
                        ):
                            onset = previous
                            break
                recent_tail = post_command[-int(config.stable_sample_count):]
                response_drift = self._calibration_median_drift_um(recent_tail)
                decision, decision_tail = self._calibration_response_decision(
                    post_command,
                    baseline_diameter_um=baseline_diameter,
                    response_threshold_um=response_threshold,
                    stable_sample_count=int(config.stable_sample_count),
                    minimum_observation_count=int(config.response_observation_limit),
                    stability_tolerance_um=float(config.stability_tolerance_um),
                    pixel_to_micron=pixel_to_micron,
                    noise_reference=baseline,
                )
                self._update_plant_calibration_experiment(
                    response_median_drift_um=response_drift,
                    response_effective_tolerance_um=effective_stability_tolerance,
                )
                if decision == "detected_stable" and onset is not None:
                    stable_tail = decision_tail
                    response_classification = decision
                    break
                if decision == "below_detection_threshold":
                    stable_tail = decision_tail
                    response_classification = decision
                    self._update_plant_calibration_experiment(
                        phase=f"{trial_id}: stable low response recorded; continuing",
                        response_detected=False,
                        response_classification=response_classification,
                        reason=(
                            f"stable response remained below {response_threshold:.3f} um after "
                            f"{len(post_command)} valid droplets; retained as a valid low-response observation"
                        ),
                    )
                    break
                last_valid_droplet = (
                    float(post_command[-1].observed_monotonic)
                    if post_command else float(command_started)
                )
                if time.monotonic() - last_valid_droplet >= float(config.vision_liveness_timeout_s):
                    raise RuntimeError(
                        f"{trial_id} 连续 {float(config.vision_liveness_timeout_s):g} 秒"
                        "没有新的有效过线液滴；已安全终止标定"
                    )
                if self._stop_event.wait(float(config.sample_poll_interval_s)):
                    break
            if not stable_tail:
                raise RuntimeError(f"{trial_id} diameter response did not become stable")
            stable_diameter = float(median(item.diameter_um for item in stable_tail))
            stable_time = float(stable_tail[-1].capture_monotonic)
            with samples_lock:
                response_observations = tuple(
                    item
                    for item in samples
                    if command_started <= item.capture_monotonic <= stable_time
                )
            response_started = onset or response_observations[0]
            delay_ms = max(0.001,(float(response_started.capture_monotonic)-float(command_started))*1000.0)
            return PlantCalibrationMeasurement(
                trial_id=trial_id,
                channel=channel,
                direction=int(direction),
                baseline_q1=float(baseline_q1),
                baseline_q2=float(baseline_q2),
                commanded_q1=float(commanded_q1),
                commanded_q2=float(commanded_q2),
                actual_q1=float(actual_q1),
                actual_q2=float(actual_q2),
                actuator_step=None if actuator_step is None else float(actuator_step),
                command_started_monotonic=float(command_started),
                readback_completed_monotonic=float(readback_completed),
                response_started_monotonic=float(response_started.capture_monotonic),
                response_stable_monotonic=stable_time,
                baseline_diameter_um=baseline_diameter,
                steady_diameter_um=stable_diameter,
                diameter_change_um=stable_diameter - baseline_diameter,
                response_delay_ms=delay_ms,
                baseline_observations=baseline,
                response_observations=response_observations,
                response_detected=response_classification == "detected_stable",
                response_classification=response_classification,
            )
        finally:
            collector_stop.set()
            collector.join(timeout=1.0)
            if update_completed:
                try:
                    self._require_calibration_lifecycle(generation, token)
                    self._apply_calibration_flow(
                        baseline_q1,
                        baseline_q2,
                        generation,
                        token,
                    )
                except Exception as exc:
                    self._log(f"[PLANT_CAL][RESTORE][FAIL] trial={trial_id} error={exc}")
                    raise

    def _validate_plant_calibration_target(self, q1: float, q2: float) -> None:
        self._require_valid_phase_flows(q1, q2)
        if float(q1) + float(q2) > float(self.pid_config.total_flow_max):
            raise ValueError(
                f"calibration target exceeds total flow limit: {q1 + q2:.6f} > "
                f"{self.pid_config.total_flow_max:.6f} uL/min"
            )

    def _plant_calibration_baseline_with_step_margin(
        self,
        config: PlantCalibrationExperimentConfig,
        q1: float,
        q2: float,
    ) -> tuple[float, float]:
        q1_low = float(self.pid_config.q1_min) + float(config.q1_step)
        q1_high = float(self.pid_config.q1_max) - float(config.q1_step)
        q2_low = float(self.pid_config.q2_min) + float(config.q2_step)
        q2_high = float(self.pid_config.q2_max) - float(config.q2_step)
        if q1_low > q1_high or q2_low > q2_high:
            raise ValueError("calibration step is too large for the configured Q1/Q2 range")

        baseline_q1 = min(max(float(q1), q1_low), q1_high)
        baseline_q2 = min(max(float(q2), q2_low), q2_high)
        # Validate all four corners because the combined-step direction is not
        # known until the individual channel sensitivities have been measured.
        for q1_direction in (-1, 1):
            for q2_direction in (-1, 1):
                self._validate_plant_calibration_target(
                    baseline_q1 + q1_direction * float(config.q1_step),
                    baseline_q2 + q2_direction * float(config.q2_step),
                )
        return baseline_q1, baseline_q2

    def run_plant_calibration_experiment(
        self,
        config: PlantCalibrationExperimentConfig,
    ) -> PlantCalibrationExperimentResult:
        """Run bounded pump steps under exclusive authority and return an unsaved result."""
        if not isinstance(config, PlantCalibrationExperimentConfig):
            config = PlantCalibrationExperimentConfig(**dict(config))
        with self._lock:
            if self._cfg is None:
                raise RuntimeError("system config is missing, call configure() first")
            if not self._is_realtime_mode():
                raise RuntimeError("pump-chip calibration requires realtime vision and pump control")
            if self._state != SystemState.INITIALIZED:
                raise RuntimeError(
                    f"plant calibration requires INITIALIZED state, current state is {self._state.value}"
                )
            if self._loop_thread and self._loop_thread.is_alive():
                raise RuntimeError("control loop must be stopped before plant calibration")
            if self._start_in_progress or self._plant_calibration_in_progress:
                raise RuntimeError("another pump-control start is already in progress")

        preflight = self.run_preflight_check()
        if not preflight["ok"]:
            raise RuntimeError(
                "plant calibration preflight failed: " + "; ".join(preflight["issues"])
            )

        with self._lock:
            if self._state != SystemState.INITIALIZED:
                raise RuntimeError("plant calibration start was superseded")
            self._plant_calibration_in_progress = True
            self._start_in_progress = True
            self._lifecycle_generation += 1
            generation = self._lifecycle_generation
            self._stop_event.clear()
            self._pause_event.clear()
            self._state = SystemState.CALIBRATING
            self._message = "pump-chip calibration is starting"

        total_trials = (
            int(config.repetitions) * 6
            + int(config.validation_repetitions) * 2
        )
        self._plant_calibration_experiment = {
            "status": "running",
            "phase": "starting",
            "reason": "",
            "completed_trials": 0,
            "total_trials": total_trials,
            "measurement_count": 0,
        }
        token: RunToken | None = None
        adapter_started = False
        pump_start_attempted = False
        stopped = False
        result: PlantCalibrationExperimentResult | None = None
        failure: Exception | None = None
        started_at = datetime.now(timezone.utc).isoformat()
        measurements: list[PlantCalibrationMeasurement] = []
        try:
            token = self._safety.begin_session()
            self._run_token = token
            set_context = getattr(self.vision_adapter, "set_run_context", None)
            if callable(set_context):
                set_context(token.session_id, token.generation)
            self.vision_adapter.start()
            adapter_started = True
            pump_start_attempted = True
            start_result = self.pump_service.start_infusion_and_verify([1, 2])
            if not start_result.ok:
                raise RuntimeError(
                    f"plant calibration pump start failed: {start_result.reason or start_result.error}"
                )
            self._pump_state.running = True
            self._refresh_pump_channels(communication_ok=True, error="")
            if not self._sync_pump_flow_readback("plant calibration start"):
                raise RuntimeError(
                    self._pump_state.last_error or "plant calibration flow readback failed"
                )
            heartbeat_timeout = self._control_heartbeat_timeout_s(
                float(config.sample_poll_interval_s),
                pump_transaction=True,
            )
            self._safety.arm(token, heartbeat_timeout_s=heartbeat_timeout)
            current_q1 = float(self._pump_state.q1_actual or self._pump_state.q1)
            current_q2 = float(self._pump_state.q2_actual or self._pump_state.q2)
            baseline_q1, baseline_q2 = self._plant_calibration_baseline_with_step_margin(
                config,
                current_q1,
                current_q2,
            )
            if abs(baseline_q1 - current_q1) > 1e-9 or abs(baseline_q2 - current_q2) > 1e-9:
                self._update_plant_calibration_experiment(
                    phase="moving to an interior calibration baseline",
                    reason=(
                        f"current flow is on a step boundary; calibration baseline adjusted to "
                        f"Q1={baseline_q1:.3f}, Q2={baseline_q2:.3f} uL/min"
                    ),
                )
                baseline_q1, baseline_q2, _command_started, _readback_completed = (
                    self._apply_calibration_flow(
                        baseline_q1,
                        baseline_q2,
                        generation,
                        token,
                    )
                )
            single_targets = (
                ("q1", 1, baseline_q1 + config.q1_step, baseline_q2),
                ("q1", -1, baseline_q1 - config.q1_step, baseline_q2),
                ("q2", 1, baseline_q1, baseline_q2 + config.q2_step),
                ("q2", -1, baseline_q1, baseline_q2 - config.q2_step),
            )
            for _channel, _direction, q1, q2 in single_targets:
                self._validate_plant_calibration_target(q1, q2)

            completed = 0
            for repeat in range(int(config.repetitions)):
                directions = (1, -1) if repeat % 2 == 0 else (-1, 1)
                for channel in ("q1", "q2"):
                    for direction in directions:
                        q1 = baseline_q1 + (direction * config.q1_step if channel == "q1" else 0.0)
                        q2 = baseline_q2 + (direction * config.q2_step if channel == "q2" else 0.0)
                        trial_id = f"{channel}-r{repeat + 1}-{'plus' if direction > 0 else 'minus'}"
                        self._update_plant_calibration_experiment(
                            phase=f"measuring {trial_id}",
                            current_trial=trial_id,
                        )
                        self._restore_plant_calibration_baseline(
                            baseline_q1=baseline_q1,
                            baseline_q2=baseline_q2,
                            generation=generation,
                            token=token,
                            next_trial_id=trial_id,
                        )
                        measurement = self._run_plant_calibration_trial(
                            config=config,
                            generation=generation,
                            token=token,
                            trial_id=trial_id,
                            channel=channel,
                            direction=direction,
                            baseline_q1=baseline_q1,
                            baseline_q2=baseline_q2,
                            commanded_q1=q1,
                            commanded_q2=q2,
                        )
                        measurements.append(measurement)
                        self._log(
                            "[PLANT_CAL][MEASUREMENT] "
                            f"trial={measurement.trial_id} channel={measurement.channel} "
                            f"direction={measurement.direction:+d} "
                            f"baseline={measurement.baseline_diameter_um:.6f}um "
                            f"steady={measurement.steady_diameter_um:.6f}um "
                            f"delta={measurement.diameter_change_um:+.6f}um "
                            f"detected={measurement.response_detected} "
                            f"classification={measurement.response_classification}"
                        )
                        completed += 1
                        self._update_plant_calibration_experiment(
                            completed_trials=completed,
                            measurement_count=len(measurements),
                        )

            q1_sensitivity, q2_sensitivity = identify_channel_sensitivities(measurements)
            q1_sign = 1.0 if q1_sensitivity > 0.0 else -1.0 if q1_sensitivity < 0.0 else 0.0
            q2_sign = 1.0 if q2_sensitivity > 0.0 else -1.0 if q2_sensitivity < 0.0 else 0.0
            base_step = min(float(config.q1_step), float(config.q2_step))
            for direction in (-1, 1):
                self._validate_plant_calibration_target(
                    baseline_q1 + direction * q1_sign * float(config.q1_step),
                    baseline_q2 + direction * q2_sign * float(config.q2_step),
                )

            for repeat in range(int(config.repetitions)):
                directions = (1, -1) if repeat % 2 == 0 else (-1, 1)
                for direction in directions:
                    q1 = baseline_q1 + direction * q1_sign * float(config.q1_step)
                    q2 = baseline_q2 + direction * q2_sign * float(config.q2_step)
                    trial_id = f"combined-r{repeat + 1}-{'plus' if direction > 0 else 'minus'}"
                    self._update_plant_calibration_experiment(
                        phase=f"measuring {trial_id}",
                        current_trial=trial_id,
                    )
                    self._restore_plant_calibration_baseline(
                        baseline_q1=baseline_q1,
                        baseline_q2=baseline_q2,
                        generation=generation,
                        token=token,
                        next_trial_id=trial_id,
                    )
                    measurement = self._run_plant_calibration_trial(
                        config=config,
                        generation=generation,
                        token=token,
                        trial_id=trial_id,
                        channel="combined",
                        direction=direction,
                        baseline_q1=baseline_q1,
                        baseline_q2=baseline_q2,
                        commanded_q1=q1,
                        commanded_q2=q2,
                        actuator_step=direction * base_step,
                    )
                    measurements.append(measurement)
                    self._log(
                        "[PLANT_CAL][MEASUREMENT] "
                        f"trial={measurement.trial_id} channel={measurement.channel} "
                        f"direction={measurement.direction:+d} "
                        f"baseline={measurement.baseline_diameter_um:.6f}um "
                        f"steady={measurement.steady_diameter_um:.6f}um "
                        f"delta={measurement.diameter_change_um:+.6f}um "
                        f"detected={measurement.response_detected} "
                        f"classification={measurement.response_classification}"
                    )
                    completed += 1
                    self._update_plant_calibration_experiment(
                        completed_trials=completed,
                        measurement_count=len(measurements),
                    )

            validation_fraction = float(config.validation_step_fraction)
            validation_base_step = base_step * validation_fraction
            for repeat in range(int(config.validation_repetitions)):
                directions = (1, -1) if repeat % 2 == 0 else (-1, 1)
                for direction in directions:
                    q1 = (
                        baseline_q1
                        + direction * q1_sign * float(config.q1_step) * validation_fraction
                    )
                    q2 = (
                        baseline_q2
                        + direction * q2_sign * float(config.q2_step) * validation_fraction
                    )
                    self._validate_plant_calibration_target(q1, q2)
                    trial_id = (
                        f"validation-r{repeat + 1}-"
                        f"{'plus' if direction > 0 else 'minus'}"
                    )
                    self._update_plant_calibration_experiment(
                        phase=f"validating {trial_id}",
                        current_trial=trial_id,
                    )
                    self._restore_plant_calibration_baseline(
                        baseline_q1=baseline_q1,
                        baseline_q2=baseline_q2,
                        generation=generation,
                        token=token,
                        next_trial_id=trial_id,
                    )
                    measurement = self._run_plant_calibration_trial(
                        config=config,
                        generation=generation,
                        token=token,
                        trial_id=trial_id,
                        channel="validation",
                        direction=direction,
                        baseline_q1=baseline_q1,
                        baseline_q2=baseline_q2,
                        commanded_q1=q1,
                        commanded_q2=q2,
                        actuator_step=direction * validation_base_step,
                    )
                    measurements.append(measurement)
                    completed += 1
                    self._update_plant_calibration_experiment(
                        completed_trials=completed,
                        measurement_count=len(measurements),
                    )

            result = build_plant_calibration_result(
                config=config,
                measurements=measurements,
                session_id=token.session_id,
                started_at=started_at,
                q1_min=float(self.pid_config.q1_min),
                q1_max=float(self.pid_config.q1_max),
                q2_min=float(self.pid_config.q2_min),
                q2_max=float(self.pid_config.q2_max),
                total_flow_max=float(self.pid_config.total_flow_max),
                min_q1_q2_gap=float(self.pid_config.min_q1_q2_gap),
            )
        except Exception as exc:
            failure = exc
        finally:
            with self._lock:
                lifecycle_current = (
                    generation == self._lifecycle_generation
                    and self._state == SystemState.CALIBRATING
                )
            if lifecycle_current:
                self._run_token = None
                self._stop_event.set()
                self._safety.request_stop("plant calibration finished")
                stopped = self._safety_stop_pump() if pump_start_attempted else True
                if stopped:
                    self._safety.confirm_stopped()
                if adapter_started:
                    try:
                        self.vision_adapter.stop()
                    except Exception as exc:
                        stopped = False
                        if failure is None:
                            failure = RuntimeError(f"plant calibration vision stop failed: {exc}")
                self._pump_state.running = False if stopped else self._pump_state.running
                self._refresh_pump_channels(
                    communication_ok=stopped,
                    error="" if stopped else "plant calibration stop could not be verified",
                )
                if not stopped and failure is None:
                    failure = RuntimeError("plant calibration pump stop could not be verified")
                if failure is None and result is not None:
                    self._update_plant_calibration_experiment(
                        status="completed",
                        phase="completed; awaiting save",
                        reason="",
                        record=result.record.to_dict(),
                    )
                    self._set_state(
                        SystemState.STOPPED,
                        message="pump-chip calibration completed; pump stop verified",
                    )
                else:
                    reason = str(failure or "plant calibration did not produce a result")
                    self._update_plant_calibration_experiment(
                        status="failed",
                        phase="failed",
                        reason=reason,
                    )
                    self._set_state(SystemState.ERROR, error=reason)
            elif failure is not None:
                self._update_plant_calibration_experiment(
                    status="cancelled",
                    phase="cancelled by pause or stop",
                    reason=str(failure),
                )
            with self._lock:
                self._plant_calibration_in_progress = False
                self._start_in_progress = False

        if failure is not None:
            raise failure
        if result is None:
            raise RuntimeError("plant calibration was cancelled before completion")
        return result

    def save_plant_calibration_experiment(
        self,
        result: PlantCalibrationExperimentResult,
        calibration_path: str,
    ) -> dict[str, Any]:
        if not isinstance(result, PlantCalibrationExperimentResult):
            raise TypeError("plant calibration result object is invalid")
        saved = save_plant_calibration_result(result, calibration_path)
        self._log(
            f"[PLANT_CAL][SAVED] record={saved['path']} "
            f"measurements={saved['measurements_path']}"
        )
        self._update_plant_calibration_experiment(
            status="saved",
            phase="saved",
            saved_path=saved["path"],
            measurements_path=saved["measurements_path"],
        )
        return saved

    def configure(self, system_config: SystemConfig) -> None:
        with self._lock:
            state = self._state
        if state not in {
            SystemState.IDLE,
            SystemState.CONFIGURED,
            SystemState.VIDEO_READY,
            SystemState.STOPPED,
            SystemState.ERROR,
        }:
            raise RuntimeError(f"state does not allow configuration: {state.value}")
        plant_calibration = (
            PlantCalibrationRecord.from_mapping(dict(system_config.plant_calibration))
            if system_config.plant_calibration
            else None
        )
        interval = int(system_config.control_interval_ms)
        if not self.runtime.min_control_interval_ms <= interval <= self.runtime.max_control_interval_ms:
            raise ValueError(
                "control_interval_ms must be user-configured in range "
                f"[{self.runtime.min_control_interval_ms}, {self.runtime.max_control_interval_ms}]"
            )

        mode = str(system_config.video_source_type or "").strip().lower()
        realtime_mode = mode not in {"file", "local", "local_video", "video"}
        if realtime_mode and not getattr(system_config, "pump_port", ""):
            raise RuntimeError("pump serial port is empty")
        if not getattr(system_config, "pump_address", None):
            system_config.pump_address = 1
        if not getattr(system_config, "pump_baudrate", None):
            system_config.pump_baudrate = 1200
        if not getattr(system_config, "pump_parity", ""):
            system_config.pump_parity = "N"

        self._apply_plant_calibration(plant_calibration)
        self._require_valid_phase_flows(system_config.initial_q1, system_config.initial_q2)

        with self._lock:
            self._cfg = system_config
            self._target_revision += 1
            self._optimizer = None
            self._optimization_candidate = None
            self._optimization_candidate_applied_monotonic = 0.0
            self._optimization_candidate_period_id = 0
            self._optimization_window_observations = []
            self._optimization_review_required = False
            self._stabilizing_until_monotonic = 0.0
            self._get_drift_supervisor().reset()
            self._plant_calibration = plant_calibration
            self._pump_state.pump_response_delay_ms = (
                None
                if plant_calibration is None
                else plant_calibration.conservative_response_delay_ms
            )
            self._pump_state.pump_response_measurement_status = (
                "unmeasured; feedforward disabled"
                if plant_calibration is None
                else (
                    f"{plant_calibration.measurement_source}; calibration="
                    f"{plant_calibration.calibration_id}; uncertainty +"
                    f"{plant_calibration.response_delay_uncertainty_ms:.0f} ms"
                )
            )
            self._error = ""
            self._message = "configured"
        self._set_state(SystemState.CONFIGURED, message="configured")

    def update_target_diameter(self, target_diameter_um: float) -> dict[str, Any]:
        """Update the PID setpoint without restarting the active experiment."""
        target = float(target_diameter_um)
        if not math.isfinite(target) or target <= 0.0:
            raise ValueError("target diameter must be finite and positive")

        allowed_states = {
            SystemState.CONFIGURED,
            SystemState.VIDEO_READY,
            SystemState.INITIALIZED,
            SystemState.RUNNING,
            SystemState.PAUSED,
            SystemState.STOPPED,
        }
        with self._lock:
            if self._cfg is None:
                raise RuntimeError("system config is missing, call configure() first")
            if self._state not in allowed_states:
                raise RuntimeError(
                    f"state does not allow target update: {self._state.value}"
                )
            previous = float(self._cfg.target_diameter)
            self._cfg.target_diameter = target
            self._target_revision += 1
            revision = self._target_revision
            state = self._state

        self._log(
            "[PID][TARGET][UPDATED] "
            f"previous={previous:.6f}um target={target:.6f}um "
            f"revision={revision} state={state.value}"
        )
        return {
            "previous_target_diameter_um": previous,
            "target_diameter_um": target,
            "target_revision": revision,
            "state": state.value,
        }

    def prepare_video(self) -> None:
        with self._lock:
            cfg = self._cfg
            adapter = self.vision_adapter
            state = self._state
        if cfg is None:
            raise RuntimeError("system config is missing, call configure() first")
        if state not in {SystemState.CONFIGURED, SystemState.VIDEO_READY}:
            raise RuntimeError(f"state does not allow video preparation: {state.value}")

        if adapter is not None:
            set_sdk_path = getattr(self.vision_service, "set_mvs_sdk_path", None)
            if callable(set_sdk_path):
                set_sdk_path(str(getattr(cfg, "mvs_sdk_path", "") or ""))
            set_backend = getattr(self.vision_service, "set_selected_backend", None)
            if callable(set_backend):
                set_backend(str(getattr(cfg, "camera_backend", "") or ""))
            set_camera_parameters = getattr(self.vision_service, "set_camera_parameters", None)
            if callable(set_camera_parameters):
                set_camera_parameters(dict(getattr(cfg, "camera_parameters", {}) or {}))
            set_roi = getattr(self.vision_service, "set_recognition_roi", None)
            if callable(set_roi):
                set_roi(dict(getattr(cfg, "recognition_roi", {}) or {}))
            set_calibration = getattr(self.vision_service, "set_calibration_metadata", None)
            if callable(set_calibration):
                set_calibration(dict(getattr(cfg, "calibration", {}) or {}))
            configure_detection = getattr(self.vision_service, "configure_detection_scale", None)
            if callable(configure_detection):
                configure_detection(
                    float(cfg.target_diameter),
                    float(cfg.pixel_to_micron),
                )
            configure_interval = getattr(self.vision_service, "configure_control_interval", None)
            if callable(configure_interval):
                configure_interval(int(cfg.control_interval_ms))
            adapter.prepare_video(
                video_source_type=cfg.video_source_type,
                video_source=cfg.video_source,
                pixel_to_micron=cfg.pixel_to_micron,
            )
        self._set_state(SystemState.VIDEO_READY, message="video ready")

    def apply_vision_tuning(
        self,
        detector_config: Any,
        channel_region_config: Any,
    ) -> dict[str, Any]:
        """Publish saved vision tuning to the live sampling pipeline."""
        apply_tuning = getattr(self.vision_service, "apply_tuning_config", None)
        if not callable(apply_tuning):
            raise RuntimeError("当前视觉服务不支持运行时应用液滴识别调参参数")
        return dict(apply_tuning(detector_config, channel_region_config) or {})

    def configure_generation_measurement(
        self,
        *,
        channel_height_um: float,
        channel_width_um: float,
        volume_correction_factor: float,
    ) -> None:
        """Apply generation-zone geometry while pumps/control are stopped."""
        values = (
            float(channel_height_um),
            float(channel_width_um),
            float(volume_correction_factor),
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("生成区通道尺寸和体积修正系数必须为有限正数")
        with self._lock:
            if self._cfg is None:
                raise RuntimeError("system config is missing, call configure() first")
            if self._state not in {
                SystemState.CONFIGURED,
                SystemState.VIDEO_READY,
                SystemState.INITIALIZED,
                SystemState.STOPPED,
            }:
                raise RuntimeError("运行或标定过程中不能更改生成区几何参数")
            roi = dict(self._cfg.recognition_roi or {})
            roi.update(
                {
                    "channel_width_um": values[1],
                    "generation_channel_height_um": values[0],
                    "generation_channel_width_um": values[1],
                    "generation_volume_correction": values[2],
                }
            )
            self._cfg.recognition_roi = roi
        setter = getattr(self.vision_service, "set_recognition_roi", None)
        if callable(setter):
            setter(roi)

    def discover_cameras(self) -> dict[str, Any]:
        self._log("[CAMERA][CALLCHAIN] frontend -> orchestrator -> vision_service -> CameraManager")
        discover = getattr(self.vision_service, "discover_cameras_result", None)
        if callable(discover):
            return discover()
        discover = getattr(self.vision_service, "refresh_cameras_result", None)
        if callable(discover):
            return discover()
        raise AttributeError("vision_service missing discover camera interface")

    def select_camera(self, unique_id: str, backend_name: str | None = None) -> dict[str, Any]:
        select = getattr(self.vision_service, "select_camera", None)
        if not callable(select):
            raise AttributeError("vision_service missing select_camera interface")
        return select(unique_id, backend_name)

    def test_camera(self, camera_parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        test = getattr(self.vision_service, "test_camera", None)
        if not callable(test):
            raise AttributeError("vision_service missing test_camera interface")
        try:
            return test(camera_config=camera_parameters or {})
        except TypeError:
            return test()

    def analyze_channel_calibration_preview(
        self,
        preview_png_base64: str,
        roi: dict[str, Any] | None = None,
        channel_width_um: float = 50.0,
        configured_pixel_to_micron: float = 1.0,
        hough_parameters: dict[str, float | int] | None = None,
    ) -> dict[str, Any]:
        analyze = getattr(self.vision_service, "analyze_channel_calibration_preview", None)
        if not callable(analyze):
            raise RuntimeError("当前视觉服务不支持测试帧管道分析")
        try:
            result = analyze(
                preview_png_base64,
                roi=roi,
                channel_width_um=channel_width_um,
                configured_pixel_to_micron=configured_pixel_to_micron,
                hough_parameters=hough_parameters,
            )
        except TypeError as exc:
            if "hough_parameters" not in str(exc):
                raise
            # Preserve compatibility with an injected legacy vision service.
            result = analyze(
                preview_png_base64,
                roi=roi,
                channel_width_um=channel_width_um,
                configured_pixel_to_micron=configured_pixel_to_micron,
            )
        return dict(result or {})

    def get_last_control_period_droplets(self) -> dict[str, Any]:
        getter = getattr(self.vision_service, "get_last_control_period_droplets", None)
        if not callable(getter):
            return {
                "period_id": 0,
                "droplet_count": 0,
                "droplets": [],
                "sample_frame_count": 0,
                "frames": [],
                "reason": "当前视觉服务不支持有效液滴回看",
            }
        return dict(getter() or {})

    def _apply_pump_serial_config(self, cfg: SystemConfig) -> None:
        serial_cfg = self.pump_service.serial_config
        serial_cfg.port = str(cfg.pump_port).strip()
        serial_cfg.address = int(cfg.pump_address)
        serial_cfg.baudrate = int(cfg.pump_baudrate)
        serial_cfg.parity = str(cfg.pump_parity or "N").strip().upper()
        if serial_cfg.parity not in {"E", "N"}:
            serial_cfg.parity = "N"

    def _require_valid_phase_flows(self, q1: float, q2: float) -> None:
        q1_f = float(q1)
        q2_f = float(q2)
        if not math.isfinite(q1_f) or not math.isfinite(q2_f) or q1_f <= 0.0 or q2_f <= 0.0:
            raise ValueError("Q1 和 Q2 必须为有限正数")
        pid_cfg = getattr(self, "pid_config", None)
        q1_value = float(q1)
        q2_value = float(q2)
        if not math.isfinite(q1_value) or not math.isfinite(q2_value):
            raise ValueError("Q1 and Q2 must be finite")
        q1_min = float(getattr(pid_cfg, "q1_min", 0.2))
        q1_max = float(getattr(pid_cfg, "q1_max", 5000.0))
        q2_min = float(getattr(pid_cfg, "q2_min", 0.2))
        q2_max = float(getattr(pid_cfg, "q2_max", 5000.0))
        if not q1_min <= q1_value <= q1_max or not q2_min <= q2_value <= q2_max:
            raise ValueError(
                f"pump flow is outside configured range: Q1 [{q1_min}, {q1_max}], "
                f"Q2 [{q2_min}, {q2_max}] uL/min"
            )
        min_gap = effective_q1_q2_gap(getattr(pid_cfg, "min_q1_q2_gap", None))
        if not q1_is_strictly_above_q2(q1_value, q2_value, min_gap):
            raise ValueError(
                f"油相 Q1 必须至少比水相 Q2 大 {min_gap:.1f} uL/min；"
                f"当前 Q1={float(q1):.6f}, Q2={float(q2):.6f}"
            )

    def run_pump_interaction_test(
        self,
        *,
        port: str,
        address: int,
        baudrate: int,
        parity: str,
        q1: float,
        q2: float,
    ) -> dict[str, Any]:
        """Exercise pump communication without initializing camera/control."""
        self._require_valid_phase_flows(q1, q2)
        serial_cfg = self.pump_service.serial_config
        serial_cfg.port = str(port or "").strip().upper()
        serial_cfg.address = int(address)
        serial_cfg.baudrate = int(baudrate)
        serial_cfg.parity = str(parity or "N").strip().upper()
        if not serial_cfg.port:
            raise ValueError("泵串口号不能为空")
        if serial_cfg.parity not in {"E", "N"}:
            raise ValueError("校验位仅支持 E 或 N")
        if float(q1) <= 0.0 or float(q2) <= 0.0:
            raise ValueError("Q1 和 Q2 必须大于 0")
        if self._state in {SystemState.OPTIMIZING, SystemState.STABILIZING, SystemState.RUNNING}:
            raise RuntimeError("系统正在运行，不能执行泵机交互测试")

        steps: list[dict[str, Any]] = []
        start_attempted = False
        stop_verified = False

        def record(name: str, ok: bool, detail: str) -> None:
            steps.append({"name": name, "ok": bool(ok), "detail": str(detail or "")})
            self._log(f"[PUMP][INTERACTION_TEST][{'OK' if ok else 'FAIL'}] step={name} detail={detail}")

        try:
            self.pump_service.disconnect()
            state = self.pump_service.connect_and_probe()
            connected = bool(state.comm_established)
            record("连接与通信探测", connected, str(state.failed or "通信正常"))
            if not connected:
                return {"ok": False, "steps": steps}

            write_result = self._apply_init_flow_rates(float(q1), float(q2))
            write_ok = bool(write_result and write_result.ok)
            write_detail = "参数下发及回读一致" if write_ok else str(
                getattr(write_result, "reason", "") or getattr(write_result, "error", "") or "参数下发失败"
            )
            record("下发泵机参数", write_ok, write_detail)
            if not write_ok:
                return {"ok": False, "steps": steps}

            # A connected start command can have taken effect even when its
            # verification reply is lost. Always protectively stop it unless
            # a verified stop has already completed.
            start_attempted = True
            start_result = self.pump_service.start_infusion_and_verify([1, 2])
            infusion_started = bool(start_result.ok)
            record(
                "启动灌注",
                infusion_started,
                "CH1/CH2 已启动并确认" if infusion_started else (start_result.reason or start_result.error),
            )
            if not infusion_started:
                return {"ok": False, "steps": steps}

            stop_result = self.pump_service.stop_system_and_verify()
            stop_ok = bool(stop_result.ok)
            if stop_ok:
                stop_verified = True
            record(
                "关闭灌注",
                stop_ok,
                "灌注已停止并确认" if stop_ok else (stop_result.reason or stop_result.error),
            )
            return {"ok": bool(stop_ok), "steps": steps}
        finally:
            if start_attempted and not stop_verified:
                try:
                    emergency_stop = self.pump_service.stop_system_and_verify()
                    record(
                        "异常保护停止",
                        bool(emergency_stop.ok),
                        "灌注已停止" if emergency_stop.ok else (emergency_stop.reason or emergency_stop.error),
                    )
                except Exception as exc:
                    record("异常保护停止", False, str(exc))

    def _default_channel_params(self, channel: int, q: float) -> ChannelParams:
        return self.pump_service.channel_params_for_flow(channel, q)

    def _to_channel_params_with_flow(self, channel: int, q: float) -> ChannelParams:
        return self.pump_service.channel_params_for_flow(channel, q)

    def _apply_init_flow_rates(self, q1: float, q2: float):
        self._require_valid_phase_flows(q1, q2)
        retries = 3
        last_res = None
        for attempt in range(1, retries + 1):
            ready = self.pump_service.prepare_parameter_write(0x03)
            if not ready.ok:
                ready.reason = ready.reason or f"initial flow prepare write failed (attempt {attempt})"
                last_res = ready
                time.sleep(0.12)
                continue

            p1 = self._to_channel_params_with_flow(1, q1)
            p2 = self._to_channel_params_with_flow(2, q2)
            q1_encoded = self._flow_from_channel_params(p1)
            q2_encoded = self._flow_from_channel_params(p2)
            try:
                self._require_valid_phase_flows(q1_encoded, q2_encoded)
            except (TypeError, ValueError) as exc:
                last_res = self.pump_service._fail(
                    f"encoded initial flows violate oil/water ordering: {exc}"
                )
                time.sleep(0.12)
                continue
            self._log(
                "[PUMP][INIT][PARAMS] "
                f"CH1 target={q1:.6f}uL/min dispense={p1.dispense_value}/unit{p1.dispense_unit} "
                f"infuse={p1.infuse_time_value}/unit{p1.infuse_time_unit} "
                f"calc={self._flow_from_channel_params(p1) or 0.0:.6f}uL/min"
            )
            w1 = self.pump_service.write_wsp_and_verify(1, p1)
            if not w1.ok:
                w1.reason = w1.reason or f"initial flow CH1 write failed (attempt {attempt})"
                last_res = w1
                time.sleep(0.12)
                continue

            self._log(
                "[PUMP][INIT][PARAMS] "
                f"CH2 target={q2:.6f}uL/min dispense={p2.dispense_value}/unit{p2.dispense_unit} "
                f"infuse={p2.infuse_time_value}/unit{p2.infuse_time_unit} "
                f"calc={self._flow_from_channel_params(p2) or 0.0:.6f}uL/min"
            )
            w2 = self.pump_service.write_wsp_and_verify(2, p2)
            if not w2.ok:
                w2.reason = w2.reason or f"initial flow CH2 write failed (attempt {attempt})"
                last_res = w2
                time.sleep(0.12)
                continue

            en = self.pump_service.enable_channels_and_verify(0x03)
            if not en.ok:
                en.reason = en.reason or f"initial flow enable CH1/CH2 failed (attempt {attempt})"
                last_res = en
                time.sleep(0.12)
                continue

            try:
                q1_hw, q2_hw = self.pump_service.get_current_q_state()
            except Exception as exc:
                last_res = self.pump_service._fail(f"initial flow final hardware readback failed: {exc}")
                time.sleep(0.12)
                continue

            ok1, reason1 = self._flow_matches("Q1", q1, q1_hw)
            ok2, reason2 = self._flow_matches("Q2", q2, q2_hw)
            if not (ok1 and ok2):
                reason = "; ".join(part for part in (reason1, reason2) if part)
                self._log(
                    "[PUMP][INIT][FLOW][VERIFY_FAIL] "
                    f"q1_set={q1:.6f} q1_hw={q1_hw:.6f} "
                    f"q2_set={q2:.6f} q2_hw={q2_hw:.6f} "
                    f"reason={reason}"
                )
                last_res = self.pump_service._fail(f"initial flow final hardware readback mismatch: {reason}")
                time.sleep(0.12)
                continue

            self._pump_state.q1 = float(q1)
            self._pump_state.q2 = float(q2)
            self._pump_state.q1_actual = float(q1_hw)
            self._pump_state.q2_actual = float(q2_hw)
            self._pump_state.last_update_ok = True
            self._pump_state.last_update_reason = "initial flow update succeeded"
            self._pump_state.last_error = ""
            self._refresh_pump_channels(communication_ok=True, error="")
            self._log(
                "[PUMP][INIT][FLOW][HARDWARE_OK] "
                f"q1_target={q1:.6f} q2_target={q2:.6f} "
                f"q1_actual={self._pump_state.q1_actual:.6f} q2_actual={self._pump_state.q2_actual:.6f}"
            )
            return w2

        return last_res

    @staticmethod
    def _flow_matches(name: str, target: float, actual: float) -> tuple[bool, str]:
        target_f = float(target)
        actual_f = float(actual)
        tolerance = max(0.05, abs(target_f) * 0.005)
        error = abs(actual_f - target_f)
        if error <= tolerance:
            return True, ""
        return (
            False,
            f"{name} target={target_f:.6f}uL/min actual={actual_f:.6f}uL/min "
            f"error={error:.6f} tolerance={tolerance:.6f}",
        )

    def _stop_pump_verified(self, source: str) -> tuple[bool, str]:
        """Stop the pump and keep the runtime state conservative on failure."""
        try:
            result = self.pump_service.stop_system_and_verify()
        except Exception as exc:
            result = None
            reason = str(exc)
        else:
            reason = result.reason or result.error or "pump stop verification failed"
        if result is not None and result.ok:
            self._pump_state.running = False
            self._pump_state.last_error = ""
            self._refresh_pump_channels(communication_ok=True, error="")
            self._log(f"[PUMP][STOP][OK] source={source}")
            return True, "pump stopped and verified"
        self._pump_state.last_error = str(reason)
        self._refresh_pump_channels(communication_ok=False, error=str(reason))
        self._log(f"[PUMP][STOP][FAIL] source={source} reason={reason}")
        return False, str(reason)

    def _start_lifecycle_is_current(self, generation: int) -> bool:
        with self._lock:
            return (
                generation == self._lifecycle_generation
                and self._state in {SystemState.INITIALIZED, SystemState.PAUSED, SystemState.STOPPED}
            )

    def _rollback_start_if_stale(
        self,
        generation: int,
        *,
        adapter_started: bool,
        recorder_started: bool,
    ) -> None:
        if self._start_lifecycle_is_current(generation):
            return
        self._rollback_start(
            adapter_started=adapter_started,
            recorder_started=recorder_started,
            stop_pump=True,
        )
        raise RuntimeError("start superseded by a lifecycle transition")

    def _update_flow_with_lifecycle_guard(self, q1: float, q2: float, generation: int):
        """Run a flow update without letting lifecycle changes trigger retries."""
        self._require_valid_phase_flows(q1, q2)
        with self._lock:
            if (
                generation != self._lifecycle_generation
                or self._state not in {
                    SystemState.CALIBRATING,
                    SystemState.OPTIMIZING,
                    SystemState.STABILIZING,
                    SystemState.RUNNING,
                }
                or self._stop_event.is_set()
                or self._pause_event.is_set()
            ):
                return None
        token = self._run_token
        interval_s = max(
            0.01,
            float(
                getattr(self._cfg, "control_interval_ms", self.runtime.default_control_interval_ms)
                if self._cfg is not None
                else self.runtime.default_control_interval_ms
            )
            / 1000.0,
        )
        transaction_watchdog_s = self._control_heartbeat_timeout_s(
            interval_s,
            pump_transaction=True,
        )
        if token is not None and not self._safety.heartbeat(
            token, timeout_s=transaction_watchdog_s
        ):
            raise RuntimeError("pump update rejected by safety supervisor")
        update_res = self.pump_service.update_flow_while_running(float(q1), float(q2))
        with self._lock:
            current = (
                generation == self._lifecycle_generation
                and self._state in {
                    SystemState.CALIBRATING,
                    SystemState.OPTIMIZING,
                    SystemState.STABILIZING,
                    SystemState.RUNNING,
                }
                and not self._stop_event.is_set()
                and not self._pause_event.is_set()
            )
        if not current:
            return None
        return update_res

    def _control_heartbeat_timeout_s(
        self,
        interval_s: float,
        *,
        pump_transaction: bool = False,
    ) -> float:
        timeout_s = max(5.0, max(0.01, float(interval_s)) * 2.5)
        if pump_transaction:
            timeout_s = max(
                timeout_s,
                float(getattr(self.runtime, "pump_update_watchdog_timeout_s", 15.0)),
            )
        return timeout_s

    @staticmethod
    def _advance_control_deadline(
        deadline: float,
        interval_s: float,
        completed_at: float,
    ) -> tuple[float, int]:
        """Advance an anchored periodic schedule without catch-up bursts."""
        interval = max(0.01, float(interval_s))
        slots = max(1, int((float(completed_at) - float(deadline)) // interval) + 1)
        return float(deadline) + slots * interval, max(0, slots - 1)

    def _rollback_start(self, *, adapter_started: bool, recorder_started: bool, stop_pump: bool) -> bool:
        """Undo resources acquired by a partially completed start."""
        self._run_token = None
        self._pump_control_enabled = False
        self._stop_event.set()
        self._pause_event.set()
        stop_ok = True
        if stop_pump and self._is_realtime_mode():
            stop_ok, _ = self._stop_pump_verified("start rollback")
        if adapter_started:
            try:
                self.vision_adapter.stop()
            except Exception as exc:
                self._log(f"[ORCH][WARN] start rollback adapter stop failed: {exc}")
        if recorder_started:
            self._pid_data_recorder.finish_session()
            self._pid_data_recorder.discard()
        return stop_ok

    def initialize_system(self) -> None:
        with self._lock:
            cfg = self._cfg
            state = self._state
        if cfg is None:
            raise RuntimeError("system config is missing, call configure() first")
        if state not in {SystemState.VIDEO_READY, SystemState.CONFIGURED, SystemState.STOPPED}:
            raise RuntimeError(f"state does not allow initialization: {state.value}")

        self._set_state(SystemState.INITIALIZING, message="initializing")
        try:
            self._pump_control_enabled = False
            if self._is_realtime_mode():
                self._apply_pump_serial_config(cfg)
                probe = self.pump_service.connect_and_probe()
                self._pump_state.connected = bool(probe.serial_connected)
                self._pump_state.comm_established = bool(probe.comm_established)
                self._pump_state.fully_ready = bool(probe.fully_ready)
                self._refresh_pump_channels(
                    communication_ok=bool(probe.comm_established),
                    error="" if probe.comm_established else str(probe.failed),
                )
                if not probe.comm_established:
                    raise RuntimeError(f"pump communication is not established: {probe.failed}")

                init_apply = self._apply_init_flow_rates(cfg.initial_q1, cfg.initial_q2)
                if init_apply is None:
                    raise RuntimeError("initial flow update did not return a result")
                if not init_apply.ok:
                    raise RuntimeError(f"initial flow update failed: {init_apply.reason or init_apply.error}")
                self._pump_control_enabled = True
                self._pump_state.last_error = ""
                self._refresh_pump_channels(communication_ok=True, error="")
                if not self._sync_pump_flow_readback("initialize", update_command=False):
                    raise RuntimeError(self._pump_state.last_error or "initial flow readback failed")
            else:
                self._message = "local video mode: skip pump initialization and PID output"

            self._pid_controller.reset()
            self._log(
                "[PID][INIT] "
                f"mode={self.pid_config.control_mode} "
                f"adaptive_min_samples={self.pid_config.adaptive_min_sample_count} "
                f"adaptive_interval={self.pid_config.adaptive_update_interval} "
                f"q1_gain={self.pid_config.q1_output_gain:.3f} "
                f"q2_gain={self.pid_config.q2_output_gain:.3f}"
            )
            self._last_control_ts = None
            self._last_control_frame_id = None
            self._last_control_period_id = None
            self._last_seen_vision_period_id = None
            self._set_state(SystemState.INITIALIZED, message="initialized")
        except Exception as e:
            self._pump_state.last_error = str(e)
            self._refresh_pump_channels(communication_ok=False, error=str(e))
            self._set_state(SystemState.ERROR, error=f"system initialization failed: {e}")
            raise

    def run_preflight_check(self) -> dict[str, Any]:
        issues: list[str] = []
        with self._lock:
            cfg = self._cfg
            state = self._state
        safety = self._safety.snapshot()
        calibration_source = "none"
        if cfg is None:
            issues.append("system configuration is missing")
        if state not in {SystemState.INITIALIZED, SystemState.PAUSED, SystemState.STOPPED}:
            issues.append(f"system state {state.value} is not ready to start")
        if safety.state in {SafetyState.ESTOP_LATCHED, SafetyState.FAULT_LATCHED}:
            issues.append(f"safety latch must be reset: {safety.reason}")
        if not safety.stop_verified:
            issues.append("pump stop has not been verified")
        if cfg is not None and self._is_realtime_mode():
            if dict(getattr(cfg, "calibration", {}) or {}):
                calibration_source = "versioned_file"
            elif self._has_verified_runtime_channel_calibration():
                calibration_source = "runtime_channel"
            else:
                issues.append(
                    "a versioned pixel calibration file or a successful runtime channel calibration "
                    "is required for realtime PID"
                )
            if not self._pump_control_enabled:
                issues.append("pump control is not initialized")
            if not self._pump_state.connected or not self._pump_state.comm_established:
                issues.append("pump communication is not established")
        return {
            "ok": not issues,
            "issues": issues,
            "state": state.value,
            "calibration_source": calibration_source,
        }

    def _has_verified_runtime_channel_calibration(self) -> bool:
        """Accept only a completed, physically anchored channel-width calibration."""
        adapter = self.vision_adapter
        if adapter is None:
            return False
        try:
            raw = adapter.get_snapshot()
        except Exception:
            return False

        def value(name: str, default: Any = None) -> Any:
            if isinstance(raw, dict):
                return raw.get(name, default)
            return getattr(raw, name, default)

        try:
            return (
                str(value("channel_calibration_status", "")) == "calibrated"
                and str(value("scale_source", "")) == "generation_channel_width"
                and float(value("channel_calibration_confidence", 0.0) or 0.0) > 0.0
                and float(value("pixel_to_micron", 0.0) or 0.0) > 0.0
                and float(value("channel_width_um", 0.0) or 0.0) > 0.0
                and float(value("channel_width_px", 0.0) or 0.0) > 0.0
            )
        except (TypeError, ValueError):
            return False

    def start(self) -> None:
        # Serialize start attempts while allowing stop() to invalidate the
        # attempt without waiting for adapter or pump I/O.
        with self._lock:
            if self._start_in_progress:
                raise RuntimeError("start is already in progress")
            self._start_in_progress = True
        try:
            self._start_impl(SystemState.RUNNING)
        finally:
            with self._lock:
                self._start_in_progress = False

    def start_optimization(self, config: BayesianOptimizationConfig) -> None:
        """Start a bounded BO commissioning run before enabling PID control."""
        with self._lock:
            if self._cfg is None:
                raise RuntimeError("system config is missing, call configure() first")
            if not self._is_realtime_mode():
                raise RuntimeError("BO requires realtime vision and pump control")
            if abs(float(config.target_diameter_um) - float(self._cfg.target_diameter)) > 1e-9:
                raise ValueError("optimizer target must equal the configured PID target")
            if float(config.q1_min) < float(self.pid_config.q1_min) or float(config.q1_max) > float(self.pid_config.q1_max):
                raise ValueError("optimizer Q1 bounds exceed controller safety limits")
            if float(config.q2_min) < float(self.pid_config.q2_min) or float(config.q2_max) > float(self.pid_config.q2_max):
                raise ValueError("optimizer Q2 bounds exceed controller safety limits")
            if float(config.min_q1_q2_gap) < float(self.pid_config.min_q1_q2_gap):
                raise ValueError("optimizer phase gap is weaker than controller safety limit")
            if float(config.total_flow_max) > float(self.pid_config.total_flow_max):
                raise ValueError("optimizer total-flow limit exceeds controller safety limit")
            calibration = getattr(self, "_plant_calibration", None)
            if calibration is not None:
                if abs(
                    float(config.measured_response_delay_ms)
                    - float(calibration.response_delay_median_ms)
                ) > 1e-9 or abs(
                    float(config.response_delay_uncertainty_ms)
                    - float(calibration.response_delay_uncertainty_ms)
                ) > 1e-9:
                    raise ValueError(
                        "optimizer response delay must match the active plant calibration"
                    )
            current_q1 = float(getattr(self._pump_state, "q1", None) or self._cfg.initial_q1)
            current_q2 = float(getattr(self._pump_state, "q2", None) or self._cfg.initial_q2)
            self._optimizer = SafeBayesianOptimizer(
                config,
                initial_point=(current_q1, current_q2),
            )
            self._optimization_candidate = None
            self._optimization_candidate_applied_monotonic = 0.0
            self._optimization_candidate_period_id = 0
            self._optimization_window_observations = []
            self._optimization_review_required = False
            self._stabilizing_until_monotonic = 0.0
            self._get_drift_supervisor().reset()
            previous_delay = self._pump_state.pump_response_delay_ms
            previous_delay_status = self._pump_state.pump_response_measurement_status
            self._pump_state.pump_response_delay_ms = float(
                config.measured_response_delay_ms + config.response_delay_uncertainty_ms
            )
            self._pump_state.pump_response_measurement_status = (
                f"{config.response_delay_source}; uncertainty +{config.response_delay_uncertainty_ms:.0f} ms"
            )
        try:
            self.start_with_mode(SystemState.OPTIMIZING)
        except Exception:
            with self._lock:
                self._optimizer = None
                self._optimization_candidate = None
                self._optimization_window_observations = []
                self._pump_state.pump_response_delay_ms = previous_delay
                self._pump_state.pump_response_measurement_status = previous_delay_status
            raise

    def start_with_mode(self, start_state: SystemState) -> None:
        if start_state not in {SystemState.OPTIMIZING, SystemState.RUNNING}:
            raise ValueError("start mode must be OPTIMIZING or RUNNING")
        with self._lock:
            if self._start_in_progress:
                raise RuntimeError("start is already in progress")
            self._start_in_progress = True
        try:
            self._start_impl(start_state)
        finally:
            with self._lock:
                self._start_in_progress = False

    def _start_impl(self, start_state: SystemState = SystemState.RUNNING) -> None:
        with self._lock:
            if self._state not in {SystemState.INITIALIZED, SystemState.PAUSED, SystemState.STOPPED}:
                raise RuntimeError(f"state does not allow start: {self._state.value}")
            if self._loop_thread and self._loop_thread.is_alive():
                raise RuntimeError(f"state does not allow start: {self._state.value}")
            adapter = self.vision_adapter
            starting_state = self._state
            generation = self._lifecycle_generation

        if (
            start_state == SystemState.RUNNING
            and self._is_realtime_mode()
            and (
                getattr(self, "_plant_calibration", None) is None
                or not bool(self.pid_config.log_sensitivity_calibrated)
            )
        ):
            raise RuntimeError(
                "没有有效的生成区 Q1/Q2 动力学标定，禁止使用默认映射启动 PID 闭环"
            )

        recorder_started = False
        adapter_started = False
        preflight = self.run_preflight_check()
        if not preflight["ok"]:
            raise RuntimeError("preflight check failed: " + "; ".join(preflight["issues"]))

        token = self._safety.begin_session()
        self._run_token = token
        with self._lock:
            context = dict(self._disturbance_context)
            configured_experiment = str(context.get("experiment_id", "")).strip()
            if not configured_experiment or configured_experiment == self._auto_disturbance_experiment_id:
                context["experiment_id"] = token.session_id
                self._auto_disturbance_experiment_id = token.session_id
            if not str(context.get("chip_id", "")).strip():
                context["chip_id"] = str(
                    dict(getattr(self._cfg, "plant_calibration", {}) or {}).get("chip_id")
                    or dict(getattr(self._cfg, "calibration", {}) or {}).get("calibration_id")
                    or getattr(self._cfg, "video_source", "")
                    or getattr(self._cfg, "camera_backend", "")
                    or "realtime-camera"
                )
            if not str(context.get("disturbance_name", "")).strip():
                context["disturbance_name"] = "closed_loop_dynamics"
            if not str(context.get("disturbance_stage", "")).strip():
                context["disturbance_stage"] = "baseline"
            self._disturbance_context = context
        set_context = getattr(adapter, "set_run_context", None)
        if callable(set_context):
            set_context(token.session_id, token.generation)
        if starting_state != SystemState.PAUSED:
            self._pid_data_recorder.begin_session(
                {
                    "system_config": asdict(self._cfg) if self._cfg is not None else {},
                    "pid_config": asdict(self.pid_config),
                }
            )
            recorder_started = True

        self._rollback_start_if_stale(
            generation,
            adapter_started=adapter_started,
            recorder_started=recorder_started,
        )
        if adapter is not None:
            adapter_started = True
            try:
                adapter.start()
            except Exception:
                self._rollback_start(adapter_started=True, recorder_started=recorder_started, stop_pump=False)
                raise

        self._rollback_start_if_stale(
            generation,
            adapter_started=adapter_started,
            recorder_started=recorder_started,
        )
        if self._is_realtime_mode():
            if not self._pump_control_enabled:
                self._rollback_start(adapter_started=adapter_started, recorder_started=recorder_started, stop_pump=False)
                raise RuntimeError("control loop is already running")
            if not self._pump_state.connected or not self._pump_state.comm_established:
                self._rollback_start(adapter_started=adapter_started, recorder_started=recorder_started, stop_pump=False)
                raise RuntimeError("pump parameters are not initialized; PID cannot start")
            self._rollback_start_if_stale(
                generation,
                adapter_started=adapter_started,
                recorder_started=recorder_started,
            )
            try:
                start_res = self.pump_service.start_infusion_and_verify([1, 2])
            except Exception:
                self._rollback_start(
                    adapter_started=adapter_started,
                    recorder_started=recorder_started,
                    stop_pump=True,
                )
                raise
            if not start_res.ok:
                reason = start_res.reason or start_res.error or "pump start infusion failed"
                self._pump_state.last_error = str(reason)
                stop_ok = self._rollback_start(
                    adapter_started=adapter_started,
                    recorder_started=recorder_started,
                    stop_pump=True,
                )
                with self._lock:
                    start_still_current = (
                        generation == self._lifecycle_generation
                        and self._state in {
                            SystemState.INITIALIZED,
                            SystemState.PAUSED,
                            SystemState.STOPPED,
                        }
                    )
                    if start_still_current:
                        self._state = SystemState.ERROR
                        self._error = str(reason) + ("" if stop_ok else "; rollback stop failed")
                self._run_token = None
                self._safety.trip(str(reason))
                if stop_ok:
                    self._safety.confirm_stopped()
                raise RuntimeError(f"pump start failed: {reason}")
            self._pump_state.running = True
            self._refresh_pump_channels(communication_ok=True, error="")
            self._rollback_start_if_stale(
                generation,
                adapter_started=adapter_started,
                recorder_started=recorder_started,
            )
            if not self._sync_pump_flow_readback("start"):
                reason = self._pump_state.last_error or "start flow readback failed"
                stop_ok = self._rollback_start(
                    adapter_started=adapter_started,
                    recorder_started=recorder_started,
                    stop_pump=True,
                )
                if not stop_ok:
                    reason = f"{reason}; rollback stop failed"
                with self._lock:
                    if (
                        generation == self._lifecycle_generation
                        and self._state in {
                            SystemState.INITIALIZED,
                            SystemState.PAUSED,
                            SystemState.STOPPED,
                        }
                    ):
                        self._state = SystemState.ERROR
                        self._error = reason
                raise RuntimeError(reason)

        self._rollback_start_if_stale(
            generation,
            adapter_started=adapter_started,
            recorder_started=recorder_started,
        )
        heartbeat_timeout = self._control_heartbeat_timeout_s(
            float(self._cfg.control_interval_ms if self._cfg else self.runtime.default_control_interval_ms)
            / 1000.0,
        )
        try:
            self._safety.arm(token, heartbeat_timeout_s=heartbeat_timeout)
        except Exception:
            self._run_token = None
            self._safety.trip("failed to arm safety supervisor after pump start")
            raise
        # Do not apply the snapshot that happened to be cached while the pump
        # was being initialized. Start feedback from the next completed vision
        # period so the first displayed state is not an avoidable stale freeze.
        try:
            baseline_recognition = self._read_recognition()
            if int(baseline_recognition.control_period_id) > 0:
                self._last_seen_vision_period_id = int(baseline_recognition.control_period_id)
        except Exception as exc:
            self._log(f"[PID][START][WARN] unable to baseline vision period: {exc}")
        with self._lock:
            if (
                generation != self._lifecycle_generation
                or self._state not in {SystemState.INITIALIZED, SystemState.PAUSED, SystemState.STOPPED}
            ):
                stale = True
            else:
                stale = False
                self._stop_event.clear()
                self._pause_event.clear()
                self._loop_thread = threading.Thread(
                    target=self._control_loop,
                    name="orchestrator-control-loop",
                    daemon=True,
                )
                self._loop_thread.start()
                self._state = start_state
        if stale:
            self._rollback_start(
                adapter_started=adapter_started,
                recorder_started=recorder_started,
                stop_pump=True,
            )
            raise RuntimeError("start superseded by a lifecycle transition")
        if start_state == SystemState.OPTIMIZING:
            self._log("[BO][START] safe operating-point optimization started; PID and feedforward disabled")
            self._set_state(SystemState.OPTIMIZING, message="optimizing operating point")
        else:
            self._log("[PID][START] PID feedback started")
            self._set_state(SystemState.RUNNING, message="running")

    def pause(self) -> None:
        with self._lock:
            if self._state not in {
                SystemState.CALIBRATING,
                SystemState.OPTIMIZING,
                SystemState.STABILIZING,
                SystemState.RUNNING,
            }:
                return
            # Invalidate both guards before any pump I/O so an in-flight
            # control step cannot publish another command after pause.
            self._pause_event.set()
            self._lifecycle_generation += 1
            self._state = SystemState.PAUSED
        self._run_token = None
        self._safety.pause("operator pause")
        stopped = self._safety_stop_pump()
        if stopped:
            self._safety.confirm_stopped()
            self._pump_state.running = False
            self._refresh_pump_channels(communication_ok=True, error="")
            self._set_state(SystemState.PAUSED, message="paused; pump stop verified")
            return
        detail = str(self._pump_state.last_error or "pump stop failed")
        reason = f"pause requested but pump stop could not be verified: {detail}"
        with self._lock:
            self._pump_control_enabled = False
            self._stop_event.set()
        self._pump_state.last_error = reason
        self._set_state(SystemState.ERROR, error=reason)
        raise RuntimeError(reason)

    def resume(self) -> None:
        with self._lock:
            if self._state != SystemState.PAUSED:
                return
            self._lifecycle_generation += 1
        safety = self._safety.snapshot()
        if not safety.stop_verified:
            raise RuntimeError("cannot leave pause before pump stop is verified")
        # Resume only unlocks the paused state. Starting infusion requires a
        # second, explicit start() action, which creates a fresh run token.
        self._pause_event.clear()
        self._set_state(
            SystemState.INITIALIZED,
            message="pause released; press start to begin a new pump run",
        )

    def stop(self) -> None:
        with self._lock:
            self._lifecycle_generation += 1
            self._state = SystemState.STOPPING
            self._run_token = None
            self._pause_event.set()
            self._stop_event.set()
        self._safety.request_stop("operator stop")
        self._log("[ORCH][STOPPING] stopping")
        t = self._loop_thread
        if t and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=float(self.runtime.stop_timeout_s))
        thread_stopped = not bool(t and t.is_alive())

        stopped = self._safety_stop_pump()
        if stopped:
            self._safety.confirm_stopped()

        adapter_error = ""
        try:
            adapter = self.vision_adapter
            if adapter is not None:
                adapter.stop()
        except Exception as exc:
            adapter_error = str(exc)
            self._log(f"[ORCH][WARN] vision stop failed: {exc}")

        self._pid_data_recorder.finish_session()
        self._loop_thread = None if thread_stopped else t
        if stopped and thread_stopped:
            self._set_state(SystemState.STOPPED, message="stopped and pump stop verified")
            return

        failures: list[str] = []
        if not stopped:
            failures.append("pump stop is not verified")
        if not thread_stopped:
            failures.append("control thread did not exit before timeout")
        if adapter_error:
            failures.append(f"vision stop failed: {adapter_error}")
        reason = "; ".join(failures)
        self._safety.trip(reason)
        self._set_state(SystemState.ERROR, error=reason)

    def get_pid_session_data_status(self) -> dict[str, Any]:
        return self._pid_data_recorder.status()

    def has_unsaved_pid_session_data(self) -> bool:
        return self._pid_data_recorder.has_unsaved_data()

    def save_pid_session_data(self, database_path: str) -> dict[str, Any]:
        result = self._pid_data_recorder.save_to_sqlite(database_path)
        self._log(
            f"[PID][DATABASE][SAVED] path={result['path']} "
            f"session={result['session_id']} records={result['record_count']}"
        )
        return result

    def load_pid_replay(self, database_path: str) -> PIDReplayData:
        return load_pid_replay(database_path)

    def discard_pid_session_data(self) -> None:
        status = self._pid_data_recorder.status()
        self._pid_data_recorder.discard()
        self._log(
            f"[PID][DATABASE][DISCARDED] session={status['session_id']} "
            f"records={status['record_count']}"
        )

    def get_snapshot(self) -> SystemSnapshot:
        with self._lock:
            rec = self._recognition
            frame = None
            if rec is not None:
                frame = FrameSnapshot(
                    frame_id=int(rec.preview_frame_id or rec.frame_id),
                    timestamp=float(rec.preview_timestamp or rec.timestamp),
                    width=int(rec.frame_width),
                    height=int(rec.frame_height),
                    valid=bool(rec.frame_png_base64),
                    frame_png_base64=rec.frame_png_base64,
                    reason=rec.reason,
                    session_id=rec.session_id,
                    run_generation=rec.run_generation,
                    capture_monotonic=rec.capture_monotonic,
                    hardware_frame_id=rec.hardware_frame_id,
                    hardware_timestamp=rec.hardware_timestamp,
                )
            return copy.deepcopy(
                SystemSnapshot(
                    system_state=self._state,
                    config=self._cfg,
                    recognition=rec,
                    pump_state=self._pump_state,
                    control=self._control,
                    message=_clean_runtime_text(self._message, "message"),
                    error=_clean_runtime_text(self._error, "error"),
                    frame=frame,
                    timestamp=time.time(),
                    disturbance_model=self.disturbance_service.get_status().to_dict(),
                    disturbance_prediction=(
                        self._last_disturbance_prediction.to_dict()
                        if self._last_disturbance_prediction is not None and hasattr(self._last_disturbance_prediction, "to_dict")
                        else None
                    ),
                    safety=asdict(self._safety.snapshot()),
                    optimization=(
                        self._optimizer.status().to_dict()
                        if self._optimizer is not None
                        else None
                    ),
                    disturbance_context=dict(self._disturbance_context),
                    plant_calibration=(
                        None
                        if getattr(self, "_plant_calibration", None) is None
                        else self._plant_calibration.to_dict()
                    ),
                    plant_calibration_experiment=dict(
                        getattr(self, "_plant_calibration_experiment", {}) or {}
                    ),
                    drift_supervisor=self._get_drift_supervisor().status().to_dict(),
                )
            )

    def get_video_frame_snapshot(self) -> FrameSnapshot | None:
        frame_providers = (
            getattr(self.vision_adapter, "get_frame_snapshot", None),
            getattr(self.vision_service, "get_frame_snapshot", None),
        )
        for get_frame in frame_providers:
            if not callable(get_frame):
                continue
            try:
                frame = get_frame()
                if frame is not None:
                    return frame
            except Exception:
                continue
        try:
            raw = self.vision_adapter.get_snapshot()
            rec = self._build_recognition_snapshot(raw)
        except Exception:
            with self._lock:
                rec = self._recognition
        if rec is None:
            return None
        return FrameSnapshot(
            frame_id=int(rec.preview_frame_id or rec.frame_id),
            timestamp=float(rec.preview_timestamp or rec.timestamp),
            width=int(rec.frame_width),
            height=int(rec.frame_height),
            valid=bool(rec.frame_png_base64),
            frame_png_base64=rec.frame_png_base64,
            reason=rec.reason,
        )

    def auto_calibrate_detection(self, duration_s: float = 3.0) -> dict[str, Any]:
        calibrate = getattr(self.vision_service, "auto_calibrate_detection", None)
        if not callable(calibrate):
            raise RuntimeError("当前视觉服务不支持自动标定")
        return dict(calibrate(duration_s) or {})

    def _build_recognition_snapshot(self, raw: Any) -> RecognitionSnapshot:
        if isinstance(raw, RecognitionSnapshot):
            return raw
        if isinstance(raw, dict):
            frame_cnt = int(raw.get("frame_droplet_count", raw.get("active_droplet_count", 0)) or 0)
            total_cnt = int(raw.get("total_droplet_count", raw.get("droplet_count", 0)) or 0)
            new_cnt = int(raw.get("new_crossing_count", 0) or 0)
            has_droplet = bool(raw.get("has_droplet", frame_cnt > 0))
            avg_raw = raw.get("frame_avg_diameter", raw.get("avg_diameter", None))
            avg_diameter = None if avg_raw is None else float(avg_raw)
            single_rate_raw = raw.get("frame_single_cell_rate", raw.get("single_cell_rate", None))
            single_rate = None if single_rate_raw is None else float(single_rate_raw)
            reason = str(raw.get("reason", raw.get("control_reason", "")) or "")
            frame_diameters = [float(v) for v in raw.get("frame_diameters", []) or []]
            crossed_track_diameters = {
                int(track_id): float(value)
                for track_id, value in dict(raw.get("crossed_track_diameters", {}) or {}).items()
            }
            crossed_track_capture_monotonic = {
                int(track_id): float(value)
                for track_id, value in dict(raw.get("crossed_track_capture_monotonic", {}) or {}).items()
            }
            crossed_track_frame_ids = {
                int(track_id): int(value)
                for track_id, value in dict(raw.get("crossed_track_frame_ids", {}) or {}).items()
            }
            diameter_sum_raw = raw.get("frame_diameter_sum", None)
            diameter_sum = (
                float(diameter_sum_raw)
                if diameter_sum_raw is not None
                else float(sum(frame_diameters))
            )
            return RecognitionSnapshot(
                frame_droplet_count=frame_cnt,
                total_droplet_count=total_cnt,
                new_crossing_count=new_cnt,
                avg_diameter=avg_diameter,
                single_cell_rate=float(single_rate or 0.0),
                valid_for_control=bool(raw.get("valid_for_control", False)),
                timestamp=float(raw.get("timestamp", time.time())),
                reason=reason,
                droplet_count=total_cnt,
                active_droplet_count=frame_cnt,
                has_droplet=has_droplet,
                control_reason=reason,
                frame_png_base64=raw.get("frame_png_base64"),
                frame_width=int(raw.get("frame_width", 0) or 0),
                frame_height=int(raw.get("frame_height", 0) or 0),
                video_source_type=str(raw.get("video_source_type", "") or ""),
                video_source=str(raw.get("video_source", "") or ""),
                frame_id=int(raw.get("frame_id", 0) or 0),
                preview_frame_id=int(raw.get("preview_frame_id", raw.get("frame_id", 0)) or 0),
                preview_timestamp=float(raw.get("preview_timestamp", raw.get("timestamp", 0.0)) or 0.0),
                frame_single_cell_count=int(raw.get("frame_single_cell_count", 0) or 0),
                frame_diameters=frame_diameters,
                crossed_track_diameters=crossed_track_diameters,
                crossed_track_capture_monotonic=crossed_track_capture_monotonic,
                crossed_track_frame_ids=crossed_track_frame_ids,
                frame_diameter_sum=diameter_sum,
                frame_avg_diameter=avg_diameter,
                frame_single_cell_rate=single_rate,
                frame_diameter_std=(
                    None
                    if raw.get("frame_diameter_std", None) is None
                    else float(raw.get("frame_diameter_std"))
                ),
                frame_diameter_cv=(
                    None
                    if raw.get("frame_diameter_cv", None) is None
                    else float(raw.get("frame_diameter_cv"))
                ),
                uniformity_valid=bool(raw.get("uniformity_valid", False)),
                uniformity_status=str(raw.get("uniformity_status", "") or "sample insufficient"),
                uniformity_reason=str(raw.get("uniformity_reason", "") or ""),
                capture_fps=float(raw.get("capture_fps", 0.0) or 0.0),
                processing_fps=float(raw.get("processing_fps", 0.0) or 0.0),
                recognition_latency_ms=float(raw.get("recognition_latency_ms", 0.0) or 0.0),
                algorithm_processing_ms=float(raw.get("algorithm_processing_ms", 0.0) or 0.0),
                replaced_processing_frames=int(raw.get("replaced_processing_frames", 0) or 0),
                pending_processing_frames=int(raw.get("pending_processing_frames", 0) or 0),
                period_replaced_processing_frames=int(raw.get("period_replaced_processing_frames", 0) or 0),
                processed_frame_count=int(raw.get("processed_frame_count", 0) or 0),
                period_processed_frames=int(raw.get("period_processed_frames", 0) or 0),
                vision_performance_status=str(raw.get("vision_performance_status", "等待视觉数据") or "等待视觉数据"),
                control_period_id=int(raw.get("control_period_id", 0) or 0),
                motion_window_frames=int(raw.get("motion_window_frames", 0) or 0),
                average_droplet_speed_um_s=(
                    None
                    if raw.get("average_droplet_speed_um_s") is None
                    else float(raw.get("average_droplet_speed_um_s"))
                ),
                speed_sample_count=int(raw.get("speed_sample_count", 0) or 0),
                droplet_generation_rate_hz=float(raw.get("droplet_generation_rate_hz", 0.0) or 0.0),
                pixel_to_micron=float(raw.get("pixel_to_micron", 0.0) or 0.0),
                scale_source=str(raw.get("scale_source", "configured") or "configured"),
                channel_width_um=(
                    None
                    if raw.get("channel_width_um") is None
                    else float(raw.get("channel_width_um"))
                ),
                channel_width_px=(
                    None
                    if raw.get("channel_width_px") is None
                    else float(raw.get("channel_width_px"))
                ),
                channel_calibration_status=str(raw.get("channel_calibration_status", "disabled") or "disabled"),
                channel_calibration_confidence=float(raw.get("channel_calibration_confidence", 0.0) or 0.0),
                channel_calibration_reason=str(raw.get("channel_calibration_reason", "") or ""),
                session_id=str(raw.get("session_id", "") or ""),
                run_generation=int(raw.get("run_generation", 0) or 0),
                capture_monotonic=float(raw.get("capture_monotonic", 0.0) or 0.0),
                hardware_frame_id=int(raw.get("hardware_frame_id", 0) or 0),
                hardware_timestamp=float(raw.get("hardware_timestamp", 0.0) or 0.0),
                raw_frame_diameters=[
                    float(value) for value in raw.get("raw_frame_diameters", []) or []
                ],
                raw_frame_diameter_cv=(
                    None
                    if raw.get("raw_frame_diameter_cv") is None
                    else float(raw.get("raw_frame_diameter_cv"))
                ),
                filtering_rule=str(raw.get("filtering_rule", "none") or "none"),
                calibration_id=str(raw.get("calibration_id", "") or ""),
                calibration_uncertainty_um_per_px=(
                    None
                    if raw.get("calibration_uncertainty_um_per_px") is None
                    else float(raw.get("calibration_uncertainty_um_per_px"))
                ),
                measurement_region=str(raw.get("measurement_region", "generation") or "generation"),
                generation_channel_height_um=float(
                    raw.get("generation_channel_height_um", 50.0) or 50.0
                ),
                generation_channel_width_um=float(
                    raw.get("generation_channel_width_um", 50.0) or 50.0
                ),
                generation_volume_correction=float(
                    raw.get("generation_volume_correction", 1.0) or 1.0
                ),
            )
        raise ValueError(f"unsupported recognition snapshot type: {type(raw).__name__}")

    def _read_recognition(self) -> RecognitionSnapshot:
        adapter = self.vision_adapter
        if adapter is None:
            raise RuntimeError("vision adapter is not configured")
        raw = adapter.get_snapshot()
        snap = self._build_recognition_snapshot(raw)
        with self._lock:
            self._recognition = snap
        return snap

    def _update_control_snapshot(self, ctrl: ControlSnapshot) -> None:
        with self._control_condition:
            self._control = ctrl
            rec = self._recognition
            cfg = self._cfg
            q1_actual = self._pump_state.q1_actual
            q2_actual = self._pump_state.q2_actual
            self._control_condition.notify_all()
        try:
            self._pid_data_recorder.record_sample(
                timestamp=float(ctrl.timestamp),
                frame_id=int(ctrl.frame_id),
                control_period_id=int(getattr(rec, "control_period_id", 0) or 0),
                q1_command_ul_min=float(ctrl.q1_command),
                q2_command_ul_min=float(ctrl.q2_command),
                q1_actual_ul_min=None if q1_actual is None else float(q1_actual),
                q2_actual_ul_min=None if q2_actual is None else float(q2_actual),
                target_diameter_um=(
                    float(ctrl.target_diameter_um)
                    if ctrl.target_diameter_um is not None
                    else None if cfg is None else float(cfg.target_diameter)
                ),
                measured_diameter_um=(
                    None if rec is None or rec.avg_diameter is None else float(rec.avg_diameter)
                ),
                diameter_error_um=float(ctrl.diameter_error),
                droplet_speed_um_s=(
                    None
                    if rec is None or rec.average_droplet_speed_um_s is None
                    else float(rec.average_droplet_speed_um_s)
                ),
                adjustment=float(ctrl.adjustment),
                pid_output=float(ctrl.pid_output),
                kp=float(ctrl.kp),
                ki=float(ctrl.ki),
                kd=float(ctrl.kd),
                adaptive_enabled=bool(ctrl.adaptive_enabled),
                adaptive_active=bool(ctrl.adaptive_active),
                feedback_frozen=bool(ctrl.freeze_feedback),
                reason=str(ctrl.reason or ""),
            )
        except Exception as exc:
            self._log(f"[PID][DATABASE][RECORD][WARN] {exc}")

    def wait_for_control_snapshot(self, after_timestamp: float = 0.0, timeout: float | None = None) -> SystemSnapshot:
        """Wait for a newer PID result, then return one coherent system view."""
        with self._control_condition:
            current = float(getattr(self._control, "timestamp", 0.0) or 0.0)
            if current <= float(after_timestamp):
                self._control_condition.wait(timeout=timeout)
        return self.get_snapshot()

    def _control_loop(self) -> None:
        token = self._run_token
        with self._lock:
            generation = self._lifecycle_generation
        interval_s = max(
            0.01,
            (self._cfg.control_interval_ms if self._cfg else self.runtime.default_control_interval_ms)
            / 1000.0,
        )
        next_deadline = time.monotonic() + interval_s
        self._log(f"[CONTROL][SCHEDULE] interval_ms={interval_s * 1000.0:.3f}")
        try:
            # Vision events can wake the loop early, but PID execution remains
            # anchored to the configured control period. Missed slots are
            # skipped rather than executed as a burst after slow hardware I/O.
            while not self._stop_event.is_set():
                interval_s = max(
                    0.01,
                    (self._cfg.control_interval_ms if self._cfg else self.runtime.default_control_interval_ms)
                    / 1000.0,
                )
                heartbeat_timeout = self._control_heartbeat_timeout_s(interval_s)
                if token is None or not self._safety.heartbeat(
                    token,
                    timeout_s=heartbeat_timeout,
                ):
                    raise RuntimeError("control loop lost its safety run token")

                waiter = getattr(self.vision_adapter, "wait_for_recognition_snapshot", None)
                remaining_s = max(0.0, next_deadline - time.monotonic())
                if callable(waiter):
                    try:
                        waiter(
                            after_period_id=int(self._last_seen_vision_period_id or 0),
                            timeout=remaining_s,
                        )
                    except (AttributeError, TypeError):
                        if self._stop_event.wait(remaining_s):
                            break
                elif self._stop_event.wait(remaining_s):
                    break
                remaining_s = max(0.0, next_deadline - time.monotonic())
                if remaining_s > 0.0 and self._stop_event.wait(remaining_s):
                    break
                with self._lock:
                    lifecycle_valid = (
                        generation == self._lifecycle_generation
                        and self._state in {SystemState.OPTIMIZING, SystemState.STABILIZING, SystemState.RUNNING}
                        and self._pump_control_enabled
                        and not self._pause_event.is_set()
                        and not self._stop_event.is_set()
                    )
                if not lifecycle_valid:
                    continue
                if not self._safety.permits(token):
                    raise RuntimeError("safety supervisor rejected control execution")
                step_started = time.monotonic()
                self.run_control_step()
                step_completed = time.monotonic()
                next_deadline, skipped_slots = self._advance_control_deadline(
                    next_deadline,
                    interval_s,
                    step_completed,
                )
                if skipped_slots > 0:
                    self._log(
                        "[CONTROL][OVERRUN] "
                        f"configured_ms={interval_s * 1000.0:.3f} "
                        f"step_ms={(step_completed - step_started) * 1000.0:.3f} "
                        f"skipped_slots={skipped_slots}"
                    )
        except Exception as exc:
            self._pump_control_enabled = False
            self._run_token = None
            self._safety.trip(f"control thread failed: {exc}")
            self._stop_event.set()
            self._pump_state.last_error = str(exc)
            self._set_state(SystemState.ERROR, error=f"control loop failed: {exc}")
        finally:
            # A control thread is never allowed to exit while leaving a pump
            # run armed, even when the exception occurred outside PID code.
            if token is not None and self._safety.permits(token):
                self._run_token = None
                self._safety.trip("control thread exited unexpectedly")

    def run_control_step(self) -> None:
        token = getattr(self, "_run_token", None)
        safety = getattr(self, "_safety", None)
        if safety is not None and (token is None or not safety.permits(token)):
            raise RuntimeError("control step rejected by safety supervisor")
        lock = getattr(self, "_lock", None)
        if lock is not None:
            with lock:
                if (
                    self._state not in {SystemState.OPTIMIZING, SystemState.STABILIZING, SystemState.RUNNING}
                    or self._stop_event.is_set()
                    or self._pause_event.is_set()
                ):
                    return
                generation = self._lifecycle_generation
        else:
            # A few pure calculation tests construct a minimal instance with
            # __new__. Production instances always own the lifecycle lock.
            generation = int(getattr(self, "_lifecycle_generation", 0))
        rec = self._read_recognition()
        if token is not None and (
            str(rec.session_id or "") != token.session_id
            or int(rec.run_generation or 0) != token.generation
        ):
            raise RuntimeError(
                "vision result belongs to a stale run "
                f"(session={rec.session_id!r}, generation={rec.run_generation})"
            )
        if int(rec.control_period_id) > 0:
            # Record the period before any early return. Otherwise a stale or
            # invalid snapshot makes the event waiter return immediately on
            # every loop and the same period is retried hundreds of times.
            self._last_seen_vision_period_id = max(
                int(self._last_seen_vision_period_id or 0),
                int(rec.control_period_id),
            )
        now = time.time()
        monotonic_now = time.monotonic()
        if self._last_control_ts is None:
            dt = (self._cfg.control_interval_ms if self._cfg else self.runtime.default_control_interval_ms) / 1000.0
        else:
            dt = max(1e-3, monotonic_now - self._last_control_ts)
        self._last_control_ts = monotonic_now

        if not self._is_realtime_mode():
            ctrl = ControlSnapshot(
                diameter_error=0.0,
                adjustment=0.0,
                q1_command=self._pump_state.q1,
                q2_command=self._pump_state.q2,
                freeze_feedback=True,
                suggested_stop=False,
                reason="local video mode: recognition display only; PID output disabled",
                timestamp=now,
            )
            self._log("[PID][FREEZE] local video mode; pump output disabled")
            self._update_control_snapshot(ctrl)
            return

        if not self._pump_control_enabled:
            ctrl = ControlSnapshot(
                diameter_error=0.0,
                adjustment=0.0,
                q1_command=self._pump_state.q1,
                q2_command=self._pump_state.q2,
                freeze_feedback=True,
                suggested_stop=False,
                reason="pump is not initialized; PID skipped",
                timestamp=now,
            )
            self._log("[PID][FREEZE] pump is not initialized; PID skipped")
            self._update_control_snapshot(ctrl)
            return

        # A default/UI scale is sufficient for preview, but not for an
        # actuator-producing feedback loop.  Only a versioned calibration or
        # a successfully measured channel scale may drive PID output.
        if str(getattr(rec, "scale_source", "configured_unverified")) == "configured_unverified":
            ctrl = ControlSnapshot(
                diameter_error=0.0,
                adjustment=0.0,
                q1_command=self._pump_state.q1,
                q2_command=self._pump_state.q2,
                freeze_feedback=True,
                suggested_stop=False,
                reason="pixel scale is not traceably calibrated; PID output disabled",
                timestamp=now,
            )
            self._log(f"[PID][FREEZE] {ctrl.reason}")
            self._update_control_snapshot(ctrl)
            return

        run_state_res = self.pump_service.read_run_state()
        if (not run_state_res.ok) or (run_state_res.parsed_reply is None):
            reason = run_state_res.error or run_state_res.reason or "pump run state read failed"
            if safety is not None:
                self._run_token = None
                safety.trip(f"pump run state read failed: {reason}")
            self._set_state(SystemState.ERROR, error=f"pump run state read failed: {reason}")
            raise RuntimeError(f"pump run state read failed: {reason}")

        running_ok, running_reason = self.pump_service.are_required_channels_running([1, 2], run_state_res.parsed_reply)
        self._refresh_pump_channels(
            channel_running=list(getattr(run_state_res.parsed_reply, "channel_running", []) or []),
            communication_ok=True,
            error="",
        )
        if not running_ok:
            if safety is not None:
                self._run_token = None
                safety.trip(f"required pump channel stopped: {running_reason}")
            self._set_state(SystemState.ERROR, error=f"required pump channel stopped: {running_reason}")
            raise RuntimeError(f"required pump channel stopped: {running_reason}")
        self._pump_state.running = True
        self._refresh_pump_channels(
            channel_running=list(getattr(run_state_res.parsed_reply, "channel_running", []) or []),
            communication_ok=True,
            error="",
        )

        if not rec.valid_for_control:
            reason = rec.reason or rec.control_reason or "recognition result invalid"
            self._reject_optimization_window_if_due(
                reason, monotonic_now, int(rec.control_period_id or 0)
            )
            ctrl = ControlSnapshot(
                diameter_error=0.0,
                adjustment=0.0,
                q1_command=self._pump_state.q1,
                q2_command=self._pump_state.q2,
                freeze_feedback=True,
                suggested_stop=False,
                reason=f"{reason}; keep current infusion, no new pump command",
                timestamp=now,
            )
            self._log(f"[PID][FREEZE] {reason}")
            self._update_control_snapshot(ctrl)
            return

        if float(rec.capture_monotonic or 0.0) > 0.0:
            recognition_age_ms = max(
                0.0,
                (monotonic_now - float(rec.capture_monotonic)) * 1000.0,
            )
        else:
            recognition_age_ms = max(0.0, (now - float(rec.timestamp or 0.0)) * 1000.0)
        # A completed vision transaction contains a five-frame batch and can
        # legitimately be older than the fixed 1.5 s guard when the configured
        # control period is several seconds. Keep the hard lower bound, but
        # allow one and a half control periods before declaring vision lost.
        control_interval_ms = float(
            self._cfg.control_interval_ms if self._cfg is not None
            else self.runtime.default_control_interval_ms
        )
        recognition_age_limit_ms = max(
            float(self.runtime.max_recognition_age_ms),
            control_interval_ms * 1.5,
        )
        if rec.timestamp <= 0.0 or recognition_age_ms > recognition_age_limit_ms:
            self._reject_optimization_window_if_due(
                "recognition result stale", monotonic_now, int(rec.control_period_id or 0)
            )
            ctrl = ControlSnapshot(
                diameter_error=0.0,
                adjustment=0.0,
                q1_command=self._pump_state.q1,
                q2_command=self._pump_state.q2,
                freeze_feedback=True,
                suggested_stop=False,
                reason=(
                    f"recognition result stale ({recognition_age_ms:.0f} ms > "
                    f"{recognition_age_limit_ms:.0f} ms); keep current infusion"
                ),
                timestamp=now,
            )
            self._log(f"[PID][FREEZE] {ctrl.reason}")
            self._update_control_snapshot(ctrl)
            return

        current_avg_diameter = rec.frame_avg_diameter if rec.frame_avg_diameter is not None else rec.avg_diameter
        if self._cfg is None or current_avg_diameter is None or current_avg_diameter <= 0.0:
            raise RuntimeError("missing parameters required for PID")

        if int(rec.control_period_id) > 0 and self._last_control_period_id == int(rec.control_period_id):
            ctrl = ControlSnapshot(
                diameter_error=0.0,
                adjustment=0.0,
                q1_command=self._pump_state.q1,
                q2_command=self._pump_state.q2,
                freeze_feedback=True,
                suggested_stop=False,
                reason="same completed control period already used for PID feedback",
                timestamp=now,
            )
            self._log(f"[PID][FREEZE] {ctrl.reason}")
            self._update_control_snapshot(ctrl)
            return

        try:
            q1_now, q2_now = self.pump_service.get_current_q_state()
            self._pump_state.q1 = float(q1_now)
            self._pump_state.q2 = float(q2_now)
            self._pump_state.q1_actual = float(q1_now)
            self._pump_state.q2_actual = float(q2_now)
            self._refresh_pump_channels(communication_ok=True, error="")
        except Exception as e:
            # A flow readback failure invalidates the control basis. Invalidate
            # the run before fail-closed stop handling; never command from
            # cached q1/q2 values.
            self._pump_state.comm_established = False
            self._pump_state.q1_actual = None
            self._pump_state.q2_actual = None
            self._pump_state.last_error = str(e)
            self._pump_control_enabled = False
            self._run_token = None
            if safety is not None:
                safety.trip(f"pump flow readback failed: {e}")
            self._stop_event.set()
            self._refresh_pump_channels(communication_ok=False, error=str(e))
            self._log(f"[ORCH][WARN] pump flow readback failed: {e}")

        if not self._pump_state.comm_established:
            reason = f"pump flow readback failed: {self._pump_state.last_error or 'communication invalid'}"
            ctrl = ControlSnapshot(
                diameter_error=0.0,
                adjustment=0.0,
                q1_command=self._pump_state.q1,
                q2_command=self._pump_state.q2,
                freeze_feedback=True,
                suggested_stop=True,
                reason=reason,
                timestamp=now,
            )
            self._update_control_snapshot(ctrl)
            self._set_state(SystemState.ERROR, error=reason)
            return

        with self._lock:
            if self._cfg is None:
                raise RuntimeError("missing parameters required for PID")
            target_diameter_um = float(self._cfg.target_diameter)
            target_revision = int(getattr(self, "_target_revision", 0))
            control_config = copy.copy(self._cfg)
            control_state = self._state

        if control_state == SystemState.OPTIMIZING:
            self._run_optimization_step(
                rec=rec,
                current_diameter_um=float(current_avg_diameter),
                generation=generation,
                token=token,
                now=now,
                monotonic_now=monotonic_now,
            )
            return
        if control_state == SystemState.STABILIZING:
            self._run_stabilizing_step(rec=rec, now=now, monotonic_now=monotonic_now)
            return

        vm = VisionMetrics(
            avg_diameter=float(current_avg_diameter),
            droplet_count=int(rec.frame_droplet_count),
            valid_for_control=bool(rec.valid_for_control),
        )
        tp = TargetParams(target_diameter=target_diameter_um)
        ps = PumpState(q1=float(self._pump_state.q1), q2=float(self._pump_state.q2))

        expected_ms = (
            float(self._cfg.control_interval_ms)
            if self._cfg is not None
            else float(self.runtime.default_control_interval_ms)
        )
        jitter_ms = abs(float(dt) * 1000.0 - expected_ms)
        disturbance_sample = self.disturbance_service.build_and_submit_sample(
            recognition=rec,
            pump_state=self._pump_state,
            control=self._control,
            config=control_config,
            system_state=control_state,
            dt=dt,
            jitter_ms=jitter_ms,
            disturbance=self._disturbance_context,
        )
        prediction = self.disturbance_service.predict(disturbance_sample)
        # A leading disturbance signal is operational context, not a learned
        # feature persisted in the sample database. This prevents historical
        # event markers from being replayed as if they were observable now.
        if prediction is not None:
            signal_declared = bool(self._disturbance_context.get("leading_signal_available", False))
            observed = float(self._disturbance_context.get("leading_signal_observed_monotonic", 0.0) or 0.0)
            elapsed_ms = max(0.0, (time.monotonic() - observed) * 1000.0) if observed > 0.0 else float("inf")
            remaining_lead_ms = max(
                0.0,
                float(self._disturbance_context.get("signal_lead_time_ms", 0.0) or 0.0) - elapsed_ms,
            )
            prediction.leading_signal_available = bool(signal_declared and remaining_lead_ms > 0.0)
            prediction.signal_lead_time_ms = remaining_lead_ms
            prediction.leading_signal_name = str(
                self._disturbance_context.get("leading_signal_name", "") or ""
            )
        self._last_disturbance_prediction = prediction

        pid_input = PIDInput(
            target_diameter_um=target_diameter_um,
            current_diameter_um=float(current_avg_diameter),
            current_q1=float(self._pump_state.q1),
            current_q2=float(self._pump_state.q2),
            dt=float(dt),
            frame_id=int(rec.frame_id),
            vision_valid=bool(rec.valid_for_control),
            pump_communication_ok=bool(self._pump_state.comm_established and not self._pump_state.last_error),
            droplet_count=int(rec.frame_droplet_count),
            disturbance_prediction=prediction,
            system_running=bool(self._state == SystemState.RUNNING),
            measurement_noise_est=float(rec.frame_diameter_cv or 0.0),
            control_jitter_ms=jitter_ms,
            pump_response_delay_ms=float(getattr(disturbance_sample, "pump_response_delay_ms", 0.0) or 0.0),
        )
        cmd = self._pid_controller.update_input(pid_input)
        drift_supervisor = self._get_drift_supervisor()
        previous_drift_recommendation = drift_supervisor.status().reoptimization_recommended
        if not cmd.freeze_feedback and not cmd.suggested_stop:
            drift_status = drift_supervisor.observe(
                target_diameter_um=target_diameter_um,
                diameter_error_um=float(cmd.diameter_error),
                integral_state=float(self._pid_controller.integral),
                integral_limit=float(self.pid_config.integral_limit),
                actuator_saturated=bool(cmd.actuator_saturated),
            )
            if drift_status.reoptimization_recommended and not previous_drift_recommendation:
                self._log(
                    "[CONTROL][DRIFT] local PID authority is persistently exhausted; "
                    "operator BO re-optimization recommended"
                )
        operating_point = self._pid_controller.operating_point
        if int(rec.frame_id) > 0:
            self._last_control_frame_id = int(rec.frame_id)
            self._last_control_period_id = int(rec.control_period_id)
        ctrl = ControlSnapshot(
            diameter_error=float(cmd.diameter_error),
            adjustment=float(cmd.adjustment),
            q1_command=float(cmd.q1),
            q2_command=float(cmd.q2),
            freeze_feedback=bool(cmd.freeze_feedback),
            suggested_stop=bool(cmd.suggested_stop),
            reason=str(cmd.reason or ""),
            timestamp=now,
            p_term=float(cmd.p_term),
            i_term=float(cmd.i_term),
            d_term=float(cmd.d_term),
            pid_output=float(cmd.pid_output),
            feedforward_output=float(cmd.feedforward_output),
            final_output=float(cmd.final_output),
            kp=float(cmd.kp),
            ki=float(cmd.ki),
            kd=float(cmd.kd),
            adaptive_active=bool(cmd.adaptive_active),
            adaptive_enabled=bool(cmd.adaptive_enabled),
            adaptive_reason=str(cmd.adaptive_reason or ""),
            feedforward_active=bool(cmd.feedforward_active),
            feedforward_reason=str(cmd.feedforward_reason or ""),
            control_mode=str(cmd.control_mode),
            q1_output_gain=float(cmd.q1_output_gain),
            q2_output_gain=float(cmd.q2_output_gain),
            frame_id=int(cmd.frame_id),
            control_period_id=int(rec.control_period_id),
            session_id=(token.session_id if token is not None else ""),
            run_generation=(token.generation if token is not None else 0),
            monotonic_timestamp=monotonic_now,
            target_diameter_um=target_diameter_um,
            control_owner=str(cmd.control_owner or "PID"),
            operating_point_q1=(None if operating_point is None else operating_point[0]),
            operating_point_q2=(None if operating_point is None else operating_point[1]),
            actuator_saturated=bool(cmd.actuator_saturated),
            requested_output=float(cmd.requested_output),
            realized_output=float(cmd.realized_output),
        )

        if cmd.freeze_feedback:
            if not ctrl.reason:
                ctrl.reason = "PID frozen; keep current infusion, no new pump command"
            elif "keep current infusion" not in ctrl.reason:
                ctrl.reason = f"{ctrl.reason}; keep current infusion, no new pump command"
            self._log(f"[PID][FREEZE] {ctrl.reason}")
            self._update_control_snapshot(ctrl)
            return

        if cmd.suggested_stop:
            self._pump_control_enabled = False
            self._run_token = None
            self._stop_event.set()
            if safety is not None:
                safety.trip(ctrl.reason or "PID suggested stop")
            stopped = self._safety_stop_pump()
            if stopped and safety is not None:
                safety.confirm_stopped()
            self._update_control_snapshot(ctrl)
            self._set_state(
                SystemState.ERROR,
                error=(ctrl.reason or "PID suggested stop")
                + ("" if stopped else "; pump stop is not verified"),
            )
            return

        try:
            self._require_valid_phase_flows(cmd.q1, cmd.q2)
        except ValueError as exc:
            ctrl.freeze_feedback = True
            ctrl.reason = f"PID command rejected: {exc}; keep current infusion"
            self._log(f"[PID][SAFETY][REJECT] {ctrl.reason}")
            self._update_control_snapshot(ctrl)
            return

        self._log(
            "[PID][UPDATE] "
            f"mode={cmd.control_mode} adaptive_enabled={cmd.adaptive_enabled} "
            f"adaptive_active={cmd.adaptive_active} "
            f"kp={cmd.kp:.6f} ki={cmd.ki:.6f} kd={cmd.kd:.6f} "
            f"q1_gain={cmd.q1_output_gain:.3f} q2_gain={cmd.q2_output_gain:.3f} "
            f"q1={cmd.q1:.6f}, q2={cmd.q2:.6f}, adj={cmd.adjustment:.6f} "
            f"owner={cmd.control_owner} saturated={cmd.actuator_saturated} "
            f"adaptive_reason={cmd.adaptive_reason} "
            f"feedforward_active={cmd.feedforward_active} "
            f"feedforward_output={cmd.feedforward_output:.6f} "
            f"feedforward_reason={cmd.feedforward_reason}"
        )
        # Revalidate immediately before hardware I/O. The safety token binds
        # session_id + run generation; the lifecycle generation closes the
        # pause/stop race in the ordinary orchestration state.
        if safety is not None and (token is None or not safety.permits(token)):
            raise RuntimeError("pump command rejected because run token was invalidated")
        with self._lock:
            if (
                generation != self._lifecycle_generation
                or self._state not in {SystemState.OPTIMIZING, SystemState.STABILIZING, SystemState.RUNNING}
                or self._stop_event.is_set()
                or self._pause_event.is_set()
            ):
                raise RuntimeError("pump command rejected because the run is stopping or paused")
            if target_revision != int(getattr(self, "_target_revision", 0)):
                ctrl.freeze_feedback = True
                ctrl.reason = (
                    "target changed during control calculation; "
                    "discard old command and recompute on the next control period"
                )
                self._log(f"[PID][TARGET][RACE] {ctrl.reason}")
                self._update_control_snapshot(ctrl)
                return
        update_res = self._update_flow_with_lifecycle_guard(cmd.q1, cmd.q2, generation)
        if update_res is None:
            return

        if not update_res.ok:
            self._pump_state.last_update_ok = False
            self._pump_state.last_update_reason = update_res.reason or "flow update failed while running"
            self._pump_state.last_error = self._pump_state.last_update_reason
            self._refresh_pump_channels(communication_ok=False, error=self._pump_state.last_update_reason)
            ctrl.reason = self._pump_state.last_update_reason
            # A two-channel update is a safety transaction. Never restart the
            # pump automatically after any partial or unverified result.
            ctrl.freeze_feedback = True
            self._pump_control_enabled = False
            self._run_token = None
            self._stop_event.set()
            if safety is not None:
                safety.trip(f"pump update transaction failed: {ctrl.reason}")
            stopped = bool(getattr(update_res, "safe_stop_verified", False))
            if not stopped:
                stopped = self._safety_stop_pump()
            if stopped and safety is not None:
                safety.confirm_stopped()
            self._pump_state.running = False if stopped else self._pump_state.running
            failure_reason = ctrl.reason
            if not stopped:
                failure_reason = f"{failure_reason}; pump stop is not verified"
            self._pump_state.last_error = failure_reason
            self._refresh_pump_channels(communication_ok=False, error=failure_reason)
            ctrl.reason = failure_reason
            self._update_control_snapshot(ctrl)
            self._set_state(SystemState.ERROR, error=failure_reason)
            return
        else:
            self._pump_state.last_update_ok = True
            self._pump_state.last_update_reason = "flow update succeeded"
            self._pump_state.last_error = ""
            self._pump_state.q1 = float(cmd.q1)
            self._pump_state.q2 = float(cmd.q2)
            self._pump_state.q1_actual = self._flow_from_channel_params(update_res.verified_q1) or float(cmd.q1)
            self._pump_state.q2_actual = self._flow_from_channel_params(update_res.verified_q2) or float(cmd.q2)
            self._refresh_pump_channels(communication_ok=True, error="")
            self._log(
                "[PUMP][UPDATE][READBACK] "
                f"q1_target={cmd.q1:.6f} q2_target={cmd.q2:.6f} "
                f"q1_actual={self._pump_state.q1_actual:.6f} q2_actual={self._pump_state.q2_actual:.6f}"
            )

        self._update_control_snapshot(ctrl)

    def _reject_optimization_window_if_due(
        self,
        reason: str,
        monotonic_now: float,
        control_period_id: int,
    ) -> None:
        with self._lock:
            optimizer = self._optimizer
            candidate = self._optimization_candidate
            state = self._state
            applied = self._optimization_candidate_applied_monotonic
        if optimizer is None or candidate is None or state != SystemState.OPTIMIZING:
            return
        elapsed_ms = max(0.0, (monotonic_now - applied) * 1000.0)
        if elapsed_ms > float(optimizer.config.candidate_timeout_ms):
            raise RuntimeError(
                f"BO candidate {candidate.candidate_id} timed out after {elapsed_ms:.0f} ms"
            )
        settle_s = float(optimizer.config.settling_time_ms) / 1000.0
        if monotonic_now - applied < settle_s:
            return
        if int(control_period_id or 0) <= int(self._optimization_candidate_period_id or 0):
            return
        self._optimization_candidate_period_id = int(control_period_id)
        self._optimization_window_observations = []
        optimizer.reject_current(reason)
        status = optimizer.status()
        self._log(f"[BO][QUALITY][REJECT] candidate={candidate.candidate_id} reason={reason}")
        if status.failed:
            raise RuntimeError(f"BO failed: {status.reason}")

    def _run_optimization_step(
        self,
        *,
        rec: RecognitionSnapshot,
        current_diameter_um: float,
        generation: int,
        token: RunToken | None,
        now: float,
        monotonic_now: float,
    ) -> None:
        optimizer = self._optimizer
        if optimizer is None:
            raise RuntimeError("OPTIMIZING state has no optimizer")
        candidate = self._optimization_candidate
        if candidate is None:
            self._apply_next_optimization_candidate(optimizer, generation, token, now, monotonic_now, rec)
            return

        elapsed_ms = max(0.0, (monotonic_now - self._optimization_candidate_applied_monotonic) * 1000.0)
        if elapsed_ms > float(optimizer.config.candidate_timeout_ms):
            raise RuntimeError(
                f"BO candidate {candidate.candidate_id} timed out after {elapsed_ms:.0f} ms"
            )
        if elapsed_ms < float(optimizer.config.settling_time_ms):
            self._update_control_snapshot(
                self._optimization_snapshot(
                    now,
                    candidate.q1,
                    candidate.q2,
                    f"BO candidate settling ({elapsed_ms:.0f}/{optimizer.config.settling_time_ms:.0f} ms)",
                )
            )
            return
        if int(rec.control_period_id or 0) <= int(self._optimization_candidate_period_id or 0):
            return

        raw_count = len(list(rec.raw_frame_diameters or []))
        accepted_count = len(list(rec.frame_diameters or []))
        if accepted_count < int(optimizer.config.minimum_valid_droplets):
            self._optimization_candidate_period_id = int(rec.control_period_id or 0)
            self._optimization_window_observations = []
            optimizer.reject_current(
                "valid droplet count below per-period BO quality threshold "
                f"({accepted_count} < {optimizer.config.minimum_valid_droplets})"
            )
            status = optimizer.status()
            if status.failed:
                raise RuntimeError(f"BO measurement-quality failure: {status.reason}")
            self._update_control_snapshot(
                self._optimization_snapshot(
                    now,
                    candidate.q1,
                    candidate.q2,
                    "BO evaluation window reset after insufficient droplets",
                )
            )
            return
        invalid_fraction = (
            max(0.0, min(1.0, float(raw_count - accepted_count) / float(raw_count)))
            if raw_count > 0
            else 0.0
        )
        observation = OptimizationObservation(
            candidate_id=int(candidate.candidate_id),
            q1=float(self._pump_state.q1_actual if self._pump_state.q1_actual is not None else candidate.q1),
            q2=float(self._pump_state.q2_actual if self._pump_state.q2_actual is not None else candidate.q2),
            diameter_um=float(current_diameter_um),
            frequency_hz=float(rec.droplet_generation_rate_hz),
            diameter_cv_percent=(None if rec.frame_diameter_cv is None else float(rec.frame_diameter_cv)),
            valid_droplets=int(len(rec.frame_diameters or [])),
            invalid_fraction=invalid_fraction,
            measurement_valid=bool(rec.valid_for_control),
            invalid_reason=str(rec.reason or rec.control_reason or ""),
        )
        self._optimization_candidate_period_id = int(rec.control_period_id or 0)
        self._optimization_window_observations.append(observation)
        required_periods = int(optimizer.config.evaluation_window_periods)
        if len(self._optimization_window_observations) < required_periods:
            self._update_control_snapshot(
                self._optimization_snapshot(
                    now,
                    candidate.q1,
                    candidate.q2,
                    "BO stable evaluation window "
                    f"({len(self._optimization_window_observations)}/{required_periods} periods)",
                )
            )
            return

        aggregated = self._aggregate_optimization_window(
            candidate.candidate_id,
            self._optimization_window_observations,
        )
        optimizer.tell(aggregated)
        status = optimizer.status()
        self._log(
            f"[BO][OBSERVE] candidate={candidate.candidate_id} q1={candidate.q1:.6f} q2={candidate.q2:.6f} "
            f"diameter={float(aggregated.diameter_um):.6f} periods={aggregated.period_count} "
            f"objective={status.objective_history[-1] if status.objective_history else float('nan'):.6f} "
            f"phase={status.phase}"
        )
        if status.failed:
            if status.best_operating_point is not None and status.failure_kind == "target_not_found":
                best = status.best_operating_point
                if abs(float(best.q1) - float(candidate.q1)) > 1e-9 or abs(float(best.q2) - float(candidate.q2)) > 1e-9:
                    self._apply_optimizer_flow(best.q1, best.q2, generation, token)
                self._optimization_candidate = None
                self._optimization_window_observations = []
                self._optimization_review_required = True
                self._set_state(
                    SystemState.STABILIZING,
                    message="BO ended without confirmed target; holding best safe point for review",
                )
                self._update_control_snapshot(
                    self._optimization_snapshot(
                        now,
                        best.q1,
                        best.q2,
                        f"BO review required ({status.failure_kind}): {status.reason}",
                    )
                )
                return
            raise RuntimeError(f"BO failed: {status.reason}")
        if status.completed:
            best = status.confirmed_operating_point
            if best is None:
                raise RuntimeError("BO completed without an operating point")
            if abs(float(best.q1) - float(candidate.q1)) > 1e-9 or abs(float(best.q2) - float(candidate.q2)) > 1e-9:
                self._apply_optimizer_flow(best.q1, best.q2, generation, token)
            self._optimization_candidate = None
            self._optimization_window_observations = []
            self._stabilizing_until_monotonic = monotonic_now + float(optimizer.config.settling_time_ms) / 1000.0
            self._set_state(SystemState.STABILIZING, message="BO completed; holding optimum before PID")
            self._update_control_snapshot(
                self._optimization_snapshot(now, best.q1, best.q2, "BO completed; stabilizing at optimum")
            )
            return
        self._optimization_candidate = None
        self._optimization_window_observations = []
        self._apply_next_optimization_candidate(optimizer, generation, token, now, monotonic_now, rec)

    @staticmethod
    def _aggregate_optimization_window(
        candidate_id: int,
        observations: list[OptimizationObservation],
    ) -> OptimizationObservation:
        if not observations:
            raise ValueError("cannot aggregate an empty BO evaluation window")

        def _median_optional(values: list[float | None]) -> float | None:
            finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
            return None if not finite else float(median(finite))

        return OptimizationObservation(
            candidate_id=int(candidate_id),
            q1=float(median(float(item.q1) for item in observations)),
            q2=float(median(float(item.q2) for item in observations)),
            diameter_um=_median_optional([item.diameter_um for item in observations]),
            frequency_hz=_median_optional([item.frequency_hz for item in observations]),
            diameter_cv_percent=_median_optional(
                [item.diameter_cv_percent for item in observations]
            ),
            valid_droplets=sum(max(0, int(item.valid_droplets)) for item in observations),
            invalid_fraction=sum(float(item.invalid_fraction) for item in observations) / len(observations),
            measurement_valid=all(bool(item.measurement_valid) for item in observations),
            invalid_reason="; ".join(
                dict.fromkeys(item.invalid_reason for item in observations if item.invalid_reason)
            ),
            period_count=len(observations),
        )

    def _apply_next_optimization_candidate(
        self,
        optimizer: SafeBayesianOptimizer,
        generation: int,
        token: RunToken | None,
        now: float,
        monotonic_now: float,
        rec: RecognitionSnapshot,
    ) -> None:
        candidate = optimizer.ask()
        self._require_valid_phase_flows(candidate.q1, candidate.q2)
        self._apply_optimizer_flow(candidate.q1, candidate.q2, generation, token)
        self._optimization_candidate = candidate
        self._optimization_candidate_applied_monotonic = monotonic_now
        self._optimization_candidate_period_id = int(rec.control_period_id or 0)
        self._optimization_window_observations = []
        self._log(
            f"[BO][APPLY] candidate={candidate.candidate_id} q1={candidate.q1:.6f} "
            f"q2={candidate.q2:.6f} reason={candidate.reason}"
        )
        self._update_control_snapshot(
            self._optimization_snapshot(now, candidate.q1, candidate.q2, candidate.reason)
        )

    def _apply_optimizer_flow(
        self,
        q1: float,
        q2: float,
        generation: int,
        token: RunToken | None,
    ) -> None:
        if token is None or not self._safety.permits(token):
            raise RuntimeError("BO pump command rejected because run token is invalid")
        result = self._update_flow_with_lifecycle_guard(float(q1), float(q2), generation)
        if result is None:
            raise RuntimeError("BO pump command superseded by lifecycle transition")
        if not result.ok:
            raise RuntimeError(f"BO pump update failed: {result.reason or result.error}")
        q1_actual = self._flow_from_channel_params(result.verified_q1) or float(q1)
        q2_actual = self._flow_from_channel_params(result.verified_q2) or float(q2)
        ok1, reason1 = self._flow_matches("Q1", q1, q1_actual)
        ok2, reason2 = self._flow_matches("Q2", q2, q2_actual)
        if not (ok1 and ok2):
            raise RuntimeError(f"BO readback mismatch: {reason1 or reason2}")
        self._pump_state.q1 = float(q1)
        self._pump_state.q2 = float(q2)
        self._pump_state.q1_actual = float(q1_actual)
        self._pump_state.q2_actual = float(q2_actual)
        self._pump_state.last_update_ok = True
        self._pump_state.last_update_reason = "BO flow update succeeded"
        self._pump_state.last_error = ""
        self._refresh_pump_channels(communication_ok=True, error="")

    def _run_stabilizing_step(
        self,
        *,
        rec: RecognitionSnapshot,
        now: float,
        monotonic_now: float,
    ) -> None:
        optimizer = self._optimizer
        if optimizer is None:
            raise RuntimeError("STABILIZING state has no optimizer")
        status = optimizer.status()
        best = status.confirmed_operating_point or status.best_operating_point
        if best is None:
            raise RuntimeError("STABILIZING state has no BO operating point")
        if bool(getattr(self, "_optimization_review_required", False)):
            self._update_control_snapshot(
                self._optimization_snapshot(
                    now,
                    best.q1,
                    best.q2,
                    "holding best safe BO point; operator review required before PID handoff",
                )
            )
            return
        if monotonic_now < self._stabilizing_until_monotonic:
            remaining_ms = (self._stabilizing_until_monotonic - monotonic_now) * 1000.0
            self._update_control_snapshot(
                self._optimization_snapshot(now, best.q1, best.q2, f"holding optimum ({remaining_ms:.0f} ms remaining)")
            )
            return
        if not rec.valid_for_control or int(len(rec.frame_diameters or [])) < int(optimizer.config.minimum_valid_droplets):
            self._update_control_snapshot(
                self._optimization_snapshot(now, best.q1, best.q2, "waiting for a valid settled confirmation window")
            )
            return
        q1_actual = float(self._pump_state.q1_actual if self._pump_state.q1_actual is not None else best.q1)
        q2_actual = float(self._pump_state.q2_actual if self._pump_state.q2_actual is not None else best.q2)
        self._pid_controller.set_operating_point(q1_actual, q2_actual)
        self._get_drift_supervisor().reset()
        self._last_control_frame_id = int(rec.frame_id or 0)
        self._last_control_period_id = int(rec.control_period_id or 0)
        self._last_control_ts = monotonic_now
        self._set_state(SystemState.RUNNING, message="BO optimum accepted; PID control enabled")
        self._update_control_snapshot(
            self._optimization_snapshot(now, q1_actual, q2_actual, "bumpless transfer complete; PID enabled")
        )
        self._log(
            f"[BO][TRANSFER] q1_bias={q1_actual:.6f} q2_bias={q2_actual:.6f}; "
            "PID reset and feedforward remains subject to deployment gates"
        )

    def accept_unconfirmed_optimization_best(self) -> None:
        """Explicitly authorize PID handoff from a safe but unconfirmed BO point."""
        with self._lock:
            if self._state != SystemState.STABILIZING or not self._optimization_review_required:
                raise RuntimeError("no unconfirmed BO operating point is awaiting review")
            optimizer = self._optimizer
            if optimizer is None or optimizer.status().best_operating_point is None:
                raise RuntimeError("BO review has no safe operating point")
            self._optimization_review_required = False
            self._stabilizing_until_monotonic = (
                time.monotonic() + float(optimizer.config.settling_time_ms) / 1000.0
            )
            self._message = "operator accepted unconfirmed BO point; stabilizing before PID"

    def _optimization_snapshot(self, now: float, q1: float, q2: float, reason: str) -> ControlSnapshot:
        return ControlSnapshot(
            diameter_error=0.0,
            adjustment=0.0,
            q1_command=float(q1),
            q2_command=float(q2),
            freeze_feedback=True,
            suggested_stop=False,
            reason=str(reason),
            timestamp=float(now),
            control_mode="BO_OPERATING_POINT",
            control_owner="BO" if self._state == SystemState.OPTIMIZING else "HOLD",
            operating_point_q1=float(q1),
            operating_point_q2=float(q2),
            target_diameter_um=(None if self._cfg is None else float(self._cfg.target_diameter)),
        )


_RUNTIME_MESSAGE_LABELS = {
    "": "",
    "configured": "参数已配置",
    "video ready": "视频已就绪",
    "initializing": "正在初始化",
    "initialized": "初始化完成",
    "running": "系统运行中",
    "bo optimization running": "BO 正在寻找安全工作点",
    "bo completed; holding optimum before pid": "BO 完成，正在稳定保持后切换 PID",
    "bo optimum accepted; pid control enabled": "最优工作点已确认，PID 已接管",
    "paused": "系统已暂停",
    "stopping": "正在停止",
    "stopped": "系统已停止",
    "local video mode: skip pump initialization and PID output": "本地视频模式：跳过泵初始化和 PID 输出",
}

_MOJIBAKE_MARKERS = set("闂閻濞缂婵濠鐎柛梺妞鈧瑜閸閹幋娴瀹绾椤")


def _clean_runtime_text(value: object, kind: str) -> str:
    text = str(value or "").strip()
    mapped = _RUNTIME_MESSAGE_LABELS.get(text.lower())
    if mapped is not None:
        return mapped
    if _looks_like_mojibake(text):
        return "发生错误，请查看运行日志" if kind == "error" else "状态信息异常，请查看运行日志"
    if len(text) > 500:
        return text[:480] + "..."
    return text


def _looks_like_mojibake(text: str) -> bool:
    if not text:
        return False
    marker_count = sum(1 for ch in text if ch in _MOJIBAKE_MARKERS)
    if marker_count >= 4:
        return True
    return marker_count > 0 and marker_count / max(1, len(text)) > 0.08

