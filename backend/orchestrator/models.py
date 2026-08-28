from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from ..pump_hardware.models import PumpChannelState
from .state import SystemState


@dataclass(slots=True)
class SystemConfig:
    target_diameter: float
    pixel_to_micron: float
    video_source_type: str
    video_source: str
    initial_q1: float
    initial_q2: float
    control_interval_ms: int
    pump_port: str = ""
    pump_address: int = 1
    pump_baudrate: int = 1200
    pump_parity: str = "N"
    mvs_sdk_path: str = ""
    camera_backend: str = ""
    camera_parameters: dict[str, float | int | str] = field(default_factory=dict)
    recognition_roi: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    detector_algorithm: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        finite_positive = {
            "target_diameter": self.target_diameter,
            "pixel_to_micron": self.pixel_to_micron,
            "initial_q1": self.initial_q1,
            "initial_q2": self.initial_q2,
        }
        for name, value in finite_positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if float(self.initial_q1) > 5000.0 or float(self.initial_q2) > 5000.0:
            raise ValueError("initial pump flows must not exceed 5000 uL/min")
        if not isinstance(self.control_interval_ms, int) or isinstance(self.control_interval_ms, bool):
            raise ValueError("control_interval_ms must be an integer number of milliseconds")
        if not 1 <= int(self.pump_address) <= 0x1F:
            raise ValueError("pump_address must be in [1, 31]")
        if int(self.pump_baudrate) <= 0:
            raise ValueError("pump_baudrate must be positive")
        parity = str(self.pump_parity or "").upper()
        if parity not in {"N", "E"}:
            raise ValueError("pump_parity must be 'N' or 'E'")
        self.pump_parity = parity
        if self.calibration:
            from ..vision.calibration import CalibrationRecord

            record = CalibrationRecord.from_mapping(dict(self.calibration))
            tolerance = max(1e-9, record.uncertainty_um_per_px * 3.0)
            if abs(float(self.pixel_to_micron) - record.pixel_to_micron) > tolerance:
                raise ValueError(
                    "pixel_to_micron disagrees with the versioned calibration record"
                )


@dataclass(slots=True)
class RecognitionSnapshot:
    frame_droplet_count: int
    total_droplet_count: int
    new_crossing_count: int
    avg_diameter: float | None
    single_cell_rate: float
    valid_for_control: bool
    timestamp: float
    reason: str
    # backward-compatible mirrors
    droplet_count: int
    active_droplet_count: int
    has_droplet: bool
    control_reason: str
    frame_png_base64: Optional[str] = None
    frame_width: int = 0
    frame_height: int = 0
    video_source_type: str = ""
    video_source: str = ""
    # frame_id/timestamp identify the frame used to produce recognition data.
    # Preview frames are published independently and must never make stale
    # recognition data look newer to the PID loop.
    frame_id: int = 0
    preview_frame_id: int = 0
    preview_timestamp: float = 0.0
    frame_single_cell_count: int = 0
    frame_diameters: list[float] = field(default_factory=list)
    frame_diameter_sum: float = 0.0
    frame_avg_diameter: float | None = None
    frame_single_cell_rate: float | None = None
    frame_diameter_std: float | None = None
    frame_diameter_cv: float | None = None
    uniformity_valid: bool = False
    uniformity_status: str = "样本不足"
    uniformity_reason: str = ""
    capture_fps: float = 0.0
    processing_fps: float = 0.0
    recognition_latency_ms: float = 0.0
    algorithm_processing_ms: float = 0.0
    replaced_processing_frames: int = 0
    pending_processing_frames: int = 0
    period_replaced_processing_frames: int = 0
    processed_frame_count: int = 0
    period_processed_frames: int = 0
    vision_performance_status: str = "等待视觉数据"
    control_period_id: int = 0
    motion_window_frames: int = 0
    average_droplet_speed_um_s: float | None = None
    speed_sample_count: int = 0
    droplet_generation_rate_hz: float = 0.0
    pixel_to_micron: float = 0.0
    scale_source: str = "configured"
    channel_width_um: float | None = None
    channel_width_px: float | None = None
    channel_calibration_status: str = "disabled"
    channel_calibration_confidence: float = 0.0
    channel_calibration_reason: str = ""
    session_id: str = ""
    run_generation: int = 0
    capture_monotonic: float = 0.0
    hardware_frame_id: int = 0
    hardware_timestamp: float = 0.0
    raw_frame_diameters: list[float] = field(default_factory=list)
    raw_frame_diameter_cv: float | None = None
    filtering_rule: str = "none"
    calibration_id: str = ""
    calibration_uncertainty_um_per_px: float | None = None


@dataclass(slots=True)
class FrameSnapshot:
    frame_id: int
    timestamp: float
    width: int
    height: int
    valid: bool
    frame_png_base64: Optional[str] = None
    frame_pgm: Optional[bytes] = None
    # Lightweight preview transport used by the independent video process.
    # Keeping this as bytes avoids the extra 33% Base64 expansion.
    frame_jpeg: Optional[bytes] = None
    reason: str = ""
    session_id: str = ""
    run_generation: int = 0
    capture_monotonic: float = 0.0
    hardware_frame_id: int = 0
    hardware_timestamp: float = 0.0


@dataclass(slots=True)
class PumpRuntimeState:
    connected: bool
    comm_established: bool
    fully_ready: bool
    q1: float
    q2: float
    running: bool
    last_error: str
    q1_actual: float | None = None
    q2_actual: float | None = None
    last_update_ok: bool = False
    last_update_reason: str = ""
    channels: dict[str, PumpChannelState] = field(default_factory=dict)
    last_readback_time: float | None = None
    readback_kind: str = "device_parameter_estimate"
    physical_flow_measured: bool = False
    pump_response_delay_ms: float | None = None
    pump_response_measurement_status: str = "unmeasured"


@dataclass(slots=True)
class ControlSnapshot:
    diameter_error: float
    adjustment: float
    q1_command: float
    q2_command: float
    freeze_feedback: bool
    suggested_stop: bool
    reason: str
    timestamp: float
    p_term: float = 0.0
    i_term: float = 0.0
    d_term: float = 0.0
    pid_output: float = 0.0
    feedforward_output: float = 0.0
    final_output: float = 0.0
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    adaptive_active: bool = False
    adaptive_enabled: bool = False
    adaptive_reason: str = ""
    feedforward_active: bool = False
    feedforward_reason: str = ""
    control_mode: str = "CLASSIC_PID"
    q1_output_gain: float = 1.0
    q2_output_gain: float = 1.0
    frame_id: int = 0
    control_period_id: int = 0
    session_id: str = ""
    run_generation: int = 0
    monotonic_timestamp: float = 0.0
    target_diameter_um: float | None = None
    control_owner: str = "PID"
    operating_point_q1: float | None = None
    operating_point_q2: float | None = None
    actuator_saturated: bool = False
    requested_output: float = 0.0
    realized_output: float = 0.0


@dataclass(slots=True)
class SystemSnapshot:
    system_state: SystemState
    config: Optional[SystemConfig]
    recognition: Optional[RecognitionSnapshot]
    pump_state: Optional[PumpRuntimeState]
    control: Optional[ControlSnapshot]
    message: str
    error: str
    frame: Optional[FrameSnapshot] = None
    timestamp: float = 0.0
    disturbance_model: Optional[dict[str, Any]] = None
    disturbance_prediction: Optional[dict[str, Any]] = None
    safety: Optional[dict[str, Any]] = None
    optimization: Optional[dict[str, Any]] = None
