from __future__ import annotations

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
    recognition_roi: dict[str, float | bool] = field(default_factory=dict)


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
    feedforward_active: bool = False
    control_mode: str = "CLASSIC_PID"
    frame_id: int = 0


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
