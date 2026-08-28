from dataclasses import dataclass, field
from typing import Any, Literal, Tuple

import os

TrackerType = Literal["nearest", "kalman"]
BeadMode = Literal["intensity", "connected"]


@dataclass
class ROIConfig:
    # Explicit/manual rectangles take priority over automatic channel-region
    # calibration. Auto-suggested preview rectangles set this to False.
    user_defined: bool = True
    # The configured HIKROBOT view places the useful channel in the lower
    # middle of the 720x540 frame. Keeping the analysis away from the channel
    # walls removes the strongest non-droplet edges before detection.
    # Detect first on the complete camera frame. A fixed crop can silently
    # exclude real droplets when the channel position changes.
    enabled: bool = False
    x_start_ratio: float = 0.0
    x_end_ratio: float = 1.0
    y_start_ratio: float = 0.48
    y_end_ratio: float = 0.84
    crop_top_ratio: float = 0.0
    # Two normalized Hough segments selected by the user. When present they
    # define a tilted quadrilateral that is rectified before all recognition.
    wall_lines: list[dict[str, Any]] = field(default_factory=list)

    def resolve(self, width: int, height: int) -> Tuple[int, int, int, int, int]:
        x0 = max(0, min(width - 1, int(width * self.x_start_ratio)))
        x1 = max(x0 + 1, min(width, int(width * self.x_end_ratio)))
        y0 = max(0, min(height - 1, int(height * self.y_start_ratio)))
        y1 = max(y0 + 1, min(height, int(height * self.y_end_ratio)))
        crop_top = max(0, min(y1 - y0 - 1, int((y1 - y0) * self.crop_top_ratio)))
        return x0, x1, y0, y1, crop_top


@dataclass
class ChannelRegionConfig:
    # Startup-only automatic channel-region calibration. Manual ROI/wall
    # selections always take priority; low-confidence results fall back to the
    # complete frame instead of blocking recognition.
    enabled: bool = True
    sample_frames: int = 12
    min_confidence: float = 0.58
    work_max_width: int = 960
    work_max_height: int = 720
    # Local RMS window used to turn thin spatial gradients into a continuous
    # high-frequency region before its high/low boundary is fitted.
    frequency_window_ratio: float = 0.025
    min_frequency_region_thickness_ratio: float = 0.055
    min_frequency_frame_support: float = 0.60
    min_region_contrast: float = 0.10
    full_region_contrast: float = 0.35
    min_region_coverage: float = 0.55
    min_coverage_advantage: float = 0.25
    canny_low: int = 35
    canny_high: int = 110
    hough_threshold: int = 42
    min_line_length_ratio: float = 0.45
    max_line_gap_ratio: float = 0.08
    max_lines: int = 40
    parallel_tolerance_degrees: float = 8.0
    min_width_ratio: float = 0.08
    max_width_ratio: float = 0.90
    max_separation_variation_ratio: float = 0.12
    high_frequency_weight: float = 0.65
    straightness_weight: float = 0.25
    geometry_weight: float = 0.10


@dataclass
class DetectorConfig:
    """Configuration for the EdgeDrawing-only droplet detector."""

    # Circle-radius bounds in original-image pixels.
    min_radius: float = 8.0
    max_radius: float = 80.0
    expected_radius: float = 0.0
    enable_expected_size_filter: bool = True
    expected_size_hard_gate: bool = False
    expected_radius_tolerance_ratio: float = 0.30

    # Generic image preparation.
    enable_intensity_normalization: bool = True
    enable_gaussian_blur: bool = True
    gaussian_blur_size: int = 5
    enable_edge_clahe: bool = True
    enable_edge_median_blur: bool = True
    edge_work_max_width: int = 760
    edge_work_max_height: int = 560

    # cv2.ximgproc.EdgeDrawing parameters.
    edge_operator: int = 0
    edge_gradient_threshold: int = 20
    edge_anchor_threshold: int = 0
    edge_scan_interval: int = 1
    edge_min_path_length: int = 10
    edge_min_line_length: int = -1
    edge_sigma: float = 1.0
    edge_line_fit_error: float = 1.0
    edge_max_line_gap: float = 6.0
    edge_max_error: float = 1.3
    edge_nfa_validation: bool = True
    edge_pf_mode: bool = False

    # Circle acceptance and output limits. Near-circular records may differ by 10%.
    edge_min_circle_ratio: float = 0.90
    edge_min_support_ratio: float = 0.15
    edge_support_neighborhood: int = 2
    edge_min_visible_ratio: float = 0.50
    edge_max_candidates: int = 120
    min_center_distance: float = 0.0
    deduplicate_distance_ratio: float = 0.60
    deduplicate_min_distance: float = 6.0
    deduplicate_overlap_ratio: float = 0.42
    diameter_min_visible_ratio: float = 0.85
    cut_line_ratio: float = 1.0


@dataclass
class KalmanConfig:
    process_noise: float = 8.0
    measurement_noise: float = 12.0
    initial_covariance: float = 25.0
    dt: float = 1.0


@dataclass
class TrackerConfig:
    tracker_type: TrackerType = "kalman"
    match_distance: float = 120.0
    min_match_distance: float = 18.0
    match_distance_radius_ratio: float = 2.0
    max_unmatched_frames: int = 5
    radius_match_ratio: float = 0.65
    radius_smoothing_alpha: float = 0.25
    confirmation_window: int = 3
    confirmation_min_hits: int = 2
    kalman: KalmanConfig = field(default_factory=KalmanConfig)


@dataclass
class BeadConfig:
    mode: BeadMode = "intensity"
    area_min: int = 5
    area_max: int = 80
    inner_radius_ratio: float = 0.82
    border_margin: int = 2
    default_droplet_radius: float = 24.0
    dark_percentile: float = 18.0
    blur_kernel: int = 5


@dataclass
class MetricsConfig:
    min_active_for_control: int = 1
    # Samples are locked once per track at the count-line crossing, so this
    # threshold counts distinct passed droplets rather than repeated frames.
    min_samples_for_control: int = 1
    realtime_window_ms: int = 500
    rolling_window: int = 120
    flow_axis: Literal["x", "y"] = "x"
    flow_direction: Literal["positive", "negative", "any"] = "positive"
    count_line_ratio: float = 0.6
    min_track_age_for_count: int = 1
    min_track_displacement_for_count: float = 8.0
    # Keep a short per-track radius history and lock its median when the
    # droplet crosses the counting line. This suppresses single-frame edge
    # jitter without adding another image-processing pass.
    diameter_samples_per_track: int = 15
    uniformity_good_threshold: float = 5.0
    uniformity_normal_threshold: float = 10.0
    # Retained for settings-file compatibility. CV is diagnostic only and no
    # longer gates PID feedback.
    max_diameter_cv_for_control: float = 25.0
    robust_mad_multiplier: float = 3.5
    no_droplet_log_interval_s: float = 2.0


@dataclass
class DebugConfig:
    enabled: bool = False
    verbose: bool = False
    # 仅影响前端显示，不影响识别/跟踪/计数/PID 数据链路。
    draw_overlay: bool = False


@dataclass
class PipelineConfig:
    roi: ROIConfig = field(default_factory=ROIConfig)
    channel_region: ChannelRegionConfig = field(default_factory=ChannelRegionConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    beads: BeadConfig = field(default_factory=BeadConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)


@dataclass
class HikrobotCameraConfig:
    mvs_sdk_path: str = os.environ.get("MVS_SDK_PATH", "")
    exposure_time: float | None = None
    gain: float | None = None
    frame_rate: float | None = None
    width: int | None = None
    height: int | None = None
    offset_x: int | None = None
    offset_y: int | None = None
    pixel_format: str | None = None
    trigger_mode: str = "Off"
    acquisition_mode: str = "Continuous"
    frame_failure_threshold: int = 10
    test_frame_count: int = 3


@dataclass
class CameraDiscoveryConfig:
    mvs_sdk_path: str = os.environ.get("MVS_SDK_PATH", "")
    opencv_scan_indices: tuple[int, ...] = tuple(range(4))
    opencv_probe_timeout_ms: int = 700


@dataclass
class CameraSystemConfig:
    sdk_paths: tuple[str, ...] = tuple(
        p for p in os.environ.get("CAMERA_SDK_PATHS", "").split(os.pathsep) if p
    )
    enabled_camera_backends: tuple[str, ...] = (
        "hikrobot",
        "gentl",
        "basler",
        "daheng",
        "flir",
        "allied_vision",
        "opencv",
    )
    preferred_backend_order: tuple[str, ...] = (
        "hikrobot",
        "gentl",
        "basler",
        "daheng",
        "flir",
        "allied_vision",
        "opencv",
    )
    gentl_producer_paths: tuple[str, ...] = tuple(
        p for p in os.environ.get("GENICAM_GENTL64_PATH", "").split(os.pathsep) if p
    )
    gentl_xml_cache_dir: str = os.environ.get("HARVESTERS_XML_FILE_DIR", "")
    opencv_scan_indices: tuple[int, ...] = tuple(range(4))
    opencv_backend_order: tuple[str, ...] = ("dshow", "msmf", "default")
    frame_timeout_ms: int = 1000
    frame_failure_threshold: int = 10
    reconnect_attempts: int = 2
    reconnect_interval_s: float = 1.0
    test_frame_count: int = 3
    default_camera_parameters: dict[str, float | int | str] = field(default_factory=dict)
    hikrobot_mvs_sdk_path: str = os.environ.get("MVS_SDK_PATH", "")
    mvs_sdk_path: str = os.environ.get("MVS_SDK_PATH", "")


def default_config() -> PipelineConfig:
    return PipelineConfig()
