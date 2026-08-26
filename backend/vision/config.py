from dataclasses import dataclass, field
from typing import Any, Literal, Tuple

import os

TrackerType = Literal["nearest", "kalman"]
DetectionMode = Literal["split_connected", "no_split"]
ThresholdMode = Literal["adaptive_gaussian", "otsu"]
BeadMode = Literal["intensity", "connected"]


@dataclass
class ROIConfig:
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
class DetectorConfig:
    # Absolute safety range. The runtime range is derived from the configured
    # target diameter and pixel calibration, so the detector is not tied to a
    # specific droplet size.
    min_radius: float = 8.0
    max_radius: float = 80.0
    expected_radius: float = 0.0
    # The accepted range follows the user-configured target and calibration.
    # It remains broad (0.65x-1.4x radius by default), so this is
    # not tied to 50 um while still rejecting unrelated channel structures.
    # Do not discard otherwise valid droplets merely because their size differs
    # from the target; the target is a control reference, not a detection gate.
    expected_size_hard_gate: bool = False
    adaptive_min_radius_ratio: float = 0.50
    adaptive_max_radius_ratio: float = 1.60
    adaptive_radius_learning_rate: float = 0.12
    adaptive_radius_min_candidates: int = 4
    min_center_distance: float = 0.0
    min_center_distance_radius_ratio: float = 0.75
    deduplicate_distance_ratio: float = 0.60
    deduplicate_min_distance: float = 6.0
    deduplicate_contained_ratio: float = 0.88
    # Deprecated JSON compatibility fields from the removed contour detector.
    circularity_threshold: float = 0.12
    min_contour_area: float = 40.0
    enable_intensity_normalization: bool = True
    enable_gaussian_blur: bool = True
    gaussian_blur_size: int = 5
    # Deprecated JSON compatibility fields from the removed binary,
    # morphology, and connected-component pipeline. They are not executed.
    threshold_mode: ThresholdMode = "adaptive_gaussian"
    adaptive_threshold_block_size: int = 31
    adaptive_threshold_c: float = 5.0
    enable_morphology: bool = True
    morphology_open_kernel: int = 3
    morphology_close_kernel: int = 5
    split_peak_threshold_ratio: float = 0.55
    split_min_radius_ratio: float = 0.65
    split_large_area_ratio: float = 1.15
    # Retained for saved-profile compatibility. The current detector always
    # uses Hough as its sole candidate source; it cannot be disabled or changed
    # into a contour fallback by these legacy flags.
    enable_hough_candidates: bool = True
    hough_fallback_only: bool = False
    enable_hough_clahe: bool = True
    enable_hough_median_blur: bool = True
    hough_dp: float = 1.2
    hough_min_distance: float = 0.0
    hough_param1: float = 100.0
    hough_param2: float = 28.0
    hough_min_radius: float = 8.0
    hough_max_radius: float = 80.0
    hough_preferred_radius: float = 0.0
    hough_edge_support_threshold: float = 0.15
    hough_edge_neighborhood: int = 2
    # Dense-droplet safeguards. A reliable target radius rejects implausible
    # Hough sizes; edge ownership rejects circles assembled from neighbouring
    # droplets' arcs.
    enable_expected_size_filter: bool = True
    expected_radius_tolerance_ratio: float = 0.30
    enable_edge_ownership_filter: bool = True
    edge_ownership_search_radius: int = 2
    edge_ownership_margin: float = 0.75
    edge_ownership_min_ratio: float = 0.55
    hough_work_max_width: int = 760
    hough_work_max_height: int = 560
    hough_max_candidates: int = 120
    # Deprecated JSON compatibility fields from the removed intensity-peak
    # candidate path. Hough is the only liquid-droplet candidate source.
    enable_intensity_peak_candidates: bool = False
    enable_intensity_peak_fallback: bool = True
    intensity_peak_kernel_radius_ratio: float = 1.25
    intensity_peak_percentile: float = 55.0
    intensity_peak_min_radius_ratio: float = 0.80
    intensity_peak_max_radius_ratio: float = 1.20
    intensity_peak_min_edge_support: float = 0.18
    intensity_peak_max_candidates: int = 120
    reject_multi_droplet_circles: bool = True
    multi_droplet_child_radius_ratio: float = 0.82
    multi_droplet_child_distance_ratio: float = 0.90
    multi_droplet_child_count: int = 2
    candidate_min_edge_support: float = 0.15
    candidate_min_ring_contrast: float = -1.0
    # In the current bright-field setup, a real droplet has a brighter inner
    # region and a darker circular boundary. Requiring this contrast rejects
    # dark junctions between neighbouring droplets that Hough may fit as
    # small circles.
    candidate_min_center_contrast: float = -1.0
    # Only use circles whose complete geometry is inside the analysis ROI;
    # partial circles at the crop/channel boundary bias diameter and create
    # false positives.
    candidate_full_circle_ratio: float = 0.85
    candidate_nms_overlap_ratio: float = 0.42
    cut_line_ratio: float = 1.0
    # Deprecated JSON compatibility field from the removed contour detector.
    detection_mode: DetectionMode = "no_split"


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
