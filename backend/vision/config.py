from dataclasses import dataclass, field
from typing import Literal, Tuple

import os

TrackerType = Literal["nearest", "kalman"]
DetectionMode = Literal["split_connected", "no_split"]
BeadMode = Literal["intensity", "connected"]


@dataclass
class ROIConfig:
    enabled: bool = False
    x_start_ratio: float = 0.0
    x_end_ratio: float = 1.0
    y_start_ratio: float = 0.0
    y_end_ratio: float = 1.0
    crop_top_ratio: float = 0.0

    def resolve(self, width: int, height: int) -> Tuple[int, int, int, int, int]:
        x0 = max(0, min(width - 1, int(width * self.x_start_ratio)))
        x1 = max(x0 + 1, min(width, int(width * self.x_end_ratio)))
        y0 = max(0, min(height - 1, int(height * self.y_start_ratio)))
        y1 = max(y0 + 1, min(height, int(height * self.y_end_ratio)))
        crop_top = max(0, min(y1 - y0 - 1, int((y1 - y0) * self.crop_top_ratio)))
        return x0, x1, y0, y1, crop_top


@dataclass
class DetectorConfig:
    min_radius: float = 22.0
    max_radius: float = 50.0
    min_center_distance: float = 30.0
    deduplicate_distance_ratio: float = 0.90
    deduplicate_min_distance: float = 6.0
    deduplicate_contained_ratio: float = 0.88
    circularity_threshold: float = 0.15
    min_contour_area: float = 60.0
    gaussian_blur_size: int = 5
    morphology_open_kernel: int = 3
    morphology_close_kernel: int = 5
    split_peak_threshold_ratio: float = 0.40
    split_large_area_ratio: float = 1.15
    enable_hough_candidates: bool = True
    hough_dp: float = 1.2
    hough_min_distance: float = 30.0
    hough_param1: float = 100.0
    hough_param2: float = 28.0
    hough_min_radius: float = 22.0
    hough_max_radius: float = 50.0
    hough_preferred_radius: float = 30.0
    hough_edge_support_threshold: float = 0.18
    hough_edge_neighborhood: int = 2
    hough_work_max_width: int = 760
    hough_work_max_height: int = 560
    hough_max_candidates: int = 120
    reject_multi_droplet_circles: bool = True
    multi_droplet_child_radius_ratio: float = 0.82
    multi_droplet_child_distance_ratio: float = 0.90
    multi_droplet_child_count: int = 2
    cut_line_ratio: float = 1.0
    detection_mode: DetectionMode = "split_connected"


@dataclass
class KalmanConfig:
    process_noise: float = 8.0
    measurement_noise: float = 12.0
    initial_covariance: float = 25.0
    dt: float = 1.0


@dataclass
class TrackerConfig:
    tracker_type: TrackerType = "nearest"
    match_distance: float = 90.0
    max_unmatched_frames: int = 8
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
    min_samples_for_control: int = 1
    realtime_window_ms: int = 500
    rolling_window: int = 120
    count_line_ratio: float = 0.6
    min_track_age_for_count: int = 3
    min_track_displacement_for_count: float = 8.0
    uniformity_good_threshold: float = 5.0
    uniformity_normal_threshold: float = 10.0


@dataclass
class DebugConfig:
    enabled: bool = False
    verbose: bool = False
    draw_helper_mask: bool = True
    # 仅影响前端显示，不影响识别/跟踪/计数/PID 数据链路。
    draw_overlay: bool = True


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
    )
    preferred_backend_order: tuple[str, ...] = (
        "hikrobot",
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
