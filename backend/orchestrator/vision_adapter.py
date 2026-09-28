from __future__ import annotations

import base64
from collections import deque
import queue
from statistics import median
import threading
import time
from dataclasses import replace
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - handled at runtime for local video only
    cv2 = None

from .models import FrameSnapshot, RecognitionSnapshot, validate_control_batch
from backend.vision.line_counter import ContinuousLineCounter

INDUSTRIAL_CAMERA_BACKENDS = {"hikrobot", "basler", "daheng", "flir", "allied_vision", "gentl"}


def _geometry_version() -> str:
    """定位几何的版本号；几何实现变化时标尺的已验证范围随之作废。"""
    from backend.vision.parallel_walls import GEOMETRY_VERSION

    return GEOMETRY_VERSION


PREVIEW_MAX_WIDTH = 640
PREVIEW_MAX_HEIGHT = 480
PREVIEW_TARGET_INTERVAL_S = 1.0 / 30.0
PREVIEW_JPEG_QUALITY = 82
MOTION_WINDOW_FRAMES = 5
PROCESSING_BATCH_QUEUE_SIZE = 2
PREVIEW_QUEUE_SIZE = 1
SAMPLING_QUEUE_SIZE = 32
GENERATION_RATE_WINDOW_S = 1.0
# Vision sampling must not be tied to the PID decision period. A 10-second PID
# period still needs continuous observations or most passing droplets vanish
# between two five-frame batches.
ANALYSIS_BATCH_INTERVAL_S = 0.05
ANALYSIS_BUSY_RETRY_S = 0.02


def _include_fitted_wall_candidates(
    candidates: list[dict[str, float | int]],
    measurement: Any,
    *,
    frame_width: int,
    frame_height: int,
    roi_x0: int,
    roi_x1: int,
    roi_y0: int,
    merge_distance_px: float,
) -> list[dict[str, float | int]]:
    """Make the two fitted walls shown in the overlay selectable in the UI."""
    result = [dict(line) for line in candidates]
    roi_width = max(1, int(roi_x1) - int(roi_x0))
    local_center_x = (roi_width - 1) * 0.5
    for center, slope in (
        (getattr(measurement, "upper_center_px", None), getattr(measurement, "upper_slope", None)),
        (getattr(measurement, "lower_center_px", None), getattr(measurement, "lower_slope", None)),
    ):
        if center is None:
            continue
        line_slope = float(slope or 0.0)
        left_y = float(roi_y0) + float(center) - line_slope * local_center_x
        right_y = float(roi_y0) + float(center) + line_slope * local_center_x
        fitted_center = 0.5 * (left_y + right_y)
        duplicate = False
        for line in result:
            existing_center = 0.5 * (
                float(line.get("y1", 0.0)) + float(line.get("y2", 0.0))
            ) * float(frame_height)
            existing_slope = float(line.get("slope", 0.0))
            if (
                abs(existing_center - fitted_center) <= max(6.0, float(merge_distance_px) * 1.5)
                and abs(existing_slope - line_slope) <= 0.06
            ):
                duplicate = True
                break
        if duplicate:
            continue
        result.append(
            {
                "id": 0,
                "x1": max(0.0, min(1.0, float(roi_x0) / float(frame_width))),
                "y1": max(0.0, min(1.0, left_y / float(frame_height))),
                "x2": max(0.0, min(1.0, float(roi_x1 - 1) / float(frame_width))),
                "y2": max(0.0, min(1.0, right_y / float(frame_height))),
                "length_ratio": float(roi_width) / float(frame_width),
                "slope": line_slope,
                "source": "fitted_wall",
            }
        )
    result.sort(key=lambda line: 0.5 * (float(line.get("y1", 0.0)) + float(line.get("y2", 0.0))))
    for index, line in enumerate(result, start=1):
        line["id"] = index
    return result


# The detector intentionally prioritizes well-scored, de-duplicated candidates.
# Matching its measured throughput keeps the queue near-empty and prevents PID
# decisions from using stale frames, while the independent preview stays 30 FPS.


@runtime_checkable
class VisionAdapterProtocol(Protocol):
    def prepare_video(self, video_source_type: str, video_source: str, pixel_to_micron: float) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def set_run_context(self, session_id: str, generation: int) -> None: ...
    def get_snapshot(self) -> RecognitionSnapshot | dict[str, Any]: ...
    def get_frame_snapshot(self) -> FrameSnapshot | None: ...
    def wait_for_recognition_snapshot(
        self, after_period_id: int = 0, timeout: float | None = None
    ) -> RecognitionSnapshot: ...


class GenericVisionAdapter:
    def __init__(self, vision_service: Any) -> None:
        self.vision_service = vision_service

    def _call(self, names: list[str], *args, **kwargs):
        if self.vision_service is None:
            raise RuntimeError("未注入 vision_service")
        for name in names:
            fn = getattr(self.vision_service, name, None)
            if callable(fn):
                return fn(*args, **kwargs)
        raise AttributeError(f"vision_service 缺少可用接口: {names}")

    def prepare_video(self, video_source_type: str, video_source: str, pixel_to_micron: float) -> None:
        try:
            self._call(
                ["prepare_video", "prepare", "setup"],
                video_source_type=video_source_type,
                video_source=video_source,
                pixel_to_micron=pixel_to_micron,
            )
        except TypeError:
            self._call(["prepare_video", "prepare", "setup"], video_source_type, video_source, pixel_to_micron)

    def start(self) -> None:
        self._call(["start", "start_loop", "run"])

    def stop(self) -> None:
        self._call(["stop", "stop_loop", "shutdown"])

    def set_run_context(self, session_id: str, generation: int) -> None:
        fn = getattr(self.vision_service, "set_run_context", None)
        if callable(fn):
            fn(session_id, generation)

    def get_snapshot(self) -> RecognitionSnapshot | dict[str, Any]:
        return self._call(["get_snapshot", "get_latest_snapshot", "read_snapshot", "pull_result", "run_once"])

    def get_frame_snapshot(self) -> FrameSnapshot | None:
        """Forward the independent live-preview frame without mixing in analysis results."""
        return self._call(["get_frame_snapshot", "get_video_frame_snapshot", "get_latest_frame_snapshot"])

    def wait_for_recognition_snapshot(
        self, after_period_id: int = 0, timeout: float | None = None
    ) -> RecognitionSnapshot:
        return self._call(
            ["wait_for_recognition_snapshot"],
            after_period_id=after_period_id,
            timeout=timeout,
        )


class PipelineVisionService:
    def __init__(self, logger: Callable[[str], None] | None = None) -> None:
        from ..vision.service import VisionCameraService

        self._log = logger or (lambda _msg: None)
        self._pipeline = None
        self._camera_service = VisionCameraService(logger=self._log)
        self._video_source_type = "camera"
        self._video_source = "0"
        self._selected_backend = ""
        self._pixel_to_micron = 1.0
        self._configured_pixel_to_micron = 1.0
        self._expected_diameter_um = 0.0
        self._session_id = ""
        self._run_generation = 0
        self._calibration_metadata: dict[str, Any] = {}
        self._channel_calibration_enabled = False
        self._channel_width_um = 50.0
        self._generation_measurement_enabled = False
        self._scale_validated = False
        self._scale_declared = False
        self._scale_scope: dict[str, Any] | None = None
        self._localization_enabled = False
        self._strict_detection_localization = False
        # 定位模式的变更记录：模式切换必须可查询，不能让几何更新把它静默改掉。
        self._localization_mode_revision = 0
        self._localization_mode_change: dict[str, Any] | None = None
        self._localizer = None
        self._localization_work_scale = 1.0
        self._camera_unique_id = ""
        self._scale_scope_change: dict[str, Any] | None = None
        self._last_localization_observation: dict[str, Any] | None = None
        self._chip_depth_um: float | None = None
        self._chip_depth_source = "unknown"
        self._chip_depth_validated = False
        self._channel_width_px: float | None = None
        self._channel_calibration_status = "disabled"
        self._channel_calibration_confidence = 0.0
        self._channel_calibration_reason = "未启用管道标定"
        self._channel_calibration_attempts = 0
        self._channel_width_samples: deque[tuple[float, float]] = deque(maxlen=5)
        self._lock = threading.RLock()
        self._pipeline_lock = threading.RLock()
        self._recognition_condition = threading.Condition(self._lock)
        self._cap = None
        self._worker: threading.Thread | None = None
        self._process_worker: threading.Thread | None = None
        self._preview_worker: threading.Thread | None = None
        self._sampling_worker: threading.Thread | None = None
        # Preserve consecutive frames inside each motion-analysis batch. If the
        # detector falls behind, an entire old batch is replaced.
        self._frame_queue: queue.Queue[list[tuple[int, float, Any]]] = queue.Queue(
            maxsize=PROCESSING_BATCH_QUEUE_SIZE
        )
        self._preview_queue: queue.Queue[tuple[int, float, Any]] = queue.Queue(maxsize=PREVIEW_QUEUE_SIZE)
        self._sampling_queue: queue.Queue[tuple[int, float, Any]] = queue.Queue(maxsize=SAMPLING_QUEUE_SIZE)
        self._capture_batch: list[tuple[int, float, Any]] = []
        self._control_batch_active = False
        self._control_batch_pending = False
        self._control_batch_id = 0
        self._control_capture_frames = 5
        self._control_analysis_frames = 5
        self._control_batch_not_before = 0.0
        self._control_batch_processed = 0
        self._control_batch_assignments: dict[int, int] = {}
        self._pinned_batch_metadata: dict[int, dict[str, Any]] = {}
        self._queued_batch_metadata: dict[int, dict[int, dict[str, Any]]] = {}
        self._batch_gallery_period = 0
        self._analysis_batch_started_at = 0.0
        self._next_analysis_batch_time = 0.0
        self._stop_event = threading.Event()
        self._last_processed_frame_id = 0
        self._last_processed_frame_timestamp = 0.0
        self._capture_frame_id = 0
        self._last_camera_packet = None
        self._frame_metadata: dict[int, dict[str, Any]] = {}
        self._last_preview_publish_time = 0.0
        self._last_processing_submit_time = 0.0
        self._camera_parameters: dict[str, float | int | str] = {}
        self._local_frame_interval_s = 0.0
        self._next_local_frame_time = 0.0
        self._capture_times: deque[float] = deque(maxlen=240)
        self._processing_times: deque[float] = deque(maxlen=240)
        self._replacement_times: deque[float] = deque(maxlen=4000)
        self._observed_radii: deque[tuple[float, float]] = deque(maxlen=4000)
        self._calibration_stats: deque[tuple[float, float, float, float, float]] = deque(maxlen=4000)
        self._replaced_processing_frames = 0
        self._processed_frame_count = 0
        self._recognition_latency_ms = 0.0
        self._algorithm_processing_ms = 0.0
        self._processing_busy = False
        self._motion_observations: deque[tuple[float, dict[int, tuple[float, float]]]] = deque(
            maxlen=MOTION_WINDOW_FRAMES
        )
        self._crossing_times: deque[float] = deque(maxlen=1000)
        self._calibration_crossing_events: dict[int, tuple[float, float, int, int, float]] = {}
        self._average_droplet_speed_um_s: float | None = None
        self._speed_sample_count = 0
        self._droplet_generation_rate_hz = 0.0
        self._last_motion_frame_id = 0
        self._droplet_gallery_periods: dict[int, list[dict[str, Any]]] = {}
        self._last_droplet_gallery: dict[str, Any] = {
            "period_id": 0,
            "droplet_count": 0,
            "droplets": [],
            "sample_frame_count": 0,
            "frames": [],
            "reason": "尚无已完成的控制周期",
        }
        self._last_gallery_period_id = 0
        self._control_interval_ms = 300
        self._line_counter = ContinuousLineCounter()
        self._counting_wall_lines: list[dict[str, float]] = []
        self._counting_line_ratio = 0.6
        self._latest = self._empty_snapshot("当前无有效液滴通过")
        self._latest_preview: FrameSnapshot | None = None

    def _ensure_pipeline(self):
        if self._pipeline is None:
            from ..vision.config import default_config
            from ..vision.pipeline import VisionPipeline

            config = default_config()
            config.detector.measurement_mode = "generation_plug"
            config.detector.generation_channel_height_um = 50.0
            config.detector.generation_channel_width_um = 50.0
            config.metrics.flow_axis = "x"
            # The commissioned generation-zone view moves detached droplets
            # from right to left. This remains configurable in the saved ROI.
            config.metrics.flow_direction = "negative"
            config.tracker.tracker_type = "nearest"
            config.tracker.match_distance = 90.0
            config.tracker.match_distance_radius_ratio = 4.0
            config.tracker.radius_match_ratio = 1.0
            config.tracker.max_unmatched_frames = 2
            config.tracker.confirmation_window = 1
            config.tracker.confirmation_min_hits = 1
            self._pipeline = VisionPipeline(config, logger=self._log)
        return self._pipeline

    def wait_for_recognition_snapshot(
        self, after_period_id: int = 0, timeout: float | None = None
    ) -> RecognitionSnapshot:
        """Wait until vision publishes a completed control period."""
        with self._recognition_condition:
            if int(self._latest.control_period_id) <= int(after_period_id):
                self._recognition_condition.wait_for(
                    lambda: (
                        self._stop_event.is_set()
                        or int(self._latest.control_period_id) > int(after_period_id)
                    ),
                    timeout=timeout,
                )
            return self._latest

    def _is_realtime_mode(self) -> bool:
        mode = self._video_source_type.strip().lower()
        return mode in {
            "camera",
            "realtime",
            "real_time",
            "live",
            "usb",
            "opencv",
            "hikrobot",
            "hikrobot_industrial_camera",
            "industrial_camera",
            "usb_camera",
        }

    def _empty_snapshot(self, reason: str) -> RecognitionSnapshot:
        return RecognitionSnapshot(
            frame_droplet_count=0,
            total_droplet_count=0,
            new_crossing_count=0,
            avg_diameter=None,
            single_cell_rate=0.0,
            valid_for_control=False,
            timestamp=time.time(),
            reason=reason,
            droplet_count=0,
            active_droplet_count=0,
            has_droplet=False,
            control_reason=reason,
            frame_png_base64=None,
            frame_width=0,
            frame_height=0,
            video_source_type=self._video_source_type,
            video_source=self._video_source,
            frame_id=0,
            preview_frame_id=0,
            preview_timestamp=0.0,
            frame_single_cell_count=0,
            frame_diameters=[],
            frame_diameter_sum=0.0,
            frame_avg_diameter=None,
            frame_single_cell_rate=None,
            frame_diameter_std=None,
            frame_diameter_cv=None,
            uniformity_valid=False,
            uniformity_status="当前无液滴",
            uniformity_reason=reason,
            pixel_to_micron=float(self._pixel_to_micron),
            scale_source=("generation_channel_width" if self._channel_calibration_status == "calibrated" else "configured"),
            measurement_region="generation",
            channel_width_um=(self._channel_width_um if self._channel_calibration_enabled else None),
            channel_width_px=self._channel_width_px,
            channel_calibration_status=self._channel_calibration_status,
            channel_calibration_confidence=self._channel_calibration_confidence,
            channel_calibration_reason=self._channel_calibration_reason,
            channel_region_status=self._ensure_pipeline().current_channel_region().status,
            channel_region_confidence=self._ensure_pipeline().current_channel_region().confidence,
            channel_region_reason=self._ensure_pipeline().current_channel_region().reason,
        )

    def _snapshot_with_error(self, reason: str) -> RecognitionSnapshot:
        with self._lock:
            current = self._latest
        if current.frame_png_base64:
            return replace(
                current,
                valid_for_control=False,
                reason=reason,
                control_reason=reason,
                uniformity_valid=False,
                uniformity_reason=reason,
            )
        return self._empty_snapshot(reason)

    def set_mvs_sdk_path(self, sdk_path: str) -> None:
        self._camera_service.set_mvs_sdk_path(sdk_path)

    def set_selected_backend(self, backend_name: str) -> None:
        self._selected_backend = str(backend_name or "").strip()

    def set_camera_parameters(self, parameters: dict[str, Any] | None) -> None:
        allowed = {"exposure", "gain", "frame_rate", "width", "height"}
        normalized: dict[str, float | int | str] = {}
        for name, value in (parameters or {}).items():
            if name not in allowed or value in (None, ""):
                continue
            normalized[name] = value
        self._camera_parameters = normalized

    def set_run_context(self, session_id: str, generation: int) -> None:
        """Bind subsequently captured frames to one control run."""
        with self._pipeline_lock, self._lock:
            self.cancel_control_batch()
            self._session_id = str(session_id or "")
            self._run_generation = int(generation)
            self._capture_batch.clear()
            self._last_motion_frame_id = 0
            self._motion_observations.clear()
            self._crossing_times.clear()
            self._calibration_crossing_events.clear()
            for work_queue in (self._preview_queue, self._sampling_queue, self._frame_queue):
                while True:
                    try:
                        work_queue.get_nowait()
                    except queue.Empty:
                        break
            self._ensure_pipeline().reset()
            self._line_counter.reset()

    def set_calibration_metadata(self, metadata: dict[str, Any] | None) -> None:
        self._calibration_metadata = dict(metadata or {})

    def configure_detection_scale(self, target_diameter_um: float, pixel_to_micron: float) -> None:
        self._expected_diameter_um = max(0.0, float(target_diameter_um))
        self._pixel_to_micron = float(pixel_to_micron) if float(pixel_to_micron) > 0.0 else 1.0
        self._configured_pixel_to_micron = self._pixel_to_micron
        pipeline = self._ensure_pipeline()
        # The control target is deliberately stored only for display/control.
        # It must not influence the image-domain Hough detector or tracking.
        # Recognition may run much slower than camera acquisition.  A droplet
        # can move farther than the old 120 px gate between processed frames.
        pipeline.config.tracker.match_distance = 180.0
        pipeline.config.tracker.match_distance_radius_ratio = 4.0
        pipeline.detector.configure_expected_diameter(
            self._expected_diameter_um,
            self._pixel_to_micron,
        )
        min_r, preferred_r, max_r = pipeline.detector.runtime_radius_range()
        self._log(
            "[VISION][DETECTOR][SCALE] "
            f"control_target_ignored={self._expected_diameter_um:.3f}um "
            f"pixel_to_micron={self._pixel_to_micron:.6f} "
            f"broad_target_size_guard=False "
            f"radius_px={min_r:.2f}/{preferred_r:.2f}/{max_r:.2f}"
        )

    def apply_tuning_config(self, detector_config: Any, channel_region_config: Any) -> dict[str, Any]:
        """Apply user tuning atomically to all subsequently sampled frames."""
        from ..vision.config import ChannelRegionConfig, DetectorConfig

        detector = DetectorConfig(**vars(detector_config))
        channel_region = ChannelRegionConfig(**vars(channel_region_config))
        with self._pipeline_lock:
            pipeline = self._ensure_pipeline()
            current = pipeline.config.detector
            # Physical geometry belongs to the camera/ROI calibration.  The
            # tuning page may change only the generation detector's image
            # thresholds, never the calibrated H/W/kappa values or mode.
            detector.measurement_mode = "generation_plug"
            detector.generation_channel_height_um = float(
                current.generation_channel_height_um
            )
            detector.generation_channel_width_um = float(
                current.generation_channel_width_um
            )
            detector.generation_volume_correction = float(
                current.generation_volume_correction
            )
            pipeline.apply_tuning_config(detector, channel_region)
            pipeline.detector.configure_expected_diameter(
                self._expected_diameter_um,
                self._pixel_to_micron,
            )
        self._log(
            "[VISION][TUNING][APPLIED] "
            f"mode={detector.measurement_mode} "
            f"center_band={detector.generation_center_band_ratio:.3f} "
            f"edge_mad={detector.generation_edge_mad_multiplier:.3f} "
            f"length_ratio={detector.generation_min_length_ratio:.3f}.."
            f"{detector.generation_max_length_ratio:.3f} "
            f"contrast_sigma={detector.generation_min_profile_contrast_sigma:.3f} "
            f"meniscus_support={detector.generation_min_meniscus_support_ratio:.3f} "
            f"capsule_outline={detector.generation_min_capsule_outline_ratio:.3f} "
            f"polarity={detector.generation_polarity} "
            f"channel_region_enabled={channel_region.enabled}"
        )
        return {
            "applied": True,
            "measurement_mode": detector.measurement_mode,
            "generation_center_band_ratio": float(detector.generation_center_band_ratio),
            "generation_edge_mad_multiplier": float(detector.generation_edge_mad_multiplier),
            "generation_min_length_ratio": float(detector.generation_min_length_ratio),
            "generation_max_length_ratio": float(detector.generation_max_length_ratio),
            "generation_min_edge_separation_ratio": float(
                detector.generation_min_edge_separation_ratio
            ),
            "generation_min_profile_contrast_sigma": float(
                detector.generation_min_profile_contrast_sigma
            ),
            "generation_min_meniscus_support_ratio": float(
                detector.generation_min_meniscus_support_ratio
            ),
            "generation_min_capsule_outline_ratio": float(
                detector.generation_min_capsule_outline_ratio
            ),
            "generation_min_raw_outline_contrast": float(detector.generation_min_raw_outline_contrast),
            "generation_polarity": detector.generation_polarity,
            "channel_region_enabled": bool(channel_region.enabled),
        }

    def configure_control_interval(self, control_interval_ms: int) -> None:
        """Use the user-selected control period as the recognition window."""
        pipeline = self._ensure_pipeline()
        pipeline.config.metrics.realtime_window_ms = max(1, int(control_interval_ms))
        self._control_interval_ms = pipeline.config.metrics.realtime_window_ms
        self._log(
            "[VISION][METRICS][WINDOW] "
            f"control_interval_ms={pipeline.config.metrics.realtime_window_ms}"
        )

    def request_control_batch(self, capture_frames: int, analysis_frames: int) -> int:
        """Arm one fresh batch; the control worker requests the next after PID."""
        validate_control_batch(capture_frames, analysis_frames)
        with self._lock:
            self._control_batch_active = True
            self._control_batch_pending = True
            self._control_batch_id += 1
            self._control_capture_frames = capture_frames
            self._control_analysis_frames = analysis_frames
            self._control_batch_processed = 0
            self._control_batch_not_before = time.monotonic()
            self._capture_batch.clear()
            self._next_analysis_batch_time = 0.0
            self._control_batch_assignments.clear()
            self._queued_batch_metadata.clear()
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break
            return self._control_batch_id

    def cancel_control_batch(self) -> None:
        """Invalidate pending results and return to continuous observation."""
        with self._lock:
            self._control_batch_active = False
            self._control_batch_pending = False
            self._control_batch_id += 1
            self._capture_batch.clear()
            self._control_batch_assignments.clear()
            self._queued_batch_metadata.clear()
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break
            self._recognition_condition.notify_all()

    def wait_for_control_batch(self, request_id: int, timeout: float = 0.1) -> RecognitionSnapshot:
        with self._recognition_condition:
            self._recognition_condition.wait_for(
                lambda: self._latest.control_batch_id == request_id
                or self._stop_event.is_set() or self._control_batch_id != request_id,
                timeout=max(0.0, timeout),
            )
            return replace(self._latest)

    def set_localization_mode(self, *, strict: bool, enabled: bool | None = None,
                              reason: str = "") -> dict:
        """显式切换当前帧定位模式。**这是唯一能改变定位模式的公开接口。**

        为什么必须独立成接口：``set_recognition_roi`` 是 ROI 的**完整几何配置**，
        而定位模式是模式开关。两者原先混在一个字典里，几何更新一带上缺省的 False
        就把严格定位静默关掉。模式开关必须有自己会被记录的入口。

        ``strict=True`` 隐含 ``enabled=True``。``strict=False`` 且 ``enabled=None``
        表示只关严格定位、保留 ``enabled`` 原值。返回切换后的
        :meth:`localization_mode_state`，供调用方与审计查询。
        """
        strict_value = bool(strict)
        if strict_value:
            if enabled is not None and not enabled:
                raise ValueError("strict 定位要求 enabled 为真")
            enabled_value = True
        elif enabled is None:
            enabled_value = bool(self._localization_enabled)
        else:
            enabled_value = bool(enabled)
        previous_strict = bool(self._strict_detection_localization)
        previous_enabled = bool(self._localization_enabled)
        self._strict_detection_localization = strict_value
        self._localization_enabled = enabled_value
        if strict_value:
            self._line_counter.reset()
        # localizer 的构建参数含 contrast_enhance=严格模式，模式一变必须重建。
        self._localizer = None
        self._localization_mode_revision += 1
        self._localization_mode_change = {
            "strict": strict_value,
            "enabled": enabled_value,
            "previous_strict": previous_strict,
            "previous_enabled": previous_enabled,
            "changed": (strict_value != previous_strict) or (enabled_value != previous_enabled),
            "reason": str(reason),
            "revision": int(self._localization_mode_revision),
        }
        return self.localization_mode_state()

    def localization_mode_state(self) -> dict:
        """当前定位模式与其最近一次变更（可查询状态）。"""
        return {
            "strict_detection_localization": bool(self._strict_detection_localization),
            "localization_enabled": bool(self._localization_enabled),
            "revision": int(self._localization_mode_revision),
            "last_change": (dict(self._localization_mode_change)
                            if self._localization_mode_change else None),
        }

    def set_recognition_roi(self, roi: dict[str, Any] | None) -> None:
        pipeline = self._ensure_pipeline()
        config = pipeline.config.roi
        values = dict(roi or {})
        config.enabled = bool(values.get("enabled", False))
        config.user_defined = bool(values.get("user_defined", config.enabled))
        channel_region = pipeline.config.channel_region
        channel_region.enabled = bool(values.get("channel_region_enabled", True))
        channel_region.sample_frames = max(1, min(48, int(values.get("channel_region_sample_frames", 12))))
        channel_region.min_confidence = max(
            0.0,
            min(1.0, float(values.get("channel_region_min_confidence", channel_region.min_confidence))),
        )
        config.x_start_ratio = max(0.0, min(1.0, float(values.get("x_start_ratio", 0.0))))
        config.x_end_ratio = max(0.0, min(1.0, float(values.get("x_end_ratio", 1.0))))
        config.y_start_ratio = max(0.0, min(1.0, float(values.get("y_start_ratio", 0.0))))
        config.y_end_ratio = max(0.0, min(1.0, float(values.get("y_end_ratio", 1.0))))
        wall_lines: list[dict[str, float]] = []
        for raw_line in list(values.get("wall_lines", []) or [])[:2]:
            if not isinstance(raw_line, dict):
                continue
            try:
                line = {
                    key: max(0.0, min(1.0, float(raw_line[key])))
                    for key in ("x1", "y1", "x2", "y2")
                }
            except (KeyError, TypeError, ValueError):
                continue
            if abs(line["x2"] - line["x1"]) + abs(line["y2"] - line["y1"]) >= 0.05:
                wall_lines.append(line)
        config.wall_lines = wall_lines if len(wall_lines) == 2 else []
        self._counting_wall_lines = [dict(line) for line in config.wall_lines]
        self._counting_line_ratio = float(pipeline.config.metrics.count_line_ratio)
        self._line_counter.reset()
        flow_direction = str(values.get("flow_direction", "negative") or "negative").lower()
        if flow_direction not in {"positive", "negative", "any"}:
            raise ValueError("生成区流动方向必须为 positive、negative 或 any")
        pipeline.config.metrics.flow_direction = flow_direction
        if config.x_end_ratio <= config.x_start_ratio or config.y_end_ratio <= config.y_start_ratio:
            raise ValueError("识别 ROI 的结束坐标必须大于开始坐标")
        config.crop_top_ratio = 0.0
        self._channel_calibration_enabled = bool(values.get("channel_calibration_enabled", False))
        self._channel_width_um = float(values.get("channel_width_um", 50.0))
        if self._channel_width_um <= 0.0:
            raise ValueError("框选区域实际高度必须大于 0 μm")
        # 生成区物理测量的开关与证据声明。默认关闭、默认不承认标尺已验证：
        # 打开即要求在采集路径上跑当前画面管壁核验，未通过就写出拒绝理由。
        self._generation_measurement_enabled = bool(values.get("generation_measurement_enabled", False))
        # 标尺的“已验证”只对它被声明时的成像/几何范围有效：相机或图像尺寸、定位几何
        # 一变，这个布尔值就作废，必须重新声明，不能沿用成永久 valid。
        self._scale_declared = bool(values.get("scale_validated", False))
        self._scale_scope = None
        # 范围作废期间不承认已验证：下一次测量会按新的成像范围重建范围。
        self._scale_validated = False
        # 定位模式**不在这里读**：它是模式开关，不是 ROI 几何。原先对缺失的
        # ``strict_detection_localization`` 取默认 False，于是任何一次几何更新都会把
        # 严格定位静默关掉（2026-09-24 修复的缺陷）。显式切换走 set_localization_mode()；
        # 这里收到这两个键就直接报错，不留任何静默关闭路径。
        for mode_key in ("localization_enabled", "strict_detection_localization"):
            if mode_key in values:
                raise ValueError(
                    f"{mode_key} 不能经 set_recognition_roi 传递：ROI 局部更新不得改变定位模式，"
                    "请改用 set_localization_mode()"
                )
        self._localization_work_scale = float(values.get("localization_work_scale", 1.0) or 1.0)
        self._localizer = None
        # 芯片深度是结构尺寸，图像里看不到，必须由调用者显式声明。
        # 刻意**不**从 generation_channel_height_um 回退：那个字段带默认值，
        # 把默认值当成“已声明深度”会让体积等效尺寸在没人声明深度时照常产出。
        declared_depth = values.get("chip_depth_um")
        try:
            declared_depth = None if declared_depth in (None, "") else float(declared_depth)
        except (TypeError, ValueError):
            declared_depth = None
        if declared_depth is None or not np.isfinite(declared_depth) or declared_depth <= 0.0:
            self._chip_depth_um = None
            self._chip_depth_source = "unknown"
            self._chip_depth_validated = False
        else:
            from ..vision.rectified_measurement import DEPTH_SOURCES

            self._chip_depth_um = declared_depth
            source = str(values.get("chip_depth_source", "declared_chip_geometry"))
            self._chip_depth_source = source if source in DEPTH_SOURCES else "unknown"
            self._chip_depth_validated = bool(values.get("chip_depth_validated", False))
        if self._channel_calibration_enabled and not config.enabled:
            raise ValueError("启用已知高度标定前必须先启用并框选 ROI")
        detector = pipeline.config.detector
        detector.measurement_mode = "generation_plug"
        detector.generation_channel_height_um = float(
            values.get("generation_channel_height_um", self._channel_width_um)
        )
        detector.generation_channel_width_um = float(
            values.get("generation_channel_width_um", self._channel_width_um)
        )
        detector.generation_volume_correction = float(
            values.get("generation_volume_correction", 1.0)
        )
        if detector.generation_channel_height_um <= 0.0 or detector.generation_channel_width_um <= 0.0:
            raise ValueError("生成区通道高度和宽度必须大于 0 μm")
        if detector.generation_volume_correction <= 0.0:
            raise ValueError("生成区体积修正系数必须大于 0")
        pipeline.channel_region_detector.reset()
        self._log(
            f"[VISION][ROI] enabled={config.enabled} "
            f"channel_region_enabled={channel_region.enabled} "
            f"channel_region_samples={channel_region.sample_frames} "
            f"x={config.x_start_ratio:.3f}-{config.x_end_ratio:.3f} "
            f"y={config.y_start_ratio:.3f}-{config.y_end_ratio:.3f}"
        )

    def declared_chip_depth(self) -> dict[str, Any]:
        """公开读取本服务声明的芯片深度证据，供调用方与自己持有的声明核对。"""
        return {
            "depth_um": self._chip_depth_um,
            "source": self._chip_depth_source,
            "validated": bool(self._chip_depth_validated),
        }

    def generation_measurement_scale(self) -> dict[str, Any]:
        """当前可用的标尺证据：来源、取值与是否已验证。"""
        calibrated = self._channel_calibration_status == "calibrated"
        return {
            "um_per_px": float(self._pixel_to_micron),
            "source": "channel_width_reference" if calibrated else "configured_optical",
            "validated": bool(self._scale_validated),
            "reference_um": float(self._channel_width_um) if calibrated else None,
            "detail": (
                f"status={self._channel_calibration_status} "
                f"configured={self._configured_pixel_to_micron:.6f} "
                f"scope={'established' if self._scale_scope else 'none'}"
            ),
        }

    def _apply_evidence_scope(self, image_shape: tuple[int, int]) -> None:
        """标尺验证状态按适用范围失效：范围变了就必须重新声明。

        范围包含图像尺寸、相机标识与本模块的几何版本。任何一项变化都会把
        ``scale_validated`` 退回 False，并在 ``_scale_scope_change`` 里记录原因。
        """
        scope = {
            "image_shape": [int(image_shape[0]), int(image_shape[1])],
            "camera": str(getattr(self, "_camera_unique_id", "") or ""),
            "geometry_version": _geometry_version(),
        }
        change = None
        if self._scale_scope is None:
            if self._scale_declared:
                self._scale_scope = scope
        elif self._scale_scope != scope:
            changed = [key for key in scope if self._scale_scope.get(key) != scope.get(key)]
            change = {"changed": changed, "previous": dict(self._scale_scope), "current": dict(scope)}
            self._scale_scope = scope
            self._scale_declared = False
            self._log(
                "[VISION][GENERATION_MEASUREMENT][SCALE_SCOPE_CHANGED] "
                f"changed={changed}；标尺验证状态已作废，需重新声明"
            )
        self._scale_validated = bool(self._scale_declared)
        self._scale_scope_change = change

    def localize_parallel_walls(self, frame, *, frame_id: int, capture_monotonic: float,
                                region=None) -> dict[str, Any]:
        """逐帧喂入平行管壁定位器并返回本帧观测（不产出可用几何）。"""
        from ..vision.parallel_walls import ParallelWallLocalizer

        if self._localizer is None:
            self._localizer = ParallelWallLocalizer(
                work_scale=self._localization_work_scale,
                contrast_enhance=self._strict_detection_localization)
        observation = self._localizer.observe(frame, frame_id=frame_id,
                                              capture_monotonic=capture_monotonic, region=region)
        self._last_localization_observation = observation
        return observation

    def current_wall_localization(self, *, now_monotonic: float | None = None) -> dict[str, Any]:
        """聚合定位器缓冲，返回定位结果（含拒绝理由）。未启用时返回 None。"""
        if self._localizer is None:
            return None
        return self._localizer.localize(now_monotonic=now_monotonic).to_dict()

    def reset_wall_localization(self) -> None:
        """清空定位证据。暂停/停止/换相机/改 ROI 时调用，避免复用旧几何。"""
        localizer = getattr(self, "_localizer", None)
        if localizer is not None:
            localizer.reset()
        self._last_localization_observation = None

    def measure_generation_zone(
        self,
        frame,
        *,
        frame_id: int,
        hardware_frame_id: int,
        capture_monotonic: float,
        time_source: str = "host_clock_proxy",
        localization_frame_id: int | None = None,
        wall_lines: list[dict[str, float]] | None = None,
        wall_source: str = "reused",
        wall_consistency: dict[str, Any] | None = None,
        verify_walls_on_this_frame: bool = True,
        duct_depth_um: float | None = None,
        duct_depth_source: str | None = None,
    ) -> dict[str, Any] | None:
        """在原始帧上做生成区物理测量，并带上标尺、几何、帧身份与拒绝理由。

        管壁来源由调用方决定：默认 ``wall_source="reused"`` 表示沿用配置里的
        ``wall_lines``，此时必须在**这一帧**上跑核验，核验不通过就返回拒绝理由；
        传入 ``wall_lines`` 可显式给出这一次要用的两条壁。

        芯片深度只有调用方显式声明时才可用。``duct_depth_um`` 为 None 时**不**回退到
        ``DetectorConfig`` 的默认尺寸——那个字段带默认值，把默认值当声明值会让体积
        等效尺寸在没人声明深度时照常产出。未声明时只输出轴向长度。
        """
        from ..vision.rectified_measurement import (
            FrameEvidence,
            ScaleEvidence,
            measure_generation_plugs,
            verify_reused_walls,
        )

        selected_walls = [dict(line) for line in (wall_lines or self._counting_wall_lines or [])]
        resolved_frame_id = int(frame_id)
        image_shape = (0, 0)
        if hasattr(frame, "shape"):
            shape = frame.shape[:2]
            image_shape = (int(shape[1]), int(shape[0]))
        self._apply_evidence_scope(image_shape)
        scale_payload = self.generation_measurement_scale()

        localization = None
        localization_payload = None
        if self._localization_enabled:
            # 同一帧只能观测一次：重复喂入同一张图会把“管内在变”抹成零变化，
            # 运动证据随即消失。调用方先观测过就复用它。
            previous = self._last_localization_observation or {}
            if int(previous.get("frame_id", -1)) != resolved_frame_id:
                self.localize_parallel_walls(frame, frame_id=resolved_frame_id,
                                            capture_monotonic=float(capture_monotonic))
            localization_payload = self.current_wall_localization(
                now_monotonic=float(capture_monotonic))
            if localization_payload is not None:
                from ..vision.parallel_walls import WallLocalization

                localization = WallLocalization.from_geometry(
                    localization_payload, localization_payload["status"],
                    localization_payload["reason"])
            if localization is None or not localization.usable:
                reason = ("localization_not_ready" if localization_payload is None
                          else f"localization_{localization_payload['status']}")
                return {
                    "valid": False, "reason": reason,
                    "localization": localization_payload,
                    "source": "vision_adapter.measure_generation_zone",
                    "scale": scale_payload,
                    "scale_scope_change": self._scale_scope_change,
                }
            selected_walls = localization.wall_lines

        if len(selected_walls) != 2:
            return {"valid": False, "reason": "wall_geometry_missing",
                    "source": "vision_adapter.measure_generation_zone",
                    "localization": localization_payload}

        depth_um = self._chip_depth_um if duct_depth_um is None else float(duct_depth_um)
        depth_source = self._chip_depth_source if duct_depth_source is None else str(duct_depth_source)

        consistency = wall_consistency
        if (localization is None and consistency is None
                and str(wall_source) == "reused" and verify_walls_on_this_frame):
            consistency = verify_reused_walls(frame, selected_walls, frame_id=resolved_frame_id)

        result = measure_generation_plugs(
            frame,
            detector=self._ensure_pipeline().detector,
            wall_lines=selected_walls,
            localization=localization,
            scale=ScaleEvidence(
                um_per_px=scale_payload["um_per_px"],
                source=scale_payload["source"],
                validated=scale_payload["validated"],
                reference_um=scale_payload["reference_um"],
                detail=scale_payload["detail"],
            ),
            frame_evidence=FrameEvidence(
                frame_id=resolved_frame_id,
                hardware_frame_id=int(hardware_frame_id),
                capture_monotonic=float(capture_monotonic),
                localization_frame_id=int(
                    resolved_frame_id if localization_frame_id is None else localization_frame_id
                ),
                time_source=str(time_source),
            ),
            duct_depth_um=None if depth_um is None else float(depth_um),
            duct_depth_source="unknown" if depth_um is None else depth_source,
            duct_depth_validated=(False if depth_um is None else bool(self._chip_depth_validated)),
            duct_width_reference_um=(
                self._channel_width_um if scale_payload["source"] == "channel_width_reference" else None
            ),
            wall_source=str(wall_source if localization is None else "localized"),
            wall_consistency=consistency,
        )
        payload = result.to_dict()
        payload["source"] = "vision_adapter.measure_generation_zone"
        payload["wall_verification"] = consistency
        payload["scale_scope_change"] = self._scale_scope_change
        if localization_payload is not None:
            payload["localization"] = localization_payload
        return payload

    def _reset_channel_calibration(self) -> None:
        self._channel_width_px = None
        self._channel_calibration_confidence = 0.0
        self._channel_calibration_attempts = 0
        self._channel_width_samples.clear()
        self._pixel_to_micron = self._configured_pixel_to_micron
        if self._channel_calibration_enabled:
            self._channel_calibration_status = "collecting"
            self._channel_calibration_reason = f"正在用用户输入的 {self._channel_width_um:.1f} μm 框选高度计算比例"
        else:
            self._channel_calibration_status = "disabled"
            self._channel_calibration_reason = "未启用管道标定"

    def _try_channel_calibration(self, frame) -> None:
        if not self._channel_calibration_enabled or self._channel_calibration_status in {"calibrated", "user_config"}:
            return
        from ..vision.channel_calibration import estimate_channel_width_px

        pipeline = self._ensure_pipeline()
        frame_h, frame_w = frame.shape[:2]
        wall_lines = list(getattr(pipeline.config.roi, "wall_lines", []) or [])
        if len(wall_lines) == 2:
            from ..vision.rectified_roi import wall_separation_px

            selected_width = wall_separation_px(frame_w, frame_h, wall_lines)
            self._channel_calibration_attempts += 1
            if selected_width is not None and selected_width > 1.0:
                scale = self._channel_width_um / selected_width
                self._pixel_to_micron = float(scale)
                pipeline.detector.configure_expected_diameter(
                    self._expected_diameter_um,
                    self._pixel_to_micron,
                )
                self._channel_width_px = float(selected_width)
                self._channel_calibration_confidence = 1.0
                self._channel_calibration_status = "calibrated"
                self._channel_calibration_reason = (
                    f"采用用户输入的框选高度与两条内壁：{self._channel_width_um:.1f} μm / "
                    f"{selected_width:.2f} px = {scale:.6f} μm/px"
                )
                self._log(
                    "[VISION][CHANNEL_CALIBRATION][SELECTED_LINES] "
                    f"width_px={selected_width:.3f} pixel_to_micron={scale:.8f}"
                )
                return
        x0, x1, y0, y1, crop_top = pipeline.config.roi.resolve(frame_w, frame_h)
        roi_frame = frame[y0 + crop_top : y1, x0:x1]
        measurement = estimate_channel_width_px(
            roi_frame,
            flow_axis=pipeline.config.metrics.flow_axis,
        )
        self._channel_calibration_attempts += 1
        if measurement.width_px is not None:
            self._channel_width_samples.append((float(measurement.width_px), float(measurement.confidence)))

        required = self._channel_width_samples.maxlen or 5
        if len(self._channel_width_samples) >= required:
            widths = [item[0] for item in self._channel_width_samples]
            center = float(median(widths))
            deviations = [abs(value - center) for value in widths]
            robust_cv = 100.0 * 1.4826 * float(median(deviations)) / max(center, 1.0)
            if robust_cv <= 3.0:
                scale = self._channel_width_um / center
                if 0.05 <= scale <= 100.0:
                    self._pixel_to_micron = float(scale)
                    pipeline.detector.configure_expected_diameter(
                        self._expected_diameter_um,
                        self._pixel_to_micron,
                    )
                    self._channel_width_px = center
                    self._channel_calibration_confidence = float(median([item[1] for item in self._channel_width_samples]))
                    self._channel_calibration_status = "calibrated"
                    self._channel_calibration_reason = (
                        f"管道内宽 {self._channel_width_um:.1f} μm / {center:.2f} px，"
                        f"标定比例 {scale:.6f} μm/px"
                    )
                    self._log(
                        "[VISION][CHANNEL_CALIBRATION][OK] "
                        f"width_um={self._channel_width_um:.3f} width_px={center:.3f} "
                        f"pixel_to_micron={scale:.8f} robust_cv={robust_cv:.3f}% "
                        f"confidence={self._channel_calibration_confidence:.3f}"
                    )
                    return

        if self._channel_calibration_attempts >= 20:
            self._channel_calibration_status = "user_config"
            failure_reason = (
                measurement.reason
                if len(self._channel_width_samples) < required
                else "多帧管壁间距不稳定；请收紧 ROI、固定相机并改善照明"
            )
            self._channel_calibration_reason = (
                f"{failure_reason}；保留用户设置的 ROI 和光学比例 "
                f"{self._configured_pixel_to_micron:.6f} μm/px"
            )
            self._log(
                "[VISION][CHANNEL_CALIBRATION][USER_CONFIG] "
                f"attempts={self._channel_calibration_attempts} samples={len(self._channel_width_samples)} "
                f"reason={self._channel_calibration_reason}"
            )

    def auto_calibrate_detection(self, duration_s: float = 3.0) -> dict[str, Any]:
        started = time.monotonic()
        with self._lock:
            baseline = self._processed_frame_count
            # Ensure calibration does not wait for the next normal control
            # period before it can collect a fresh five-frame sample.
            self._next_analysis_batch_time = 0.0
        while time.monotonic() - started < max(0.5, float(duration_s)) and not self._stop_event.is_set():
            time.sleep(0.05)
        with self._lock:
            radii = [radius for sample_time, radius in self._observed_radii if sample_time >= started]
            stats = [item for item in self._calibration_stats if item[0] >= started]
            processed = self._processed_frame_count - baseline
        if processed < 3 or len(radii) < 3:
            raise RuntimeError(f"自动标定样本不足：处理 {processed} 帧，仅识别到 {len(radii)} 个液滴样本")
        sorted_radii = sorted(float(value) for value in radii)
        radius_median = sorted_radii[len(sorted_radii) // 2]
        deviations = sorted(abs(value - radius_median) for value in sorted_radii)
        radius_mad = deviations[len(deviations) // 2]
        robust_cv = 100.0 * 1.4826 * radius_mad / max(1.0, radius_median)
        if robust_cv > 35.0:
            raise RuntimeError(
                f"自动标定检测到的移动目标尺寸不稳定（稳健 CV={robust_cv:.1f}%），"
                "请缩小 ROI、排除气泡和反光后重试"
            )
        detector = self._ensure_pipeline().detector
        generation_mode = detector._config.measurement_mode == "generation_plug"
        preferred = radius_median if generation_mode else detector.calibrate_preferred_radius(radii)
        brightness = sorted(item[1] for item in stats)
        noise = sorted(item[2] for item in stats)
        center_contrasts = sorted(item[3] for item in stats)
        ring_contrasts = sorted(item[4] for item in stats)
        median = lambda values: float(values[len(values) // 2]) if values else 0.0
        center_median = median(center_contrasts)
        ring_median = median(ring_contrasts)
        polarity = "亮心暗边" if center_median >= 0.0 else "暗心亮边"
        detector_config = self._ensure_pipeline().config.detector
        # Only enable a polarity threshold when the measured separation is
        # strong; otherwise retain polarity-neutral detection.
        detector_config.candidate_min_center_contrast = max(0.01, center_median * 0.35) if center_median >= 0.04 else -1.0
        detector_config.candidate_min_ring_contrast = max(0.005, ring_median * 0.30) if ring_median >= 0.025 else -1.0
        result = {
            "ok": True,
            "sample_count": len(radii),
            "processed_frames": processed,
            "preferred_radius_px": preferred,
            "preferred_diameter_px": preferred * 2.0,
            "radius_robust_cv_percent": robust_cv,
            "background_brightness": median(brightness),
            "noise_sigma": median(noise),
            "polarity": polarity,
            "center_contrast_threshold": detector_config.candidate_min_center_contrast,
            "ring_contrast_threshold": detector_config.candidate_min_ring_contrast,
            "measurement_region": "generation" if generation_mode else "observation",
            "measurement_model": (
                "square_channel_plug_volume_to_equivalent_diameter"
                if generation_mode
                else "hough_circle_diameter"
            ),
            "volume_correction": float(detector_config.generation_volume_correction),
        }
        self._log(f"[VISION][CALIBRATION] {result}")
        return result

    @staticmethod
    def _rate(times: deque[float], now: float) -> float:
        recent = [value for value in times if now - value <= 1.0]
        return float(len(recent))

    def _diagnostics(self) -> dict[str, float | int | str]:
        now = time.monotonic()
        with self._lock:
            period_start = now - self._control_interval_ms / 1000.0
            capture_fps = self._rate(self._capture_times, now)
            processing_fps = self._rate(self._processing_times, now)
            period_frames = sum(value >= period_start for value in self._processing_times)
            period_replaced = sum(value >= period_start for value in self._replacement_times)
            analysis_frames = self._control_analysis_frames if self._control_batch_active else MOTION_WINDOW_FRAMES
            pending_frames = (
                int(self._frame_queue.qsize()) * analysis_frames
                + (max(0, analysis_frames - self._control_batch_processed) if self._processing_busy and self._control_batch_active
                   else MOTION_WINDOW_FRAMES if self._processing_busy else 0)
                + len(self._capture_batch)
            )
            if self._control_batch_active:
                if self._control_batch_pending:
                    status = f"批次 {self._control_batch_id}：采集 {len(self._capture_batch)}/{self._control_capture_frames} 帧"
                elif self._latest.control_batch_id == self._control_batch_id:
                    status = f"批次 {self._control_batch_id}：已完成，等待 PID／下一周期"
                else:
                    status = f"批次 {self._control_batch_id}：分析 {self._control_batch_processed}/{analysis_frames} 帧"
            elif capture_fps <= 0.0:
                status = "相机没有新画面"
            elif period_frames <= 0:
                status = "识别线程未完成处理"
            elif self._recognition_latency_ms > self._control_interval_ms or (capture_fps >= 3.0 and processing_fps < capture_fps * 0.5):
                status = "识别线程来不及，正在替换旧帧"
            elif self._latest.frame_droplet_count <= 0:
                status = "画面正常更新，但当前未识别到液滴"
            else:
                status = "视觉性能正常"
            return {
                "capture_fps": capture_fps,
                "processing_fps": processing_fps,
                "recognition_latency_ms": self._recognition_latency_ms,
                "algorithm_processing_ms": self._algorithm_processing_ms,
                "replaced_processing_frames": self._replaced_processing_frames,
                "pending_processing_frames": pending_frames,
                "period_replaced_processing_frames": period_replaced,
                "processed_frame_count": self._processed_frame_count,
                "period_processed_frames": period_frames,
                "vision_performance_status": status,
            }

    def discover_cameras_result(self) -> dict[str, Any]:
        return self._camera_service.discover_cameras_result()

    def refresh_cameras_result(self) -> dict[str, Any]:
        return self._camera_service.refresh_cameras_result()

    def get_camera_devices(self) -> list[dict[str, Any]]:
        return self._camera_service.get_camera_devices()

    def select_camera(self, unique_id: str, backend_name: str | None = None) -> dict[str, Any]:
        self._video_source = str(unique_id or "")
        self._selected_backend = str(backend_name or "")
        return self._camera_service.select_camera(unique_id, backend_name)

    def test_camera(self, camera_config: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._camera_service.test_camera(camera_config=camera_config or self._camera_parameters)

    def analyze_channel_calibration_preview(
        self,
        preview_png_base64: str,
        roi: dict[str, Any] | None = None,
        channel_width_um: float = 50.0,
        configured_pixel_to_micron: float = 1.0,
        hough_parameters: dict[str, float | int] | None = None,
    ) -> dict[str, Any]:
        """Analyze and annotate the synchronized camera-test frame."""
        if cv2 is None:
            raise RuntimeError("OpenCV/cv2 未安装，无法分析管道标定")
        from ..vision.channel_calibration import (
            ChannelWidthMeasurement,
            detect_wall_line_candidates,
            estimate_channel_width_px,
            normalize_hough_line_parameters,
            suggest_channel_roi,
        )
        from ..vision.rectified_roi import wall_lines_bbox, wall_separation_px

        encoded = str(preview_png_base64 or "").strip()
        if not encoded:
            raise ValueError("测试帧为空，无法分析管道")
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("测试帧解码失败")

        values = dict(roi or {})
        applied_hough_parameters = normalize_hough_line_parameters(hough_parameters)
        selected_wall_lines = [dict(line) for line in list(values.get("wall_lines", []) or [])[:2] if isinstance(line, dict)]
        if len(selected_wall_lines) != 2:
            selected_wall_lines = []
        used_user_roi = bool(selected_wall_lines or (values.get("enabled", False) and values.get("user_defined", False)))
        suggestion = None
        if selected_wall_lines:
            bbox = wall_lines_bbox(selected_wall_lines)
            if bbox:
                values.update({"enabled": True, **bbox, "user_defined": True})
        elif not used_user_roi:
            suggestion = suggest_channel_roi(
                frame,
                flow_axis="x",
                hough_parameters=applied_hough_parameters,
            )
            if suggestion is not None:
                values.update(
                    {
                        "enabled": True,
                        "x_start_ratio": suggestion.x_start_ratio,
                        "y_start_ratio": suggestion.y_start_ratio,
                        "x_end_ratio": suggestion.x_end_ratio,
                        "y_end_ratio": suggestion.y_end_ratio,
                        "user_defined": False,
                    }
                )

        height, width = frame.shape[:2]
        enabled = bool(values.get("enabled", False))
        x0 = max(0, min(width - 1, int(width * float(values.get("x_start_ratio", 0.0)))))
        x1 = max(x0 + 1, min(width, int(width * float(values.get("x_end_ratio", 1.0)))))
        y0 = max(0, min(height - 1, int(height * float(values.get("y_start_ratio", 0.0)))))
        y1 = max(y0 + 1, min(height, int(height * float(values.get("y_end_ratio", 1.0)))))
        selected_width = wall_separation_px(width, height, selected_wall_lines) if selected_wall_lines else None
        measurement = (
            ChannelWidthMeasurement(selected_width, 1.0, "ok")
            if selected_width is not None
            else (
                estimate_channel_width_px(
                    frame[y0:y1, x0:x1],
                    flow_axis="x",
                    hough_parameters=applied_hough_parameters,
                )
                if enabled
                else None
            )
        )
        ok = bool(measurement is not None and measurement.width_px is not None)
        reference_um = max(1e-9, float(channel_width_um))
        measured_scale = reference_um / float(measurement.width_px) if ok else None
        fallback_scale = max(1e-9, float(configured_pixel_to_micron))

        overlay = frame.copy()
        color = (70, 210, 70) if ok else (40, 80, 230)
        hough_lines = detect_wall_line_candidates(
            frame,
            hough_parameters=applied_hough_parameters,
        )
        if ok and measurement is not None and not selected_wall_lines:
            hough_lines = _include_fitted_wall_candidates(
                hough_lines,
                measurement,
                frame_width=width,
                frame_height=height,
                roi_x0=x0,
                roi_x1=x1,
                roi_y0=y0,
                merge_distance_px=float(applied_hough_parameters["merge_distance_px"]),
            )
        for candidate in hough_lines:
            p1 = (int(round(float(candidate["x1"]) * width)), int(round(float(candidate["y1"]) * height)))
            p2 = (int(round(float(candidate["x2"]) * width)), int(round(float(candidate["y2"]) * height)))
            cv2.line(overlay, p1, p2, (255, 190, 40), 1, cv2.LINE_AA)
            cv2.putText(overlay, str(candidate["id"]), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 220, 80), 1, cv2.LINE_AA)
        if enabled:
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), color, 2)
        if selected_wall_lines:
            for line in selected_wall_lines:
                p1 = (int(round(float(line["x1"]) * width)), int(round(float(line["y1"]) * height)))
                p2 = (int(round(float(line["x2"]) * width)), int(round(float(line["y2"]) * height)))
                cv2.line(overlay, p1, p2, (0, 165, 255), 4, cv2.LINE_AA)
        elif ok and measurement is not None:
            local_center_x = (x1 - x0 - 1) * 0.5
            for center, slope in (
                (measurement.upper_center_px, measurement.upper_slope),
                (measurement.lower_center_px, measurement.lower_slope),
            ):
                if center is None:
                    continue
                line_slope = float(slope or 0.0)
                left_y = int(round(y0 + float(center) - line_slope * local_center_x))
                right_y = int(round(y0 + float(center) + line_slope * local_center_x))
                cv2.line(overlay, (x0, left_y), (x1 - 1, right_y), (40, 255, 255), 2)
        label = (
            f"{'USER' if used_user_roi else 'AUTO'} ROI | {reference_um:.1f}um / "
            f"{float(measurement.width_px):.2f}px = {float(measured_scale):.6f}um/px"
            if ok and measurement is not None
            else f"USER SETTINGS | optical scale {fallback_scale:.6f}um/px"
        )
        cv2.rectangle(overlay, (6, 6), (min(width - 6, 6 + max(360, len(label) * 8)), 34), (0, 0, 0), -1)
        cv2.putText(overlay, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        encoded_ok, encoded_overlay = cv2.imencode(".png", overlay)
        if not encoded_ok:
            raise RuntimeError("管道标定预览编码失败")

        applied_roi = {
            "enabled": enabled,
            "x_start_ratio": x0 / float(width),
            "y_start_ratio": y0 / float(height),
            "x_end_ratio": x1 / float(width),
            "y_end_ratio": y1 / float(height),
            "user_defined": used_user_roi,
            "channel_calibration_enabled": bool(values.get("channel_calibration_enabled", True)),
            "channel_width_um": reference_um,
            "wall_lines": selected_wall_lines,
        }
        reason = "ok" if ok else (
            measurement.reason if measurement is not None else "自动检测未找到可信管道区域"
        )
        return {
            "ok": ok,
            "used_user_roi": used_user_roi,
            "auto_suggested": bool(suggestion is not None and not used_user_roi),
            "roi": applied_roi,
            "channel_width_um": reference_um,
            "channel_width_px": (None if measurement is None else measurement.width_px),
            "pixel_to_micron": measured_scale if measured_scale is not None else fallback_scale,
            "confidence": (0.0 if measurement is None else measurement.confidence),
            "reason": reason,
            "fallback_to_configured_scale": not ok,
            "hough_lines": hough_lines,
            "hough_parameters": applied_hough_parameters,
            "overlay_png_base64": base64.b64encode(encoded_overlay.tobytes()).decode("ascii"),
        }

    def get_camera_status(self) -> dict[str, Any]:
        return self._camera_service.get_camera_status()

    def prepare_video(self, video_source_type: str, video_source: str, pixel_to_micron: float) -> None:
        self.stop()
        with self._lock:
            self._video_source_type = str(video_source_type or "camera")
            self._video_source = str(video_source or "0")
            self._pixel_to_micron = float(pixel_to_micron) if float(pixel_to_micron) > 0 else 1.0
            self._configured_pixel_to_micron = self._pixel_to_micron
            # 相机标识进入标尺的适用范围：换设备后旧标尺不再作数。
            self._camera_unique_id = str(video_source or "")
            # 换视频源等于换采集条件：旧的定位证据与标尺范围一起作废。
            self.reset_wall_localization()
            self._scale_scope = None
            self._scale_validated = False
            self._last_processed_frame_id = 0
            self._last_processed_frame_timestamp = 0.0
            self._capture_frame_id = 0
            self._last_preview_publish_time = 0.0
            self._last_processing_submit_time = 0.0
            self._analysis_batch_started_at = 0.0
            self._next_analysis_batch_time = 0.0
            self._capture_times.clear()
            self._processing_times.clear()
            self._observed_radii.clear()
            self._calibration_stats.clear()
            self._replaced_processing_frames = 0
            self._replacement_times.clear()
            self._processed_frame_count = 0
            self._recognition_latency_ms = 0.0
            self._algorithm_processing_ms = 0.0
            self._local_frame_interval_s = 0.0
            self._next_local_frame_time = 0.0
            self._ensure_pipeline().reset()
            self._reset_channel_calibration()
            self._line_counter.reset()
            self._droplet_gallery_periods.clear()
            self._last_droplet_gallery = {
                "period_id": 0,
                "droplet_count": 0,
                "droplets": [],
                "sample_frame_count": 0,
                "frames": [],
                "reason": "尚无已完成的控制周期",
            }
            self._last_gallery_period_id = 0
            self._latest = self._empty_snapshot("视频输入已准备，等待识别")
            self._latest_preview = None

        if self._is_realtime_mode():
            backend = self._selected_backend or _backend_from_mode(self._video_source_type)
            selected = self._camera_service.select_camera(self._video_source, backend or None)
            _require_industrial_camera(selected)
            self._log(
                "[VISION][CAMERA][SELECTED] "
                f"source=industrial_camera backend={selected.get('selected_backend') or selected.get('backend_name')} "
                f"vendor={selected.get('manufacturer')} model={selected.get('model')} "
                f"serial={selected.get('serial_number')} unique_id={selected.get('unique_id')}"
            )
            self._camera_service.open_camera()
            if self._camera_parameters:
                applied = self._camera_service.configure_camera(self._camera_parameters)
                self._log(f"[VISION][CAMERA][PARAMETERS] {applied}")
            self._camera_service.start_camera_stream()
            deadline = time.monotonic() + 3.0
            packet = self._camera_service.get_latest_frame()
            while time.monotonic() < deadline and (not packet.valid or packet.image is None):
                time.sleep(0.03)
                packet = self._camera_service.get_latest_frame()
            if not packet.valid or packet.image is None:
                raise RuntimeError(packet.error or "实时相机未产生有效帧")
            self._log(
                "[VISION][CAMERA][FRAME][OK] "
                f"backend={packet.source_backend} frame_id={packet.frame_id} "
                f"width={packet.width} height={packet.height} pixel_format={packet.pixel_format}"
            )
            try:
                self._publish_video_frame(
                    packet.image,
                    frame_id=int(packet.frame_id or 0),
                    timestamp=float(packet.timestamp or time.time()),
                )
            except Exception as exc:
                self._log(f"[VISION][PREVIEW][WARN] initial frame process failed: {exc}")
            self.start()
            self._log("[VISION][PREVIEW][START] realtime preview loop started")

    def _open_capture(self):
        if cv2 is None:
            raise RuntimeError("OpenCV/cv2 未安装，无法读取本地视频")
        cap = cv2.VideoCapture(self._video_source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {self._video_source}")
        return cap

    def start(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            if not self._is_realtime_mode():
                self._cap = self._open_capture()
                fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
                if 0.5 <= fps <= 240.0:
                    self._local_frame_interval_s = 1.0 / fps
                else:
                    self._local_frame_interval_s = 1.0 / 30.0
                self._next_local_frame_time = time.monotonic()
            self._frame_queue = queue.Queue(maxsize=PROCESSING_BATCH_QUEUE_SIZE)
            self._preview_queue = queue.Queue(maxsize=PREVIEW_QUEUE_SIZE)
            self._sampling_queue = queue.Queue(maxsize=SAMPLING_QUEUE_SIZE)
            self._capture_batch = []
            self._analysis_batch_started_at = 0.0
            self._next_analysis_batch_time = 0.0
            self._motion_observations.clear()
            self._crossing_times.clear()
            self._calibration_crossing_events.clear()
            self._average_droplet_speed_um_s = None
            self._speed_sample_count = 0
            self._droplet_generation_rate_hz = 0.0
            self._last_motion_frame_id = 0
            self._stop_event.clear()
            self._line_counter.reset()
            self._worker = threading.Thread(target=self._capture_loop, name="vision-capture-loop", daemon=True)
            self._process_worker = threading.Thread(target=self._process_loop, name="vision-processing-loop", daemon=True)
            self._preview_worker = threading.Thread(target=self._preview_loop, name="vision-preview-loop", daemon=True)
            self._sampling_worker = threading.Thread(target=self._sampling_loop, name="vision-sampling-loop", daemon=True)
            self._worker.start()
            self._process_worker.start()
            self._preview_worker.start()
            self._sampling_worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        # 定位证据与本次采集绑定：停止后不清空就可能被下一次会话的测量复用。
        self.reset_wall_localization()
        worker = self._worker
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=1.0)
        process_worker = self._process_worker
        if process_worker is not None and process_worker.is_alive() and process_worker is not threading.current_thread():
            process_worker.join(timeout=1.0)
        preview_worker = self._preview_worker
        if preview_worker is not None and preview_worker.is_alive() and preview_worker is not threading.current_thread():
            preview_worker.join(timeout=1.0)
        sampling_worker = self._sampling_worker
        if sampling_worker is not None and sampling_worker.is_alive() and sampling_worker is not threading.current_thread():
            sampling_worker.join(timeout=1.0)
        with self._lock:
            cap = self._cap
            self._cap = None
            self._worker = None
            self._process_worker = None
            self._preview_worker = None
            self._sampling_worker = None
        if cap is not None:
            cap.release()
        try:
            self._camera_service.stop_camera_stream()
            self._camera_service.close_camera()
        except Exception:
            pass

    def _encode_png_base64(self, frame) -> tuple[str | None, int, int]:
        if cv2 is None:
            return None, 0, 0
        try:
            preview = self._resize_preview_frame(frame)
            # Compression level 0 minimizes CPU latency. The preview is an
            # in-process UI stream, so a larger payload is preferable to
            # stalling acquisition/Tk on deflate work.
            ok, buf = cv2.imencode(".png", preview, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            if not ok:
                return None, int(preview.shape[1]), int(preview.shape[0])
            return base64.b64encode(buf.tobytes()).decode("ascii"), int(preview.shape[1]), int(preview.shape[0])
        except Exception:
            return None, 0, 0

    def _encode_jpeg(self, frame) -> tuple[bytes | None, int, int]:
        """Encode a compact preview frame for cross-process transport."""
        if cv2 is None:
            return None, 0, 0
        try:
            preview = self._resize_preview_frame(frame)
            if getattr(preview.dtype, "name", "") != "uint8":
                preview = cv2.normalize(preview, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            if preview.ndim == 3 and int(preview.shape[2]) == 4:
                preview = cv2.cvtColor(preview, cv2.COLOR_BGRA2BGR)
            height, width = int(preview.shape[0]), int(preview.shape[1])
            ok, encoded = cv2.imencode(
                ".jpg",
                preview,
                [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_JPEG_QUALITY],
            )
            if not ok:
                return None, width, height
            return encoded.tobytes(), width, height
        except Exception:
            return None, 0, 0

    def _encode_pgm(self, frame) -> tuple[bytes | None, int, int]:
        if cv2 is None:
            return None, 0, 0
        try:
            preview = self._resize_preview_frame(frame)
            gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY) if preview.ndim == 3 else preview
            if gray.dtype.name != "uint8":
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            height, width = int(gray.shape[0]), int(gray.shape[1])
            payload = f"P5\n{width} {height}\n255\n".encode("ascii") + gray.tobytes()
            return payload, width, height
        except Exception:
            return None, 0, 0

    def _resize_preview_frame(self, frame):
        try:
            height = int(frame.shape[0])
            width = int(frame.shape[1])
        except Exception:
            return frame
        if width <= 0 or height <= 0:
            return frame
        scale = min(1.0, PREVIEW_MAX_WIDTH / float(width), PREVIEW_MAX_HEIGHT / float(height))
        if scale >= 0.999:
            return frame
        target = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(frame, target, interpolation=cv2.INTER_AREA)

    def _read_next_frame(self) -> tuple[bool, Any, str]:
        if self._is_realtime_mode():
            packet = self._camera_service.get_latest_frame()
            if not packet.valid or packet.image is None:
                return False, None, packet.error or "相机取帧异常"
            frame_id = int(packet.frame_id or 0)
            timestamp = float(packet.timestamp or 0.0)
            if (
                frame_id > 0
                and timestamp > 0.0
                and frame_id == self._last_processed_frame_id
                and timestamp <= self._last_processed_frame_timestamp
            ):
                return False, None, ""
            self._last_processed_frame_id = frame_id
            self._last_processed_frame_timestamp = timestamp or time.time()
            self._last_camera_packet = packet
            return True, packet.image, ""

        with self._lock:
            cap = self._cap
        if cap is None:
            return False, None, "视频源未打开"
        ok, frame = cap.read()
        if not ok:
            return False, None, "本地视频读取结束"
        return True, frame, ""

    def _snapshot_from_frame(
        self,
        frame,
        *,
        frame_png_base64: str | None = None,
        frame_width: int = 0,
        frame_height: int = 0,
        frame_id: int | None = None,
        timestamp: float | None = None,
        encode_frame: bool = False,
    ) -> RecognitionSnapshot:
        with self._lock:
            acquisition_meta = dict(self._pinned_batch_metadata.get(
                int(frame_id or 0), self._frame_metadata.get(int(frame_id or 0), {})
            ))
        measurement_time = self._acquisition_time(acquisition_meta)
        with self._pipeline_lock:
            wall_kwargs: dict[str, Any] = {}
            if self._strict_detection_localization:
                resolved_frame_id = int(frame_id or 0)
                self.localize_parallel_walls(frame, frame_id=resolved_frame_id,
                                            capture_monotonic=measurement_time)
                geometry = self.current_wall_localization(now_monotonic=measurement_time) or {}
                usable = geometry.get("status") == "localized" and len(geometry.get("wall_lines", [])) == 2
                wall_kwargs = {
                    "current_wall_lines": geometry["wall_lines"] if usable else [],
                    "wall_rejection_reason": str(geometry.get("reason", "localization_not_ready")),
                }
            if not self._strict_detection_localization:
                with self._lock:
                    self._try_channel_calibration(frame)
            result = self._ensure_pipeline().process_frame(frame, timestamp=measurement_time,
                                                           **wall_kwargs)
        observed_ids = {int(track_id) for track_id, _ in result.tracking.matched_pairs}
        observed_ids.update(int(track_id) for track_id in result.tracking.new_track_ids)
        with self._lock:
            sample_time = measurement_time
            calibration_frame = (result.analysis_frame if self._strict_detection_localization
                                 else self._ensure_pipeline().rectify_selected_channel(frame))
            if calibration_frame is None:
                calibration_frame = frame
            if (not self._strict_detection_localization
                    and self._ensure_pipeline().config.roi.enabled
                    and not self._ensure_pipeline().config.roi.wall_lines):
                frame_h, frame_w = frame.shape[:2]
                x0, x1, y0, y1, crop_top = self._ensure_pipeline().config.roi.resolve(frame_w, frame_h)
                calibration_frame = frame[y0 + crop_top : y1, x0:x1]
            gray = (
                cv2.cvtColor(calibration_frame, cv2.COLOR_BGR2GRAY)
                if cv2 is not None and len(calibration_frame.shape) == 3
                else calibration_frame
            )
            normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX) if cv2 is not None else gray
            calibration_tracks = [
                track
                for track in result.tracking.active_tracks
                if int(track.id) in observed_ids
                and int(track.age) >= 2
                and (
                    float(track.velocity[0]) * float(track.velocity[0])
                    + float(track.velocity[1]) * float(track.velocity[1])
                ) ** 0.5 >= 2.0
            ]
            for track in calibration_tracks:
                _cx = float(track.position[0])
                _cy = float(track.position[1])
                radius = float(track.metadata.get("observed_radius", track.radius))
                self._observed_radii.append((sample_time, float(radius)))
                center = self._ensure_pipeline().detector._center_contrast(normalized, _cx, _cy, radius)
                ring = self._ensure_pipeline().detector._ring_contrast(normalized, _cx, _cy, radius)
                self._calibration_stats.append((sample_time, float(gray.mean()), float(gray.std()), center, ring))
        control = result.metrics.control
        self._update_droplet_gallery(
            result,
            frame_id=int(frame_id if frame_id is not None else result.frame_index),
            timestamp=float(timestamp or result.timestamp),
        )
        avg_px = control.frame_avg_diameter
        active_count = int(result.metrics.control.frame_droplet_count)
        total_count = int(result.metrics.control.total_droplet_count)
        new_cross = int(result.metrics.control.new_crossing_count)
        self._update_motion_measurements(
            result.tracking,
            observed_ids,
            measurement_time,
            len(control.crossed_track_ids),
            int(frame_id if frame_id is not None else result.frame_index),
        )
        has_droplet = active_count > 0
        control_reason = str(result.metrics.control.reason or "")
        frame_b64 = frame_png_base64
        width = int(frame_width or 0)
        height = int(frame_height or 0)
        if encode_frame and frame_b64 is None:
            frame_b64, width, height = self._encode_png_base64(frame)
        scale = float(self._pixel_to_micron)
        frame_diameters = [float(value) * scale for value in control.frame_diameters]
        current_crossed_track_diameters = {
            int(track_id): float(value) * scale
            for track_id, value in dict(control.crossed_track_diameters).items()
        }
        raw_frame_diameters = [
            float(value) * scale for value in control.raw_frame_diameters
        ]
        frame_avg_diameter = (float(avg_px) * scale) if avg_px is not None else None
        frame_diameter_sum = float(control.frame_diameter_sum) * scale
        frame_diameter_std = (
            float(control.frame_diameter_std) * scale
            if control.frame_diameter_std is not None
            else None
        )
        diagnostics = self._diagnostics()
        resolved_frame_id = int(frame_id if frame_id is not None else result.frame_index)
        frame_meta = acquisition_meta
        capture_monotonic = measurement_time
        with self._lock:
            # 启用新测量路径时**不再往校准过线缓存里写旧路径的尺寸**：那份缓存会进入
            # 标定记录，混入旧口径的尺寸比缺数据更危险。
            if not self._generation_measurement_enabled:
                for track_id, diameter_um in current_crossed_track_diameters.items():
                    self._calibration_crossing_events[int(track_id)] = (
                        float(diameter_um),
                        capture_monotonic,
                        resolved_frame_id,
                        int(control.period_id),
                        control.crossed_track_sample_starts[int(track_id)],
                    )
            while len(self._calibration_crossing_events) > 4000:
                self._calibration_crossing_events.pop(next(iter(self._calibration_crossing_events)))
            calibration_events = dict(self._calibration_crossing_events)
        crossed_track_diameters = {track_id:item[0] for track_id,item in calibration_events.items()}
        crossed_track_capture_monotonic = {track_id:item[1] for track_id,item in calibration_events.items()}
        crossed_track_frame_ids = {track_id:item[2] for track_id,item in calibration_events.items()}
        if self._generation_measurement_enabled:
            # 新路径没有轨迹 ID，这些逐轨迹字段在新路径下没有对应语义，因此交给下游空值，
            # 而不是让旧口径的尺寸继续流下去。
            raw_frame_diameters = []
            crossed_track_diameters = {}
            crossed_track_capture_monotonic = {}
            crossed_track_frame_ids = {}
        frequency = self._line_counter.window(control.window_start_time, control.window_end_time)
        calibrated = bool(self._calibration_metadata or self._channel_calibration_status == "calibrated")
        geometry = self._ensure_pipeline().config.detector
        quality_valid = bool(calibrated and frame_diameters and all(
            np.isfinite(value) and value > 0 for value in (
                scale, geometry.generation_channel_height_um,
                geometry.generation_channel_width_um, geometry.generation_volume_correction,
            )
        ))
        if self._strict_detection_localization and not self._generation_measurement_enabled:
            quality_valid = False
            frame_diameters = []
            frame_avg_diameter = None
            frame_diameter_std = None
            frame_diameter_sum = 0.0
        quality_reason = ("每滴一个有效等效直径" if quality_valid else
                          "比例未标定：尺寸仅供预览" if not calibrated else "无有效尺寸数据或几何参数无效")
        valid_size_sample_count = control.sample_size if quality_valid else 0
        generation_measurement = None
        if self._generation_measurement_enabled:
            generation_measurement = self.measure_generation_zone(
                frame,
                frame_id=resolved_frame_id,
                hardware_frame_id=int(frame_meta.get("hardware_frame_id", 0) or 0),
                capture_monotonic=float(capture_monotonic),
                time_source="host_clock_proxy",
            )
        # 启用新路径时，尺寸列与质量门槛由它决定：拒绝就写出空尺寸并标记为不可用，
        # 不允许旧管线尺寸继续参与记录与标定。
        if self._generation_measurement_enabled:
            measurement = generation_measurement or {"valid": False, "reason": "not_evaluated"}
            if not measurement.get("valid"):
                quality_valid = False
                quality_reason = f"生成区测量未通过：{measurement.get('reason')}"
                frame_diameters = []
                frame_avg_diameter = None
                frame_diameter_std = None
                frame_diameter_sum = 0.0
                valid_size_sample_count = 0
            elif not measurement.get("has_complete_plug"):
                quality_valid = False
                quality_reason = "生成区测量有效但没有完整柱塞"
                frame_diameters = []
                frame_avg_diameter = None
                frame_diameter_std = None
                frame_diameter_sum = 0.0
                valid_size_sample_count = 0
            else:
                measured = [float(value) for value
                            in measurement.get("equivalent_diameters_um", [])]
                frame_diameters = measured
                frame_avg_diameter = float(np.mean(measured)) if measured else None
                frame_diameter_std = (float(np.std(measured)) if len(measured) > 1 else None)
                frame_diameter_sum = float(np.sum(measured))
                quality_valid = True
                quality_reason = "生成区完整柱塞等效直径（定位+全分辨率扶正）"
                valid_size_sample_count = int(measurement.get("complete_plug_count", 0) or 0)
        return RecognitionSnapshot(
            frame_droplet_count=active_count,
            total_droplet_count=total_count,
            new_crossing_count=new_cross,
            avg_diameter=frame_avg_diameter,
            single_cell_rate=float(control.frame_single_cell_rate or 0.0),
            valid_for_control=bool(result.metrics.control.valid_for_control and has_droplet and quality_valid),
            timestamp=float(timestamp or time.time()),
            reason=control_reason if quality_valid else quality_reason,
            droplet_count=total_count,
            active_droplet_count=control.current_frame_droplet_count,
            has_droplet=has_droplet,
            control_reason=control_reason,
            frame_png_base64=frame_b64,
            frame_width=width,
            frame_height=height,
            video_source_type=self._video_source_type,
            video_source=self._video_source,
            frame_id=resolved_frame_id,
            preview_frame_id=resolved_frame_id,
            preview_timestamp=float(timestamp or result.timestamp),
            frame_single_cell_count=int(control.frame_single_cell_count),
            frame_diameters=frame_diameters,
            crossed_track_diameters=crossed_track_diameters,
            crossed_track_capture_monotonic=crossed_track_capture_monotonic,
            crossed_track_sample_starts={key: item[4] for key, item in calibration_events.items()},
            crossed_track_frame_ids=crossed_track_frame_ids,
            frame_diameter_sum=frame_diameter_sum,
            frame_avg_diameter=frame_avg_diameter,
            frame_single_cell_rate=control.frame_single_cell_rate,
            frame_diameter_std=frame_diameter_std,
            frame_diameter_cv=control.frame_diameter_cv,
            raw_frame_diameters=raw_frame_diameters,
            raw_frame_diameter_cv=control.raw_frame_diameter_cv,
            filtering_rule=control.filtering_rule,
            session_id=str(frame_meta.get("session_id", "") or ""),
            run_generation=int(frame_meta.get("run_generation", 0) or 0),
            capture_monotonic=capture_monotonic,
            hardware_frame_id=int(frame_meta.get("hardware_frame_id", 0) or 0),
            hardware_timestamp=float(frame_meta.get("hardware_timestamp", 0.0) or 0.0),
            uniformity_valid=bool(control.uniformity_valid),
            uniformity_status=str(control.uniformity_status or ""),
            uniformity_reason=str(control.uniformity_reason or ""),
            control_period_id=int(control.period_id),
            motion_window_frames=len(self._motion_observations),
            average_droplet_speed_um_s=self._average_droplet_speed_um_s,
            speed_sample_count=self._speed_sample_count,
            droplet_generation_rate_hz=float(frequency.rate_hz or 0.0),
            frequency_valid=frequency.valid,
            frequency_reason=frequency.reason,
            frequency_passage_count=frequency.count,
            measurement_window_start=control.window_start_time,
            measurement_window_end=control.window_end_time,
            measurement_sample_start=control.sample_start_time,
            measurement_sample_end=control.sample_end_time,
            processing_completed_monotonic=time.monotonic(),
            current_frame_droplet_count=control.current_frame_droplet_count,
            window_passage_count=control.window_passage_count,
            valid_size_sample_count=(valid_size_sample_count if self._generation_measurement_enabled
                                     else (control.sample_size if quality_valid else 0)),
            measurement_quality_valid=quality_valid,
            measurement_quality_reason=quality_reason,
            generation_measurement=generation_measurement,
            pixel_to_micron=scale,
            scale_source=(
                "generation_channel_width"
                if self._channel_calibration_status == "calibrated"
                else ("calibration_file" if self._calibration_metadata else "configured_unverified")
            ),
            channel_width_um=(self._channel_width_um if self._channel_calibration_enabled else None),
            channel_width_px=self._channel_width_px,
            channel_calibration_status=self._channel_calibration_status,
            channel_calibration_confidence=self._channel_calibration_confidence,
            channel_calibration_reason=self._channel_calibration_reason,
            channel_region_status=result.channel_region.status,
            channel_region_confidence=result.channel_region.confidence,
            channel_region_reason=result.channel_region.reason,
            calibration_id=str(self._calibration_metadata.get("calibration_id", "") or ""),
            calibration_uncertainty_um_per_px=(
                None
                if self._calibration_metadata.get("uncertainty_um_per_px") is None
                else float(self._calibration_metadata["uncertainty_um_per_px"])
            ),
            measurement_region="generation",
            generation_channel_height_um=float(
                self._ensure_pipeline().config.detector.generation_channel_height_um
            ),
            generation_channel_width_um=float(
                self._ensure_pipeline().config.detector.generation_channel_width_um
            ),
            generation_volume_correction=float(
                self._ensure_pipeline().config.detector.generation_volume_correction
            ),
            **diagnostics,
        )

    def _update_droplet_gallery(
        self,
        result,
        *,
        frame_id: int | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Store every sampled recognition frame for the previous-period viewer."""
        control = result.metrics.control
        completed_period = int(control.period_id)
        with self._lock:
            if not self._batch_gallery_period and completed_period > self._last_gallery_period_id:
                period_frames = self._droplet_gallery_periods.pop(completed_period, [])
                unique_valid_ids = {
                    int(track_id)
                    for item in period_frames
                    for track_id in list(item.get("valid_track_ids", []) or [])
                }
                self._last_droplet_gallery = {
                    "period_id": completed_period,
                    "droplet_count": len(unique_valid_ids),
                    "droplets": [],
                    "sample_frame_count": len(period_frames),
                    "frames": period_frames,
                    "reason": "ok" if period_frames else "该控制周期没有可回看的采样识别帧",
                }
                self._last_gallery_period_id = completed_period
                for old_period in [key for key in self._droplet_gallery_periods if key <= completed_period]:
                    self._droplet_gallery_periods.pop(old_period, None)

            # The metrics transition publishes period N before processing the
            # current frame into period N+1, so the current sample belongs to
            # completed_period + 1.
            target_period = self._batch_gallery_period or completed_period + 1
            period_frames = self._droplet_gallery_periods.setdefault(target_period, [])
            if len(period_frames) >= 300:
                return

            valid_ids = {
                int(track_id)
                for track_id in list(getattr(control, "valid_track_ids", []) or [])
            }
            crossed_ids = {
                int(track_id)
                for track_id in list(getattr(control, "crossed_track_ids", []) or [])
            }
            analysis_frame = result.analysis_frame
            frame_h, frame_w = analysis_frame.shape[:2]
            annotated = (
                cv2.cvtColor(analysis_frame, cv2.COLOR_GRAY2BGR)
                if analysis_frame.ndim == 2
                else analysis_frame.copy()
            )
            tracks = {int(track.id): track for track in result.tracking.active_tracks}
            valid_ids = {
                track_id
                for track_id in valid_ids
                if track_id in tracks
                and float(
                    tracks[track_id].metadata.get(
                        "observed_radius",
                        tracks[track_id].radius,
                    )
                ) > 1.0
            }
            crossed_ids.intersection_update(valid_ids)
            valid_diameters_um: list[float] = []
            valid_droplets: list[dict[str, float | int]] = []
            for track_id in sorted(valid_ids):
                track = tracks.get(track_id)
                if track is None:
                    continue
                radius = float(track.metadata.get("observed_radius", track.radius))
                if radius <= 1.0:
                    continue
                cx, cy = float(track.position[0]), float(track.position[1])
                diameter_um = radius * 2.0 * float(self._pixel_to_micron)
                plug_length_px = float(track.metadata.get("plug_length_px", 0.0) or 0.0)
                valid_diameters_um.append(diameter_um)
                valid_droplets.append(
                    {
                        "track_id": int(track_id),
                        "center_x_px": cx,
                        "center_y_px": cy,
                        "radius_px": radius,
                        "diameter_um": diameter_um,
                        **(
                            {
                                "plug_length_px": plug_length_px,
                                "plug_length_um": plug_length_px * float(self._pixel_to_micron),
                            }
                            if plug_length_px > 0.0
                            else {}
                        ),
                    }
                )
                color = (0, 165, 255) if track_id in crossed_ids else (40, 220, 70)
                thickness = 3 if track_id in crossed_ids else 2
                if plug_length_px > 0.0:
                    half_length = plug_length_px * 0.5
                    if frame_w >= frame_h:
                        top_left = (max(0, int(round(cx - half_length))), 1)
                        bottom_right = (min(frame_w - 1, int(round(cx + half_length))), frame_h - 2)
                    else:
                        top_left = (1, max(0, int(round(cy - half_length))))
                        bottom_right = (frame_w - 2, min(frame_h - 1, int(round(cy + half_length))))
                    cv2.rectangle(annotated, top_left, bottom_right, color, thickness, cv2.LINE_AA)
                else:
                    cv2.circle(
                        annotated,
                        (int(round(cx)), int(round(cy))),
                        max(2, int(round(radius))),
                        color,
                        thickness,
                        cv2.LINE_AA,
                    )

            metrics_config = self._ensure_pipeline().config.metrics
            axis = str(metrics_config.flow_axis).strip().lower()
            line_ratio = float(metrics_config.count_line_ratio)
            if axis == "y":
                line_position = min(frame_h - 1, max(0, int(round(frame_h * line_ratio))))
                cv2.line(annotated, (0, line_position), (frame_w - 1, line_position), (255, 210, 40), 1)
            else:
                line_position = min(frame_w - 1, max(0, int(round(frame_w * line_ratio))))
                cv2.line(annotated, (line_position, 0), (line_position, frame_h - 1), (255, 210, 40), 1)

            resolved_frame_id = int(frame_id if frame_id is not None else result.frame_index)
            encoded_ok, encoded = cv2.imencode(
                ".jpg",
                annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), 88],
            )
            if not encoded_ok:
                return
            period_frames.append(
                {
                    "frame_id": resolved_frame_id,
                    "timestamp": float(timestamp if timestamp is not None else result.timestamp),
                    "valid_droplet_count": len(valid_ids),
                    "crossed_droplet_count": len(crossed_ids),
                    "valid_track_ids": sorted(valid_ids),
                    "crossed_track_ids": sorted(crossed_ids),
                    "valid_droplets": valid_droplets,
                    "average_diameter_um": (
                        float(np.mean(valid_diameters_um)) if valid_diameters_um else None
                    ),
                    "image_jpeg_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
                    "width": int(frame_w),
                    "height": int(frame_h),
                }
            )
            if self._batch_gallery_period and completed_period == self._batch_gallery_period:
                self._last_droplet_gallery = {
                    "period_id": completed_period,
                    "droplet_count": len({key for item in period_frames for key in item["valid_track_ids"]}),
                    "droplets": [], "sample_frame_count": len(period_frames),
                    "frames": list(period_frames), "reason": "ok",
                }
                self._last_gallery_period_id = completed_period
                self._droplet_gallery_periods = {
                    key: value for key, value in self._droplet_gallery_periods.items() if key > completed_period
                }

    def get_last_control_period_droplets(self) -> dict[str, Any]:
        with self._lock:
            return {
                "period_id": int(self._last_droplet_gallery.get("period_id", 0)),
                "droplet_count": int(self._last_droplet_gallery.get("droplet_count", 0)),
                "droplets": [dict(item) for item in self._last_droplet_gallery.get("droplets", [])],
                "sample_frame_count": int(self._last_droplet_gallery.get("sample_frame_count", 0)),
                "frames": [dict(item) for item in self._last_droplet_gallery.get("frames", [])],
                "reason": str(self._last_droplet_gallery.get("reason", "") or ""),
            }

    def _update_motion_measurements(
        self,
        tracking,
        observed_ids: set[int],
        timestamp: float,
        new_crossings: int,
        frame_id: int = 0,
    ) -> None:
        if frame_id > 0 and self._last_motion_frame_id > 0 and frame_id != self._last_motion_frame_id + 1:
            self._motion_observations.clear()
        if frame_id > 0:
            self._last_motion_frame_id = frame_id
        positions = {
            int(track.id): (float(track.position[0]), float(track.position[1]))
            for track in tracking.active_tracks
            if int(track.id) in observed_ids
        }
        self._motion_observations.append((timestamp, positions))

        speeds: list[float] = []
        if len(self._motion_observations) == MOTION_WINDOW_FRAMES:
            axis = str(self._ensure_pipeline().config.metrics.flow_axis).strip().lower()
            axis_index = 1 if axis == "y" else 0
            track_ids = set.intersection(
                *(set(frame_positions) for _, frame_positions in self._motion_observations)
            )
            first_time, first_positions = self._motion_observations[0]
            last_time, last_positions = self._motion_observations[-1]
            elapsed = last_time - first_time
            if elapsed > 0.0:
                for track_id in track_ids:
                    displacement_px = abs(
                        last_positions[track_id][axis_index] - first_positions[track_id][axis_index]
                    )
                    speeds.append(displacement_px * float(self._pixel_to_micron) / elapsed)
        self._speed_sample_count = len(speeds)
        self._average_droplet_speed_um_s = sorted(speeds)[len(speeds) // 2] if speeds else None

        for _ in range(max(0, int(new_crossings))):
            self._crossing_times.append(timestamp)
        cutoff = timestamp - GENERATION_RATE_WINDOW_S
        while self._crossing_times and self._crossing_times[0] < cutoff:
            self._crossing_times.popleft()
        if len(self._crossing_times) >= 2:
            intervals = [
                later - earlier
                for earlier, later in zip(self._crossing_times, list(self._crossing_times)[1:])
                if later > earlier
            ]
            self._droplet_generation_rate_hz = (
                1.0 / float(median(intervals)) if intervals else 0.0
            )
        else:
            self._droplet_generation_rate_hz = 0.0

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._is_realtime_mode():
                self._pace_local_video()
            ok, frame, error = self._read_next_frame()
            if not ok:
                if error and self._is_realtime_mode():
                    with self._lock:
                        self._latest = self._snapshot_with_error(error)
                elif error:
                    break
                # A repeated latest-frame snapshot is normal while polling a
                # 100 FPS camera. Sleeping 30 ms here skipped roughly three
                # camera frames and capped the effective acquisition rate.
                time.sleep(0.001 if not error else 0.03)
                continue
            try:
                packet = self._last_camera_packet if self._is_realtime_mode() else None
                with self._lock:
                    if packet is not None and int(packet.frame_id or 0) > 0:
                        frame_id = int(packet.frame_id)
                    else:
                        self._capture_frame_id += 1
                        frame_id = self._capture_frame_id
                timestamp = float(packet.timestamp) if packet is not None else time.time()
                capture_monotonic = (
                    float(packet.host_monotonic_timestamp)
                    if packet is not None and float(packet.host_monotonic_timestamp or 0.0) > 0.0
                    else time.monotonic()
                )
                with self._lock:
                    self._capture_times.append(capture_monotonic)
                    self._frame_metadata[frame_id] = {
                        "capture_monotonic": capture_monotonic,
                        "hardware_frame_id": int(getattr(packet, "hardware_frame_id", 0) or 0),
                        "hardware_timestamp": float(getattr(packet, "hardware_timestamp_ticks", 0) or 0),
                        "session_id": self._session_id,
                        "run_generation": self._run_generation,
                    }
                    while len(self._frame_metadata) > 512:
                        self._frame_metadata.pop(next(iter(self._frame_metadata)))
                if self._should_publish_preview(capture_monotonic):
                    self._submit_preview_frame(frame_id, timestamp, frame)
                self._submit_sampling_frame(frame_id, timestamp, frame)
            except Exception as exc:
                self._log(f"[VISION][WARN] capture frame failed: {exc}")
                time.sleep(0.02)
        with self._lock:
            cap = self._cap
            self._cap = None
            self._worker = None
        if cap is not None:
            cap.release()

    def _preview_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame_id, timestamp, frame = self._preview_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self._publish_video_frame(frame, frame_id, timestamp)
            except Exception as exc:
                self._log(f"[VISION][WARN] preview frame failed: {exc}")

    def _sampling_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame_id, timestamp, frame = self._sampling_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                with self._lock:
                    frame_meta = dict(self._frame_metadata.get(int(frame_id), {}))
                sequence_id = int(frame_meta.get("hardware_frame_id", 0) or frame_id)
                measurement_time = self._acquisition_time(frame_meta)
                if not self._strict_detection_localization:
                    self._line_counter.observe_frame(frame, sequence_id, measurement_time,
                                                     self._counting_wall_lines,
                                                     line_ratio=self._counting_line_ratio)
            except Exception as exc:
                self._line_counter.reset()
                self._log(f"[VISION][FREQUENCY][INVALID] {exc}")
            self._submit_processing_frame(frame_id, timestamp, frame)

    def _process_loop(self) -> None:
        previous_control_batch = False
        while not self._stop_event.is_set():
            try:
                batch = self._frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            request_id = 0
            try:
                with self._lock:
                    request_id = self._control_batch_assignments.pop(batch[-1][0], 0)
                    if self._control_batch_active and request_id != self._control_batch_id:
                        continue
                    if request_id and not self._control_batch_active:
                        continue
                    self._processing_busy = True
                    if request_id:
                        self._pinned_batch_metadata = self._queued_batch_metadata.pop(batch[-1][0], {})
                with self._pipeline_lock:
                    metrics = self._ensure_pipeline().metrics
                    if request_id:
                        self._batch_gallery_period = metrics.begin_frame_batch(len(batch))
                        with self._lock:
                            # Discard an unfinished continuous window or a
                            # cancelled batch before recording this batch.
                            self._droplet_gallery_periods.clear()
                    elif previous_control_batch:
                        metrics.cancel_frame_batch()
                    previous_control_batch = bool(request_id)
                batch_snapshot = None
                for frame_id, timestamp, frame in batch:
                    if self._stop_event.is_set() or (
                        request_id and (not self._control_batch_active or request_id != self._control_batch_id)
                    ):
                        batch_snapshot = None
                        break
                    processing_started = time.perf_counter()
                    batch_snapshot = self._snapshot_from_frame(
                        frame,
                        frame_id=frame_id,
                        timestamp=timestamp,
                    )
                    with self._lock:
                        completed_at = time.monotonic()
                        frame_meta = self._pinned_batch_metadata.get(
                            int(frame_id), self._frame_metadata.get(int(frame_id), {})
                        )
                        capture_monotonic = float(frame_meta.get("capture_monotonic", 0.0) or 0.0)
                        self._algorithm_processing_ms = max(0.0, (time.perf_counter() - processing_started) * 1000.0)
                        self._processing_times.append(completed_at)
                        self._processed_frame_count += 1
                        if request_id:
                            self._control_batch_processed += 1
                        self._recognition_latency_ms = (
                            max(0.0, (completed_at - capture_monotonic) * 1000.0)
                            if capture_monotonic > 0.0
                            else self._algorithm_processing_ms
                        )
                    # Candidate scoring contains Python loops. Yield briefly so
                    # the preview producer and GUI are not starved by analysis.
                    time.sleep(0.002)
                # A complete batch is one analysis transaction. Publish only
                # after all frames have updated the pipeline's accumulated data.
                if batch_snapshot is not None:
                    with self._recognition_condition:
                        if self._control_batch_active and request_id != self._control_batch_id:
                            continue
                        if request_id and not self._control_batch_active:
                            continue
                        preview = self._latest_preview
                        self._latest = replace(
                            batch_snapshot,
                            control_batch_id=request_id,
                            batch_capture_frames=self._control_capture_frames if request_id else 0,
                            batch_analysis_frames=len(batch) if request_id else 0,
                            frame_png_base64=(preview.frame_png_base64 if preview else None),
                            frame_width=(preview.width if preview else 0),
                            frame_height=(preview.height if preview else 0),
                            preview_frame_id=(preview.frame_id if preview else 0),
                            preview_timestamp=(preview.timestamp if preview else 0.0),
                            **self._diagnostics(),
                        )
                        self._recognition_condition.notify_all()
            except Exception as exc:
                self._log(f"[VISION][WARN] processing frame failed: {exc}")
                with self._recognition_condition:
                    if request_id and (not self._control_batch_active or request_id != self._control_batch_id):
                        continue
                    self._latest = replace(self._snapshot_with_error(str(exc)), control_batch_id=request_id)
                    self._recognition_condition.notify_all()
            finally:
                with self._lock:
                    self._processing_busy = False
                    self._pinned_batch_metadata.clear()
                    self._batch_gallery_period = 0

    @staticmethod
    def _acquisition_time(metadata: dict[str, Any]) -> float:
        value = metadata.get("capture_monotonic")
        if value is None or not np.isfinite(float(value)) or not 0 < float(value) <= time.monotonic():
            raise ValueError("missing or invalid monotonic acquisition time")
        return float(value)

    def _publish_video_frame(self, frame, frame_id: int, timestamp: float) -> None:
        display_frame = (None if self._strict_detection_localization
                         else self._ensure_pipeline().rectify_selected_channel(frame))
        if display_frame is None:
            display_frame = frame
        frame_jpeg, width, height = self._encode_jpeg(display_frame)
        if frame_jpeg is None:
            return
        with self._lock:
            frame_meta = self._frame_metadata.get(int(frame_id), {})
            self._latest_preview = FrameSnapshot(
                frame_id=int(frame_id),
                timestamp=float(timestamp),
                width=width,
                height=height,
                valid=True,
                frame_png_base64=None,
                frame_pgm=None,
                frame_jpeg=frame_jpeg,
                reason="",
                session_id=str(frame_meta.get("session_id", "") or ""),
                run_generation=int(frame_meta.get("run_generation", 0) or 0),
                capture_monotonic=float(frame_meta.get("capture_monotonic", 0.0) or 0.0),
                hardware_frame_id=int(frame_meta.get("hardware_frame_id", 0) or 0),
                hardware_timestamp=float(frame_meta.get("hardware_timestamp", 0.0) or 0.0),
            )

    def _submit_preview_frame(self, frame_id: int, timestamp: float, frame) -> None:
        item = (int(frame_id), float(timestamp), frame)
        try:
            self._preview_queue.put_nowait(item)
            return
        except queue.Full:
            pass
        # Preview is best-effort. Replacing an old frame keeps acquisition
        # latency bounded even when PNG encoding or the UI is temporarily slow.
        try:
            self._preview_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._preview_queue.put_nowait(item)
        except queue.Full:
            pass

    def _submit_sampling_frame(self, frame_id: int, timestamp: float, frame) -> None:
        item = (int(frame_id), float(timestamp), frame)
        try:
            self._sampling_queue.put_nowait(item)
            return
        except queue.Full:
            pass
        # Sampling must not block camera acquisition. If the lightweight
        # sampler ever falls behind, discard the oldest frame and let its
        # continuity check restart the current five-frame burst.
        try:
            self._sampling_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._sampling_queue.put_nowait(item)
        except queue.Full:
            pass
    def _pace_local_video(self) -> None:
        interval = float(self._local_frame_interval_s)
        if interval <= 0.0:
            return
        now = time.monotonic()
        due = float(self._next_local_frame_time or now)
        if due > now:
            self._stop_event.wait(min(due - now, interval))
            now = time.monotonic()
        self._next_local_frame_time = max(due + interval, now)

    def _submit_processing_frame(self, frame_id: int, timestamp: float, frame) -> None:
        with self._lock:
            self._submit_processing_frame_locked(frame_id, timestamp, frame)

    def _submit_processing_frame_locked(self, frame_id: int, timestamp: float, frame) -> None:
        now = time.monotonic()
        if self._control_batch_active:
            if not self._control_batch_pending:
                return
            try:
                acquired = self._acquisition_time(self._frame_metadata.get(int(frame_id), {}))
            except ValueError:
                self._capture_batch.clear()
                return
            if acquired < self._control_batch_not_before:
                return
            if int(frame.nbytes) * self._control_capture_frames > 256 * 1024 * 1024:
                self._control_batch_pending = False
                self._latest = replace(
                    self._snapshot_with_error("采集批次超过 256 MiB，请减少帧数或图像分辨率"),
                    control_batch_id=self._control_batch_id,
                )
                self._recognition_condition.notify_all()
                return
        if self._capture_batch and int(frame_id) != int(self._capture_batch[-1][0]) + 1:
            self._capture_batch = []
        if self._control_batch_active and self._capture_batch:
            previous_id = self._capture_batch[-1][0]
            previous_hw = self._frame_metadata.get(previous_id, {}).get("hardware_frame_id", 0)
            current_hw = self._frame_metadata.get(int(frame_id), {}).get("hardware_frame_id", 0)
            if previous_hw and current_hw and int(current_hw) != int(previous_hw) + 1:
                self._capture_batch.clear()
        if not self._capture_batch:
            if now < self._next_analysis_batch_time:
                return
            with self._lock:
                processing_unavailable = self._processing_busy or not self._frame_queue.empty()
            if processing_unavailable:
                # Do not build a backlog, but retry soon. The PID period only
                # controls metrics aggregation and must never create a blind
                # interval in visual tracking.
                self._next_analysis_batch_time = now + ANALYSIS_BUSY_RETRY_S
                return
            self._analysis_batch_started_at = now
        self._capture_batch.append((int(frame_id), float(timestamp), frame))
        capture_count = self._control_capture_frames if self._control_batch_active else MOTION_WINDOW_FRAMES
        if len(self._capture_batch) < capture_count:
            return
        item = self._capture_batch
        self._capture_batch = []
        if self._control_batch_active:
            item = item[-self._control_analysis_frames:]
            self._control_batch_assignments[item[-1][0]] = self._control_batch_id
            self._queued_batch_metadata[item[-1][0]] = {
                entry[0]: dict(self._frame_metadata.get(entry[0], {})) for entry in item
            }
            self._control_batch_pending = False
        self._next_analysis_batch_time = self._analysis_batch_started_at + ANALYSIS_BATCH_INTERVAL_S
        try:
            self._frame_queue.put_nowait(item)
            return
        except queue.Full:
            pass
        # Recognition must never block camera acquisition or the monitor. Keep
        # the newest pending frames when processing temporarily falls behind.
        try:
            replaced = self._frame_queue.get_nowait()
            with self._lock:
                self._replaced_processing_frames += len(replaced)
                self._replacement_times.extend([time.monotonic()] * len(replaced))
        except queue.Empty:
            pass
        try:
            self._frame_queue.put_nowait(item)
        except queue.Full:
            pass

    def _should_publish_preview(self, timestamp: float) -> bool:
        with self._lock:
            last = float(self._last_preview_publish_time or 0.0)
            current = float(timestamp)
            if last <= 0.0 or current < last:
                self._last_preview_publish_time = current
                return True
            elapsed = current - last
            if elapsed < PREVIEW_TARGET_INTERVAL_S:
                return False
            # Advance the target clock instead of restarting it from the
            # selected capture frame. At 100 FPS this alternates 30/40 ms
            # selections and averages 30 FPS instead of collapsing to 25 FPS.
            intervals = max(1, int(elapsed / PREVIEW_TARGET_INTERVAL_S))
            self._last_preview_publish_time = last + intervals * PREVIEW_TARGET_INTERVAL_S
            return True

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            ok, frame, error = self._read_next_frame()
            if not ok:
                if error and self._is_realtime_mode():
                    with self._lock:
                        self._latest = self._snapshot_with_error(error)
                elif error:
                    break
                time.sleep(0.03)
                continue
            try:
                # Synchronous compatibility loop: stamp acquisition at read
                # completion, before any recognition work.
                acquired_at = time.monotonic()
                with self._lock:
                    self._capture_frame_id += 1
                    frame_id = self._capture_frame_id
                    self._frame_metadata[frame_id] = {"capture_monotonic": acquired_at}
                    while len(self._frame_metadata) > 512:
                        self._frame_metadata.pop(next(iter(self._frame_metadata)))
                snapshot = self._snapshot_from_frame(frame, frame_id=frame_id, timestamp=time.time())
                with self._lock:
                    self._latest = snapshot
            except Exception as exc:
                self._log(f"[VISION][WARN] 帧处理失败: {exc}")
                time.sleep(0.02)
        with self._lock:
            cap = self._cap
            self._cap = None
            self._worker = None
        if cap is not None:
            cap.release()

    def get_snapshot(self) -> RecognitionSnapshot:
        with self._lock:
            return replace(
                self._latest,
                frame_diameters=list(self._latest.frame_diameters),
                crossed_track_diameters=dict(self._latest.crossed_track_diameters),
                crossed_track_capture_monotonic=dict(self._latest.crossed_track_capture_monotonic),
                crossed_track_sample_starts=dict(self._latest.crossed_track_sample_starts),
                crossed_track_frame_ids=dict(self._latest.crossed_track_frame_ids),
                **self._diagnostics(),
            )

    def get_frame_snapshot(self) -> FrameSnapshot | None:
        with self._lock:
            if self._latest_preview is None:
                return None
            return replace(self._latest_preview)

    def run_once(self) -> RecognitionSnapshot:
        return self.get_snapshot()


def _backend_from_mode(mode: str) -> str:
    value = str(mode or "").strip().lower()
    aliases = {"alliedvision": "allied_vision"}
    value = aliases.get(value, value)
    return value if value in {"hikrobot", "basler", "daheng", "flir", "allied_vision", "gentl", "opencv"} else ""


def _require_industrial_camera(device: dict[str, Any]) -> None:
    backend = str(device.get("selected_backend", "") or device.get("backend_name", "") or "").strip().lower()
    device_type = str(device.get("device_type", "") or "").strip().lower()
    if backend in INDUSTRIAL_CAMERA_BACKENDS and device_type == "industrial_camera":
        return
    raise RuntimeError(
        "Realtime video must use an industrial camera. "
        f"Selected backend={backend or '--'}, device_type={device_type or '--'}."
    )
