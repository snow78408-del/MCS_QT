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

from .models import FrameSnapshot, RecognitionSnapshot

INDUSTRIAL_CAMERA_BACKENDS = {"hikrobot", "basler", "daheng", "flir", "allied_vision", "gentl"}
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
        self._channel_width_um = 430.0
        self._channel_width_px: float | None = None
        self._channel_calibration_status = "disabled"
        self._channel_calibration_confidence = 0.0
        self._channel_calibration_reason = "未启用管道标定"
        self._channel_calibration_attempts = 0
        self._channel_width_samples: deque[tuple[float, float]] = deque(maxlen=5)
        self._lock = threading.RLock()
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
        self._latest = self._empty_snapshot("当前无有效液滴通过")
        self._latest_preview: FrameSnapshot | None = None

    def _ensure_pipeline(self):
        if self._pipeline is None:
            from ..vision.config import default_config
            from ..vision.pipeline import VisionPipeline

            self._pipeline = VisionPipeline(default_config(), logger=self._log)
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
            scale_source=("channel_430um" if self._channel_calibration_status == "calibrated" else "configured"),
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
        with self._lock:
            self._session_id = str(session_id or "")
            self._run_generation = int(generation)
            self._capture_batch.clear()
            self._last_motion_frame_id = 0
            self._motion_observations.clear()
            self._crossing_times.clear()
            for work_queue in (self._preview_queue, self._sampling_queue, self._frame_queue):
                while True:
                    try:
                        work_queue.get_nowait()
                    except queue.Empty:
                        break
            self._ensure_pipeline().reset()

    def set_calibration_metadata(self, metadata: dict[str, Any] | None) -> None:
        self._calibration_metadata = dict(metadata or {})

    def configure_detection_scale(self, target_diameter_um: float, pixel_to_micron: float) -> None:
        self._expected_diameter_um = max(0.0, float(target_diameter_um))
        self._pixel_to_micron = float(pixel_to_micron) if float(pixel_to_micron) > 0.0 else 1.0
        self._configured_pixel_to_micron = self._pixel_to_micron
        pipeline = self._ensure_pipeline()
        # The control target is deliberately stored only for display/control.
        # It must not influence detector gates, candidate scores, or tracking.
        pipeline.config.detector.expected_size_hard_gate = False
        pipeline.config.detector.edge_work_max_width = 480
        pipeline.config.detector.edge_work_max_height = 360
        pipeline.config.detector.edge_max_candidates = 40
        # Recover weaker, partially illuminated boundaries while retaining an
        # EdgeDrawing perimeter-support check against isolated false circles.
        pipeline.config.detector.edge_gradient_threshold = 16
        pipeline.config.detector.edge_min_support_ratio = 0.12
        pipeline.config.detector.diameter_min_visible_ratio = 0.75
        # Recognition may run much slower than camera acquisition.  A droplet
        # can move farther than the old 120 px gate between processed frames.
        pipeline.config.tracker.match_distance = 180.0
        pipeline.config.tracker.match_distance_radius_ratio = 4.0
        min_r, preferred_r, max_r = pipeline.detector.runtime_radius_range()
        self._log(
            "[VISION][DETECTOR][SCALE] "
            f"control_target_ignored={self._expected_diameter_um:.3f}um "
            f"pixel_to_micron={self._pixel_to_micron:.6f} "
            f"broad_target_size_guard=False "
            f"radius_px={min_r:.2f}/{preferred_r:.2f}/{max_r:.2f}"
        )

    def configure_control_interval(self, control_interval_ms: int) -> None:
        """Use the user-selected control period as the recognition window."""
        pipeline = self._ensure_pipeline()
        pipeline.config.metrics.realtime_window_ms = max(1, int(control_interval_ms))
        self._control_interval_ms = pipeline.config.metrics.realtime_window_ms
        self._log(
            "[VISION][METRICS][WINDOW] "
            f"control_interval_ms={pipeline.config.metrics.realtime_window_ms}"
        )

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
        if config.x_end_ratio <= config.x_start_ratio or config.y_end_ratio <= config.y_start_ratio:
            raise ValueError("识别 ROI 的结束坐标必须大于开始坐标")
        config.crop_top_ratio = 0.0
        self._channel_calibration_enabled = bool(values.get("channel_calibration_enabled", False))
        self._channel_width_um = float(values.get("channel_width_um", 430.0))
        if self._channel_width_um <= 0.0:
            raise ValueError("管道内宽必须大于 0 μm")
        if self._channel_calibration_enabled and not config.enabled:
            raise ValueError("启用 430 μm 管道标定前必须先启用并框选 ROI")
        pipeline.channel_region_detector.reset()
        self._log(
            f"[VISION][ROI] enabled={config.enabled} "
            f"channel_region_enabled={channel_region.enabled} "
            f"channel_region_samples={channel_region.sample_frames} "
            f"x={config.x_start_ratio:.3f}-{config.x_end_ratio:.3f} "
            f"y={config.y_start_ratio:.3f}-{config.y_end_ratio:.3f}"
        )

    def _reset_channel_calibration(self) -> None:
        self._channel_width_px = None
        self._channel_calibration_confidence = 0.0
        self._channel_calibration_attempts = 0
        self._channel_width_samples.clear()
        self._pixel_to_micron = self._configured_pixel_to_micron
        if self._channel_calibration_enabled:
            self._channel_calibration_status = "collecting"
            self._channel_calibration_reason = f"正在从 ROI 中拟合 {self._channel_width_um:.1f} μm 管道内壁"
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
                self._channel_width_px = float(selected_width)
                self._channel_calibration_confidence = 1.0
                self._channel_calibration_status = "calibrated"
                self._channel_calibration_reason = (
                    f"采用用户选择的两条管壁：{self._channel_width_um:.1f} μm / "
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
        preferred = self._ensure_pipeline().detector.calibrate_preferred_radius(radii)
        brightness = sorted(item[1] for item in stats)
        noise = sorted(item[2] for item in stats)
        center_contrasts = sorted(item[3] for item in stats)
        ring_contrasts = sorted(item[4] for item in stats)
        median = lambda values: float(values[len(values) // 2]) if values else 0.0
        center_median = median(center_contrasts)
        ring_median = median(ring_contrasts)
        polarity = "亮心暗边" if center_median >= 0.0 else "暗心亮边"
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
            pending_frames = (
                int(self._frame_queue.qsize()) * MOTION_WINDOW_FRAMES
                + (MOTION_WINDOW_FRAMES if self._processing_busy else 0)
                + len(self._capture_batch)
            )
            if capture_fps <= 0.0:
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
        channel_width_um: float = 430.0,
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
            self._average_droplet_speed_um_s = None
            self._speed_sample_count = 0
            self._droplet_generation_rate_hz = 0.0
            self._last_motion_frame_id = 0
            self._stop_event.clear()
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
            self._try_channel_calibration(frame)
        result = self._ensure_pipeline().process_frame(frame, timestamp=timestamp)
        observed_ids = {int(track_id) for track_id, _ in result.tracking.matched_pairs}
        observed_ids.update(int(track_id) for track_id in result.tracking.new_track_ids)
        with self._lock:
            sample_time = time.monotonic()
            calibration_frame = self._ensure_pipeline().rectify_selected_channel(frame)
            if calibration_frame is None:
                calibration_frame = frame
            if self._ensure_pipeline().config.roi.enabled and not self._ensure_pipeline().config.roi.wall_lines:
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
            float(timestamp or result.timestamp),
            new_cross,
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
        frame_meta = self._frame_metadata.get(resolved_frame_id, {})
        return RecognitionSnapshot(
            frame_droplet_count=active_count,
            total_droplet_count=total_count,
            new_crossing_count=new_cross,
            avg_diameter=frame_avg_diameter,
            single_cell_rate=float(control.frame_single_cell_rate or 0.0),
            valid_for_control=bool(result.metrics.control.valid_for_control and has_droplet),
            timestamp=float(timestamp or time.time()),
            reason=control_reason,
            droplet_count=total_count,
            active_droplet_count=active_count,
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
            capture_monotonic=float(frame_meta.get("capture_monotonic", 0.0) or 0.0),
            hardware_frame_id=int(frame_meta.get("hardware_frame_id", 0) or 0),
            hardware_timestamp=float(frame_meta.get("hardware_timestamp", 0.0) or 0.0),
            uniformity_valid=bool(control.uniformity_valid),
            uniformity_status=str(control.uniformity_status or ""),
            uniformity_reason=str(control.uniformity_reason or ""),
            control_period_id=int(control.period_id),
            motion_window_frames=len(self._motion_observations),
            average_droplet_speed_um_s=self._average_droplet_speed_um_s,
            speed_sample_count=self._speed_sample_count,
            droplet_generation_rate_hz=self._droplet_generation_rate_hz,
            pixel_to_micron=scale,
            scale_source=(
                "channel_430um"
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
            if completed_period > self._last_gallery_period_id:
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
            target_period = completed_period + 1
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
            for track_id in sorted(valid_ids):
                track = tracks.get(track_id)
                if track is None:
                    continue
                radius = float(track.metadata.get("observed_radius", track.radius))
                if radius <= 1.0:
                    continue
                cx, cy = float(track.position[0]), float(track.position[1])
                diameter_um = radius * 2.0 * float(self._pixel_to_micron)
                valid_diameters_um.append(diameter_um)
                color = (0, 165, 255) if track_id in crossed_ids else (40, 220, 70)
                thickness = 3 if track_id in crossed_ids else 2
                cv2.circle(
                    annotated,
                    (int(round(cx)), int(round(cy))),
                    max(2, int(round(radius))),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    f"ID {track_id}  {diameter_um:.1f}um",
                    (
                        max(2, int(round(cx - radius))),
                        max(16, int(round(cy - radius - 5))),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1,
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
            header = f"Frame {resolved_frame_id} | valid {len(valid_ids)} | crossed {len(crossed_ids)}"
            cv2.rectangle(annotated, (0, 0), (min(frame_w - 1, 340), 25), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                header,
                (7, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
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
                    "average_diameter_um": (
                        float(np.mean(valid_diameters_um)) if valid_diameters_um else None
                    ),
                    "image_jpeg_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
                    "width": int(frame_w),
                    "height": int(frame_h),
                }
            )

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
        self._droplet_generation_rate_hz = len(self._crossing_times) / GENERATION_RATE_WINDOW_S

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
            self._submit_processing_frame(frame_id, timestamp, frame)

    def _process_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                batch = self._frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                with self._lock:
                    self._processing_busy = True
                batch_snapshot = None
                for frame_id, timestamp, frame in batch:
                    processing_started = time.perf_counter()
                    batch_snapshot = self._snapshot_from_frame(
                        frame,
                        frame_id=frame_id,
                        timestamp=timestamp,
                    )
                    with self._lock:
                        completed_at = time.monotonic()
                        frame_meta = self._frame_metadata.get(int(frame_id), {})
                        capture_monotonic = float(frame_meta.get("capture_monotonic", 0.0) or 0.0)
                        self._algorithm_processing_ms = max(0.0, (time.perf_counter() - processing_started) * 1000.0)
                        self._processing_times.append(completed_at)
                        self._processed_frame_count += 1
                        self._recognition_latency_ms = (
                            max(0.0, (completed_at - capture_monotonic) * 1000.0)
                            if capture_monotonic > 0.0
                            else self._algorithm_processing_ms
                        )
                    # Candidate scoring contains Python loops. Yield briefly so
                    # the preview producer and Tk main loop are not starved by
                    # a five-frame analysis burst.
                    time.sleep(0.002)
                # A five-frame window is one analysis transaction. Publish only
                # after all frames have updated the pipeline's accumulated data.
                if batch_snapshot is not None:
                    with self._recognition_condition:
                        preview = self._latest_preview
                        self._latest = replace(
                            batch_snapshot,
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
            finally:
                with self._lock:
                    self._processing_busy = False

    def _publish_video_frame(self, frame, frame_id: int, timestamp: float) -> None:
        display_frame = self._ensure_pipeline().rectify_selected_channel(frame)
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
        now = time.monotonic()
        if self._capture_batch and int(frame_id) != int(self._capture_batch[-1][0]) + 1:
            self._capture_batch = []
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
        if len(self._capture_batch) < MOTION_WINDOW_FRAMES:
            return
        item = self._capture_batch
        self._capture_batch = []
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
                snapshot = self._snapshot_from_frame(frame)
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
