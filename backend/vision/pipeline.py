from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

try:
    from .bead_counter import BeadCounter, BeadResult
    from .channel_region import ChannelRegionDetector, ChannelRegionResult
    from .config import ChannelRegionConfig, DetectorConfig, PipelineConfig
    from .detector import DetectionResult, DropletDetector
    from .kalman_tracker import KalmanTracker
    from .metrics import MetricsCalculator, MetricsResult
    from .nearest_tracker import NearestTracker
    from .tracker import BaseTracker, TrackingResult
except ImportError:
    from bead_counter import BeadCounter, BeadResult
    from channel_region import ChannelRegionDetector, ChannelRegionResult
    from config import ChannelRegionConfig, DetectorConfig, PipelineConfig
    from detector import DetectionResult, DropletDetector
    from kalman_tracker import KalmanTracker
    from metrics import MetricsCalculator, MetricsResult
    from nearest_tracker import NearestTracker
    from tracker import BaseTracker, TrackingResult


@dataclass
class VisionResult:
    frame_index: int
    timestamp: float
    detections: DetectionResult
    tracking: TrackingResult
    beads: BeadResult
    metrics: MetricsResult
    annotated_frame: np.ndarray
    analysis_frame: np.ndarray
    channel_region: ChannelRegionResult


class VisionPipeline:
    def __init__(self, config: PipelineConfig, logger: Callable[[str], None] | None = None) -> None:
        self.config = config
        self._log = logger or (lambda _msg: None)
        self.channel_region_detector = ChannelRegionDetector(config.channel_region)
        self.detector = DropletDetector(config.detector, config.debug)
        self.bead_counter = BeadCounter(config.beads, config.debug)
        self.metrics = MetricsCalculator(config.metrics, logger=self._log)
        self.tracker: BaseTracker = self._build_tracker(config)
        self._frame_index = 0

    def configure_expected_diameter(self, diameter_um: float, pixel_to_micron: float) -> None:
        self.detector.configure_expected_diameter(diameter_um, pixel_to_micron)

    def apply_tuning_config(
        self,
        detector_config: DetectorConfig,
        channel_region_config: ChannelRegionConfig,
    ) -> None:
        """Apply saved tuning to subsequent frames without resetting run metrics."""
        detector = DetectorConfig(**vars(detector_config))
        channel_region = ChannelRegionConfig(**vars(channel_region_config))
        # Build the detector before publishing it so invalid input cannot leave
        # the live pipeline half-updated.
        replacement_detector = DropletDetector(detector, self.config.debug)
        self.config.detector = detector
        self.config.channel_region = channel_region
        self.detector = replacement_detector
        # A resolved channel region defines the track coordinate system. Keep
        # that result for the current run and use the new calibration settings
        # at the next reset; restarting it mid-period could clear tracking and
        # cumulative metrics. During startup collection it is safe to restart.
        self.channel_region_detector.config = channel_region
        if self._frame_index == 0 or self.channel_region_detector.result.status == "collecting":
            self.channel_region_detector.reset()

    def _build_tracker(self, config: PipelineConfig) -> BaseTracker:
        if config.tracker.tracker_type == "kalman":
            return KalmanTracker(config.tracker)
        return NearestTracker(config.tracker)

    def reset(self) -> None:
        self.channel_region_detector.reset()
        self.detector.reset_adaptive_size()
        self.tracker.reset()
        self.metrics.reset()
        self._frame_index = 0

    def process_frame(self, frame: np.ndarray, *, timestamp: float | None = None) -> VisionResult:
        self._frame_index += 1
        channel_region = self._update_channel_region(frame)
        roi_frame, roi_offset = self._apply_roi(frame)

        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY) if roi_frame.ndim == 3 else roi_frame
        if channel_region.status == "collecting":
            # The channel-region check is a true prerequisite: startup samples
            # must not produce full-frame detections or PID-valid measurements
            # before the effective channel has been established.
            detections = DetectionResult(
                centers=[],
                radii=[],
                debug_image=np.empty((0, 0, 3), dtype=np.uint8),
                helper_mask=np.zeros(gray.shape[:2], dtype=np.uint8),
                diameter_valid=[],
            )
        else:
            detections = self.detector.detect(gray)
        tracking = self.tracker.update(detections.centers, detections.radii, timestamp=timestamp)
        self._align_tracks_to_current_detections(tracking, detections)
        confirmed_observed_tracks = [
            track
            for track in tracking.active_tracks
            if "detection_index" in track.metadata
            and bool(track.is_confirmed)
            and int(track.age) >= int(self.config.metrics.min_track_age_for_count)
        ]
        beads = self.bead_counter.count(confirmed_observed_tracks, gray, detections.helper_mask)
        metrics = self.metrics.update(
            tracking,
            beads,
            frame_height=int(roi_frame.shape[0]),
            frame_width=int(roi_frame.shape[1]),
            timestamp=timestamp,
        )
        # The realtime system no longer renders recognition guides. Keep the
        # source reference so analysis does not allocate another full frame.
        annotated = frame

        return VisionResult(
            frame_index=self._frame_index,
            timestamp=float(timestamp if timestamp is not None else time()),
            detections=detections,
            tracking=tracking,
            beads=beads,
            metrics=metrics,
            annotated_frame=annotated,
            analysis_frame=roi_frame,
            channel_region=channel_region,
        )

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = False,
        max_frames: Optional[int] = None,
    ) -> List[VisionResult]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        return self._process_capture(cap, fps=fps, output_path=output_path, display=display, max_frames=max_frames)

    def process_camera(
        self,
        camera_index: int = 0,
        output_path: Optional[str] = None,
        display: bool = False,
        max_frames: Optional[int] = None,
    ) -> List[VisionResult]:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")
        return self._process_capture(cap, fps=30.0, output_path=output_path, display=display, max_frames=max_frames)

    def _process_capture(
        self,
        cap: cv2.VideoCapture,
        fps: float,
        output_path: Optional[str],
        display: bool,
        max_frames: Optional[int],
    ) -> List[VisionResult]:
        results: List[VisionResult] = []
        writer: Optional[cv2.VideoWriter] = None

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                result = self.process_frame(frame)
                results.append(result)

                if output_path:
                    if writer is None:
                        h, w = result.annotated_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 30.0, (w, h))
                    writer.write(result.annotated_frame)

                if display:
                    cv2.imshow("VisionPipeline", result.annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if max_frames is not None and len(results) >= max_frames:
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        return results

    def current_channel_region(self) -> ChannelRegionResult:
        manual_lines = list(getattr(self.config.roi, "wall_lines", []) or [])
        if len(manual_lines) == 2:
            return ChannelRegionResult("manual", 1.0, "采用用户选择的两条管壁", manual_lines)
        if self.config.roi.enabled and bool(getattr(self.config.roi, "user_defined", True)):
            return ChannelRegionResult("manual", 1.0, "采用用户设置的矩形 ROI")
        return self.channel_region_detector.result

    def _update_channel_region(self, frame: np.ndarray) -> ChannelRegionResult:
        current = self.current_channel_region()
        if current.status == "manual":
            return current
        previous_status = self.channel_region_detector.result.status
        result = self.channel_region_detector.add_frame(frame)
        if result.status != previous_status and result.status in {"calibrated", "fallback"}:
            self._log(
                "[VISION][CHANNEL_REGION] "
                f"status={result.status} confidence={result.confidence:.3f} reason={result.reason}"
            )
            if result.status == "calibrated":
                # The coordinate system changes from full-frame to rectified
                # ROI once startup calibration completes. Old tracks must not
                # be matched across that boundary.
                self.tracker.reset()
                self.metrics.reset()
        return result

    def _effective_wall_lines(self) -> list[dict[str, object]]:
        manual_lines = list(getattr(self.config.roi, "wall_lines", []) or [])
        if len(manual_lines) == 2:
            return manual_lines
        result = self.channel_region_detector.result
        return list(result.wall_lines) if result.status == "calibrated" else []

    def _apply_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        rectified = self.rectify_selected_channel(frame)
        if rectified is not None:
            return rectified, (0, 0)
        if not self.config.roi.enabled or not bool(getattr(self.config.roi, "user_defined", True)):
            return frame.copy(), (0, 0)

        h, w = frame.shape[:2]
        x0, x1, y0, y1, crop_top = self.config.roi.resolve(w, h)
        cropped = frame[y0:y1, x0:x1]
        if crop_top > 0:
            cropped = cropped[crop_top:, :]
        return cropped, (x0, y0 + crop_top)

    def rectify_selected_channel(self, frame: np.ndarray) -> np.ndarray | None:
        wall_lines = self._effective_wall_lines()
        if len(wall_lines) != 2:
            return None
        try:
            from .rectified_roi import rectify_channel_frame
        except ImportError:
            from rectified_roi import rectify_channel_frame
        return rectify_channel_frame(frame, wall_lines)

    @staticmethod
    def _align_tracks_to_current_detections(
        tracking: TrackingResult,
        detections: DetectionResult,
    ) -> None:
        detection_by_track = {
            int(track_id): int(detection_index)
            for track_id, detection_index in tracking.matched_pairs
        }
        for track in tracking.active_tracks:
            detection_index = detection_by_track.get(int(track.id))
            if detection_index is None:
                track.metadata.pop("detection_index", None)
                track.metadata.pop("observed_radius", None)
                track.metadata.pop("diameter_valid", None)
                continue
            if detection_index < 0 or detection_index >= len(detections.centers):
                continue
            track.position = np.asarray(detections.centers[detection_index], dtype=np.float32)
            track.metadata["detection_index"] = float(detection_index)
            diameter_valid = (
                bool(detections.diameter_valid[detection_index])
                if detection_index < len(detections.diameter_valid)
                else True
            )
            track.metadata["diameter_valid"] = 1.0 if diameter_valid else 0.0
            if diameter_valid:
                track.metadata["observed_radius"] = float(detections.radii[detection_index])
            else:
                track.metadata.pop("observed_radius", None)
