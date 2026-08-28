from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

try:
    from .algorithms import get_algorithm
    from .bead_counter import BeadCounter, BeadResult
    from .config import PipelineConfig
    from .detector import DetectionResult
    from .kalman_tracker import KalmanTracker
    from .metrics import MetricsCalculator, MetricsResult
    from .nearest_tracker import NearestTracker
    from .tracker import BaseTracker, TrackingResult
except ImportError:
    from algorithms import get_algorithm
    from bead_counter import BeadCounter, BeadResult
    from config import PipelineConfig
    from detector import DetectionResult
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


class VisionPipeline:
    def __init__(self, config: PipelineConfig, logger: Callable[[str], None] | None = None) -> None:
        self.config = config
        self._log = logger or (lambda _msg: None)
        self.algorithm_id = "hybrid_v1"
        self.algorithm_parameters = {}
        self.detector = get_algorithm(self.algorithm_id).detector_factory(config.detector, config.debug)
        self.bead_counter = BeadCounter(config.beads, config.debug)
        self.metrics = MetricsCalculator(config.metrics, logger=self._log)
        self.tracker: BaseTracker = self._build_tracker(config)
        self._frame_index = 0

    def configure_algorithm(self, plugin_id: str, parameters: dict | None = None) -> None:
        plugin = get_algorithm(plugin_id)
        algorithm_config = plugin.build_config(parameters)
        self.algorithm_id = plugin.plugin_id
        self.algorithm_parameters = plugin.serialize_config(algorithm_config)
        self.detector = plugin.detector_factory(algorithm_config, self.config.debug)
        # Preserve the legacy public config field for the built-in detector.
        if plugin.plugin_id == "hybrid_v1":
            self.config.detector = algorithm_config

    def configure_expected_diameter(self, diameter_um: float, pixel_to_micron: float) -> None:
        callback = getattr(self.detector, "configure_expected_diameter", None)
        if callable(callback):
            callback(diameter_um, pixel_to_micron)

    def _build_tracker(self, config: PipelineConfig) -> BaseTracker:
        if config.tracker.tracker_type == "kalman":
            return KalmanTracker(config.tracker)
        return NearestTracker(config.tracker)

    def reset(self) -> None:
        reset_detector = getattr(self.detector, "reset_adaptive_size", None)
        if callable(reset_detector):
            reset_detector()
        self.tracker.reset()
        self.metrics.reset()
        self._frame_index = 0

    def process_frame(self, frame: np.ndarray, *, timestamp: float | None = None) -> VisionResult:
        self._frame_index += 1
        roi_frame, roi_offset = self._apply_roi(frame)

        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY) if roi_frame.ndim == 3 else roi_frame
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

    def _apply_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        rectified = self.rectify_selected_channel(frame)
        if rectified is not None:
            return rectified, (0, 0)
        if not self.config.roi.enabled:
            return frame.copy(), (0, 0)

        h, w = frame.shape[:2]
        x0, x1, y0, y1, crop_top = self.config.roi.resolve(w, h)
        cropped = frame[y0:y1, x0:x1]
        if crop_top > 0:
            cropped = cropped[crop_top:, :]
        return cropped, (x0, y0 + crop_top)

    def rectify_selected_channel(self, frame: np.ndarray) -> np.ndarray | None:
        wall_lines = list(getattr(self.config.roi, "wall_lines", []) or [])
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
