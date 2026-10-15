"""Bounded one-frame-per-interval diagnostics, without a video writer."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from backend.vision.config import DebugConfig, DetectorConfig
from backend.vision.detector import DropletDetector
from backend.vision.rectified_measurement import detect_rectified_generation_plugs
from backend.vision.rectified_roi import rectify_channel_frame
from backend.vision.measurement_statistics import CandidateFrameStatistics


class SampledFrameRecorder:
    def __init__(self, *, directory: Path, interval_s: float, walls: list) -> None:
        if not np.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("sample interval must be positive and finite")
        self.directory = Path(directory)
        self.interval_s = float(interval_s)
        self.walls = walls
        self.detector = DropletDetector(DetectorConfig(
            measurement_mode="generation_plug",
            generation_phase_mode="transverse_bounded"), DebugConfig())
        self.accepted = 0
        self._last_timestamp: float | None = None
        self._next_timestamp: float | None = None
        self._shape: tuple | None = None
        self._stream = None
        self.max_sample_gap_s = 0.0
        self.candidate_statistics = CandidateFrameStatistics()

    def prepare(self, frame: object) -> None:
        image = getattr(frame, "image", None)
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise ValueError("sample frame must be a uint8 image")
        self._shape = image.shape
        self.directory.mkdir(parents=True, exist_ok=True)
        # Verify storage before the lifecycle permits starting the pump.
        if not cv2.imwrite(str(self.directory / "pre_start.png"), image):
            raise OSError("cannot save pre-start frame")
        self._stream = (self.directory / "samples.ndjson").open("x", encoding="utf-8")

    def submit(self, frame: object) -> None:
        image = getattr(frame, "image", None)
        timestamp = float(getattr(frame, "host_monotonic_timestamp", 0.0) or 0.0)
        if not np.isfinite(timestamp) or timestamp <= 0:
            raise ValueError("sample requires a valid host timestamp")
        if self._stream is None:
            self.prepare(frame)
        if image is None or image.shape != self._shape:
            raise ValueError("sample image dimensions changed")
        if self._next_timestamp is not None and timestamp < self._next_timestamp:
            return
        roi = rectify_channel_frame(image, self.walls)
        if roi is None:
            raise ValueError("cannot rectify sampled frame")
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        _, trace = detect_rectified_generation_plugs(gray, detector=self.detector)
        name = f"sample_{self.accepted:06d}.png"
        if not cv2.imwrite(str(self.directory / name), image):
            raise OSError("cannot save sampled frame")
        record = {
            "sample_index": self.accepted, "host_monotonic_timestamp": timestamp,
            "hardware_frame_id": int(getattr(frame, "hardware_frame_id", 0) or 0),
            "image": name, "candidate_intervals": trace["selected_intervals"],
            "interval_source": trace.get("interval_source"),
            "endpoint_validated": False, "absence_confirmed": False,
            "phase_diagnostics": trace.get("phase_diagnostics", {}),
            "phase_size_rejected": trace.get("phase_size_rejected", []),
            "observation_status": self.candidate_statistics.observe(
                trace["selected_intervals"], width=gray.shape[1]),
        }
        self._stream.write(json.dumps(record) + "\n")
        self._stream.flush()
        if self._last_timestamp is not None:
            self.max_sample_gap_s = max(self.max_sample_gap_s, timestamp - self._last_timestamp)
        self._last_timestamp = timestamp
        # Skip missed sample slots; never burst or invent images for past times.
        self._next_timestamp = timestamp + self.interval_s
        self.accepted += 1

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def summary(self) -> dict:
        return {
            "sample_interval_s": self.interval_s, "sampled_frames": self.accepted,
            "max_sample_gap_s": self.max_sample_gap_s, "raw_video_saved": False,
            "sampling_premise_ok": self.accepted > 0,
            "sampling_use": "sparse_diagnostic_frames_not_continuous_flow_measurement",
            "measurement_accepted": False,
            "candidate_statistics": self.candidate_statistics.to_dict(),
        }
