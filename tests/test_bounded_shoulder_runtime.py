"""Weak outer shoulders must reach both live generation entry points."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.vision.capsule_profile import bounded_shoulder_intervals
from backend.vision.config import DebugConfig, DetectorConfig, PipelineConfig
from backend.vision.detector import DropletDetector
from backend.vision.pipeline import VisionPipeline
from backend.vision.rectified_measurement import (
    FrameEvidence,
    ScaleEvidence,
    measure_generation_plugs,
)


def _weak_frame(spans: list[tuple[int, int]]) -> np.ndarray:
    """A bright centre and paired dark shoulders with a visible carrier gap."""
    band = np.full((53, 570), 48.0, dtype=np.float32)
    band[11:19] = 50.0
    band[34:43] = 50.0
    band[23:30] = 45.0
    for left, right in spans:
        band[11:19, left:right] = 46.0
        band[34:43, left:right] = 46.0
        band[23:30, left:right] = 52.0
    return np.uint8(np.rint(cv2.GaussianBlur(band, (0, 0), 1.2)))


def _camera_frame(spans: list[tuple[int, int]]) -> tuple[np.ndarray, list[dict]]:
    frame = np.full((240, 720), 48, dtype=np.uint8)
    frame[100:153, 100:670] = _weak_frame(spans)
    lines = [
        {"x1": 100 / 720, "y1": row / 240,
         "x2": 669 / 720, "y2": row / 240}
        for row in (100, 126)
    ]
    return frame, lines


def _detector() -> DropletDetector:
    return DropletDetector(
        DetectorConfig(measurement_mode="generation_plug",
                       generation_channel_height_um=26.0,
                       generation_channel_width_um=26.0), DebugConfig())


def test_paired_outer_shoulders_keep_full_length_and_reject_cut_objects() -> None:
    spans = [(22, 109), (163, 262), (312, 408), (465, 550)]
    assert bounded_shoulder_intervals(_weak_frame(spans), 52.0) == [
        (21, 110), (162, 263), (311, 409), (464, 551)]
    assert bounded_shoulder_intervals(_weak_frame([]), 52.0) == []
    assert bounded_shoulder_intervals(_weak_frame([(465, 570)]), 52.0) == []
    assert bounded_shoulder_intervals(_weak_frame([(10, 55)]), 52.0) == []


@pytest.mark.parametrize("reverse_walls", [False, True])
def test_live_pipeline_uses_outer_band_but_keeps_inner_wall_width(reverse_walls: bool) -> None:
    frame, walls = _camera_frame([(22, 109), (163, 262), (312, 408), (465, 550)])
    if reverse_walls:
        walls = [{"x1": wall["x2"], "y1": wall["y2"],
                  "x2": wall["x1"], "y2": wall["y1"]} for wall in walls]
    config = PipelineConfig()
    config.detector.measurement_mode = "generation_plug"
    config.detector.generation_channel_height_um = 26.0
    config.detector.generation_channel_width_um = 26.0
    pipeline = VisionPipeline(config)
    result = pipeline.process_frame(frame, timestamp=1.0, current_wall_lines=walls)

    assert result.analysis_frame.shape == (26, 569)
    assert result.detections.plug_lengths_px == ([87.0, 98.0, 101.0, 89.0]
                                                  if reverse_walls else [89.0, 101.0, 98.0, 87.0])
    assert result.detections.diameter_valid == [True] * 4
    assert result.detection_geometry["channel_width_px"] == 25.0
    reported = result.detection_geometry["detector_reported"]
    assert reported["interval_source"] == "bounded_raw_shoulders"
    assert reported["outer_observation_used"] is True


def test_physical_measurement_uses_same_candidates_without_inventing_scale() -> None:
    frame, walls = _camera_frame([(22, 109), (163, 262), (312, 408), (465, 550)])
    result = measure_generation_plugs(
        frame, detector=_detector(), wall_lines=walls,
        scale=ScaleEvidence(None, "unknown", False),
        frame_evidence=FrameEvidence(1, 2, 10.0, 1), duct_depth_um=None,
    )

    assert result.plug_lengths_px == (89.0, 101.0, 98.0, 87.0)
    assert result.complete_plug_flags == (True,) * 4
    assert result.detection_trace["interval_source"] == "bounded_raw_shoulders"
    assert result.axes.transverse_index_span == 25.0
    assert result.valid is False
    assert result.plug_lengths_um == ()
