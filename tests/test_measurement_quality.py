from __future__ import annotations

import numpy as np
import pytest

from backend.orchestrator.vision_adapter import PipelineVisionService
from backend.vision.bead_counter import BeadResult
from backend.vision.config import MetricsConfig
from backend.vision.metrics import MetricsCalculator
from backend.vision.tracker import DropletTrack, TrackingResult


def empty_beads():
    return BeadResult([], 0, np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1), np.uint8))


@pytest.mark.parametrize("value", [None, 0, -1, float("nan"), float("inf"), 101])
def test_acquisition_time_is_never_replaced_by_processing_time(monkeypatch, value):
    monkeypatch.setattr("backend.orchestrator.vision_adapter.time.monotonic", lambda: 100)
    with pytest.raises(ValueError, match="acquisition time"):
        PipelineVisionService._acquisition_time({"capture_monotonic": value})
    assert PipelineVisionService._acquisition_time({"capture_monotonic": 90}) == 90


@pytest.mark.parametrize("bad_time", [1.0, .9, float("nan"), float("inf")])
def test_duplicate_or_invalid_frame_does_not_mutate_statistics(bad_time):
    metrics = MetricsCalculator(MetricsConfig(realtime_window_ms=500))
    track = DropletTrack(id=1, position=np.array([80, 50]), radius=25, age=5)
    tracking = TrackingResult([track], [(1, 0)], [], [], 1)
    metrics.update(tracking, empty_beads(), 100, 200, 1.0)
    track.position = np.array([140, 50])
    with pytest.raises(ValueError):
        metrics.update(tracking, empty_beads(), 100, 200, bad_time)
    assert metrics._total_counted == 0
    metrics.update(tracking, empty_beads(), 100, 200, 1.1)
    result = metrics.update(TrackingResult([], [], [], [1], 1), empty_beads(), 100, 200, 1.51).control
    assert result.sample_size == 1
    assert result.sample_start_time == 1.0
    assert result.sample_end_time == 1.1


@pytest.mark.parametrize("radius", [float("nan"), float("inf"), -1, 0])
def test_passage_without_valid_size_is_not_a_size_sample(radius):
    metrics = MetricsCalculator(MetricsConfig(realtime_window_ms=500))
    track = DropletTrack(id=1, position=np.array([80, 50]), radius=radius, age=5)
    tracking = TrackingResult([track], [(1, 0)], [], [], 1)
    metrics.update(tracking, empty_beads(), 100, 200, 1.0)
    track.position = np.array([140, 50])
    metrics.update(tracking, empty_beads(), 100, 200, 1.1)
    result = metrics.update(TrackingResult([], [], [], [1], 1), empty_beads(), 100, 200, 1.51).control
    assert result.window_passage_count == result.total_droplet_count == 1
    assert result.current_frame_droplet_count == result.sample_size == 0
    assert result.average_diameter is None
    assert not result.valid_for_control


def test_discarded_old_track_measurements_do_not_extend_feedback_wait():
    metrics = MetricsCalculator(MetricsConfig(realtime_window_ms=500, diameter_samples_per_track=2))
    track = DropletTrack(id=1, position=np.array([80, 50]), radius=25, age=5)
    tracking = TrackingResult([track], [(1, 0)], [], [], 1)
    for timestamp, x in [(1.0, 80), (1.1, 90), (1.2, 100), (1.3, 140)]:
        track.position = np.array([x, 50])
        metrics.update(tracking, empty_beads(), 100, 200, timestamp)
    result = metrics.update(TrackingResult([], [], [], [1], 1), empty_beads(), 100, 200, 1.51).control
    assert result.sample_start_time == 1.2
    assert result.sample_end_time == 1.3
    assert result.sample_size == 1
