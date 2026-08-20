from __future__ import annotations

import unittest

import numpy as np

from backend.vision.bead_counter import BeadResult, DropletBead
from backend.vision.config import MetricsConfig
from backend.vision.metrics import MetricsCalculator
from backend.vision.tracker import DropletTrack, TrackingResult


def _beads(track_ids: list[int]) -> BeadResult:
    return BeadResult(
        droplets=[DropletBead(droplet_id=track_id, bead_count=1) for track_id in track_ids],
        total_beads=len(track_ids),
        debug_image=np.zeros((1, 1, 3), dtype=np.uint8),
        candidate_mask=np.zeros((1, 1), dtype=np.uint8),
    )


class MetricsTrackWindowTests(unittest.TestCase):
    def test_repeated_frames_of_one_track_count_as_one_sample(self) -> None:
        metrics = MetricsCalculator(MetricsConfig(realtime_window_ms=500))
        track = DropletTrack(id=7, position=np.array([100, 50]), radius=25, age=5)
        tracking = TrackingResult([track], [(7, 0)], [], [], 1)

        metrics.update(tracking, _beads([7]), 100, 200, timestamp=1.0)
        track.radius = 26
        metrics.update(tracking, _beads([7]), 100, 200, timestamp=1.1)
        empty = TrackingResult([], [], [], [7], 1)
        completed = metrics.update(empty, _beads([]), 100, 200, timestamp=1.51)

        self.assertEqual(completed.control.period_id, 1)
        self.assertEqual(completed.control.sample_size, 1)
        self.assertEqual(completed.control.frame_droplet_count, 1)
        self.assertAlmostEqual(completed.control.average_diameter or 0.0, 51.0)
        self.assertEqual(completed.control.frame_single_cell_rate, 100.0)

    def test_distinct_tracks_count_as_distinct_samples(self) -> None:
        metrics = MetricsCalculator(MetricsConfig(realtime_window_ms=500))
        track_a = DropletTrack(id=7, position=np.array([100, 50]), radius=25, age=5)
        track_b = DropletTrack(id=8, position=np.array([140, 50]), radius=30, age=5)
        tracking = TrackingResult([track_a, track_b], [(7, 0), (8, 1)], [], [], 2)

        metrics.update(tracking, _beads([7, 8]), 100, 200, timestamp=1.0)
        empty = TrackingResult([], [], [], [7, 8], 2)
        result = metrics.update(empty, _beads([]), 100, 200, timestamp=1.51)

        self.assertEqual(result.control.sample_size, 2)
        self.assertEqual(result.control.frame_droplet_count, 2)
        self.assertAlmostEqual(result.control.average_diameter or 0.0, 55.0)

    def test_completed_control_periods_do_not_overlap(self) -> None:
        metrics = MetricsCalculator(MetricsConfig(realtime_window_ms=300))
        track_a = DropletTrack(id=7, position=np.array([100, 50]), radius=25, age=5)
        first = TrackingResult([track_a], [(7, 0)], [], [], 1)
        metrics.update(first, _beads([7]), 100, 200, timestamp=1.0)

        empty = TrackingResult([], [], [], [7], 1)
        first_period = metrics.update(empty, _beads([]), 100, 200, timestamp=1.31)
        self.assertEqual(first_period.control.frame_droplet_count, 1)

        track_b = DropletTrack(id=8, position=np.array([140, 50]), radius=30, age=5)
        second = TrackingResult([track_b], [(8, 0)], [], [], 2)
        metrics.update(second, _beads([8]), 100, 200, timestamp=1.4)
        second_period = metrics.update(empty, _beads([]), 100, 200, timestamp=1.61)

        self.assertEqual(second_period.control.period_id, 2)
        self.assertEqual(second_period.control.frame_droplet_count, 1)
        self.assertAlmostEqual(second_period.control.average_diameter or 0.0, 60.0)

    def test_crossing_is_reported_in_exact_completed_period(self) -> None:
        config = MetricsConfig(realtime_window_ms=300, flow_axis="x", count_line_ratio=0.6)
        metrics = MetricsCalculator(config)
        track = DropletTrack(id=7, position=np.array([100, 50]), radius=25, age=5)
        tracking = TrackingResult([track], [(7, 0)], [], [], 1)
        metrics.update(tracking, _beads([7]), 100, 200, timestamp=1.0)
        track.position = np.array([140, 50])
        metrics.update(tracking, _beads([7]), 100, 200, timestamp=1.1)

        empty = TrackingResult([], [], [], [7], 1)
        completed = metrics.update(empty, _beads([]), 100, 200, timestamp=1.31)

        self.assertEqual(completed.control.new_crossing_count, 1)
        self.assertEqual(completed.control.total_droplet_count, 1)


if __name__ == "__main__":
    unittest.main()
