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
        track.position = np.array([140, 50])
        crossing = metrics.update(tracking, _beads([7]), 100, 200, timestamp=1.1)
        self.assertEqual(crossing.control.crossed_track_diameters,{7:51.0})
        empty = TrackingResult([], [], [], [7], 1)
        completed = metrics.update(empty, _beads([]), 100, 200, timestamp=1.51)

        self.assertEqual(completed.control.period_id, 1)
        self.assertEqual(completed.control.sample_size, 1)
        self.assertEqual(completed.control.frame_droplet_count, 1)
        self.assertAlmostEqual(completed.control.average_diameter or 0.0, 51.0)
        self.assertEqual(completed.control.frame_single_cell_rate, 100.0)

    def test_distinct_tracks_count_as_distinct_samples(self) -> None:
        metrics = MetricsCalculator(MetricsConfig(realtime_window_ms=500))
        track_a = DropletTrack(id=7, position=np.array([80, 50]), radius=25, age=5)
        track_b = DropletTrack(id=8, position=np.array([80, 50]), radius=30, age=5)
        tracking = TrackingResult([track_a, track_b], [(7, 0), (8, 1)], [], [], 2)

        metrics.update(tracking, _beads([7, 8]), 100, 200, timestamp=1.0)
        track_a.position = np.array([140, 50])
        track_b.position = np.array([140, 50])
        metrics.update(tracking, _beads([7, 8]), 100, 200, timestamp=1.1)
        empty = TrackingResult([], [], [], [7, 8], 2)
        result = metrics.update(empty, _beads([]), 100, 200, timestamp=1.51)

        self.assertEqual(result.control.sample_size, 2)
        self.assertEqual(result.control.frame_droplet_count, 2)
        self.assertAlmostEqual(result.control.average_diameter or 0.0, 55.0)

    def test_completed_control_periods_do_not_overlap(self) -> None:
        metrics = MetricsCalculator(MetricsConfig(realtime_window_ms=300))
        track_a = DropletTrack(id=7, position=np.array([80, 50]), radius=25, age=5)
        first = TrackingResult([track_a], [(7, 0)], [], [], 1)
        metrics.update(first, _beads([7]), 100, 200, timestamp=1.0)
        track_a.position = np.array([140, 50])
        metrics.update(first, _beads([7]), 100, 200, timestamp=1.1)

        empty = TrackingResult([], [], [], [7], 1)
        first_period = metrics.update(empty, _beads([]), 100, 200, timestamp=1.31)
        self.assertEqual(first_period.control.frame_droplet_count, 1)

        track_b = DropletTrack(id=8, position=np.array([80, 50]), radius=30, age=5)
        second = TrackingResult([track_b], [(8, 0)], [], [], 2)
        metrics.update(second, _beads([8]), 100, 200, timestamp=1.4)
        track_b.position = np.array([140, 50])
        metrics.update(second, _beads([8]), 100, 200, timestamp=1.5)
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

    def test_false_size_outlier_does_not_freeze_diameter_feedback(self) -> None:
        config = MetricsConfig(
            realtime_window_ms=500,
            max_diameter_cv_for_control=25.0,
        )
        metrics = MetricsCalculator(config)
        radii = [24.5, 25.0, 25.5, 24.8, 70.0]
        tracks = [
            DropletTrack(id=index, position=np.array([80, 50]), radius=radius, age=5)
            for index, radius in enumerate(radii, start=1)
        ]
        tracking = TrackingResult(
            tracks,
            [(track.id, index) for index, track in enumerate(tracks)],
            [],
            [],
            len(tracks),
        )
        track_ids = [track.id for track in tracks]
        metrics.update(tracking, _beads(track_ids), 100, 200, timestamp=1.0)
        for track in tracks:
            track.position = np.array([140, 50])
        metrics.update(tracking, _beads(track_ids), 100, 200, timestamp=1.1)
        empty = TrackingResult([], [], [], track_ids, len(tracks))
        completed = metrics.update(empty, _beads([]), 100, 200, timestamp=1.51)

        self.assertEqual(completed.control.frame_droplet_count, 5)
        self.assertEqual(completed.control.sample_size, 4)
        self.assertTrue(completed.control.valid_for_control)
        self.assertLess(completed.control.frame_diameter_cv or 100.0, 25.0)
        self.assertAlmostEqual(completed.control.average_diameter or 0.0, 49.9, places=1)

    def test_high_diameter_cv_is_reported_but_does_not_freeze_feedback(self) -> None:
        config = MetricsConfig(
            realtime_window_ms=500,
            max_diameter_cv_for_control=1.0,
        )
        metrics = MetricsCalculator(config)
        radii = [10.0, 20.0, 30.0, 40.0]
        tracks = [
            DropletTrack(id=index, position=np.array([80, 50]), radius=radius, age=5)
            for index, radius in enumerate(radii, start=1)
        ]
        tracking = TrackingResult(
            tracks,
            [(track.id, index) for index, track in enumerate(tracks)],
            [],
            [],
            len(tracks),
        )
        track_ids = [track.id for track in tracks]
        metrics.update(tracking, _beads(track_ids), 100, 200, timestamp=1.0)
        for track in tracks:
            track.position = np.array([140, 50])
        metrics.update(tracking, _beads(track_ids), 100, 200, timestamp=1.1)
        empty = TrackingResult([], [], [], track_ids, len(tracks))
        completed = metrics.update(empty, _beads([]), 100, 200, timestamp=1.51)

        self.assertGreater(completed.control.frame_diameter_cv or 0.0, 25.0)
        self.assertTrue(completed.control.valid_for_control)
        self.assertEqual(completed.control.reason, "ok")
        self.assertAlmostEqual(completed.control.average_diameter or 0.0, 50.0)

    def test_track_seen_but_not_crossed_is_not_a_pid_sample(self) -> None:
        metrics = MetricsCalculator(MetricsConfig(realtime_window_ms=300))
        track = DropletTrack(id=7, position=np.array([60, 50]), radius=25, age=5)
        tracking = TrackingResult([track], [(7, 0)], [], [], 1)
        metrics.update(tracking, _beads([7]), 100, 200, timestamp=1.0)
        track.position = np.array([90, 50])
        metrics.update(tracking, _beads([7]), 100, 200, timestamp=1.1)
        completed = metrics.update(
            TrackingResult([], [], [], [7], 0),
            _beads([]),
            100,
            200,
            timestamp=1.31,
        )

        self.assertEqual(completed.control.frame_droplet_count, 0)
        self.assertEqual(completed.control.sample_size, 0)
        self.assertFalse(completed.control.valid_for_control)


if __name__ == "__main__":
    unittest.main()
