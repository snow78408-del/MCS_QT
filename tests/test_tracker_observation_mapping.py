from __future__ import annotations

import unittest

import numpy as np

from backend.vision.config import TrackerConfig
from backend.vision.kalman_tracker import KalmanTracker
from backend.vision.nearest_tracker import NearestTracker


class TrackerObservationMappingTests(unittest.TestCase):
    def test_new_tracks_keep_current_detection_mapping(self) -> None:
        for tracker_type in (NearestTracker, KalmanTracker):
            with self.subTest(tracker=tracker_type.__name__):
                tracker = tracker_type(TrackerConfig())
                result = tracker.update(
                    [np.array([120.0, 80.0], dtype=np.float32)],
                    [25.0],
                )

                self.assertEqual(result.new_track_ids, [1])
                self.assertEqual(result.matched_pairs, [(1, 0)])

    def test_tracks_require_two_hits_inside_three_frame_window(self) -> None:
        for tracker_type in (NearestTracker, KalmanTracker):
            with self.subTest(tracker=tracker_type.__name__):
                tracker = tracker_type(TrackerConfig(confirmation_window=3, confirmation_min_hits=2))
                first = tracker.update(
                    [np.array([120.0, 80.0], dtype=np.float32)],
                    [25.0],
                )
                first_confirmed = first.active_tracks[0].is_confirmed
                missed = tracker.update([], [])
                missed_confirmed = missed.active_tracks[0].is_confirmed
                confirmed = tracker.update(
                    [np.array([121.0, 80.0], dtype=np.float32)],
                    [25.0],
                )

                self.assertFalse(first_confirmed)
                self.assertFalse(missed_confirmed)
                self.assertTrue(confirmed.active_tracks[0].is_confirmed)
                self.assertEqual(confirmed.active_tracks[0].observation_history, [True, False, True])


if __name__ == "__main__":
    unittest.main()
