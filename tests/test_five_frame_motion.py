from __future__ import annotations

import unittest
from collections import deque
from types import SimpleNamespace

from backend.orchestrator.vision_adapter import (
    MOTION_WINDOW_FRAMES,
    PipelineVisionService,
)


class FiveFrameMotionTests(unittest.TestCase):
    def test_five_consecutive_frames_produce_speed_and_generation_rate(self) -> None:
        service = PipelineVisionService.__new__(PipelineVisionService)
        service._motion_observations = deque(maxlen=MOTION_WINDOW_FRAMES)
        service._crossing_times = deque(maxlen=1000)
        service._pixel_to_micron = 2.0
        service._average_droplet_speed_um_s = None
        service._speed_sample_count = 0
        service._droplet_generation_rate_hz = 0.0
        service._last_motion_frame_id = 0
        pipeline = SimpleNamespace(config=SimpleNamespace(metrics=SimpleNamespace(flow_axis="x")))
        service._ensure_pipeline = lambda: pipeline

        for index in range(5):
            track = SimpleNamespace(id=7, position=(float(index), 12.0))
            tracking = SimpleNamespace(active_tracks=[track])
            service._update_motion_measurements(
                tracking,
                {7},
                timestamp=10.0 + index * 0.01,
                new_crossings=1 if index in {2, 4} else 0,
                frame_id=index + 1,
            )

        self.assertEqual(service._speed_sample_count, 1)
        self.assertAlmostEqual(service._average_droplet_speed_um_s or 0.0, 200.0)
        self.assertAlmostEqual(service._droplet_generation_rate_hz, 2.0)

    def test_frame_gap_restarts_the_five_frame_window(self) -> None:
        service = PipelineVisionService.__new__(PipelineVisionService)
        service._motion_observations = deque(maxlen=MOTION_WINDOW_FRAMES)
        service._crossing_times = deque(maxlen=1000)
        service._pixel_to_micron = 1.0
        service._average_droplet_speed_um_s = None
        service._speed_sample_count = 0
        service._droplet_generation_rate_hz = 0.0
        service._last_motion_frame_id = 0
        pipeline = SimpleNamespace(config=SimpleNamespace(metrics=SimpleNamespace(flow_axis="x")))
        service._ensure_pipeline = lambda: pipeline
        tracking = SimpleNamespace(active_tracks=[SimpleNamespace(id=1, position=(0.0, 0.0))])

        service._update_motion_measurements(tracking, {1}, 1.0, 0, frame_id=1)
        service._update_motion_measurements(tracking, {1}, 1.1, 0, frame_id=3)

        self.assertEqual(len(service._motion_observations), 1)
        self.assertIsNone(service._average_droplet_speed_um_s)


if __name__ == "__main__":
    unittest.main()
