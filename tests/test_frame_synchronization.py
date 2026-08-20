from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from backend.orchestrator.vision_adapter import MOTION_WINDOW_FRAMES, PipelineVisionService


class FrameSynchronizationTests(unittest.TestCase):
    def test_preview_publish_does_not_advance_recognition_identity(self) -> None:
        service = PipelineVisionService()
        service._latest = service._empty_snapshot("test")
        service._latest.frame_id = 7
        service._latest.timestamp = 100.0
        frame = np.zeros((20, 30), dtype=np.uint8)

        service._publish_video_frame(frame, frame_id=12, timestamp=120.0)
        snapshot = service.get_snapshot()
        preview = service.get_frame_snapshot()

        self.assertEqual(snapshot.frame_id, 7)
        self.assertEqual(snapshot.timestamp, 100.0)
        self.assertEqual(snapshot.preview_frame_id, 0)
        self.assertEqual(snapshot.preview_timestamp, 0.0)
        self.assertIsNotNone(preview)
        self.assertEqual(preview.frame_id, 12)
        self.assertEqual(preview.timestamp, 120.0)
        self.assertIsNone(preview.frame_png_base64)
        self.assertIsNone(preview.frame_pgm)
        self.assertTrue(preview.frame_jpeg)

    def test_preview_clock_averages_thirty_fps_for_hundred_fps_capture(self) -> None:
        service = PipelineVisionService()

        selected = [
            index
            for index in range(100)
            if service._should_publish_preview(1000.0 + index / 100.0)
        ]

        self.assertGreaterEqual(len(selected), 30)
        self.assertLessEqual(len(selected), 31)

    def test_diagnostics_are_computed_when_snapshot_is_read(self) -> None:
        service = PipelineVisionService()
        service._processing_busy = True

        snapshot = service.get_snapshot()

        self.assertEqual(snapshot.pending_processing_frames, MOTION_WINDOW_FRAMES)

    def test_preview_queue_replaces_stale_frame_without_blocking(self) -> None:
        service = PipelineVisionService()
        first = np.zeros((10, 10), dtype=np.uint8)
        newest = np.ones((10, 10), dtype=np.uint8)

        service._submit_preview_frame(1, 1.0, first)
        service._submit_preview_frame(2, 2.0, newest)

        frame_id, timestamp, frame = service._preview_queue.get_nowait()
        self.assertEqual(frame_id, 2)
        self.assertEqual(timestamp, 2.0)
        self.assertTrue(np.array_equal(frame, newest))

    def test_analysis_collects_only_one_five_frame_batch_per_control_period(self) -> None:
        service = PipelineVisionService()
        service._control_interval_ms = 1000
        frame = np.zeros((10, 10), dtype=np.uint8)

        with patch("backend.orchestrator.vision_adapter.time.monotonic", side_effect=[0.0] * 5 + [0.2] * 5):
            for frame_id in range(1, 11):
                service._submit_processing_frame(frame_id, float(frame_id), frame)

        self.assertEqual(service._frame_queue.qsize(), 1)
        batch = service._frame_queue.get_nowait()
        self.assertEqual([item[0] for item in batch], [1, 2, 3, 4, 5])
        self.assertEqual(service._capture_batch, [])


if __name__ == "__main__":
    unittest.main()
