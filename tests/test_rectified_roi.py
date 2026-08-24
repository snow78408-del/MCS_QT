from __future__ import annotations

import unittest

import cv2
import numpy as np

from backend.vision.config import default_config
from backend.vision.pipeline import VisionPipeline
from backend.vision.rectified_roi import rectify_channel_frame, wall_lines_bbox, wall_separation_px


class RectifiedRoiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.full((300, 640, 3), 180, dtype=np.uint8)
        cv2.line(self.frame, (10, 55), (630, 75), (30, 30, 30), 3)
        cv2.line(self.frame, (10, 220), (630, 240), (30, 30, 30), 3)
        self.lines = [
            {"x1": 10 / 640, "y1": 55 / 300, "x2": 630 / 640, "y2": 75 / 300},
            {"x1": 10 / 640, "y1": 220 / 300, "x2": 630 / 640, "y2": 240 / 300},
        ]

    def test_two_tilted_walls_are_rectified_to_horizontal_image(self) -> None:
        rectified = rectify_channel_frame(self.frame, self.lines)

        self.assertIsNotNone(rectified)
        self.assertAlmostEqual(rectified.shape[0], 165, delta=2)
        self.assertGreater(rectified.shape[1], 600)
        self.assertAlmostEqual(wall_separation_px(640, 300, self.lines) or 0.0, 165, delta=2)

    def test_pipeline_uses_same_rectification_before_detection(self) -> None:
        config = default_config()
        config.roi.enabled = True
        config.roi.wall_lines = self.lines
        pipeline = VisionPipeline(config)

        roi, offset = pipeline._apply_roi(self.frame)

        self.assertEqual(offset, (0, 0))
        self.assertAlmostEqual(roi.shape[0], 165, delta=2)
        self.assertGreater(roi.shape[1], 600)

    def test_axis_aligned_bbox_is_retained_for_compatibility(self) -> None:
        bbox = wall_lines_bbox(self.lines)

        self.assertIsNotNone(bbox)
        self.assertLess(bbox["y_start_ratio"], 55 / 300)
        self.assertGreater(bbox["y_end_ratio"], 240 / 300)


if __name__ == "__main__":
    unittest.main()
