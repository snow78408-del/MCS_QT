from __future__ import annotations

import unittest

import cv2
import numpy as np

from backend.vision.config import ChannelRegionConfig, DetectorConfig
from backend.vision.tuning import (
    TuningFrame,
    annotate_frame,
    detect_frame,
    evaluate_config,
    grid_search,
    inspect_frame,
)


class VisionTuningTests(unittest.TestCase):
    def _frames(self):
        frames = []
        for index in range(3):
            image = np.full((120, 220, 3), 190, dtype=np.uint8)
            cv2.circle(image, (70 + index, 60), 20, (45, 45, 45), 3)
            frames.append(TuningFrame(index, image))
        return frames

    def test_evaluation_and_annotation_are_detector_only(self):
        frames = self._frames()
        evaluation = evaluate_config(frames, DetectorConfig(), expected_count=1)
        self.assertEqual(evaluation.processed_frames, 3)
        self.assertGreater(evaluation.mean_count, 0)
        annotated = annotate_frame(frames[0].image, detect_frame(frames[0].image, DetectorConfig()))
        self.assertEqual(annotated.shape, frames[0].image.shape)

    def test_pipeline_inspection_exposes_each_processing_stage(self):
        frame = self._frames()[0].image
        result, stages = inspect_frame(frame, DetectorConfig())
        self.assertEqual(len(stages), 8)
        self.assertEqual(stages[0].name, "1. 原始图像")
        self.assertEqual(stages[-1].name, "8. 简单过滤结果")
        self.assertIn("Hough 输入预处理", stages[4].name)
        self.assertIn("Hough 边缘支撑", stages[5].name)
        self.assertIn("Hough 原始圆", stages[6].name)
        self.assertTrue(all(stage.image.size > 0 for stage in stages))
        self.assertEqual(len(result.centers), len(result.radii))

    def test_channel_region_is_exposed_as_a_major_stage_before_detection(self):
        frames = []
        for index in range(12):
            image = np.full((240, 520, 3), 180, dtype=np.uint8)
            cv2.line(image, (3, 45), (516, 58), (35, 35, 35), 3)
            cv2.line(image, (3, 185), (516, 198), (35, 35, 35), 3)
            cv2.circle(image, (80 + index * 10, 120), 20, (90, 90, 90), 2)
            frames.append(image)

        _result, stages = inspect_frame(
            frames[0],
            DetectorConfig(),
            channel_config=ChannelRegionConfig(),
            channel_frames=frames,
        )

        self.assertEqual(len(stages), 12)
        self.assertIn("管道检定", stages[0].name)
        self.assertIn("高频信号", stages[1].name)
        self.assertIn("直线性质", stages[2].name)
        self.assertIn("有效区域", stages[3].name)
        self.assertTrue(stages[4].name.startswith("B1."))

    def test_optional_preprocessing_steps_can_be_skipped(self):
        frame = self._frames()[0].image
        config = DetectorConfig(
            enable_intensity_normalization=False,
            enable_gaussian_blur=False,
        )
        _, stages = inspect_frame(frame, config)
        np.testing.assert_array_equal(stages[2].image, stages[1].image)
        np.testing.assert_array_equal(stages[3].image, stages[2].image)
        self.assertEqual(stages[2].parameters, "已跳过")
        self.assertEqual(stages[3].parameters, "已跳过")

    def test_grid_search_returns_best_first_and_does_not_mutate_base(self):
        frames = self._frames()
        base = DetectorConfig()
        results = grid_search(
            frames,
            base,
            {"hough_param2": [20, 28], "hough_edge_support_threshold": [0.1, 0.2]},
            expected_count=1,
        )
        self.assertEqual(len(results), 4)
        self.assertGreaterEqual(results[0].score, results[-1].score)
        self.assertEqual(base.hough_param2, 28.0)

    def test_unknown_search_field_is_rejected(self):
        with self.assertRaises(ValueError):
            grid_search(self._frames(), DetectorConfig(), {"not_a_parameter": [1]})


if __name__ == "__main__":
    unittest.main()
