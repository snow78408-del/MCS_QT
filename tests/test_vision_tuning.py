from __future__ import annotations

import unittest

import cv2
import numpy as np

from backend.vision.config import DetectorConfig
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
        self.assertEqual(len(stages), 10)
        self.assertEqual(stages[0].name, "1. 原始图像")
        self.assertEqual(stages[-1].name, "10. 磁珠辅助掩膜")
        self.assertTrue(all(stage.image.size > 0 for stage in stages))
        self.assertEqual(len(result.centers), len(result.radii))

    def test_grid_search_returns_best_first_and_does_not_mutate_base(self):
        frames = self._frames()
        base = DetectorConfig()
        results = grid_search(
            frames,
            base,
            {"min_radius": [8, 18], "circularity_threshold": [0.1, 0.8]},
            expected_count=1,
        )
        self.assertEqual(len(results), 4)
        self.assertGreaterEqual(results[0].score, results[-1].score)
        self.assertEqual(base.min_radius, 8.0)

    def test_unknown_search_field_is_rejected(self):
        with self.assertRaises(ValueError):
            grid_search(self._frames(), DetectorConfig(), {"not_a_parameter": [1]})


if __name__ == "__main__":
    unittest.main()
