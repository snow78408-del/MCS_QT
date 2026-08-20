from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from backend.vision.config import default_config
from backend.vision.detector import DropletDetector
from backend.vision.pipeline import VisionPipeline


class MultiscaleDropletDetectorTests(unittest.TestCase):
    def test_detects_small_medium_and_large_droplets(self) -> None:
        for radius in (12, 25, 45):
            with self.subTest(radius=radius):
                image = np.full((240, 520), 190, dtype=np.uint8)
                cv2.circle(image, (140, 120), radius, 45, 3)
                cv2.circle(image, (380, 120), radius, 45, 3)
                image = cv2.GaussianBlur(image, (3, 3), 0)

                config = default_config()
                detector = DropletDetector(config.detector, config.debug)
                detector.configure_expected_diameter(float(radius * 2), 1.0)
                result = detector.detect(image)

                self.assertEqual(len(result.centers), 2)
                self.assertTrue(all(abs(float(value) - radius) <= max(3.0, radius * 0.20) for value in result.radii))

    def test_target_size_changes_preference_without_excluding_other_sizes(self) -> None:
        config = default_config()
        detector = DropletDetector(config.detector, config.debug)
        detector.configure_expected_diameter(20.0, 1.0)
        small_range = detector.runtime_radius_range()
        detector.configure_expected_diameter(80.0, 1.0)
        large_range = detector.runtime_radius_range()

        self.assertEqual(small_range, (8.0, 10.0, 80.0))
        self.assertEqual(large_range, (8.0, 40.0, 80.0))

    def test_realtime_pipeline_reuses_source_frame_without_overlay(self) -> None:
        config = default_config()
        pipeline = VisionPipeline(config)
        frame = np.full((240, 520, 3), 190, dtype=np.uint8)

        result = pipeline.process_frame(frame, timestamp=1.0)

        self.assertIs(result.annotated_frame, frame)

    def test_intensity_detector_runs_only_as_empty_result_fallback(self) -> None:
        config = default_config()
        config.detector.enable_hough_candidates = False
        config.detector.enable_intensity_peak_candidates = False
        config.detector.enable_intensity_peak_fallback = True
        detector = DropletDetector(config.detector, config.debug)
        image = np.full((100, 120), 180, dtype=np.uint8)
        fallback_center = np.array([60.0, 50.0], dtype=np.float32)

        with patch.object(detector, "_detect_split_connected", return_value=([], [])), patch.object(
            detector,
            "_detect_intensity_peak_candidates",
            return_value=([fallback_center], [15.0]),
        ) as fallback:
            detector.detect(image)

        fallback.assert_called_once()

    def test_hough_is_skipped_when_contour_candidate_is_valid(self) -> None:
        config = default_config()
        config.detector.hough_fallback_only = True
        config.detector.enable_intensity_peak_candidates = False
        detector = DropletDetector(config.detector, config.debug)
        image = np.full((100, 120), 180, dtype=np.uint8)
        contour_center = np.array([60.0, 50.0], dtype=np.float32)

        with patch.object(
            detector,
            "_detect_split_connected",
            return_value=([contour_center], [15.0]),
        ), patch.object(
            detector,
            "_score_and_suppress_candidates",
            side_effect=lambda _image, centers, radii: (centers, radii),
        ), patch.object(detector, "_detect_hough_candidates") as hough:
            result = detector.detect(image)

        hough.assert_not_called()
        self.assertEqual(len(result.centers), 1)


if __name__ == "__main__":
    unittest.main()
