from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from backend.vision.config import default_config
from backend.vision.detector import DropletDetector
from backend.vision.pipeline import VisionPipeline


class DropletDetectorTests(unittest.TestCase):
    def test_default_detector_finds_droplets_in_configured_range(self) -> None:
        image = np.full((220, 360), 190, dtype=np.uint8)
        cv2.circle(image, (100, 110), 24, 45, 3)
        cv2.circle(image, (260, 110), 24, 45, 3)

        config = default_config()
        result = DropletDetector(config.detector, config.debug).detect(image)

        self.assertEqual(len(result.centers), 2)
        self.assertTrue(all(abs(float(radius) - 24.0) <= 4.0 for radius in result.radii))

    def test_control_target_does_not_change_detector_output_domain(self) -> None:
        config = default_config()
        detector = DropletDetector(config.detector, config.debug)
        detector.configure_expected_diameter(20.0, 1.0)
        small_range = detector.runtime_radius_range()
        detector.configure_expected_diameter(80.0, 1.0)

        self.assertEqual(small_range, detector.runtime_radius_range())
        self.assertEqual(small_range, (18.0, 24.0, 32.0))

    def test_hough_uses_requested_algorithm_parameters(self) -> None:
        config = default_config()
        config.detector.min_radius = 18
        config.detector.max_radius = 32
        config.detector.min_center_distance = 32
        config.detector.sensitivity = 0.96
        detector = DropletDetector(config.detector, config.debug)
        image = np.full((120, 180), 180, dtype=np.uint8)

        with patch("backend.vision.detector.cv2.HoughCircles", return_value=None) as hough:
            detector.detect(image)

        kwargs = hough.call_args.kwargs
        self.assertEqual(kwargs["dp"], 1.2)
        self.assertEqual(kwargs["minDist"], 32.0)
        self.assertEqual(kwargs["param1"], 75)
        self.assertAlmostEqual(kwargs["param2"], 21.0)
        self.assertEqual(kwargs["minRadius"], 18)
        self.assertEqual(kwargs["maxRadius"], 32)

    def test_hough_results_are_rounded_and_stably_sorted(self) -> None:
        config = default_config()
        detector = DropletDetector(config.detector, config.debug)
        image = np.full((120, 180), 180, dtype=np.uint8)
        circles = np.asarray([[[91.2, 70.4, 22.6], [80.6, 30.2, 20.4], [20.1, 30.4, 19.6]]])

        with patch("backend.vision.detector.cv2.HoughCircles", return_value=circles):
            result = detector.detect(image)

        self.assertEqual([center.tolist() for center in result.centers], [[20.0, 30.0], [81.0, 30.0], [91.0, 70.0]])
        self.assertEqual(result.radii, [20.0, 20.0, 23.0])

    def test_preprocessing_matches_illumination_correction_pipeline(self) -> None:
        image = np.arange(120 * 180, dtype=np.uint8).reshape(120, 180)
        detector = DropletDetector(default_config().detector, default_config().debug)

        actual = detector._preprocess(image)
        background = cv2.GaussianBlur(image, (0, 0), sigmaX=25, sigmaY=25)
        expected = cv2.addWeighted(image, 1.0, background, -1.0, 128)
        expected = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(expected)
        expected = cv2.GaussianBlur(expected, (7, 7), 1.4)

        np.testing.assert_array_equal(actual, expected)

    def test_invalid_sensitivity_is_rejected(self) -> None:
        config = default_config()
        config.detector.sensitivity = 1.1
        with self.assertRaisesRegex(ValueError, "敏感度"):
            DropletDetector(config.detector, config.debug).detect(np.zeros((80, 80), dtype=np.uint8))

    def test_disabling_hough_disables_droplet_detection(self) -> None:
        config = default_config()
        config.detector.enable_hough_candidates = False
        image = np.full((220, 360), 190, dtype=np.uint8)
        cv2.circle(image, (120, 110), 24, 45, 3)

        result = DropletDetector(config.detector, config.debug).detect(image)

        self.assertEqual(result.centers, [])

    def test_realtime_pipeline_reuses_source_frame_without_overlay(self) -> None:
        config = default_config()
        config.channel_region.enabled = False
        pipeline = VisionPipeline(config)
        frame = np.full((240, 520, 3), 190, dtype=np.uint8)

        result = pipeline.process_frame(frame, timestamp=1.0)

        self.assertIs(result.annotated_frame, frame)

    def test_partial_circle_has_no_valid_diameter(self) -> None:
        detector = DropletDetector(default_config().detector, default_config().debug)
        center = np.array([5.0, 60.0], dtype=np.float32)
        self.assertFalse(detector._candidate_diameter_valid((120, 160), center, 20.0))


if __name__ == "__main__":
    unittest.main()
