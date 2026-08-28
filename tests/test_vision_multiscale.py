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

    def test_control_target_does_not_change_detector_output_domain(self) -> None:
        config = default_config()
        detector = DropletDetector(config.detector, config.debug)
        detector.configure_expected_diameter(20.0, 1.0)
        small_range = detector.runtime_radius_range()
        detector.configure_expected_diameter(80.0, 1.0)
        large_range = detector.runtime_radius_range()

        self.assertEqual(small_range, large_range)
        self.assertEqual(small_range, (8.0, float((8.0 * 80.0) ** 0.5), 80.0))

    def test_expected_size_filter_only_rejects_wrong_radius_in_hard_mode(self) -> None:
        config = default_config()
        config.detector.expected_radius = 20.0
        config.detector.expected_radius_tolerance_ratio = 0.25
        config.detector.expected_size_hard_gate = True
        detector = DropletDetector(config.detector, config.debug)
        candidates = [(40.0, 40.0, 19.0), (80.0, 40.0, 28.0), (120.0, 40.0, 12.0)]

        filtered = detector._filter_expected_size(candidates)

        self.assertEqual(filtered, [(40.0, 40.0, 19.0)])

    def test_expected_size_soft_mode_preserves_other_radii(self) -> None:
        config = default_config()
        config.detector.expected_radius = 20.0
        config.detector.expected_radius_tolerance_ratio = 0.25
        detector = DropletDetector(config.detector, config.debug)
        candidates = [(40.0, 40.0, 10.0), (80.0, 40.0, 20.0), (120.0, 40.0, 30.0)]

        self.assertEqual(detector._filter_expected_size(candidates), candidates)

    def test_overlapping_hough_circles_are_deduplicated(self) -> None:
        detector = DropletDetector(default_config().detector, default_config().debug)
        centers = [
            np.array([50.0, 70.0], dtype=np.float32),
            np.array([53.0, 71.0], dtype=np.float32),
            np.array([110.0, 70.0], dtype=np.float32),
        ]

        filtered_centers, filtered_radii = detector._deduplicate(centers, [25.0, 24.0, 25.0])

        self.assertEqual(len(filtered_centers), 2)
        self.assertEqual(filtered_radii, [25.0, 25.0])

    def test_realtime_pipeline_reuses_source_frame_without_overlay(self) -> None:
        config = default_config()
        pipeline = VisionPipeline(config)
        frame = np.full((240, 520, 3), 190, dtype=np.uint8)

        result = pipeline.process_frame(frame, timestamp=1.0)

        self.assertIs(result.annotated_frame, frame)

    def test_disabling_hough_disables_droplet_detection(self) -> None:
        config = default_config()
        config.detector.enable_hough_candidates = False
        image = np.full((220, 360), 190, dtype=np.uint8)
        cv2.circle(image, (120, 110), 28, 45, 3)

        result = DropletDetector(config.detector, config.debug).detect(image)

        self.assertEqual(result.centers, [])
        self.assertFalse(hasattr(DropletDetector, "_detect_contour_candidates"))
        self.assertFalse(hasattr(DropletDetector, "_split_contour_watershed"))

    def test_hough_detects_touching_droplets(self) -> None:
        config = default_config()
        config.detector.min_radius = 10.0
        config.detector.max_radius = 50.0
        image = np.full((220, 360), 190, dtype=np.uint8)
        cv2.circle(image, (130, 110), 28, 45, 3)
        cv2.circle(image, (178, 110), 28, 45, 3)

        result = DropletDetector(config.detector, config.debug).detect(image)

        self.assertEqual(len(result.centers), 2)
        self.assertTrue(all(abs(float(radius) - 28.0) <= 4.0 for radius in result.radii))

    def test_parallel_channel_edges_are_not_droplets(self) -> None:
        config = default_config()
        config.detector.enable_hough_candidates = False
        image = np.full((220, 360), 190, dtype=np.uint8)
        cv2.line(image, (10, 70), (350, 70), 45, 3)
        cv2.line(image, (10, 150), (350, 150), 45, 3)

        result = DropletDetector(config.detector, config.debug).detect(image)

        self.assertEqual(result.centers, [])

    def test_partial_edge_candidate_can_track_but_has_no_valid_diameter(self) -> None:
        config = default_config()
        detector = DropletDetector(config.detector, config.debug)
        image = np.full((120, 160), 190, dtype=np.uint8)
        cv2.circle(image, (5, 60), 20, 45, 3)

        center = np.array([5.0, 60.0], dtype=np.float32)

        self.assertGreaterEqual(detector._circle_visible_ratio(image.shape, 5.0, 60.0, 20.0), 0.5)
        self.assertFalse(detector._candidate_diameter_valid(image.shape, center, 20.0))

    def test_hough_runs_on_every_frame(self) -> None:
        config = default_config()
        detector = DropletDetector(config.detector, config.debug)
        image = np.full((100, 120), 180, dtype=np.uint8)
        hough_center = np.array([60.0, 50.0], dtype=np.float32)

        with patch.object(
            detector,
            "_detect_hough_candidates",
            return_value=([hough_center], [15.0]),
        ) as hough:
            first = detector.detect(image)
            second = detector.detect(image)

        self.assertEqual(hough.call_count, 2)
        self.assertEqual(len(first.centers), 1)
        self.assertEqual(len(second.centers), 1)


if __name__ == "__main__":
    unittest.main()
