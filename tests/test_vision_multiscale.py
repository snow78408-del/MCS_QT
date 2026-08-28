from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from backend.vision.config import default_config
from backend.vision.detector import CircleDetection, DropletDetector
from backend.vision.pipeline import VisionPipeline


class EdgeDrawingDropletDetectorTests(unittest.TestCase):
    def test_detector_config_contains_only_circle_detection_fields(self) -> None:
        fields = vars(default_config().detector)

        self.assertTrue(any(name.startswith("edge_") for name in fields))
        self.assertFalse(any("hough" in name for name in fields))
        self.assertEqual(fields["edge_min_circle_ratio"], 0.90)

    def test_detects_small_medium_and_large_circular_droplets(self) -> None:
        for radius in (12, 25, 45):
            with self.subTest(radius=radius):
                image = np.full((240, 520), 190, dtype=np.uint8)
                cv2.circle(image, (140, 120), radius, 45, 3)
                cv2.circle(image, (380, 120), radius, 45, 3)
                image = cv2.GaussianBlur(image, (3, 3), 0)

                result = DropletDetector(
                    default_config().detector,
                    default_config().debug,
                ).detect(image)

                self.assertEqual(len(result.centers), 2)
                self.assertTrue(
                    all(abs(value - radius) <= max(3.0, radius * 0.20) for value in result.radii)
                )

    def test_near_circle_record_is_accepted_and_flat_record_is_rejected(self) -> None:
        detector = DropletDetector(default_config().detector, default_config().debug)
        records = np.array(
            [
                [[50.0, 60.0, 15.0, 0.0, 0.0, 1.0]],
                [[90.0, 60.0, 0.0, 20.0, 18.0, 1.0]],
                [[130.0, 60.0, 0.0, 30.0, 15.0, 1.0]],
            ],
            dtype=np.float64,
        )

        circles = detector._parse_circle_records(records)

        self.assertEqual(len(circles), 2)
        self.assertAlmostEqual(circles[0].radius, 15.0)
        self.assertAlmostEqual(circles[1].radius, 19.0)

    def test_expected_size_gate_is_only_applied_in_hard_mode(self) -> None:
        config = default_config()
        config.detector.expected_radius = 20.0
        config.detector.expected_radius_tolerance_ratio = 0.25
        detector = DropletDetector(config.detector, config.debug)

        self.assertTrue(detector._expected_size_valid(10.0))
        config.detector.expected_size_hard_gate = True
        self.assertTrue(detector._expected_size_valid(19.0))
        self.assertFalse(detector._expected_size_valid(28.0))

    def test_control_target_does_not_change_detector_radius_domain(self) -> None:
        config = default_config()
        detector = DropletDetector(config.detector, config.debug)
        detector.configure_expected_diameter(20.0, 1.0)
        first = detector.runtime_radius_range()
        detector.configure_expected_diameter(80.0, 1.0)

        self.assertEqual(first, detector.runtime_radius_range())
        self.assertEqual(first, (8.0, float((8.0 * 80.0) ** 0.5), 80.0))

    def test_near_duplicate_circles_are_deduplicated(self) -> None:
        detector = DropletDetector(default_config().detector, default_config().debug)
        candidates = [
            CircleDetection(np.array([50.0, 70.0], np.float32), 25.0),
            CircleDetection(np.array([53.0, 71.0], np.float32), 24.0),
            CircleDetection(np.array([110.0, 70.0], np.float32), 25.0),
        ]

        self.assertEqual(len(detector._deduplicate(candidates)), 2)

    def test_touching_droplets_are_detected_as_two_circles(self) -> None:
        image = np.full((220, 360), 190, dtype=np.uint8)
        cv2.circle(image, (130, 110), 28, 45, 3)
        cv2.circle(image, (178, 110), 28, 45, 3)

        result = DropletDetector(
            default_config().detector,
            default_config().debug,
        ).detect(image)

        self.assertEqual(len(result.centers), 2)
        self.assertTrue(all(abs(radius - 28.0) <= 4.0 for radius in result.radii))

    def test_parallel_channel_edges_are_not_circle_candidates(self) -> None:
        image = np.full((220, 360), 190, dtype=np.uint8)
        cv2.line(image, (10, 70), (350, 70), 45, 3)
        cv2.line(image, (10, 150), (350, 150), 45, 3)

        result = DropletDetector(
            default_config().detector,
            default_config().debug,
        ).detect(image)

        self.assertEqual(result.centers, [])

    def test_partial_circle_can_track_but_has_no_valid_diameter(self) -> None:
        detector = DropletDetector(default_config().detector, default_config().debug)
        item = CircleDetection(np.array([5.0, 60.0], np.float32), 20.0)

        self.assertGreaterEqual(detector._circle_visible_ratio((120, 160), item), 0.5)
        self.assertFalse(detector._circle_fully_visible((120, 160), item))

    def test_edge_drawing_runs_on_every_frame(self) -> None:
        detector = DropletDetector(default_config().detector, default_config().debug)
        image = np.full((100, 120), 180, dtype=np.uint8)
        item = CircleDetection(np.array([60.0, 50.0], np.float32), 15.0)

        with patch.object(detector, "_detect_candidates", return_value=[item]) as detect:
            first = detector.detect(image)
            second = detector.detect(image)

        self.assertEqual(detect.call_count, 2)
        self.assertEqual(len(first.centers), 1)
        self.assertEqual(len(second.centers), 1)

    def test_realtime_pipeline_reuses_source_frame_without_overlay(self) -> None:
        pipeline = VisionPipeline(default_config())
        frame = np.full((240, 520, 3), 190, dtype=np.uint8)

        result = pipeline.process_frame(frame, timestamp=1.0)

        self.assertIs(result.annotated_frame, frame)


if __name__ == "__main__":
    unittest.main()
