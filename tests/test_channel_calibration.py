from __future__ import annotations

import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np
import base64

from backend.vision.config import ChannelRegionConfig, DetectorConfig
from backend.vision.channel_calibration import (
    detect_wall_line_candidates,
    estimate_channel_width_px,
    suggest_channel_roi,
)
from backend.vision.channel_calibration import ChannelWidthMeasurement
from backend.orchestrator.vision_adapter import PipelineVisionService
from backend.orchestrator.vision_adapter import _include_fitted_wall_candidates
from backend.orchestrator.service import OrchestratorService


class ChannelCalibrationTests(unittest.TestCase):
    def test_orchestrator_forwards_saved_tuning_to_vision_service(self) -> None:
        calls = []
        vision = SimpleNamespace(
            apply_tuning_config=lambda detector, channel: (
                calls.append((detector, channel)) or {"applied": True}
            )
        )
        service = OrchestratorService(vision_service=vision)
        detector = DetectorConfig(min_radius=23.0, max_radius=41.0)
        channel = ChannelRegionConfig(canny_low=30)

        result = service.apply_vision_tuning(detector, channel)

        self.assertTrue(result["applied"])
        self.assertEqual(calls, [(detector, channel)])

    def test_runtime_tuning_replaces_detector_for_subsequent_samples(self) -> None:
        service = PipelineVisionService()
        pipeline = service._ensure_pipeline()
        tracker = pipeline.tracker
        metrics = pipeline.metrics

        applied = service.apply_tuning_config(
            DetectorConfig(
                measurement_mode="observation_circle",
                generation_channel_height_um=90.0,
                generation_channel_width_um=80.0,
                generation_volume_correction=1.4,
                generation_center_band_ratio=0.54,
                generation_edge_mad_multiplier=2.6,
                generation_min_profile_contrast_sigma=0.48,
            ),
            ChannelRegionConfig(enabled=False, canny_low=28),
        )

        self.assertTrue(applied["applied"])
        self.assertEqual(pipeline.detector._config.measurement_mode, "generation_plug")
        self.assertAlmostEqual(pipeline.detector._config.generation_channel_height_um, 50.0)
        self.assertAlmostEqual(pipeline.detector._config.generation_channel_width_um, 50.0)
        self.assertAlmostEqual(pipeline.detector._config.generation_volume_correction, 1.0)
        self.assertAlmostEqual(pipeline.detector._config.generation_center_band_ratio, 0.54)
        self.assertAlmostEqual(pipeline.detector._config.generation_edge_mad_multiplier, 2.6)
        self.assertAlmostEqual(pipeline.detector._config.generation_min_profile_contrast_sigma, 0.48)
        self.assertEqual(applied["measurement_mode"], "generation_plug")
        self.assertFalse(pipeline.channel_region_detector.config.enabled)
        self.assertEqual(pipeline.channel_region_detector.config.canny_low, 28)
        self.assertIs(pipeline.tracker, tracker)
        self.assertIs(pipeline.metrics, metrics)

    def test_user_square_channel_width_populates_missing_height_and_width(self) -> None:
        service = PipelineVisionService()

        service.set_recognition_roi(
            {
                "enabled": True,
                "channel_width_um": 70.0,
                "channel_calibration_enabled": False,
            }
        )

        detector = service._ensure_pipeline().config.detector
        self.assertEqual(service._channel_width_um, 70.0)
        self.assertEqual(detector.generation_channel_height_um, 70.0)
        self.assertEqual(detector.generation_channel_width_um, 70.0)

    def test_fitted_lower_wall_is_added_to_clickable_candidates(self) -> None:
        measurement = ChannelWidthMeasurement(
            180.0,
            0.9,
            "ok",
            upper_center_px=10.0,
            lower_center_px=190.0,
            upper_slope=0.0,
            lower_slope=0.0,
        )
        candidates = [
            {
                "id": 1,
                "x1": 0.0,
                "y1": 30 / 240,
                "x2": 0.998,
                "y2": 30 / 240,
                "slope": 0.0,
            }
        ]

        result = _include_fitted_wall_candidates(
            candidates,
            measurement,
            frame_width=640,
            frame_height=240,
            roi_x0=0,
            roi_x1=640,
            roi_y0=20,
            merge_distance_px=4.0,
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[-1]["source"], "fitted_wall")
        self.assertAlmostEqual(float(result[-1]["y1"]), 210 / 240)

    def test_custom_hough_parameters_are_applied_to_line_transform(self) -> None:
        image = np.full((240, 640), 180, dtype=np.uint8)
        with patch(
            "backend.vision.channel_calibration.cv2.HoughLinesP",
            return_value=None,
        ) as hough:
            result = detect_wall_line_candidates(
                image,
                hough_parameters={
                    "canny_low": 20,
                    "canny_high": 70,
                    "hough_threshold": 17,
                    "min_line_length_ratio": 0.25,
                    "max_line_gap_ratio": 0.10,
                    "max_tilt_degrees": 45,
                    "merge_distance_px": 8,
                    "max_lines": 12,
                },
            )

        self.assertEqual(result, [])
        call = hough.call_args
        self.assertEqual(call.kwargs["threshold"], 17)
        self.assertEqual(call.kwargs["minLineLength"], 160)
        self.assertEqual(call.kwargs["maxLineGap"], 64)

    def test_measures_tilted_channel_walls_despite_circular_clutter(self) -> None:
        image = np.full((230, 640), 180, dtype=np.uint8)
        cv2.line(image, (4, 10), (635, 22), 35, 3)
        cv2.line(image, (4, 208), (635, 220), 35, 3)
        for row in (58, 110, 162):
            for x in range(30, 630, 55):
                cv2.circle(image, (x, row + x // 100), 23, 95, 2)

        result = estimate_channel_width_px(image)

        self.assertEqual(result.reason, "ok")
        self.assertIsNotNone(result.width_px)
        self.assertAlmostEqual(result.width_px or 0.0, 198.0, delta=4.0)
        self.assertGreater(result.confidence, 0.68)

    def test_rejects_roi_without_both_outer_walls(self) -> None:
        image = np.full((230, 640), 180, dtype=np.uint8)
        cv2.line(image, (4, 12), (635, 20), 35, 3)

        result = estimate_channel_width_px(image)

        self.assertIsNone(result.width_px)
        self.assertNotEqual(result.reason, "ok")

    def test_five_stable_startup_frames_replace_configured_scale(self) -> None:
        service = PipelineVisionService()
        service.set_recognition_roi(
            {
                "enabled": True,
                "x_start_ratio": 0.0,
                "x_end_ratio": 1.0,
                "y_start_ratio": 0.1,
                "y_end_ratio": 0.9,
                "channel_calibration_enabled": True,
                "channel_width_um": 430.0,
            }
        )
        service.configure_detection_scale(100.0, 1.725)
        service._reset_channel_calibration()
        frame = np.zeros((240, 640), dtype=np.uint8)
        measured = ChannelWidthMeasurement(215.0, 0.9, "ok")

        with patch(
            "backend.vision.channel_calibration.estimate_channel_width_px",
            return_value=measured,
        ):
            for _ in range(5):
                service._try_channel_calibration(frame)

        self.assertEqual(service._channel_calibration_status, "calibrated")
        self.assertAlmostEqual(service._pixel_to_micron, 2.0)
        self.assertAlmostEqual(service._channel_width_px or 0.0, 215.0)

    def test_test_frame_auto_suggests_roi_and_returns_overlay(self) -> None:
        image = np.full((300, 640), 180, dtype=np.uint8)
        cv2.line(image, (3, 50), (636, 62), 35, 3)
        cv2.line(image, (3, 245), (636, 257), 35, 3)
        for row in (100, 155, 210):
            for x in range(30, 630, 55):
                cv2.circle(image, (x, row), 22, 90, 2)
        suggestion = suggest_channel_roi(image)
        self.assertIsNotNone(suggestion)

        encoded_ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(encoded_ok)
        service = PipelineVisionService()
        result = service.analyze_channel_calibration_preview(
            base64.b64encode(encoded.tobytes()).decode("ascii"),
            {"enabled": False, "user_defined": False},
            430.0,
            1.725,
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["auto_suggested"])
        self.assertFalse(result["used_user_roi"])
        self.assertTrue(result["overlay_png_base64"])
        self.assertGreaterEqual(len(result["hough_lines"]), 2)

    def test_test_frame_returns_normalized_custom_hough_parameters(self) -> None:
        image = np.full((200, 400), 180, dtype=np.uint8)
        encoded_ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(encoded_ok)

        result = PipelineVisionService().analyze_channel_calibration_preview(
            base64.b64encode(encoded.tobytes()).decode("ascii"),
            {"enabled": False, "user_defined": False},
            430.0,
            1.725,
            {"canny_low": 15, "canny_high": 60, "hough_threshold": 22},
        )

        self.assertEqual(result["hough_parameters"]["canny_low"], 15)
        self.assertEqual(result["hough_parameters"]["canny_high"], 60)
        self.assertEqual(result["hough_parameters"]["hough_threshold"], 22)

    def test_user_roi_has_priority_over_auto_suggestion(self) -> None:
        image = np.full((300, 640), 180, dtype=np.uint8)
        cv2.line(image, (3, 50), (636, 62), 35, 3)
        cv2.line(image, (3, 245), (636, 257), 35, 3)
        encoded_ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(encoded_ok)
        roi = {
            "enabled": True,
            "user_defined": True,
            "x_start_ratio": 0.0,
            "x_end_ratio": 1.0,
            "y_start_ratio": 0.13,
            "y_end_ratio": 0.90,
        }

        result = PipelineVisionService().analyze_channel_calibration_preview(
            base64.b64encode(encoded.tobytes()).decode("ascii"),
            roi,
            430.0,
            1.725,
        )

        self.assertTrue(result["used_user_roi"])
        self.assertFalse(result["auto_suggested"])
        self.assertAlmostEqual(result["roi"]["y_start_ratio"], 0.13, places=2)

    def test_failed_auto_calibration_keeps_user_scale_without_pid_gate(self) -> None:
        service = PipelineVisionService()
        service.set_recognition_roi(
            {
                "enabled": True,
                "channel_calibration_enabled": True,
                "channel_width_um": 430.0,
            }
        )
        service.configure_detection_scale(100.0, 1.725)
        service._reset_channel_calibration()
        failed = ChannelWidthMeasurement(None, 0.0, "not found")
        frame = np.zeros((240, 640), dtype=np.uint8)

        with patch(
            "backend.vision.channel_calibration.estimate_channel_width_px",
            return_value=failed,
        ):
            for _ in range(20):
                service._try_channel_calibration(frame)

        self.assertEqual(service._channel_calibration_status, "user_config")
        self.assertAlmostEqual(service._pixel_to_micron, 1.725)

    def test_selected_hough_lines_define_physical_width(self) -> None:
        image = np.full((300, 640), 180, dtype=np.uint8)
        cv2.line(image, (10, 55), (630, 75), 35, 3)
        cv2.line(image, (10, 220), (630, 240), 35, 3)
        encoded_ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(encoded_ok)
        lines = [
            {"x1": 10 / 640, "y1": 55 / 300, "x2": 630 / 640, "y2": 75 / 300},
            {"x1": 10 / 640, "y1": 220 / 300, "x2": 630 / 640, "y2": 240 / 300},
        ]

        result = PipelineVisionService().analyze_channel_calibration_preview(
            base64.b64encode(encoded.tobytes()).decode("ascii"),
            {"enabled": True, "user_defined": True, "wall_lines": lines},
            430.0,
            1.725,
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["used_user_roi"])
        self.assertEqual(len(result["roi"]["wall_lines"]), 2)
        self.assertAlmostEqual(result["channel_width_px"], 165.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
