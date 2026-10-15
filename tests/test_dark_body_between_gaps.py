from __future__ import annotations

import cv2
import numpy as np
import backend.vision.detector as detector_module

from backend.vision.capsule_profile import (dark_body_intervals,
                                           transverse_body_intervals)
from backend.vision.config import DebugConfig, DetectorConfig
from backend.vision.detector import DropletDetector
from backend.vision.rectified_measurement import detect_rectified_generation_plugs


def _dark_plugs_between_bright_gaps() -> np.ndarray:
    height, width = 28, 600
    x = np.arange(width, dtype=np.float32)
    field = np.full((height, width), 78.0, dtype=np.float32)
    field += 2.0 * np.sin(x / 90.0)[None, :]
    field[2:7] -= 3.0
    field[21:26] += 3.0
    for left in (40, 150, 260, 370, 480):
        field[:, left:left + 30] += 15.0
        field[2:7, left:left + 30] += 8.0
        field[21:26, left:left + 30] -= 8.0
    # These horizontal walls must not be interpreted as menisci.
    field[:2] -= 8.0
    field[-2:] -= 8.0
    return np.uint8(np.clip(cv2.GaussianBlur(field, (0, 0), 1.4), 0, 255))


def test_dark_bodies_are_measured_between_carrier_gaps() -> None:
    frame = _dark_plugs_between_bright_gaps()
    intervals = dark_body_intervals(frame, 27.0)
    assert len(intervals) == 4
    for (left, right), expected in zip(intervals,
                                       ((70, 150), (180, 260),
                                        (290, 370), (400, 480))):
        assert abs(left - expected[0]) <= 8
        assert abs(right - expected[1]) <= 8


def test_dark_body_detector_does_not_invent_plugs_in_uniform_channel() -> None:
    blank = np.full((28, 600), 86, dtype=np.uint8)
    assert dark_body_intervals(blank, 27.0) == []


def test_axial_illumination_stripes_are_not_plugs() -> None:
    x = np.arange(600, dtype=np.float32)
    field = np.tile(85.0 + 10.0 * np.sin(x / 17.0), (28, 1))
    assert dark_body_intervals(np.uint8(field), 27.0) == []


def test_runtime_generation_detector_uses_dark_body_endpoints() -> None:
    frame = _dark_plugs_between_bright_gaps()
    detector = DropletDetector(DetectorConfig(measurement_mode="generation_plug"),
                               DebugConfig())
    result, trace = detect_rectified_generation_plugs(frame, detector=detector)
    assert trace["interval_source"] == "dark_body_between_carrier_gaps"
    assert len(trace["selected_intervals"]) == 4
    assert all(65 <= interval[2] <= 95 for interval in trace["selected_intervals"])
    assert len(result.plug_lengths_px) == 4


def test_transverse_reversal_recovers_weak_centre_plugs() -> None:
    height, width = 28, 600
    x = np.arange(width, dtype=np.float32)
    image = np.tile(90.0 + 4.0 * np.sin(x / 42.0), (height, 1))
    image[2:7] -= 10.0
    image[21:26] += 10.0
    for left in (40, 160, 280, 400, 520):
        image[2:7, left:left + 35] += 18.0
        image[21:26, left:left + 35] -= 18.0
    image = np.uint8(np.clip(cv2.GaussianBlur(image, (0, 0), 1.0), 0, 255))
    assert dark_body_intervals(image, 27.0) == []
    intervals = transverse_body_intervals(image, 27.0)
    assert len(intervals) == 4
    detector = DropletDetector(DetectorConfig(measurement_mode="generation_plug"),
                               DebugConfig())
    result, trace = detect_rectified_generation_plugs(image, detector=detector)
    assert trace["interval_source"] == "transverse_body_between_carrier_gaps"
    assert len(result.plug_lengths_px) == 4


def test_transverse_reversal_rejects_shared_illumination_stripes() -> None:
    x = np.arange(600, dtype=np.float32)
    image = np.uint8(np.tile(90.0 + 12.0 * np.sin(x / 25.0), (28, 1)))
    assert transverse_body_intervals(image, 27.0) == []


def test_generation_rejects_intervals_no_longer_than_channel_height(monkeypatch) -> None:
    frame = np.full((32, 220), 90, dtype=np.uint8)
    monkeypatch.setattr(detector_module, "dark_body_intervals",
                        lambda _frame, _width: [(20, 38), (80, 140)])
    detector = DropletDetector(DetectorConfig(measurement_mode="generation_plug"),
                               DebugConfig())
    result, trace = detect_rectified_generation_plugs(frame, detector=detector)
    assert trace["minimum_length_px"] == 31.0
    assert trace["short_intervals_rejected"] == [(20, 38, 18.0)]
    assert trace["selected_intervals"] == [(80, 140, 60.0)]
    assert result.plug_lengths_px == [60.0]


def test_user_review_frames_do_not_merge_droplets_or_measure_continuous_phase():
    from pathlib import Path
    root = Path(__file__).parent / "fixtures" / "phase_review"
    detector = DropletDetector(DetectorConfig(
        measurement_mode="generation_plug", generation_phase_mode="transverse_bounded"), DebugConfig())
    # User-corrected outer menisci replace the former core-only 39--41 px
    # expectations. Matching old algorithm output was not a validity test.
    for index, expected in [(109, [(81, 148), (175, 241), (264, 330)]),
                            (121, [(91, 149), (177, 245), (268, 336)]),
                            (129, [(90, 154), (176, 240), (265, 329)])]:
        frame = cv2.imread(str(root / f"channel_{index}.png"), cv2.IMREAD_GRAYSCALE)
        result, trace = detect_rectified_generation_plugs(frame, detector=detector)
        assert len(trace["selected_intervals"]) == len(expected)
        for (a, b, _), (left, right) in zip(trace["selected_intervals"], expected):
            assert abs(a - left) <= 10 and abs(b - right) <= 10
            assert abs((b - a) - (right - left)) <= 10
        assert trace["interval_source"] == "calibrated_phase_outer_endpoints"
    for index in (188, 190, 195, 425):
        frame = cv2.imread(str(root / f"channel_{index}.png"), cv2.IMREAD_GRAYSCALE)
        result, trace = detect_rectified_generation_plugs(frame, detector=detector)
        assert result.plug_lengths_px == []
        assert trace["interval_source"] == "transverse_phase_unresolved"
    assert detector._phase_length_history and max(detector._phase_length_history) < 90


def test_calibrated_phase_does_not_fall_back_to_wrong_brightness_pairs(monkeypatch):
    frame = np.full((28, 220), 90, dtype=np.uint8)
    monkeypatch.setattr(detector_module, "dark_body_intervals", lambda *_: [(20, 180)])
    detector = DropletDetector(DetectorConfig(
        measurement_mode="generation_plug", generation_phase_mode="transverse_bounded"), DebugConfig())
    result, trace = detect_rectified_generation_plugs(frame, detector=detector)
    assert result.plug_lengths_px == []
    assert trace["interval_source"] == "transverse_phase_unresolved"
