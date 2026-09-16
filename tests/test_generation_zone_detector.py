from __future__ import annotations

import math

import numpy as np
import pytest
import cv2

from backend.vision.config import DebugConfig, DetectorConfig
from backend.vision.detector import DropletDetector


def test_generation_plug_length_is_converted_to_equivalent_diameter() -> None:
    frame=np.full((60,500),100,dtype=np.uint8)
    frame[8:52,80:220]=190
    frame[8:52,300:450]=190
    config=DetectorConfig(
        measurement_mode="generation_plug",
        generation_channel_height_um=50.0,
        generation_channel_width_um=50.0,
        generation_min_length_ratio=2.5,
        generation_polarity="brighter",
    )
    detector=DropletDetector(config,DebugConfig())
    detector.configure_expected_diameter(105.0,1.25)

    result=detector.detect(frame)

    assert result.plug_lengths_px == pytest.approx([140.0,151.0],abs=2.0)
    first_length_um=result.plug_lengths_px[0]*1.25
    effective_area=50.0*50.0-(4.0-math.pi)*(4.0/50.0)**-2
    expected_um=(6.0*effective_area*(first_length_um-50.0/3.0)/math.pi)**(1.0/3.0)
    assert result.equivalent_diameters_px[0]*1.25 == pytest.approx(expected_um)
    assert all(result.diameter_valid)


def test_generation_detector_does_not_use_pid_target_as_a_size_gate() -> None:
    frame=np.full((60,300),100,dtype=np.uint8)
    frame[8:52,60:200]=190
    config=DetectorConfig(
        measurement_mode="generation_plug",
        generation_min_length_ratio=2.5,
        generation_polarity="brighter",
    )
    detector=DropletDetector(config,DebugConfig())
    detector.configure_expected_diameter(10.0,1.25)
    first=detector.detect(frame)
    detector.configure_expected_diameter(500.0,1.25)
    second=detector.detect(frame)

    assert first.plug_lengths_px == second.plug_lengths_px
    assert first.equivalent_diameters_px == second.equivalent_diameters_px


def test_generation_detector_rejects_gap_and_keeps_full_capsule_outlines() -> None:
    frame = np.full((60, 420), 150, dtype=np.uint8)
    frame[8:52, 40:150] = 90
    frame[8:52, 230:360] = 90
    detector = DropletDetector(
        DetectorConfig(
            measurement_mode="generation_plug",
            generation_min_length_ratio=1.5,
            # Deliberately opposite to the droplets: a complete 2-D capsule
            # must remain valid despite a phase-contrast polarity reversal.
            generation_polarity="brighter",
        ),
        DebugConfig(),
    )
    detector.configure_expected_diameter(100.0, 1.25)

    result = detector.detect(frame)

    assert result.plug_lengths_px == pytest.approx([110.0, 130.0], abs=2.0)
    assert all(abs(length - 80.0) > 5.0 for length in result.plug_lengths_px)


@pytest.mark.parametrize("inverted", [False, True])
def test_weak_center_meniscus_is_recovered_from_capsule_body(inverted: bool) -> None:
    frame = np.full((40, 400), 100, dtype=np.uint8)
    # Strong longitudinal outlines with weak centre contrast at both ends.
    for left, right in [(40, 140), (220, 320)]:
        cv2.rectangle(frame, (left, 7), (right, 32), 180, 2)
        frame[14:26, left-2:left+3] = 100
        frame[14:26, right-2:right+3] = 100
    if inverted:
        frame = 255-frame
    detector = DropletDetector(DetectorConfig(measurement_mode="generation_plug"), DebugConfig())
    detector.configure_expected_diameter(100.0, 1.5)
    result = detector.detect(frame)
    assert len(result.plug_lengths_px) == 2
    assert result.plug_lengths_px == pytest.approx([102, 102], abs=5)


def test_partial_capsules_and_empty_uniform_channel_are_not_sizes() -> None:
    detector = DropletDetector(DetectorConfig(measurement_mode="generation_plug"), DebugConfig())
    frame = np.full((40, 400), 100, dtype=np.uint8)
    assert not detector.detect(frame).plug_lengths_px
    frame[5:35, :100] = 180
    frame[5:35, 300:] = 180
    assert not detector.detect(frame).plug_lengths_px


def test_body_outline_validation_does_not_depend_on_enhancement(monkeypatch) -> None:
    frame = np.full((40, 400), 100, dtype=np.uint8)
    frame[6:34, 80:220] = 180
    detector = DropletDetector(DetectorConfig(measurement_mode="generation_plug"), DebugConfig())
    # Simulate enhancement flattening the boundary; original evidence survives.
    monkeypatch.setattr(detector, "_preprocess", lambda image: np.full_like(image, 100))
    assert detector.detect(frame).plug_lengths_px == pytest.approx([140], abs=3)


def test_vectorized_outline_support_matches_column_reference() -> None:
    gradient = np.random.default_rng(4).uniform(0, 10, size=(35, 150))
    expected = 0
    for column in range(12, 128):
        rows = np.flatnonzero(gradient[4:31, column] >= 8)
        expected += int(len(rows) >= 2 and 5 <= rows[-1]-rows[0] <= 38)
    actual = DropletDetector._capsule_outline_support(
        gradient, left=10, right=130, row_margin=4, edge_threshold=8, reference_width_px=30)
    assert actual == expected / 116


@pytest.mark.parametrize("phase", [0.0, 0.7, 1.4, 2.1])
def test_fixed_walls_shading_and_compression_texture_are_not_capsules(phase):
    y, x = np.mgrid[:40, :500]
    illumination = 9 * np.sin(x / 14 + phase) + 5 * np.cos(x / 47)
    walls = 25 * np.exp(-((y - 5) / 1.8) ** 2) - 30 * np.exp(-((y - 34) / 2) ** 2)
    texture = 2 * np.sin(x / 7 + y / 4) + np.random.default_rng(42).normal(0, .5, x.shape)
    gray = np.uint8(np.clip(100 + illumination + walls + texture, 0, 255))
    ok, jpeg = cv2.imencode('.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, 75])
    assert ok
    gray = cv2.imdecode(jpeg, cv2.IMREAD_GRAYSCALE)
    detector = DropletDetector(DetectorConfig(measurement_mode="generation_plug",
        generation_min_length_ratio=.9), DebugConfig())
    detector.configure_expected_diameter(100, 1.5)
    assert not detector.detect(gray).centers


@pytest.mark.parametrize("polarity", [-1, 1])
@pytest.mark.parametrize("background", [40, 150])
def test_visible_raw_outline_survives_brightness_and_polarity_changes(polarity, background):
    frame = np.full((40, 400), background, np.uint8)
    cv2.rectangle(frame, (50, 7), (180, 32), background + polarity * 24, 2)
    # Keep weak/missing central menisci while retaining their complete outline.
    frame[14:26, 48:53] = background
    frame[14:26, 178:183] = background
    detector = DropletDetector(DetectorConfig(measurement_mode="generation_plug"), DebugConfig())
    detector.configure_expected_diameter(100, 1.5)
    assert detector.detect(frame).plug_lengths_px == pytest.approx([132], abs=5)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), 256])
def test_raw_contrast_threshold_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="raw_outline_contrast"):
        DropletDetector(DetectorConfig(generation_min_raw_outline_contrast=value), DebugConfig())


@pytest.mark.parametrize("polarity", [-1, 1])
@pytest.mark.parametrize("contrast", [24, 80])
@pytest.mark.parametrize("vertical", [False, True])
def test_close_capsules_do_not_use_neighbour_as_background(polarity, contrast, vertical):
    frame = np.full((40, 420), 100, np.uint8)
    for left in (40, 178):
        cv2.rectangle(frame, (left, 7), (left + 130, 32), 100 + polarity * contrast, 2)
    detector = DropletDetector(DetectorConfig(measurement_mode="generation_plug"), DebugConfig())
    detector.configure_expected_diameter(100, 1.5)
    result = detector.detect(frame.T.copy() if vertical else frame)
    assert result.plug_lengths_px == pytest.approx([134, 134], abs=3)
    assert all(result.diameter_valid)


@pytest.mark.parametrize("neighbour", ["partial", "too_short"])
def test_rejected_neighbour_is_excluded_from_carrier_reference(neighbour):
    frame = np.full((40, 300), 100, np.uint8)
    # The left body cannot be measured, but it is still not carrier fluid.
    left = -30 if neighbour == "partial" else 40
    cv2.rectangle(frame, (left, 7), (70, 32), 180, 2)
    cv2.rectangle(frame, (78, 7), (208, 32), 180, 2)
    detector = DropletDetector(DetectorConfig(measurement_mode="generation_plug"), DebugConfig())
    detector.configure_expected_diameter(100, 1.5)
    assert detector.detect(frame).plug_lengths_px == pytest.approx([134], abs=3)


def test_unresolved_carrier_gap_does_not_borrow_neighbour_pixels():
    frame = np.full((40, 420), 100, np.uint8)
    for left in (40, 176):
        cv2.rectangle(frame, (left, 7), (left + 130, 32), 180, 2)
    detector = DropletDetector(DetectorConfig(measurement_mode="generation_plug"), DebugConfig())
    detector.configure_expected_diameter(100, 1.5)
    assert not detector.detect(frame).centers
