from __future__ import annotations

import math

import pytest

from backend.vision.config import DebugConfig, DetectorConfig
from backend.vision.detector import DropletDetector
from backend.vision.plug_geometry import (
    PlugGeometry,
    effective_area_px2,
    equivalent_sphere_diameter_px,
    plug_length_px_from_equivalent_diameter,
    plug_volume_px3,
    sphere_diameter_um,
)

# Measured on the 2026-09-18 observation window; see
# docs/plant_model_measured_20260919.md.
DUCT_GAP_PX = 27.011
SCALE_UM_PER_PX = 1.8511
MEASURED_PLUG_LENGTH_PX = 107.0


def test_square_duct_area_collapses_to_rounded_rectangle_term() -> None:
    assert effective_area_px2(50.0, 50.0) == pytest.approx(0.94635 * 50.0 * 50.0, rel=1e-4)


def test_volume_is_the_source_of_the_equivalent_diameter() -> None:
    volume = plug_volume_px3(120.0, 30.0, 30.0)
    diameter = equivalent_sphere_diameter_px(120.0, 30.0, 30.0)
    assert math.pi * diameter ** 3 / 6.0 == pytest.approx(volume)


def test_equivalent_diameter_round_trips_through_its_inverse() -> None:
    for length in (70.0, 94.0, 107.0, 160.0, 300.0):
        diameter = equivalent_sphere_diameter_px(length, 27.0, 27.0)
        assert plug_length_px_from_equivalent_diameter(diameter, 27.0, 27.0) == pytest.approx(
            length
        )


def test_measured_window_plug_reports_93_6_um() -> None:
    geometry = PlugGeometry(MEASURED_PLUG_LENGTH_PX, DUCT_GAP_PX, SCALE_UM_PER_PX)
    assert geometry.duct_width_um == pytest.approx(50.0, abs=0.01)
    assert geometry.length_um == pytest.approx(198.07, abs=0.01)
    assert geometry.aspect_ratio == pytest.approx(3.96, abs=0.01)
    assert geometry.equivalent_diameter_px == pytest.approx(50.557, abs=0.01)
    assert geometry.equivalent_diameter_um == pytest.approx(93.586, abs=0.02)
    assert geometry.volume_nl == pytest.approx(0.4292, abs=0.001)


def test_pid_target_of_110_um_is_a_far_longer_plug_than_the_working_point() -> None:
    target_length = plug_length_px_from_equivalent_diameter(
        110.0 / 1.725, DUCT_GAP_PX, DUCT_GAP_PX
    )
    assert target_length == pytest.approx(205.65, abs=0.05)
    assert target_length > 1.9 * MEASURED_PLUG_LENGTH_PX


def test_sphere_diameter_matches_equivalent_diameter() -> None:
    volume = plug_volume_px3(MEASURED_PLUG_LENGTH_PX, DUCT_GAP_PX, DUCT_GAP_PX)
    assert sphere_diameter_um(volume) == pytest.approx(
        equivalent_sphere_diameter_px(MEASURED_PLUG_LENGTH_PX, DUCT_GAP_PX, DUCT_GAP_PX)
    )


def test_detector_delegates_to_the_shared_formula() -> None:
    for correction in (1.0, 1.08, 1.4):
        detector = DropletDetector(
            DetectorConfig(generation_volume_correction=correction), DebugConfig()
        )
        for length in (70.0, 107.0, 240.0):
            assert detector._plug_equivalent_diameter_px(length, 27.011, 27.011) == pytest.approx(
                equivalent_sphere_diameter_px(length, 27.011, 27.011, correction)
            )


def test_degenerate_plugs_are_rejected_not_clamped() -> None:
    with pytest.raises(ValueError):
        plug_volume_px3(5.0, 27.0, 27.0)
    with pytest.raises(ValueError):
        plug_volume_px3(107.0, 27.0, 27.0, correction=0.0)
    with pytest.raises(ValueError):
        effective_area_px2(0.0, 27.0)
    detector = DropletDetector(DetectorConfig(), DebugConfig())
    assert detector._plug_equivalent_diameter_px(0.0, 27.0, 27.0) is None
    assert detector._plug_equivalent_diameter_px(5.0, 27.0, 27.0) is None
