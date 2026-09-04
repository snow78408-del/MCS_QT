from __future__ import annotations

import math

import numpy as np
import pytest

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
