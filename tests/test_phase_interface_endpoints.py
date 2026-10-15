from __future__ import annotations

import numpy as np
import pytest

from backend.vision.capsule_profile import calibrated_phase_intervals
from backend.vision.config import DebugConfig, DetectorConfig
from backend.vision.detector import DropletDetector
from backend.vision.rectified_measurement import detect_rectified_generation_plugs, trace_summary


def phase_frame() -> np.ndarray:
    x = np.arange(360, dtype=float)
    signal = np.full(x.shape, -24.0)
    for left, right in [(45, 105), (155, 215), (265, 325)]:
        signal += 48 * (1 / (1 + np.exp(-(x - left) / 4))
                        - 1 / (1 + np.exp(-(x - right) / 4)))
    frame = np.full((32, 360), 128.0)
    frame[2:8] -= signal / 2
    frame[24:30] += signal / 2
    return np.rint(frame).astype(np.uint8)


def test_review_threshold_contracts_optical_tails_without_fixed_pixel_trim() -> None:
    frame = phase_frame()
    old = calibrated_phase_intervals(frame, 31)
    new = calibrated_phase_intervals(frame, 31, support_fraction=.1)
    assert len(old) == len(new) == 3
    assert all(a < c < d < b for (a, b), (c, d) in zip(old, new))
    shifted = np.pad(frame, ((0, 0), (17, 0)), mode="edge")
    assert calibrated_phase_intervals(shifted, 31, support_fraction=.1) == [
        (a + 17, b + 17) for a, b in new]
    assert calibrated_phase_intervals(np.full_like(frame, 128), 31, support_fraction=.1) == []


@pytest.mark.parametrize("fraction", [0, .5, -.1, float("nan"), float("inf")])
def test_rejects_invalid_support_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="support fraction"):
        calibrated_phase_intervals(phase_frame(), 31, support_fraction=fraction)
    with pytest.raises(ValueError, match="support fraction"):
        DropletDetector(DetectorConfig(generation_phase_support_fraction=fraction), DebugConfig())


def test_explicit_setting_is_recorded_in_measurement_trace() -> None:
    detector = DropletDetector(DetectorConfig(
        measurement_mode="generation_plug", generation_phase_mode="transverse_bounded",
        generation_phase_support_fraction=.1), DebugConfig())
    _, trace = detect_rectified_generation_plugs(phase_frame(), detector=detector)
    assert trace_summary(trace)["phase_support_fraction"] == .1
    assert len(trace["selected_intervals"]) == 3
