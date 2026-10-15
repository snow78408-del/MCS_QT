from __future__ import annotations

import numpy as np
import pytest

from backend.vision.capsule_profile import calibrated_phase_intervals
from backend.vision.config import DebugConfig, DetectorConfig
from backend.vision.detector import DropletDetector
from backend.vision.rectified_measurement import detect_rectified_generation_plugs


def phase_frame(centres: tuple[int, ...], *, jet_end: int = 0) -> np.ndarray:
    """Calibrated shoulder ordering with a carrier-separated droplet train."""
    height, width = 53, 1100
    x = np.arange(width)
    signal = np.full(width, -12.0)
    for centre in centres:
        signal += 48.0 * np.exp(-0.5 * ((x - centre) / 20.0) ** 2)
    signal[:jet_end] = 36.0
    frame = np.full((height, width), 100.0)
    frame[max(1, int(height * .08)):int(height * .25)] -= signal / 2
    frame[int(height * .76):int(height * .94)] += signal / 2
    return np.uint8(np.clip(frame, 0, 255))


@pytest.mark.parametrize("offset", [0, 180, 420])
def test_complete_droplets_follow_moving_formation_region(offset: int) -> None:
    centres = (100 + offset, 260 + offset)
    detector = DropletDetector(DetectorConfig(
        measurement_mode="generation_plug",
        generation_phase_mode="transverse_bounded"), DebugConfig())
    result, trace = detect_rectified_generation_plugs(
        phase_frame(centres), detector=detector)
    assert len(result.plug_lengths_px) == 2
    for (left, right, length), centre in zip(trace["selected_intervals"], centres):
        assert abs((left + right) / 2 - centre) <= 2
        assert 95 < length < 125


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("jet_end", [100, 350, 650])
def test_jet_dominated_frame_keeps_separated_droplets(reverse: bool, jet_end: int) -> None:
    frame = phase_frame((jet_end + 150, jet_end + 330), jet_end=jet_end)
    if reverse:
        frame = np.ascontiguousarray(frame[:, ::-1])
    detector = DropletDetector(DetectorConfig(
        measurement_mode="generation_plug",
        generation_phase_mode="transverse_bounded"), DebugConfig())
    result, trace = detect_rectified_generation_plugs(frame, detector=detector)
    assert len(result.plug_lengths_px) == 2
    assert all(95 < length < 125 for _, _, length in trace["selected_intervals"])
    for left, right, _ in trace["selected_intervals"]:
        if reverse:
            assert right < frame.shape[1] - jet_end
        else:
            assert left > jet_end


@pytest.mark.parametrize("centre", [0, 1099])
def test_edge_truncated_body_is_not_a_complete_droplet(centre: int) -> None:
    assert calibrated_phase_intervals(phase_frame((centre,)), 52.0) == []


def test_continuous_jet_is_not_a_complete_droplet() -> None:
    assert calibrated_phase_intervals(phase_frame((), jet_end=1100), 52.0) == []


def test_connected_thread_is_rejected_at_each_position() -> None:
    for centre in (100, 400, 800):
        frame = phase_frame((centre,))
        # A persistent centre/shoulder contrast in both flanks represents
        # unresolved thread evidence, even with a local bulge in the phase.
        frame[18:34] += 30
        assert calibrated_phase_intervals(frame, 52.0) == []
