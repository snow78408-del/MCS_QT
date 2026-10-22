from __future__ import annotations

import numpy as np
import pytest

from backend.vision.capsule_profile import calibrated_phase_intervals
from backend.vision.config import DebugConfig, DetectorConfig
from backend.vision.detector import DropletDetector


def phase_image(signal: np.ndarray) -> np.ndarray:
    frame = np.full((32, len(signal)), 128.0)
    frame[2:8] -= signal / 2
    frame[24:30] += signal / 2
    return np.rint(frame).astype(np.uint8)


def test_internal_peaks_without_carrier_do_not_split_a_body() -> None:
    x = np.arange(360)
    signal = np.full(360, -24.0)
    signal[110:200] = 20
    signal += 16 * np.exp(-((x - 130) / 7) ** 2)
    signal += 18 * np.exp(-((x - 170) / 7) ** 2)
    intervals = calibrated_phase_intervals(phase_image(signal), 31)
    assert len(intervals) == 1
    assert intervals[0][0] < 115 and intervals[0][1] > 195


@pytest.mark.parametrize('fraction', [.02, .1])
def test_compact_trace_preserves_measurements_without_preprocessing(monkeypatch, fraction) -> None:
    signal = np.full(360, -24.0)
    for a, b in [(30, 95), (140, 205), (250, 315)]:
        signal[a:b] = 24
    full, compact = [DropletDetector(DetectorConfig(
        measurement_mode='generation_plug', generation_phase_mode='transverse_bounded',
        generation_phase_support_fraction=fraction), DebugConfig()) for _ in range(2)]

    def forbidden(*args):
        pytest.fail('Unused enhanced profile calculated on the compact path')

    monkeypatch.setattr(compact, '_preprocess', forbidden)
    for shift in [0, 2, 4]:
        frame = np.roll(phase_image(signal), shift, axis=1)
        trace = {}
        a = full.detect(frame, channel_width_px=31, flow_axis='x')
        b = compact.detect(frame, channel_width_px=31, flow_axis='x',
                           profile_diagnostics=False, trace=trace)
        assert a.plug_lengths_px == b.plug_lengths_px
        assert a.equivalent_diameters_px == b.equivalent_diameters_px
        assert a.diameter_valid == b.diameter_valid
        np.testing.assert_array_equal(a.centers, b.centers)
        np.testing.assert_array_equal(a.helper_mask, b.helper_mask)
        assert full._phase_length_history == compact._phase_length_history
        assert trace['corrected'] is None
        assert trace['profile_diagnostics_computed'] is False


def test_endpoint_tightening_is_bounded_and_preserves_explicit_stronger_support() -> None:
    x = np.arange(360, dtype=float)
    signal = np.full(360, -24.0)
    for a, b in [(45, 105), (155, 215), (265, 325)]:
        signal += 48 * (1 / (1 + np.exp(-(x-a)/4)) - 1 / (1 + np.exp(-(x-b)/4)))
    frame = phase_image(signal)
    original = calibrated_phase_intervals(frame, 31, endpoint_support_fraction=.02)
    refined = calibrated_phase_intervals(frame, 31)
    assert len(original) == len(refined) == 3
    for (a, b), (c, d) in zip(original, refined):
        assert a <= c < d <= b and max(c-a, b-d) <= 31*.25
    assert calibrated_phase_intervals(frame, 31, support_fraction=.1) == calibrated_phase_intervals(
        frame, 31, support_fraction=.1, endpoint_support_fraction=.1)


@pytest.mark.parametrize('value', [0, -.1, .5, float('nan'), float('inf')])
def test_invalid_endpoint_support(value) -> None:
    with pytest.raises(ValueError, match='endpoint support'):
        calibrated_phase_intervals(np.zeros((32, 360), np.uint8), 31,
                                   endpoint_support_fraction=value)
