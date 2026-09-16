from __future__ import annotations

import numpy as np
import pytest

from backend.vision.line_counter import ContinuousLineCounter


def calibrated_counter() -> ContinuousLineCounter:
    counter = ContinuousLineCounter(warmup_frames=8)
    for i, value in enumerate([2, 2, 30, 30, 2, 2, 30, 30, 2]):
        counter.observe_score(value, i, i * .01)
    return counter


def test_full_passages_once_and_half_open_window() -> None:
    counter = calibrated_counter()
    for i, score in enumerate([2, 30, 30, 20, 2, 2, 30, 30, 2], start=9):
        counter.observe_score(score, i, i * .01)
    first = counter.window(.09, .17)
    assert first.valid
    assert first.count == 1  # event at .17 belongs to the NEXT window
    assert first.rate_hz == pytest.approx(12.5)
    assert counter.window(.13, .17).count == 1


def test_gap_invalidates_window_and_does_not_count_partial_passage() -> None:
    counter = calibrated_counter()
    counter.observe_score(30, 9, .09)
    counter.observe_score(2, 11, .11)
    counter.observe_score(2, 12, .12)
    result = counter.window(.08, .12)
    assert not result.valid
    assert result.rate_hz is None
    assert counter.window(.11, .12).valid is False


def test_constant_background_warmup_and_history_expiry_are_invalid() -> None:
    counter = ContinuousLineCounter(history_limit=10, warmup_frames=8)
    for i in range(30):
        counter.observe_score(30, i, i * .01)
    assert not counter.window(.22, .28).valid
    assert not counter.window(0, .28).valid
    assert not counter.window(.28, .4).valid


def test_low_contrast_texture_does_not_self_calibrate_as_droplet_phases() -> None:
    counter = ContinuousLineCounter(warmup_frames=8)
    for i in range(100):
        counter.observe_score(8 if i % 4 in (1, 2) else 2, i, i * .01)
    assert not counter.passage_times()
    assert not counter.window(.5, .9).valid
    # A later visible phase sequence can calibrate without a restart or delay.
    for i in range(100, 140):
        counter.observe_score(30 if i % 4 in (1, 2) else 2, i, i * .01)
    assert counter.passage_times()
    assert counter.window(1.2, 1.35).valid


def test_reset_and_time_reversal_discard_old_events() -> None:
    counter = calibrated_counter()
    counter.observe_score(30, 9, .09)
    counter.observe_score(2, 10, .1)
    assert counter.window(.08, .1).valid
    counter.observe_score(2, 11, .01)
    assert not counter.window(.08, .1).valid
    counter.reset()
    assert not counter.window(.08, .1).valid


def test_gate_uses_only_narrow_strip_and_requires_manual_walls() -> None:
    counter = ContinuousLineCounter(warmup_frames=8)
    frame = np.full((100, 400), 100, np.uint8)
    counter.observe_frame(frame, 0, 0, [])
    assert not counter.window(0, 1).valid
    walls = [dict(x1=0, y1=.2, x2=1, y2=.2), dict(x1=0, y1=.8, x2=1, y2=.8)]
    for i in range(20):
        image = frame.copy()
        if i % 4 in (1, 2):
            image[30:70, 235:245] = 180
        counter.observe_frame(image, i, i * .01, walls)
    assert counter._maps[0].shape == (60, 5)
    assert counter.window(.12, .19).valid
    assert counter.window(.12, .19).count == 1


def test_completed_size_window_bounds_match_frequency_query() -> None:
    from backend.vision.config import MetricsConfig
    from backend.vision.metrics import MetricsCalculator

    metrics = MetricsCalculator(MetricsConfig(realtime_window_ms=100))
    metrics._update_realtime_window(crossed_track_diameters={}, crossed_track_bead_counts={}, timestamp=1.0)
    assert metrics._completed_bounds == (None, None)
    metrics._update_realtime_window(crossed_track_diameters={1: 50.0}, crossed_track_bead_counts={}, timestamp=1.05)
    completed = metrics._update_realtime_window(crossed_track_diameters={}, crossed_track_bead_counts={}, timestamp=1.11)
    assert completed[0] == [50.0]
    assert metrics._completed_bounds == pytest.approx((1.0, 1.1))
    metrics.reset()
    assert metrics._completed_bounds == (None, None)


def test_long_analysis_stall_does_not_relabel_old_sizes_as_current() -> None:
    from backend.vision.config import MetricsConfig
    from backend.vision.metrics import MetricsCalculator

    metrics = MetricsCalculator(MetricsConfig(realtime_window_ms=100))
    metrics._update_realtime_window(crossed_track_diameters={1: 50}, crossed_track_bead_counts={}, timestamp=1.0)
    completed = metrics._update_realtime_window(crossed_track_diameters={}, crossed_track_bead_counts={}, timestamp=1.35)
    assert completed[0] == []
    assert metrics._completed_bounds == pytest.approx((1.2, 1.3))


def test_sampling_counts_before_size_admission_with_acquisition_time() -> None:
    from backend.orchestrator.vision_adapter import PipelineVisionService

    service = PipelineVisionService()
    frame = np.zeros((40, 100), np.uint8)
    order = []

    class Counter:
        def observe_frame(self, image, frame_id, timestamp, walls, *, line_ratio):
            assert line_ratio == .6
            order.append(("count", frame_id, timestamp))

    service._line_counter = Counter()
    service._sampling_queue.put((1, 12345.0, frame))
    service._frame_metadata[1] = {"capture_monotonic": 10.0, "hardware_frame_id": 99}

    def submit(frame_id, timestamp, image):
        order.append(("size", frame_id, timestamp))
        service._stop_event.set()

    service._submit_processing_frame = submit
    service._sampling_loop()
    assert order == [("count", 99, 10.0), ("size", 1, 12345.0)]


def test_ten_second_period_has_continuous_counts_but_no_early_completion() -> None:
    from backend.vision.config import MetricsConfig
    from backend.vision.metrics import MetricsCalculator

    counter = ContinuousLineCounter()
    metrics = MetricsCalculator(MetricsConfig(realtime_window_ms=10_000))
    for i in range(2001):
        timestamp = i / 100
        counter.observe_score(30 if i % 4 in (1, 2) else 2, i, timestamp)
        if i % 5 == 0:
            completed = metrics._update_realtime_window(
                crossed_track_diameters={}, crossed_track_bead_counts={}, timestamp=timestamp)
            assert completed[-1] == i // 1000
    assert metrics._completed_bounds == (10.0, 20.0)
    frequency = counter.window(*metrics._completed_bounds)
    assert frequency.valid
    assert frequency.count == 250
    assert frequency.rate_hz == 25.0
