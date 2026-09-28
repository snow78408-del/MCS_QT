from __future__ import annotations

import numpy as np
import pytest

from backend.vision.flow_locking import GapEstimate, VelocityEstimate
from backend.vision.offline_analysis import periodic_phase_average, screen_velocity, spatial_pitch, velocity_windows


def moving_profiles(frames: int = 80, pitch: float = 80.4, speed: float = -23.7) -> np.ndarray:
    x = np.arange(401)
    phase = 2 * np.pi * (x[None, :] - speed * np.arange(frames)[:, None]) / pitch
    return 80 + 20 * np.cos(phase) + 6 * np.cos(2 * phase + 0.4) + 0.03 * x


def test_long_record_preserves_shape_without_cumulative_extrapolation() -> None:
    result = periodic_phase_average(moving_profiles(frames=1500), 80.4)
    assert result.usable
    assert len(result.accepted_frames) > 1450
    assert result.min_bin_support > 3
    assert result.contrast_gray > 38
    assert all(0 <= value < 80.4 for value in result.frame_phase_px)
    template = 20 * np.cos(2 * np.pi * np.arange(80) / 80) + 6 * np.cos(4 * np.pi * np.arange(80) / 80 + 0.4)
    # Shape is recovered up to arbitrary phase, even after hundreds of periods of travel.
    best = max(np.corrcoef(result.profile, np.roll(template, n))[0, 1] for n in range(80))
    assert best > 0.995


def test_integer_alias_has_identical_phase_average() -> None:
    first = periodic_phase_average(moving_profiles(speed=18), 80.4)
    alias = periodic_phase_average(moving_profiles(speed=18 + 80.4), 80.4)
    assert alias.profile == pytest.approx(first.profile, abs=1e-8)


def test_static_structure_cannot_produce_temporal_phase_evidence() -> None:
    values = np.tile(np.sin(np.arange(401) / 10), (30, 1))
    result = periodic_phase_average(values, 80)
    assert not result.usable
    assert result.intensity_duty is None


def test_bad_frames_are_reported_not_silently_averaged() -> None:
    values = moving_profiles()
    # A blank frame contains no moving periodic shape after background removal.
    values[-10:] = values[:-10].mean(axis=0)
    result = periodic_phase_average(values, 80.4)
    assert set(range(70, 80)).issubset(result.rejected_frames)


def test_pitch_is_measured_in_pixels_without_a_scale() -> None:
    pitch, strength = spatial_pitch(moving_profiles(), 40, 110)
    assert pitch == pytest.approx(80.4, abs=0.5)
    assert strength > 0.9


def estimate(rate: float = -48, *, peak: float = 10, residual: float = 0.2) -> VelocityEstimate:
    return VelocityEstimate(px_per_frame=rate, ok=True, residual_px=residual,
                            detail=[GapEstimate(gap, rate * gap, rate, peak, residual, True) for gap in (1, 2, 3)])


def test_good_correlation_and_direction_do_not_resolve_alias_order() -> None:
    result = screen_velocity(estimate(), 165, direction=-1)
    assert result["status"] == "ALIAS_UNRESOLVED"
    assert -213 in result["candidates"]
    assert -48 in result["candidates"]
    assert result["selected_px_per_frame"] is None
    assert not result["control_authorized"]


def test_independent_bound_can_select_branch_but_is_recorded_as_conditional() -> None:
    result = screen_velocity(estimate(70), 165, direction=-1, max_displacement=120,
                             bound_source="synthetic known displacement bound")
    assert result["status"] == "CONDITIONAL"
    assert result["selected_px_per_frame"] == -95
    assert result["candidates_exhaustive"]


@pytest.mark.parametrize("peak,residual", [(1, 0.1), (10, 25)])
def test_weak_or_inconsistent_estimates_are_rejected(peak: float, residual: float) -> None:
    result = screen_velocity(estimate(peak=peak, residual=residual), 165)
    assert result["status"] == "REJECTED"
    assert not result["candidates"]


def test_an_overwide_bound_preserves_ambiguity() -> None:
    result = screen_velocity(estimate(), 165, direction=-1, max_displacement=400, bound_source="test")
    assert result["status"] == "ALIAS_UNRESOLVED"


def test_an_incompatible_bound_rejects_instead_of_clamping() -> None:
    result = screen_velocity(estimate(), 165, max_displacement=10, bound_source="test")
    assert result["status"] == "REJECTED"


def test_bounds_need_provenance() -> None:
    with pytest.raises(ValueError):
        screen_velocity(estimate(), 165, max_displacement=50)


def test_windows_keep_tail_and_do_not_invent_true_frequency() -> None:
    rows = velocity_windows(moving_profiles(frames=145, pitch=80, speed=-10), 0.01,
                            window=60, step=60, minimum_pitch=40, maximum_pitch=110)
    assert [row["start_frame"] for row in rows] == [0, 60, 85]
    assert all(row["status"] == "ALIAS_UNRESOLVED" for row in rows)
    assert all(row["selected_passage_frequency_hz"] is None for row in rows)


def test_sudden_motion_change_is_not_a_steady_velocity() -> None:
    values = np.concatenate([moving_profiles(frames=40, pitch=80, speed=-5),
                             moving_profiles(frames=40, pitch=80, speed=-30)])
    row = velocity_windows(values, 0.01, window=80, step=80,
                           minimum_pitch=40, maximum_pitch=110)[0]
    assert row["status"] == "REJECTED"
    assert row["selected_px_per_frame"] is None
    assert not row["steady_candidate"]


def test_static_window_has_no_velocity_candidate() -> None:
    values = np.tile(np.sin(np.arange(401) / 10), (30, 1))
    rows = velocity_windows(values, 0.01, window=30, step=30)
    assert rows[0]["status"] == "REJECTED"
    assert rows[0]["selected_px_per_frame"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_input_is_rejected(value: float) -> None:
    values = moving_profiles()
    values[0, 0] = value
    with pytest.raises(ValueError):
        periodic_phase_average(values, 80.4)
