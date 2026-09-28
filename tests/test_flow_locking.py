"""Offline regression tests for ROI-free channel locking and velocity measurement.

Everything here runs without a camera, a pump or a display.  The reference
recording is used when it is present and skipped otherwise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.vision.flow_locking import (
    ChannelLock,
    activity_band,
    centreline_profiles,
    channel_start,
    channel_profiles,
    duct_centre_line,
    duct_walls,
    dynamic_shift,
    estimate_velocity,
    measure_flow,
    pitch_from_autocorrelation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CANDIDATES = (
    REPO_ROOT / "output" / "crops" / "ref_video.npy",
    Path(r"E:\MCS_QT\output\crops\ref_video.npy"),
    Path(r"C:\missing-test-data\ref_video.npy"),
)


def _reference_path() -> Path | None:
    for candidate in REFERENCE_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def train_stack(frames: int = 48, height: int = 140, width: int = 800, band: tuple[int, int] = (50, 100),
                pitch: float = 150.0, droplet_length: float = 105.0, speed: float = 8.0,
                static_level: float = 140.0, amplitude: float = 55.0, noise: float = 2.0,
                seed: int = 1) -> np.ndarray:
    """Static walls and texture plus a flat-top droplet train moving ``speed`` px/frame."""
    random = np.random.default_rng(seed)
    columns = np.arange(width, dtype=np.float32)
    background = np.full((height, width), static_level, np.float32)
    background += random.normal(0.0, noise, (height, width)).astype(np.float32)
    background[band[0] - 6:band[0] - 4, :] += 60.0
    background[band[1] + 5:band[1] + 7, :] += 60.0
    rows = np.zeros((height, 1), np.float32)
    rows[band[0]:band[1] + 1] = 1.0
    kernel = np.exp(-0.5 * (np.arange(-9, 10) / 2.5) ** 2)
    kernel /= kernel.sum()
    stack = np.empty((frames, height, width), np.float32)
    for index in range(frames):
        phase = np.mod(columns - index * speed, pitch)
        train = np.convolve((phase < droplet_length).astype(np.float32), kernel, mode="same")
        stack[index] = background + amplitude * rows * train[None, :]
    return stack


def smooth_train_stack(frames: int = 48, height: int = 140, width: int = 720,
                       band: tuple[int, int] = (50, 100), pitch: float = 160.0,
                       droplet_width: float = 80.0, speed: float = 30.37,
                       noise: float = 3.0, seed: int = 2) -> np.ndarray:
    """A nearly sinusoidal train: the degenerate case for a whitened correlation."""
    random = np.random.default_rng(seed)
    columns = np.arange(width, dtype=np.float32)
    background = np.full((height, width), 140.0, np.float32)
    background += random.normal(0.0, noise, (height, width)).astype(np.float32)
    rows = np.zeros((height, 1), np.float32)
    rows[band[0]:band[1] + 1] = 1.0
    stack = np.empty((frames, height, width), np.float32)
    for index in range(frames):
        train = np.zeros(width, np.float32)
        for k in range(-3, int(width / pitch) + 4):
            centre = k * pitch + 0.5 * droplet_width + index * speed
            train += np.exp(-0.5 * ((columns - centre) / (droplet_width / 3.0)) ** 2)
        stack[index] = background + 55.0 * rows * train[None, :]
    return stack


def profiles_of(stack: np.ndarray):
    top, bot, _ = activity_band(stack)
    profiles = channel_profiles(stack, top, bot)
    pitch, strength = pitch_from_autocorrelation(profiles)
    return profiles, pitch, strength


def test_moving_train_is_measured_through_a_static_background() -> None:
    """The static channel walls dominate the raw signal; they must not pin the peak at zero."""
    stack = train_stack(speed=12.0)
    measurement = measure_flow(stack, dt=1.0 / 320.0)

    assert measurement.verdict == "FLOWING"
    assert measurement.velocity.ok
    assert measurement.velocity.px_per_frame == pytest.approx(12.0, abs=0.5)
    assert measurement.band_height_px >= 8


def test_sign_convention_follows_the_column_axis() -> None:
    right = measure_flow(train_stack(speed=12.0), dt=1.0 / 320.0)
    left = measure_flow(train_stack(speed=-12.0), dt=1.0 / 320.0)

    assert right.velocity.px_per_frame > 0
    assert left.velocity.px_per_frame < 0


def test_periodic_train_is_not_reported_as_frozen() -> None:
    """Whitening every spectral bin makes a smooth periodic train read as 0.02 px/frame."""
    stack = smooth_train_stack(speed=30.37)
    measurement = measure_flow(stack, dt=1.0 / 320.0)

    assert measurement.verdict == "FLOWING"
    assert 0.75 * 30.37 < abs(measurement.velocity.px_per_frame) < 1.15 * 30.37


def test_direction_lock_recovers_rates_above_half_a_pitch() -> None:
    """Above half a pitch per frame a periodic train is ambiguous without a direction prior."""
    stack = train_stack(speed=120.0, pitch=150.0)
    profiles, pitch, strength = profiles_of(stack)
    assert strength > 0.5

    free = estimate_velocity(profiles, 1.0 / 320.0, pitch=pitch, gaps=(1, 2, 3))
    locked = estimate_velocity(profiles, 1.0 / 320.0, pitch=pitch, gaps=(1, 2, 3), direction=1.0)

    assert free.px_per_frame == pytest.approx(120.0 - pitch, abs=0.5)   # the honest alias
    assert locked.ok
    assert locked.px_per_frame == pytest.approx(120.0, abs=0.5)
    assert locked.confidence == "direction_locked"
    assert locked.alias_margin > 0.45
    assert locked.agreeing_gaps == [1, 2, 3]


def test_channel_lock_keeps_a_fast_train_signed_correctly() -> None:
    """The direction is learned while the flow is slow and then held as it speeds up.

    At 0.6 pitch per frame the frame-gap evidence alone cannot tell the true rate
    from its alias, so a lock that learns the direction from a fast window gets
    it backwards.  Learning it from a slow window and holding it is what keeps
    the sign stable; this is also the order the field protocol runs in.
    """
    lock = ChannelLock(window=48)
    for frame in train_stack(frames=48, speed=12.0, pitch=150.0):
        lock.push(frame)

    assert lock.ready
    lock.lock()
    slow = lock.measure(dt=1.0 / 320.0)

    assert slow.velocity.px_per_frame == pytest.approx(12.0, abs=0.5)
    assert lock.direction == pytest.approx(1.0)

    for frame in train_stack(frames=48, speed=90.0, pitch=150.0):
        lock.push(frame)
    measurement = lock.measure(dt=1.0 / 320.0)

    assert measurement.verdict == "FLOWING"
    assert measurement.velocity.px_per_frame == pytest.approx(90.0, abs=1.0)
    assert measurement.velocity.confidence == "direction_locked"


def test_activity_band_returns_one_contiguous_run() -> None:
    """Two separate active regions must not be merged into one band spanning both."""
    random = np.random.default_rng(4)
    stack = np.zeros((20, 200, 400), np.float32)
    stack[:, 20:40] += random.normal(0.0, 40.0, (20, 20, 400)).astype(np.float32)
    stack[:, 120:140] += random.normal(0.0, 25.0, (20, 20, 400)).astype(np.float32)

    top, bot, _ = activity_band(stack)

    assert (top, bot) == (20, 39)


def test_channel_start_skips_the_active_chamber() -> None:
    random = np.random.default_rng(3)
    stack = np.zeros((16, 60, 400), np.float32)
    stack[:, 10:50, :120] += random.normal(0.0, 45.0, (16, 40, 120)).astype(np.float32)
    stack[:, 10:50, 260:] += random.normal(0.0, 12.0, (16, 40, 140)).astype(np.float32)

    x0 = channel_start(stack, 10, 50)

    assert 120 <= x0 <= 260


def test_a_static_scene_is_never_reported_as_flowing() -> None:
    random = np.random.default_rng(5)
    stack = np.full((40, 120, 400), 120.0, np.float32)
    stack += random.normal(0.0, 2.0, (40, 120, 400)).astype(np.float32)

    measurement = measure_flow(stack, dt=1.0 / 320.0)

    assert measurement.verdict != "FLOWING"
    assert abs(measurement.velocity.px_per_frame) < 0.5


def test_a_frozen_droplet_train_is_reported_as_no_signal() -> None:
    """The zero-flow control: droplets are visible but nothing moves, so there is no flow."""
    stack = train_stack(speed=0.0)
    measurement = measure_flow(stack, dt=1.0 / 320.0)

    assert measurement.verdict == "NO_SIGNAL"
    assert measurement.velocity.ok is False


def test_dynamic_shift_reports_the_synthetic_displacement() -> None:
    stack = train_stack(frames=16, speed=6.0)
    profiles, _, _ = profiles_of(stack)

    shift, peak = dynamic_shift(profiles, 1)

    assert shift == pytest.approx(6.0, abs=0.3)
    assert peak > 5.0


@pytest.mark.skipif(_reference_path() is None, reason="reference recording is not available")
def test_reference_recording_regression() -> None:
    """63 frames of a real chip whose known displacement is 47.6 px/frame towards lower columns."""
    stack = np.load(_reference_path()).astype(np.float32)
    measurement = measure_flow(stack, dt=1.0 / 101.613)

    assert measurement.verdict == "FLOWING"
    assert measurement.velocity.px_per_frame == pytest.approx(-48.0, abs=1.0)
    assert measurement.velocity.residual_px < 1.0
    assert len(measurement.velocity.agreeing_gaps) >= 2


@pytest.mark.skipif(_reference_path() is None, reason="reference recording is not available")
def test_reference_measurement_is_stable_when_the_direction_is_known() -> None:
    stack = np.load(_reference_path()).astype(np.float32)

    free = measure_flow(stack, dt=1.0 / 101.613)
    locked = measure_flow(stack, dt=1.0 / 101.613, direction=-1.0)

    assert locked.velocity.px_per_frame == pytest.approx(free.velocity.px_per_frame, abs=0.2)


def test_measurement_serializes_to_json() -> None:
    measurement = measure_flow(train_stack(speed=10.0), dt=1.0 / 320.0)

    payload = measurement.to_json()

    assert '"verdict": "FLOWING"' in payload
    assert '"velocity"' in payload


def slanted_stack(slope: float = 0.20, rows: int = 260, cols: int = 420, frames: int = 40,
                  speed: float = 5.0, pitch: int = 60) -> np.ndarray:
    """A droplet train inside a duct that runs diagonally across the sensor.

    The 2026-09-18 rig has a slanted duct (slope +0.063 on the sensor), which is
    exactly the case a fixed row band handles badly.
    """
    rng = np.random.default_rng(11)
    base = np.full((rows, cols), 45.0, dtype=np.float32)
    base += np.linspace(0.0, 8.0, cols, dtype=np.float32)[None, :]
    base += 3.0 * np.sin(np.arange(rows, dtype=np.float32)[:, None] / 9.0)
    row_axis = np.arange(rows, dtype=np.float32)[:, None]
    stack = np.empty((frames, rows, cols), dtype=np.float32)
    for index in range(frames):
        centre = 110.0 + slope * np.arange(cols)
        position = (np.arange(cols) + speed * index) % pitch
        gate = np.exp(-((position - pitch * 0.5) ** 2) / (2 * 9.0 ** 2))
        frame = base + 26.0 * gate[None, :] * np.exp(
            -((row_axis - centre[None, :]) ** 2) / (2 * 5.0 ** 2))
        stack[index] = frame
    return stack + rng.normal(0.0, 0.4, stack.shape).astype(np.float32)


def test_duct_centre_line_follows_a_slanted_channel() -> None:
    stack = slanted_stack(slope=0.20)
    top, bot, _ = activity_band(stack)
    centre, half = duct_centre_line(stack, top, bot)
    fitted = np.polyfit(np.arange(stack.shape[2], dtype=float), centre, 1)
    assert abs(fitted[0] - 0.20) < 0.02
    assert 3.0 < float(np.mean(half)) < 9.0


def test_centreline_sampling_beats_the_fixed_band_on_a_slanted_channel() -> None:
    stack = slanted_stack(slope=0.20)
    top, bot, _ = activity_band(stack)
    band = channel_profiles(stack, top, bot)
    centre, _ = duct_centre_line(stack, top, bot)
    along = centreline_profiles(stack, centre, 2)
    band_contrast = float(np.percentile(band, 95) - np.percentile(band, 5))
    line_contrast = float(np.percentile(along, 95) - np.percentile(along, 5))
    assert line_contrast > 1.5 * band_contrast


def test_measure_flow_prefers_the_centreline_on_a_slanted_channel() -> None:
    stack = slanted_stack(slope=0.20)
    measurement = measure_flow(stack, dt=1.0 / 320.0)
    assert measurement.profile_source == "centreline"
    assert abs(measurement.pitch_px - 60.0) < 5.0
    assert measurement.velocity.px_per_frame < 0.0


def tilted_duct(frames: int = 40, height: int = 200, width: int = 700, gap: float = 27.0,
                slope: float = 0.066, mid0: float = 100.0, wall: float = 40.0,
                amplitude: float = 20.0, speed: float = 1.0, noise: float = 2.0,
                seed: int = 3) -> np.ndarray:
    """A tilted duct: two bright wall lines plus a slow train between them."""
    random = np.random.default_rng(seed)
    rows = np.arange(height)[:, None]
    columns = np.arange(width)[None, :]
    stack = np.full((frames, height, width), 140.0, np.float32)
    stack += random.normal(0.0, noise, stack.shape).astype(np.float32)
    mid = mid0 + slope * columns
    for offset in (-0.5 * gap, 0.5 * gap):
        stack += wall * np.exp(-0.5 * ((rows - (mid + offset)) / 0.9) ** 2)[None, :, :]
    phase = np.mod(columns - speed * np.arange(frames)[:, None, None], 150.0)
    inside = (np.abs(rows - mid) < 0.5 * gap - 1).astype(np.float32)
    stack += amplitude * (phase < 100).astype(np.float32) * inside[None]
    return stack


def ramping_train(frames: int = 90, height: int = 200, width: int = 800, pitch: float = 150.0,
                  droplet: float = 105.0, first_speed: float = 8.0, accel: float = 0.35,
                  noise: float = 2.0, seed: int = 4) -> np.ndarray:
    """A droplet train whose speed grows linearly, like a syringe pump coming up."""
    random = np.random.default_rng(seed)
    columns = np.arange(width, dtype=np.float32)
    background = np.full((height, width), 140.0, np.float32)
    background += random.normal(0.0, noise, (height, width)).astype(np.float32)
    background[44:46, :] += 60.0
    background[105:107, :] += 60.0
    inside = np.zeros((height, 1), np.float32)
    inside[50:101] = 1.0
    kernel = np.exp(-0.5 * (np.arange(-9, 10) / 2.5) ** 2)
    kernel /= kernel.sum()
    position = np.zeros(frames)
    for index in range(1, frames):
        position[index] = position[index - 1] + first_speed + accel * index
    stack = np.empty((frames, height, width), np.float32)
    for index in range(frames):
        phase = np.mod(columns - position[index], pitch)
        train = np.convolve((phase < droplet).astype(np.float32), kernel, mode="same")
        stack[index] = background + 55.0 * inside * train[None, :]
    return stack


def test_duct_walls_measures_a_tilted_duct() -> None:
    """The wall gap is the only static length in the frame; a scale can only trust it."""
    walls = duct_walls(np.asarray(tilted_duct(gap=27.0, slope=0.066), np.float32))

    assert walls.ok
    assert walls.gap_px == pytest.approx(27.0, abs=0.6)
    assert walls.gap_sd_px < 0.5
    assert walls.mid_slope == pytest.approx(0.066, abs=0.004)
    assert walls.columns_used > 500


def test_duct_walls_needs_no_motion_at_all() -> None:
    """Walls are static: a stopped flow must not stop the scale measurement."""
    stack = tilted_duct(frames=1, amplitude=0.0)
    stack = np.repeat(stack, 20, axis=0)
    walls = duct_walls(np.asarray(stack, np.float32), top=80, bot=130)

    assert walls.ok
    assert walls.gap_px == pytest.approx(27.0, abs=0.3)


def test_duct_walls_refuses_a_featureless_frame() -> None:
    """No wall lines means no duct: noise must not be fitted into a channel."""
    random = np.random.default_rng(11)
    stack = 140.0 + random.normal(0.0, 2.0, (20, 200, 700)).astype(np.float32)
    walls = duct_walls(np.asarray(stack, np.float32), top=80, bot=130)

    assert not walls.ok
    assert "grey levels" in walls.reason


def test_ramping_train_reports_its_window_mean() -> None:
    """A pump ramp inside the window averages out; it is not a wrong answer."""
    stack = ramping_train(accel=0.35)
    early = measure_flow(stack[0:20], dt=1.0 / 320.0)
    late = measure_flow(stack[60:90], dt=1.0 / 320.0)

    assert early.verdict == "FLOWING"
    assert late.verdict == "FLOWING"
    # speeds sweep 8..15 px/frame over the first window and 29..40 over the last
    assert 8.0 < early.velocity.px_per_frame < 15.5
    assert 29.0 < late.velocity.px_per_frame < 40.0


def test_train_past_half_a_pitch_reports_the_alias_not_the_truth() -> None:
    """Above half a pitch per frame a periodic train has no prior-free answer.

    The estimator returns the smallest alias, which is why a direction has to be
    learned while the flow is slow (see ``ChannelLock``): a train doing 100 px/frame
    on a 150 px pitch is reported as -50 px/frame, and every gap agrees because
    the pattern is strictly periodic.  This test pins that behaviour down so it
    can never be mistaken for a measurement.
    """
    stack = ramping_train(frames=60, first_speed=100.0, accel=0.0)
    measurement = measure_flow(stack, dt=1.0 / 320.0)
    rate = measurement.velocity.px_per_frame
    pitch = measurement.pitch_px

    assert measurement.verdict == "FLOWING"
    assert abs(rate) < 0.5 * pitch
    wrapped = (rate - 100.0 + 0.5 * pitch) % pitch - 0.5 * pitch
    assert abs(wrapped) < 0.15 * pitch
