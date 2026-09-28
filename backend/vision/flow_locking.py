"""Lock onto the flow channel from the image itself and measure its velocity.

The ROI that the operator draws in the application is a snapshot of where the
channel was at that moment; as soon as the stage or the chip moves it is wrong,
and a wrong ROI silently turns every downstream calibration into noise.  This
module therefore never trusts a saved rectangle: it finds the channel in the
frames it is given.

The recipe, and why each step is needed:

1. ``activity_band`` looks for the rows that carry *temporal* activity.  Static
   walls, dust and stage texture are identical in every frame, so they never
   appear in a temporal-activity map, while anything that moves always does.
2. ``channel_start`` drops the generation chamber that usually sits at one end
   of the field of view.  The chamber is the most active region of all but its
   contents swirl instead of translating, so including it corrupts the
   displacement estimate.
3. ``dynamic_shift`` measures the displacement with an averaged phase
   correlation computed on the *dynamic part only*.  Subtracting the per-column
   time mean is essential: the static channel walls carry most of the energy and
   otherwise pin the correlation peak at zero lag, which makes a flowing channel
   look frozen.
4. The correlation is whitened with a spectral floor (``alpha``).  Pure
   whitening divides by the magnitude of every spectral bin, including bins that
   carry no signal at all, and on a strongly periodic droplet train that
   degenerates: a synthetic train moving 30 px/frame is reported as 0.02 px/frame
   without the floor and as 27.5 px/frame with it.
5. ``estimate_velocity`` cross-checks several frame gaps and only accepts gaps
   whose displacement stays below half a droplet pitch.  Beyond that the
   periodic pattern makes the peak ambiguous, which is the failure mode that
   makes the reported velocity flip sign at high flow rates.
6. When the flow direction is known, ``direction`` restricts the search to that
   half of the lag axis.  The unambiguous range then becomes a full pitch
   instead of half of one: on the 2026-09-18 recording this removes the sign
   flips that appear above 45 mm/s at 320 fps, where the per-frame displacement
   reaches 0.5 pitch.
7. ``duct_walls`` measures the two static wall lines on the time mean.  That gap is
   the only static, sharp length in the frame, so it is what a micron-per-pixel
   scale has to be tied to; a scale derived from the droplet train is circular
   because the droplet length is itself a function of the flow.
8. ``measure_flow`` reports the intensity statistics (``image_range``,
   ``moving_rms``, ``moving_pp``) but does not gate on them.  A grey-level gate was
   implemented and then refuted by the data: the long capture's three windows
   disagree by 26%, which looks like noise, but a scan of seven 200-frame windows
   shows a monotone ramp from -55.3 to -81.3 px/frame that a straight line fits to
   0.9%.  The disagreeing long window is a window with a ramp inside it.  What
   separates good from bad is the frame-gap residual, and that is already the
   acceptance test.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_ALPHA = 0.1
DEFAULT_MAX_LAG = 180


@dataclass(frozen=True)
class GapEstimate:
    """Displacement measured at one frame gap, checked against the accepted rate."""

    gap: int
    shift_px: float
    per_frame_px: float
    peak_ratio: float
    residual_px: float
    agrees: bool


@dataclass(frozen=True)
class VelocityEstimate:
    """Velocity in px/frame, resolved against the droplet pitch and cross-checked."""

    px_per_frame: float = 0.0
    px_per_second: float = 0.0
    ok: bool = False
    confidence: str = "none"
    agreeing_gaps: list[int] = field(default_factory=list)
    residual_px: float = 0.0
    alias_margin: float = 0.0
    direction_locked: bool = False
    detail: list[GapEstimate] = field(default_factory=list)


@dataclass(frozen=True)
class FlowMeasurement:
    """Everything one lock-and-measure pass produced."""

    verdict: str
    reason: str
    frames: int
    dt_s: float
    pixel_to_micron: float
    band_rows: tuple
    band_height_px: int
    channel_x0: int
    pitch_px: float
    pitch_strength: float
    velocity: VelocityEstimate
    droplets_in_view: int
    droplet_length_px: float
    frequency_hz: float
    um_per_second: float
    mm_per_second: float
    droplets_per_second: float
    profile_source: str = "band"
    image_range: float = 0.0
    moving_rms: float = 0.0
    moving_pp: float = 0.0
    duct: "DuctWalls | None" = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _as_stack(stack: np.ndarray) -> np.ndarray:
    array = np.asarray(stack, np.float32)
    if array.ndim != 3:
        raise ValueError(f"expected a (frames, rows, columns) stack, got shape {array.shape}")
    if array.shape[0] < 3:
        raise ValueError("need at least 3 frames to measure motion")
    return array


def activity_band(stack: np.ndarray, frac: float = 0.35, min_height: int = 10):
    """Rows that contain moving content, from the temporal-activity map.

    The band is the *contiguous run* of rows carrying the most activity, not the
    span from the first to the last row above the threshold: two unrelated
    active regions would otherwise be merged into one band that contains neither
    of them properly.
    """
    stack = _as_stack(stack)
    dynamic = np.abs(stack - stack.mean(axis=0, keepdims=True))
    per_row = dynamic.mean(axis=0).mean(axis=1)
    per_row = per_row - np.median(per_row)
    mask = per_row > frac * per_row.max()
    runs = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if (not value or index == len(mask) - 1) and start is not None:
            end = index if not value else index + 1
            runs.append((start, end))
            start = None
    runs = [(a, b) for a, b in runs if b - a >= min_height]
    if not runs:
        return 0, stack.shape[1] - 1, per_row
    best = max(runs, key=lambda run: float(per_row[run[0]:run[1]].sum()))
    return int(best[0]), int(best[1]) - 1, per_row


def channel_start(stack: np.ndarray, top: int, bot: int, factor: float = 1.6, run: int = 60) -> int:
    """First column of the quiet stretch: excludes the generation chamber."""
    stack = _as_stack(stack)
    band = stack[:, top:bot + 1]
    activity = np.abs(band - band.mean(axis=0, keepdims=True)).mean(axis=(0, 1))
    reference = np.median(activity[len(activity) // 2:])
    if reference <= 0:
        return 0
    quiet = activity < factor * reference
    for x in range(max(1, len(quiet) - run)):
        if quiet[x:x + run].all():
            return x
    return 0


def channel_profiles(stack: np.ndarray, top: int, bot: int,
                     x_lo: int = 0, x_hi: int | None = None) -> np.ndarray:
    """Mean intensity along the channel per frame: shape (frames, width)."""
    stack = _as_stack(stack)
    stop = stack.shape[2] if x_hi is None else x_hi
    band = stack[:, top:bot + 1, x_lo:stop]
    return band.mean(axis=1).astype(np.float32)


def duct_centre_line(stack: np.ndarray, top: int, bot: int, margin: int = 24,
                     window: int = 14, x_lo: int = 0):
    """Row of the moving liquid column for every column, and its half width.

    The duct of the 2026-09-18 rig is not horizontal: it runs from about row 252
    at x=173 to row 287 at x=719 (slope +0.063, residual 1.5 px).  Averaging a
    fixed row band therefore mixes the droplet signal with the wall above and
    below it, which flattens the profile contrast from 7.6 to 2.3 grey levels --
    enough to make any threshold-based length estimate meaningless.

    The line is found on the difference between the 90th and the 10th percentile
    over time: a pixel that the droplet train modulates shows up there, while
    static walls and illumination gradients cancel out.  The search walks column
    by column so it cannot jump to an unrelated bright structure in the frame.
    """
    stack = _as_stack(stack)
    upper = np.percentile(stack, 90.0, axis=0)
    lower = np.percentile(stack, 10.0, axis=0)
    signal = np.asarray(upper - lower, np.float32)
    rows, width = signal.shape
    first = max(0, top - margin)
    last = min(rows, bot + margin + 1)
    centre = np.zeros(width, dtype=np.float64)
    half_width = np.zeros(width, dtype=np.float64)
    if width == 0 or last - first < 3:
        return centre, half_width
    seed = max(0, min(width - 1, x_lo))
    current = first + int(np.argmax(signal[first:last, seed]))
    for x in range(width):
        a = max(first, current - window)
        b = min(last, current + window + 1)
        if b - a < 3:
            a, b = first, last
        column = signal[a:b, x]
        k = int(np.argmax(column))
        if 0 < k < len(column) - 1:
            y0, y1, y2 = float(column[k - 1]), float(column[k]), float(column[k + 1])
            denominator = y0 - 2.0 * y1 + y2
            offset = 0.5 * (y0 - y2) / denominator if abs(denominator) > 1e-9 else 0.0
        else:
            offset = 0.0
        current = a + k
        centre[x] = current + offset
        peak = float(column[k])
        base = float(np.percentile(column, 10.0))
        target = base + 0.5 * (peak - base)
        left = k
        while left > 0 and column[left] > target:
            left -= 1
        right = k
        while right < len(column) - 1 and column[right] > target:
            right += 1
        half_width[x] = max(1.0, 0.5 * (right - left))
    return centre, half_width


def centreline_profiles(stack: np.ndarray, centre: np.ndarray, half: int,
                        x_lo: int = 0, x_hi: int | None = None) -> np.ndarray:
    """Mean intensity along ``centre`` instead of a fixed row band.

    On the 2026-09-18 recording this lifts the axial profile contrast from 2.3
    to 7.6 grey levels and the autocorrelation peak from 0.497 to 0.650, while
    the pitch (173 px) and the velocity stay the same.
    """
    stack = _as_stack(stack)
    stop = stack.shape[2] if x_hi is None else min(x_hi, stack.shape[2])
    start = max(0, x_lo)
    if stop <= start:
        return np.empty((stack.shape[0], 0), dtype=np.float32)
    centre = np.asarray(centre, np.float64)
    rows = stack.shape[1]
    out = np.empty((stack.shape[0], stop - start), dtype=np.float32)
    for x in range(start, stop):
        column = int(round(float(centre[x])))
        a = max(0, column - half)
        b = min(rows, column + half + 1)
        out[:, x - start] = stack[:, a:b, x].mean(axis=1)
    return out


@dataclass(frozen=True)
class DuctWalls:
    """The two static walls of the duct, measured on the time-averaged frame.

    ``gap_px`` is the distance between the two bright wall lines.  It is the only
    length in the field of view that is both static and sharp, which makes it the
    only sound thing to tie a scale to: the droplet train cannot serve that role
    because its own length is a function of the flow.

    The walls of the 2026-09-18 rig are two bright lines that run across the
    frame at slope +0.065 -- the duct is not horizontal, and a fixed row band
    therefore mixes wall, droplet and background.  The time mean keeps the walls
    and cancels the droplets, so the walls are read off the time mean.  The
    subtraction of the per-row mean is required: the illumination ramp across the
    sensor is ~10 grey levels, several times the wall contrast, and without it
    the tracker follows the ramp instead of the walls.
    """

    ok: bool = False
    gap_px: float = 0.0
    gap_sd_px: float = 0.0
    gap_slope: float = 0.0
    mid_row_at_x_lo: float = 0.0
    mid_slope: float = 0.0
    x_lo: int = 0
    x_hi: int = 0
    columns_used: int = 0
    residual_px: float = 0.0
    contrast_gray: float = 0.0
    reason: str = ""


def _best_tilt(plane: np.ndarray, xs: np.ndarray, coarse: float = 0.005,
               fine: float = 0.0005, span: float = 0.15) -> float:
    """Slope that makes the duct walls horizontal.

    A duct on this rig is tilted, so a row profile taken as-is smears each wall
    over (slope x width) rows.  Shearing the mean frame until the walls stack up
    into single rows turns the tilt into a one-parameter search, and it is a
    global search: no column-by-column walk is involved, so a bright chamber or a
    dark droplet cannot pull the walls apart.
    """
    rows = plane.shape[0]
    reference = float(xs[0])
    offsets = xs - reference

    def score(slope: float) -> float:
        shift = np.round(slope * offsets).astype(np.int64)
        index = np.clip(np.arange(rows)[:, None] + shift[None, :], 0, rows - 1)
        profile = plane[index, xs[None, :]].mean(axis=1)
        return float(np.percentile(profile, 99.5) - np.median(profile))

    best = max(np.arange(-span, span + fine, coarse), key=score)
    best = max(np.arange(best - coarse, best + coarse + 1e-9, fine), key=score)
    return float(best)


def _shear(plane: np.ndarray, xs: np.ndarray, slope: float):
    rows = plane.shape[0]
    shift = np.round(slope * (xs - float(xs[0]))).astype(np.int64)
    index = np.clip(np.arange(rows)[:, None] + shift[None, :], 0, rows - 1)
    return plane[index, xs[None, :]], shift


def _two_peaks(profile: np.ndarray, first: int, last: int, reach: int, min_gap: float):
    """Two best separated local maxima of a row profile, or (None, None)."""
    lo = max(1, first - reach)
    hi = min(profile.size - 1, last + reach)
    segment = profile[lo:hi]
    rows = np.arange(lo, hi)
    peaks = [int(r) for r in rows[1:-1]
             if segment[r - lo] >= segment[r - lo - 1] and segment[r - lo] >= segment[r - lo + 1]]
    if not peaks:
        return None, None
    peaks.sort(key=lambda r: -profile[r])
    top = peaks[0]
    for other in peaks[1:]:
        if abs(other - top) >= min_gap:
            return (min(top, other), max(top, other))
    return None, None


def duct_walls(stack: np.ndarray, top: int | None = None, bot: int | None = None,
               reach: int = 30, min_gap: float = 4.0, track: int = 4,
               x_lo: int | None = None, x_hi: int | None = None,
               min_columns: int = 40, min_contrast: float = 1.0) -> DuctWalls:
    """Locate the two bright wall lines and return the gap between them in px.

    Two local maxima are tracked per column inside a window that follows the
    previous column, so the pair cannot swap sides or run off onto an unrelated
    bright structure.  Both the centre line and the gap are then fitted with
    outlier rejection; ``gap_sd_px`` is the scatter of the gap about that fit and
    is the honest uncertainty of the width.
    """
    stack = _as_stack(stack)
    rows, width = stack.shape[1], stack.shape[2]
    if top is None or bot is None:
        top, bot, _ = activity_band(stack)
    plane = stack.mean(axis=0)
    plane = plane - plane.mean(axis=1)[:, None]
    stop = width if x_hi is None else min(int(x_hi), width)
    begin = channel_start(stack, top, bot) if x_lo is None else int(x_lo)
    begin = max(0, min(begin, max(0, stop - 1)))
    if stop - begin < 20 or rows < 8:
        return DuctWalls(reason="too few columns to fit a duct", x_lo=begin, x_hi=stop - 1)

    xs = np.arange(begin, stop)
    slope = _best_tilt(plane, xs)
    sheared, _ = _shear(plane, xs, slope)
    profile = sheared.mean(axis=1)
    upper, lower = _two_peaks(profile, top, bot, reach, min_gap)
    if upper is None:
        return DuctWalls(reason="no pair of parallel wall lines near the band",
                         x_lo=begin, x_hi=stop - 1, mid_slope=slope)
    nominal = float(lower - upper)
    mids = np.full(xs.size, np.nan)
    gaps = np.full(xs.size, np.nan)
    peaks = np.full(xs.size, np.nan)
    for index in range(xs.size):
        column = sheared[:, index]
        found = []
        for target in (upper, lower):
            a = max(0, target - track)
            b = min(rows, target + track + 1)
            if b - a < 3:
                found = []
                break
            window = column[a:b]
            k = int(np.argmax(window))
            if 0 < k < window.size - 1:
                y0, y1, y2 = float(window[k - 1]), float(window[k]), float(window[k + 1])
                denom = y0 - 2.0 * y1 + y2
                offset = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
            else:
                offset = 0.0
            found.append((a + k + offset, float(window[k])))
        if len(found) != 2:
            continue
        (row_a, peak_a), (row_b, peak_b) = found
        if row_b - row_a < min_gap or abs((row_b - row_a) - nominal) > max(3.0, 0.25 * nominal):
            continue
        mids[index] = 0.5 * (row_a + row_b)
        gaps[index] = row_b - row_a
        peaks[index] = 0.5 * (peak_a + peak_b)

    usable = np.isfinite(gaps)
    if int(usable.sum()) < min_columns:
        return DuctWalls(reason=f"only {int(usable.sum())} columns carried two wall lines",
                         x_lo=begin, x_hi=stop - 1, mid_slope=slope,
                         columns_used=int(usable.sum()),
                         gap_px=float(np.nanmedian(gaps)) if usable.any() else 0.0)
    width_px = float(np.median(gaps[usable]))
    spread = 1.4826 * float(np.median(np.abs(gaps[usable] - width_px)))
    keep = usable & (np.abs(gaps - width_px) <= max(3.0 * spread, 1.0))
    count = int(keep.sum())
    if count < min_columns:
        return DuctWalls(reason=f"only {count} columns survived outlier rejection",
                         x_lo=begin, x_hi=stop - 1, mid_slope=slope, columns_used=count)
    gap_px = float(np.median(gaps[keep]))
    gap_sd = 1.4826 * float(np.median(np.abs(gaps[keep] - gap_px)))
    mid_px = float(np.median(mids[keep]))
    mid_sd = 1.4826 * float(np.median(np.abs(mids[keep] - mid_px)))
    baseline = float(np.median(sheared[:, np.flatnonzero(keep)]))
    contrast = float(np.median(peaks[keep]) - baseline)
    if contrast < min_contrast:
        return DuctWalls(reason=f"wall lines are only {contrast:.2f} grey levels above the "
                                f"background (< {min_contrast:g})",
                         x_lo=begin, x_hi=stop - 1, mid_slope=slope, columns_used=count,
                         gap_px=gap_px, gap_sd_px=gap_sd, contrast_gray=contrast)
    first = int(np.flatnonzero(keep)[0])
    return DuctWalls(ok=True,
                     gap_px=gap_px,
                     gap_sd_px=gap_sd,
                     gap_slope=0.0,
                     mid_row_at_x_lo=mid_px,
                     mid_slope=slope,
                     x_lo=int(begin + first), x_hi=int(begin + np.flatnonzero(keep)[-1]),
                     columns_used=count, residual_px=mid_sd,
                     contrast_gray=contrast,
                     reason="two wall lines fitted")


def dynamic_shift(profiles: np.ndarray, gap: int, max_lag: int = DEFAULT_MAX_LAG,
                  alpha: float = DEFAULT_ALPHA, direction: float | None = None,
                  pitch: float | None = None):
    """Displacement over ``gap`` frames, averaged over all frame pairs.

    Sign convention: the returned value is the displacement of the pattern in
    the direction of increasing column index, so a leftward flow is negative.

    ``direction`` (-1 or +1) restricts the search to that half of the lag axis,
    which doubles the range that stays unambiguous on a periodic droplet train.
    ``pitch`` further restricts it to one pitch in that direction.
    """
    profiles = np.asarray(profiles, np.float32)
    if profiles.ndim != 2 or profiles.shape[0] <= gap + 1:
        return 0.0, 0.0
    dynamic = profiles - profiles.mean(axis=0, keepdims=True)
    frames, width = dynamic.shape
    window = np.hanning(width).astype(np.float32)
    spectrum = np.zeros(width // 2 + 1, dtype=np.complex128)
    pairs = 0
    for index in range(frames - gap):
        first = dynamic[index] * window
        second = dynamic[index + gap] * window
        first = first - first.mean()
        second = second - second.mean()
        cross = np.fft.rfft(first) * np.conj(np.fft.rfft(second))
        magnitude = np.abs(cross)
        cross /= magnitude + alpha * (magnitude.mean() + 1e-12)
        spectrum += cross
        pairs += 1
    if pairs == 0:
        return 0.0, 0.0
    spectrum /= pairs
    corr = np.fft.fftshift(np.fft.irfft(spectrum, n=width))
    centre = width // 2
    reach = int(round(1.02 * pitch)) if pitch else 0
    span = int(min(centre - 1, max(max_lag, reach)))
    segment = corr[centre - span:centre + span + 1]
    lags = np.arange(-span, span + 1, dtype=np.float64)
    if direction is not None and direction != 0:
        # displacement = -lag, so "moving towards lower columns" means lag > 0.
        # The reachable displacement scales with the gap, so the limits do too.
        scale = pitch if pitch else float(span)
        limit = 0.98 * scale * gap
        guard = 0.05 * scale * gap
        if direction < 0:
            keep = (lags >= -guard) & (lags <= limit)
        else:
            keep = (lags <= guard) & (lags >= -limit)
        segment = np.where(keep, segment, -np.inf)
    peak_index = int(np.argmax(segment))
    offset = 0.0
    if 0 < peak_index < len(segment) - 1:
        y0, y1, y2 = segment[peak_index - 1], segment[peak_index], segment[peak_index + 1]
        if np.isfinite(y0) and np.isfinite(y2):
            denom = y0 - 2 * y1 + y2
            if abs(denom) > 1e-12:
                offset = 0.5 * (y0 - y2) / denom
    lag = (peak_index - span) + offset
    reference = float(np.abs(corr).mean())
    peak_ratio = float(segment[peak_index]) / max(1e-9, reference) if np.isfinite(segment[peak_index]) else 0.0
    return -float(lag), peak_ratio


def pitch_from_autocorrelation(profiles: np.ndarray, min_lag: int = 25, max_lag: int = 400,
                               sample_frames: int = 60):
    """Droplet pitch in px and the relative strength of its autocorrelation peak."""
    profiles = np.asarray(profiles, np.float32)
    if profiles.ndim != 2 or profiles.shape[1] < min_lag + 2:
        return 0.0, 0.0
    dynamic = profiles - profiles.mean(axis=0, keepdims=True)
    accum = np.zeros(profiles.shape[1])
    used = min(sample_frames, len(dynamic))
    for row in dynamic[:used]:
        row = row - row.mean()
        accum += np.correlate(row, row, "full")[len(row) - 1:]
    accum /= max(1, used)
    accum /= max(1e-9, accum[0])
    best = (0.0, 0)
    for lag in range(min_lag, min(len(accum) - 1, max_lag)):
        if accum[lag] > accum[lag - 1] and accum[lag] > accum[lag + 1]:
            if accum[lag] > best[0]:
                best = (float(accum[lag]), lag)
    return float(best[1]), best[0]


def _wrap_residual(predicted: float, measured: float, pitch: float) -> float:
    """Signed distance from ``measured`` to ``predicted`` modulo one pitch."""
    error = predicted - measured
    return (error + 0.5 * pitch) % pitch - 0.5 * pitch


def estimate_velocity(profiles: np.ndarray, dt: float, pitch: float = 0.0,
                      gaps=(1, 2, 3), direction: float | None = None,
                      max_lag: int = DEFAULT_MAX_LAG, alpha: float = DEFAULT_ALPHA) -> VelocityEstimate:
    """Velocity in px/frame, resolved against the droplet pitch and cross-checked.

    A droplet train is periodic with the pitch, so a single frame-gap
    measurement only fixes the displacement modulo that pitch.  The 63-frame
    reference recording moves 47.99 px/frame, yet its 3-frame estimate wraps to
    +6.6 px/frame.  Every candidate rate is therefore scored by how well it
    predicts *all* frame gaps at once, with each prediction compared modulo one
    pitch, and the cheapest candidate wins.  Candidates come from a fine grid
    over the reachable range rather than from one gap, because a single noisy
    gap must not be able to define the answer on its own.

    Which rates are reachable depends on what is known:

    * direction unknown -- the rate stays below half a pitch per frame, the
      point beyond which a periodic pattern cannot be resolved without a prior;
    * direction known -- up to a full pitch per frame in that direction, which
      is what keeps the sign stable at high flow rates.  On the 2026-09-18
      recording the free estimator flips sign above 45 mm/s at 320 fps while
      the direction-locked one stays monotonic.
    """
    profiles = np.asarray(profiles, np.float32)
    raw = []
    for gap in gaps:
        if profiles.shape[0] <= gap + 4:
            continue
        shift, peak = dynamic_shift(profiles, gap, max_lag=max_lag, alpha=alpha,
                                    direction=direction, pitch=pitch)
        raw.append((int(gap), float(shift), float(peak)))
    if not raw:
        return VelocityEstimate(direction_locked=direction is not None)

    # 10% of a pitch is far too loose to certify a rate: two candidates that differ
    # by that much are on different pitch periods, which is exactly the ambiguity the
    # multi-gap score exists to reject.  On the 2026-09-18 rig a correct lock leaves
    # residuals of 0.06-0.12 px (reference recording, 0.8% error) while an unstable one
    # leaves 1.3-18 px, so the split is wide and the tight end costs nothing.
    tolerance = max(0.03 * pitch, 1.5) if pitch > 0 else 1.5

    if pitch <= 0:
        median = float(np.median([shift / gap for gap, shift, _ in raw]))
        detail = [GapEstimate(gap=gap, shift_px=round(shift, 3), per_frame_px=round(shift / gap, 3),
                              peak_ratio=round(peak, 3), residual_px=0.0, agrees=True)
                  for gap, shift, peak in raw]
        return VelocityEstimate(px_per_frame=median, px_per_second=median / dt if dt > 0 else 0.0,
                                ok=len(raw) >= 2,
                                confidence="multi_gap" if len(raw) >= 2 else "single_gap",
                                agreeing_gaps=[gap for gap, _, _ in raw],
                                direction_locked=direction is not None, detail=detail)

    limit = 0.98 * pitch if direction is not None else 0.5 * pitch
    step = max(0.02, pitch / 4000.0)
    rates = np.arange(-limit, limit + step, step, dtype=np.float64)
    if direction is not None:
        rates = rates[rates * direction > 0]
    if rates.size == 0:
        return VelocityEstimate(ok=False, confidence="alias_only",
                                direction_locked=direction is not None)

    mirror = 0.5 * pitch
    cost = np.zeros_like(rates)
    total_weight = 0.0
    for gap, shift, peak in raw:
        weight = max(1.0, peak)
        predicted = rates * gap - shift
        predicted = (predicted + mirror) % pitch - mirror
        cost += np.abs(predicted) * weight
        total_weight += weight
    cost /= max(1e-9, total_weight)
    best_index = int(np.argmin(cost))
    rate = float(rates[best_index])
    if 0 < best_index < len(rates) - 1:
        y0, y1, y2 = float(cost[best_index - 1]), float(cost[best_index]), float(cost[best_index + 1])
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            rate += 0.5 * (y0 - y2) / denom * step

    detail = []
    for gap, shift, peak in raw:
        residual = (rate * gap - shift + mirror) % pitch - mirror
        detail.append(GapEstimate(gap=gap, shift_px=round(shift, 3),
                                  per_frame_px=round(shift / gap, 3), peak_ratio=round(peak, 3),
                                  residual_px=round(float(residual), 3),
                                  agrees=bool(abs(residual) <= tolerance)))
    agreeing = [item for item in detail if item.agrees]
    margin = abs(rate) / pitch
    if len(agreeing) >= 2 and cost[best_index] <= tolerance:
        ok = True
        confidence = "direction_locked" if (direction is not None and margin > 0.45) else "multi_gap"
    elif len(agreeing) >= 1 and agreeing[0].peak_ratio >= 6.0 and cost[best_index] <= tolerance:
        ok = True
        confidence = "single_gap"
    else:
        ok = False
        confidence = "disagree"
    return VelocityEstimate(px_per_frame=rate, px_per_second=rate / dt if dt > 0 else 0.0,
                            ok=ok, confidence=confidence,
                            agreeing_gaps=[item.gap for item in agreeing],
                            residual_px=round(float(cost[best_index]), 3),
                            alias_margin=round(float(margin), 3),
                            direction_locked=direction is not None, detail=detail)


def droplet_metrics(profiles: np.ndarray, velocity_px_per_frame: float, dt: float,
                    pitch: float = 0.0, min_length: int = 15) -> dict:
    """Droplet length, count in view and passage frequency."""
    profiles = np.asarray(profiles, np.float32)
    lo, hi = np.percentile(profiles, [5, 95])
    threshold = 0.5 * (lo + hi)
    counts, lengths = [], []
    for row in profiles:
        mask = row > threshold
        runs, start = [], None
        for index, value in enumerate(mask):
            if value and start is None:
                start = index
            if (not value or index == len(mask) - 1) and start is not None:
                end = index if not value else index + 1
                runs.append(end - start)
                start = None
        counts.append(len(runs))
        lengths.extend(size for size in runs if size >= min_length)
    effective_pitch = pitch
    if effective_pitch <= 0:
        effective_pitch, _ = pitch_from_autocorrelation(profiles)
    frequency = abs(velocity_px_per_frame) / effective_pitch / dt if effective_pitch > 0 and dt > 0 else 0.0
    return {
        "droplets_in_view_median": int(np.median(counts)) if counts else 0,
        "droplet_length_px_median": float(np.median(lengths)) if lengths else 0.0,
        "pitch_px_autocorr": float(effective_pitch),
        "frequency_hz": float(frequency),
        "droplets_per_second": float(frequency),
    }


def measure_flow(stack: np.ndarray, dt: float, pixel_to_micron: float = 1.725,
                 gaps=(1, 2, 3), direction: float | None = None, label: str = "",
                 max_lag: int = DEFAULT_MAX_LAG, alpha: float = DEFAULT_ALPHA) -> FlowMeasurement:
    """Lock the channel and measure the flow, without any saved ROI.

    ``image_range``, ``moving_rms`` and ``moving_pp`` are reported because they are
    what an operator needs to judge a capture, but they are *not* used as a gate:
    a gate on them was tried and refuted.  The 2026-09-18 long capture carries
    only 1.27 grey levels of moving content on a fixed row band against the
    reference recording's 1.81, which looked like a usable threshold, yet the
    centreline profile of the same capture carries 9.76 -- as much as the
    reference's 9.77 -- and a window scan shows why.  Its velocity rises
    monotonically from -55.3 to -81.3 px/frame over 4.7 s and a straight line fits
    that to 0.9%, so the capture is an actuator transient that the estimator reads
    correctly, not noise.  Only the frame-gap residuals separate a good window
    from a bad one, and they are already checked in ``estimate_velocity``.
    """
    stack = _as_stack(stack)
    low, high = np.percentile(stack, [0.1, 99.9])
    image_range = float(high - low)
    top, bot, _ = activity_band(stack)
    height = bot - top + 1
    blank = VelocityEstimate()
    if height < 8:
        return FlowMeasurement("NO_BAND", "no moving band found in the field of view",
                               int(stack.shape[0]), dt, pixel_to_micron, (top, bot), height, 0,
                               0.0, 0.0, blank, 0, 0.0, 0.0, 0.0, 0.0, 0.0, "band",
                               image_range=image_range)
    duct = duct_walls(stack, top, bot)
    x0 = channel_start(stack, top, bot)
    profiles = channel_profiles(stack, top, bot, x_lo=x0)
    pitch, strength = pitch_from_autocorrelation(profiles)
    profile_source = "band"
    centre, _ = duct_centre_line(stack, top, bot, x_lo=x0)
    along = centreline_profiles(stack, centre, 2, x_lo=x0)
    if along.shape[1] >= 60:
        line_pitch, line_strength = pitch_from_autocorrelation(along)
        band_contrast = float(np.percentile(profiles, 95) - np.percentile(profiles, 5))
        line_contrast = float(np.percentile(along, 95) - np.percentile(along, 5))
        if (line_pitch >= 20.0 and line_strength > 0.25
                and line_contrast > band_contrast * 1.3):
            profiles, pitch, strength = along, line_pitch, line_strength
            profile_source = "centreline"
    dynamic = profiles - profiles.mean(axis=0, keepdims=True)
    moving_rms = float(np.sqrt((dynamic ** 2).mean()))
    moving_pp = float(np.percentile(dynamic, 97.5) - np.percentile(dynamic, 2.5))
    quality = dict(image_range=image_range, moving_rms=moving_rms, moving_pp=moving_pp,
                   duct=duct)
    if pitch < 20.0 or strength < 0.25:
        flat = float(np.abs(dynamic).mean())
        if flat < 1.0:
            verdict, reason = "NO_SIGNAL", "no moving content: the field of view is static"
        else:
            verdict, reason = "NO_TRAIN", "content moves but no periodic droplet train was found"
        return FlowMeasurement(verdict, reason, int(stack.shape[0]), dt, pixel_to_micron,
                               (top, bot), height, x0, pitch, strength, blank, 0, 0.0, 0.0,
                               0.0, 0.0, 0.0, "band", **quality)
    velocity = estimate_velocity(profiles, dt, pitch=pitch, gaps=gaps, direction=direction,
                                 max_lag=max_lag, alpha=alpha)
    metrics = droplet_metrics(profiles, velocity.px_per_frame, dt, pitch=pitch)
    um_per_second = velocity.px_per_second * pixel_to_micron
    if not velocity.ok:
        verdict = {"alias_only": "AMBIGUOUS", "disagree": "DISAGREE"}.get(velocity.confidence, "WEAK_SIGNAL")
        reason = {
            "AMBIGUOUS": "no reachable rate explains every frame gap",
            "DISAGREE": "frame gaps disagree on the displacement",
        }.get(verdict, "correlation peak too weak to trust")
    elif abs(velocity.px_per_second) < 5.0:
        verdict, reason = "STATIC", "no measurable displacement"
    else:
        verdict, reason = "FLOWING", "locked and measured"
    return FlowMeasurement(verdict, reason, int(stack.shape[0]), dt, pixel_to_micron,
                           (top, bot), height, x0, pitch, strength, velocity,
                           metrics["droplets_in_view_median"], metrics["droplet_length_px_median"],
                           metrics["frequency_hz"], um_per_second, um_per_second / 1000.0,
                           metrics["droplets_per_second"], profile_source, **quality)


class ChannelLock:
    """Reusable lock for a live stream: lock once, then measure frame windows.

    A saved ROI is never used.  The flow direction is learned from the first
    window that yields a usable measurement and is then held.  Learn it while
    the flow is slow: above half a pitch per frame the frame-gap evidence alone
    cannot tell the true rate from its alias, so a lock that first sees a fast
    window can adopt the wrong sign.  Holding the direction is what keeps the
    sign stable once the flow speeds past the alias limit.
    """

    def __init__(self, pixel_to_micron: float = 1.725, gaps=(1, 2, 3),
                 window: int = 60, max_lag: int = DEFAULT_MAX_LAG, alpha: float = DEFAULT_ALPHA):
        self.pixel_to_micron = float(pixel_to_micron)
        self.gaps = tuple(gaps)
        self.window = int(window)
        self.max_lag = int(max_lag)
        self.alpha = float(alpha)
        self.band_rows: tuple[int, int] | None = None
        self.channel_x0: int | None = None
        self.pitch_px: float = 0.0
        self.direction: float | None = None
        self.duct: DuctWalls | None = None
        self._buffer: list[np.ndarray] = []

    def push(self, frame: np.ndarray) -> None:
        """Append one frame; the buffer keeps at most ``window`` frames."""
        self._buffer.append(np.asarray(frame, np.float32))
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

    @property
    def ready(self) -> bool:
        return len(self._buffer) >= self.window

    def lock(self, stack: np.ndarray | None = None) -> dict:
        """(Re)find the channel band, the quiet start column and the pitch."""
        data = _as_stack(np.stack(self._buffer) if stack is None else stack)
        top, bot, _ = activity_band(data)
        self.band_rows = (top, bot)
        self.channel_x0 = channel_start(data, top, bot)
        profiles = channel_profiles(data, top, bot, x_lo=self.channel_x0)
        self.pitch_px, strength = pitch_from_autocorrelation(profiles)
        duct = duct_walls(data, top, bot)
        self.duct = duct
        return {"band_rows": self.band_rows, "channel_x0": self.channel_x0,
                "pitch_px": self.pitch_px, "pitch_strength": strength,
                "duct_gap_px": duct.gap_px, "duct_gap_sd_px": duct.gap_sd_px,
                "duct_ok": duct.ok}

    def measure(self, dt: float, stack: np.ndarray | None = None) -> FlowMeasurement:
        """Measure the current buffer, learning the direction if it is unknown."""
        data = _as_stack(np.stack(self._buffer) if stack is None else stack)
        if self.band_rows is None:
            self.lock(data)
        top, bot = self.band_rows
        profiles = channel_profiles(data, top, bot, x_lo=self.channel_x0 or 0)
        if self.pitch_px <= 0:
            self.pitch_px, _ = pitch_from_autocorrelation(profiles)
        if self.direction is None and self.pitch_px > 0:
            free = estimate_velocity(profiles, dt, pitch=self.pitch_px, gaps=self.gaps,
                                     max_lag=self.max_lag, alpha=self.alpha)
            if free.ok and abs(free.px_per_frame) > 0.5:
                self.direction = -1.0 if free.px_per_frame < 0 else 1.0
        return measure_flow(data, dt, pixel_to_micron=self.pixel_to_micron, gaps=self.gaps,
                            direction=self.direction, max_lag=self.max_lag, alpha=self.alpha)


def main() -> int:
    parser = argparse.ArgumentParser(description="Lock the channel in a frame stack and measure its velocity.")
    parser.add_argument("--video", required=True, help="offline .npy stack of frames")
    parser.add_argument("--rate", type=float, default=320.0, help="frames per second")
    parser.add_argument("--scale", type=float, default=1.725, help="micron per pixel")
    parser.add_argument("--gaps", default="1,2,3")
    parser.add_argument("--direction", default="auto", choices=("auto", "left", "right"))
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    stack = np.load(args.video).astype(np.float32)
    dt = 1.0 / args.rate
    direction = None if args.direction == "auto" else (-1.0 if args.direction == "left" else 1.0)
    gaps = tuple(int(token) for token in args.gaps.split(",") if token.strip())
    measurement = measure_flow(stack, dt, pixel_to_micron=args.scale, gaps=gaps, direction=direction)
    print(measurement.to_json())
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(measurement.to_json(), encoding="utf-8")
        print("WROTE " + str(target))
    return 0 if measurement.verdict == "FLOWING" else 1


if __name__ == "__main__":
    raise SystemExit(main())
