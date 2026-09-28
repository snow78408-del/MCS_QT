"""Offline periodic-image analysis; image agreement does not resolve alias order."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .flow_locking import VelocityEstimate, estimate_velocity


def _profiles(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 3 or array.shape[1] < 16:
        raise ValueError("profiles must contain at least three frames and 16 columns")
    if not np.isfinite(array).all():
        raise ValueError("profiles must be finite")
    return array


def spatial_pitch(profiles: np.ndarray, minimum: int = 25, maximum: int = 230) -> tuple[float, float]:
    """Unbiased spatial autocorrelation peak, excluding the search endpoints."""
    values = _profiles(profiles)
    if minimum < 2 or maximum <= minimum + 2:
        raise ValueError("invalid pitch search interval")
    values = values - values.mean(axis=0)
    values -= values.mean(axis=1, keepdims=True)
    power = float(np.mean(values ** 2))
    stop = min(maximum, values.shape[1] // 2)
    if power <= 1e-12 or stop <= minimum + 2:
        return 0.0, 0.0
    lags = np.arange(minimum, stop + 1)
    corr = np.array([np.mean(values[:, :-lag] * values[:, lag:]) / power for lag in lags])
    peaks = np.flatnonzero((corr[1:-1] > corr[:-2]) & (corr[1:-1] >= corr[2:])) + 1
    if not peaks.size:
        return 0.0, 0.0
    index = int(peaks[np.argmax(corr[peaks])])
    left, middle, right = corr[index - 1:index + 2]
    offset = 0.5 * (left - right) / (left - 2 * middle + right)
    return float(lags[index] + offset), float(middle)


@dataclass(frozen=True)
class PhaseAverage:
    pitch_px: float
    profile: list[float]
    frame_phase_px: list[float]
    frame_correlation: list[float]
    accepted_frames: list[int]
    rejected_frames: list[int]
    min_bin_support: float
    contrast_gray: float
    intensity_duty: float | None
    usable: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def periodic_phase_average(profiles: np.ndarray, pitch: float, min_correlation: float = 0.6) -> PhaseAverage:
    """Fold actual samples into a period, then align each frame independently.

    Phase is modulo pitch, NOT cumulative displacement. Linear bin deposition
    and circular interpolation never invent out-of-frame endpoint plateaus.
    Intensity duty is diagnostic only: optical brightness is not a phase label.
    """
    values = _profiles(profiles)
    if not math.isfinite(pitch) or not 4 <= pitch <= values.shape[1] / 2:
        raise ValueError("at least two spatial periods must be visible")
    if not 0 < min_correlation <= 1:
        raise ValueError("min_correlation must be in (0, 1]")
    # Remove static illumination/walls, without tracking a displacement in time.
    dynamic = values - values.mean(axis=0)
    dynamic -= dynamic.mean(axis=1, keepdims=True)
    bins = int(round(pitch))
    coordinate = (np.arange(values.shape[1]) % pitch) * bins / pitch
    lower = np.floor(coordinate).astype(int) % bins
    fraction = coordinate - np.floor(coordinate)
    upper = (lower + 1) % bins
    support = np.bincount(lower, weights=1 - fraction, minlength=bins)
    support += np.bincount(upper, weights=fraction, minlength=bins)
    if np.any(support <= 0):
        raise ValueError("phase bins lack sample support")
    folded = np.stack([
        (np.bincount(lower, weights=row * (1 - fraction), minlength=bins)
         + np.bincount(upper, weights=row * fraction, minlength=bins)) / support
        for row in dynamic
    ])
    folded -= folded.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(folded, axis=1)
    nonzero = np.flatnonzero(norms > 1e-8)
    if len(nonzero) < 3:
        return PhaseAverage(pitch, [], [], [], [], list(range(len(values))), float(support.min()),
                            0.0, None, False, "no temporal signal")
    # A median-energy frame avoids privileging a single bright outlier.
    template = folded[nonzero[np.argsort(norms[nonzero])[len(nonzero) // 2]]]
    template_fft = np.fft.rfft(template)
    phases, correlations, aligned, accepted = [], [], [], []
    grid = np.arange(bins)
    for index, row in enumerate(folded):
        corr = np.fft.irfft(np.fft.rfft(row) * np.conj(template_fft), n=bins)
        peak = int(np.argmax(corr))
        left, middle, right = corr[(peak - 1) % bins], corr[peak], corr[(peak + 1) % bins]
        curvature = left - 2 * middle + right
        offset = 0.5 * (left - right) / curvature if abs(curvature) > 1e-12 else 0.0
        shift = (peak + offset) % bins
        quality = float(middle / max(norms[index] * np.linalg.norm(template), 1e-12))
        phases.append(float(shift * pitch / bins))
        correlations.append(float(np.clip(quality, -1, 1)))
        if norms[index] > 1e-8 and quality >= min_correlation:
            aligned.append(np.interp((grid + shift) % bins, grid, row, period=bins))
            accepted.append(index)
    usable = len(accepted) >= max(3, math.ceil(len(values) * 0.5))
    mean = np.mean(aligned, axis=0) if aligned else np.zeros(bins)
    contrast = float(np.ptp(mean))
    duty = float(np.mean(mean > (mean.max() + mean.min()) / 2)) if usable and contrast > 1e-8 else None
    accepted_set = set(accepted)
    return PhaseAverage(pitch, mean.tolist(), phases, correlations, accepted,
                        [i for i in range(len(values)) if i not in accepted_set], float(support.min()),
                        contrast, duty, usable, "periodic intensity profile only" if usable else "inconsistent frame shapes")


def screen_velocity(estimate: VelocityEstimate, pitch: float, *, direction: int | None = None,
                    max_displacement: float | None = None, bound_source: str = "") -> dict:
    """Require independent speed bounds before choosing a periodic alias branch."""
    if not math.isfinite(pitch) or pitch <= 0 or direction not in (None, -1, 1):
        raise ValueError("positive finite pitch and direction -1/1/None required")
    if max_displacement is not None:
        if not math.isfinite(max_displacement) or max_displacement <= 0 or not bound_source.strip():
            raise ValueError("a positive finite bound requires an independent source")
    rate = estimate.px_per_frame
    strong = [item for item in estimate.detail if item.agrees and item.peak_ratio >= 3]
    consistent = (estimate.ok and math.isfinite(rate) and math.isfinite(estimate.residual_px)
                  and estimate.residual_px <= max(1.5, 0.03 * pitch) and len(strong) >= 2)
    result = {"status": "REJECTED", "reason": "weak or inconsistent frame gaps",
              "candidate_px_per_frame": rate if math.isfinite(rate) else None,
              "selected_px_per_frame": None, "alias_period_px_per_frame": pitch,
              "independent_bound_px_per_frame": max_displacement, "bound_source": bound_source,
              "candidates": [], "candidates_exhaustive": False, "control_authorized": False}
    if not consistent:
        return result
    if max_displacement is None:
        orders = range(-2, 3)
    else:
        first = math.ceil((-max_displacement - rate) / pitch)
        last = math.floor((max_displacement - rate) / pitch)
        if last - first > 10000:
            raise ValueError("speed bound spans too many alias branches")
        orders = range(first, last + 1)
    candidates = [float(rate + n * pitch) for n in orders
                  if direction is None or (rate + n * pitch) * direction > 0]
    result.update(candidates=candidates, candidates_exhaustive=max_displacement is not None)
    if max_displacement is None or len(candidates) > 1:
        result.update(status="ALIAS_UNRESOLVED", reason="periodic images do not determine alias order")
    elif not candidates:
        result.update(reason="no branch satisfies the independent bound and direction")
    else:
        result.update(status="CONDITIONAL", reason="unique only under the supplied independent bound",
                      selected_px_per_frame=candidates[0])
    return result


def velocity_windows(profiles: np.ndarray, dt: float, *, window: int = 200, step: int = 200,
                     minimum_pitch: int = 25, maximum_pitch: int = 230,
                     direction: int | None = None, max_displacement: float | None = None,
                     bound_source: str = "") -> list[dict]:
    """Keep rejected and unresolved windows; never silently smooth their signs."""
    values = _profiles(profiles)
    if not math.isfinite(dt) or dt <= 0 or window < 12 or step < 1 or step > window:
        raise ValueError("positive dt, window >= 12 and 1 <= step <= window required")
    if len(values) < 12:
        return []
    size = min(window, len(values))
    starts = list(range(0, len(values) - size + 1, step))
    if starts[-1] != len(values) - size:
        starts.append(len(values) - size)
    rows = []
    for start in starts:
        segment = values[start:start + size]
        pitch, strength = spatial_pitch(segment, minimum_pitch, maximum_pitch)
        row = {"start_frame": start, "stop_frame_exclusive": start + size,
               "time_center_s": (start + (size - 1) / 2) * dt,
               "pitch_px": pitch, "pitch_strength": strength}
        if pitch <= 0 or strength < 0.25:
            rows.append(dict(row, status="REJECTED", reason="no spatial periodic signal",
                             selected_px_per_frame=None, control_authorized=False))
            continue
        estimate = estimate_velocity(segment, dt, pitch=pitch, direction=direction)
        row.update(screen_velocity(estimate, pitch, direction=direction,
                                   max_displacement=max_displacement, bound_source=bound_source))
        row["estimate"] = asdict(estimate)
        # A mean over a changing window must not masquerade as a steady point.
        halves = [estimate_velocity(part, dt, pitch=pitch, direction=direction)
                  for part in np.array_split(segment, 2)]
        delta = abs((halves[1].px_per_frame - halves[0].px_per_frame + pitch / 2) % pitch - pitch / 2)
        row["half_window_change_px_per_frame"] = float(delta)
        row["steady_candidate"] = bool(all(part.ok for part in halves) and delta < max(0.5, 0.01 * pitch))
        if not all(part.ok for part in halves) or delta > 0.15 * pitch:
            row.update(status="REJECTED", reason="unstable or inconsistent subwindows", selected_px_per_frame=None)
        selected = row["selected_px_per_frame"]
        row["selected_px_per_second"] = None if selected is None else selected / dt
        row["selected_passage_frequency_hz"] = None if selected is None else abs(selected) / dt / pitch
        rows.append(row)
    return rows
