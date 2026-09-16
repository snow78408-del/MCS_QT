from __future__ import annotations

import cv2
import numpy as np


def transverse_contrast(gray: np.ndarray) -> np.ndarray:
    """Axial outline signal, independent of bright/dark droplet polarity."""
    height = gray.shape[0]
    margin = max(1, int(round(height * 0.08)))
    inner = gray[margin:height - margin].astype(np.float32)
    if inner.shape[0] < 3:
        return np.zeros(gray.shape[1], dtype=np.float32)
    return np.asarray(
        np.percentile(inner, 95, axis=0) - np.percentile(inner, 5, axis=0),
        dtype=np.float32,
    )


def capsule_intervals(gray: np.ndarray) -> list[tuple[int, int]] | None:
    """Hysteretic capsule bodies; None means insufficient phase separation.

    Strong body seeds distinguish capsules from inter-droplet gaps. A lower
    threshold locates the outline ends without requiring a centre-band peak.
    Edge-touching intervals remain present so callers can reject partial drops.
    """
    smoothed = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    signal = transverse_contrast(smoothed)
    low, high = np.percentile(signal, [10, 90])
    spread = float(high - low)
    if spread < 3.0 or high < max(4.0, low * 1.5):
        return None
    normalized = np.uint8(np.clip((signal - low) * 255.0 / spread, 0, 255))
    threshold, _ = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    seeds = signal >= low + max(0.4, threshold / 255.0) * spread
    support = signal >= low + 0.20 * spread
    transitions = np.diff(np.r_[False, support, False].astype(np.int8))
    starts, stops = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
    return [
        (int(start), int(stop - 1))
        for start, stop in zip(starts, stops)
        if np.count_nonzero(seeds[start:stop]) >= max(3, (stop - start) * 0.4)
    ]


def raw_outline_contrast(
    smoothed: np.ndarray,
    left: int,
    right: int,
    *,
    background_start: int = 0,
    background_stop: int | None = None,
) -> float:
    """Contour change relative to each flank, in original grayscale units.

    Subtracting adjacent transverse profiles cancels stationary walls; using
    their 5–95% range also cancels a uniform illumination step. Both flanks
    must provide evidence, so a single boundary is not a complete capsule.
    Input is lightly smoothed original data, never normalized or CLAHE data.
    Background bounds exclude neighbouring capsule bodies, including partial
    or size-rejected ones. Fewer than two carrier columns is insufficient
    evidence, rather than permission to sample the next droplet.
    """
    height, width = smoothed.shape
    margin = max(1, int(height * 0.12))
    pad = max(3, int(round(height * 0.30)))
    trim = max(2, int(round((right - left) * 0.15)))
    body = smoothed[margin:height - margin, left + trim:right - trim]
    flank_start = min(left, max(0, left - pad, background_start))
    flank_stop = max(right + 1, min(
        width, right + pad + 1,
        width if background_stop is None else background_stop,
    ))
    flanks = (
        smoothed[margin:height - margin, flank_start:left],
        smoothed[margin:height - margin, right + 1:flank_stop],
    )
    if body.shape[0] < 3 or body.shape[1] < 3 or any(part.shape[1] < 2 for part in flanks):
        return 0.0
    inside = np.median(body, axis=1)
    contrasts = []
    for part in flanks:
        difference = inside - np.median(part, axis=1)
        low, high = np.percentile(difference, [5, 95])
        contrasts.append(float(high - low))
    return min(contrasts)
