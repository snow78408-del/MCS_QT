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
