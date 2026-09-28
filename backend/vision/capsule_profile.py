from __future__ import annotations

import cv2
import numpy as np


def dark_body_intervals(
    gray: np.ndarray,
    reference_width_px: float,
) -> list[tuple[int, int]]:
    """Find dark plugs bounded by two separate bright carrier gaps.

    The older outline path can select a bright carrier gap or a narrow halo
    fragment as a whole plug in low-contrast recordings.  This path first
    locates complete bright gaps, then measures the dark body *between* gaps.
    Slowly varying illumination is removed before comparison. Bodies touching
    either analysis edge are omitted.
    """
    if gray.ndim != 2:
        raise ValueError("phase plateau input must be grayscale")
    height, width = gray.shape
    channel = float(reference_width_px)
    if (height < 18 or width < max(100, height * 3)
            or not np.isfinite(channel) or channel < 14 or channel > height):
        return []
    upper = max(0, int(round(height * 0.20)))
    lower = min(height, int(round(height * 0.80)))
    profile = np.median(gray[upper:lower].astype(np.float32), axis=0)
    profile = cv2.GaussianBlur(profile.reshape(1, -1), (0, 0), 2.0).ravel()
    upper_shoulder = np.mean(gray[max(1, int(height * .08)):int(height * .25)], axis=0)
    lower_shoulder = np.mean(gray[int(height * .76):int(height * .94)], axis=0)
    transverse = cv2.GaussianBlur(
        (lower_shoulder - upper_shoulder).astype(np.float32).reshape(1, -1),
        (0, 0), 2.0).ravel()
    background = cv2.GaussianBlur(
        profile.reshape(1, -1), (0, 0), max(35.0, channel * 2.2)).ravel()
    residual = profile - background
    radius = max(16, int(round(channel * 1.35)))
    bright = np.flatnonzero(
        (residual[1:-1] >= residual[:-2])
        & (residual[1:-1] > residual[2:])) + 1
    peaks: list[int] = []
    for index in sorted(bright, key=lambda point: residual[point], reverse=True):
        if index < radius or index >= width - radius:
            continue
        if residual[index] < 3.0:
            continue
        left_low = float(np.min(residual[index - radius:index]))
        right_low = float(np.min(residual[index + 1:index + radius + 1]))
        if min(float(residual[index]) - left_low,
               float(residual[index]) - right_low) < 5.0:
            continue
        if all(abs(index - prior) >= channel * 0.9 for prior in peaks):
            peaks.append(int(index))
    peaks.sort()
    gaps: list[tuple[int, int]] = []
    for peak in peaks:
        left = peak
        right = peak
        while left > 0 and residual[left - 1] > 0.0:
            left -= 1
        while right + 1 < width and residual[right + 1] > 0.0:
            right += 1
        if (left > 0 and right < width - 1
                and channel * 0.25 <= right - left <= channel * 2.5
                and (not gaps or left > gaps[-1][1])):
            gaps.append((left, right))
    intervals: list[tuple[int, int]] = []
    for left_gap, right_gap in zip(gaps, gaps[1:]):
        left, right = left_gap[1], right_gap[0]
        length = right - left
        if not channel * 1.1 <= length <= channel * 12.0:
            continue
        margin = max(3, int(round(channel * 0.15)))
        core = residual[left + margin:right - margin]
        if core.size < 5 or float(np.mean(core < 0.0)) < 0.55:
            continue
        if float(np.percentile(core, 35)) > -1.0:
            continue
        body_inner = transverse[left + margin:right - margin]
        gap_reference = 0.5 * (
            float(np.median(transverse[left_gap[0]:left_gap[1] + 1]))
            + float(np.median(transverse[right_gap[0]:right_gap[1] + 1])))
        # A fixed illumination stripe brightens all rows together. A real
        # phase boundary reverses its upper/lower shoulder contrast across
        # the carrier gap and body, as seen in the rectified channel.
        if float(np.median(body_inner)) - gap_reference < 8.0:
            continue
        intervals.append((left, right))
    return intervals


def transverse_body_intervals(
    gray: np.ndarray,
    reference_width_px: float,
) -> list[tuple[int, int]]:
    """Find complete plugs when their centre has too little axial contrast.

    In some low-carrier-flow images the upper/lower wall contrast changes
    sign strongly at each carrier gap, while the centre intensity barely
    changes. Require two full negative-contrast gaps and a positive-contrast
    body between them; axial illumination changes shared by all rows cancel.
    """
    if gray.ndim != 2:
        raise ValueError("transverse body input must be grayscale")
    height, width = gray.shape
    channel = float(reference_width_px)
    if (height < 18 or width < max(100, height * 3)
            or not np.isfinite(channel) or channel < 14 or channel > height):
        return []
    upper = np.mean(gray[max(1, int(height * .08)):int(height * .25)], axis=0)
    lower = np.mean(gray[int(height * .76):int(height * .94)], axis=0)
    contrast = cv2.GaussianBlur(
        (lower - upper).astype(np.float32).reshape(1, -1), (0, 0), 2.0
    ).ravel()
    mask = contrast < -3.0
    changes = np.diff(np.r_[0, mask.astype(np.int8), 0])
    gaps = [
        (int(left), int(right))
        for left, right in zip(np.flatnonzero(changes == 1),
                               np.flatnonzero(changes == -1))
        if (left > 3 and right < width - 3
            and channel * .25 <= right - left <= channel * 3.0
            and float(np.median(contrast[left:right])) < -5.0)
    ]
    intervals: list[tuple[int, int]] = []
    for (_, left), (right, _) in zip(gaps, gaps[1:]):
        if not channel * 1.1 <= right - left <= channel * 12.0:
            continue
        body = contrast[left:right]
        if (float(np.median(body)) > 8.0
                and float(np.mean(body > 3.0)) >= .65):
            intervals.append((left, right))
    return intervals


def bounded_shoulder_intervals(
    gray: np.ndarray,
    reference_width_px: float,
) -> list[tuple[int, int]]:
    """Find complete weak-contrast plugs from both raw transverse shoulders.

    A bright centre alone can be a halo or carrier gap. Both shoulders must
    agree on a strong body seed; each endpoint may then grow only a short
    distance into weak support. All lengths and inlet margins scale with the
    measured channel width, so no recording-specific image coordinates enter
    the runtime detector. Intervals touching the analysis edge are incomplete.
    """
    height, width = gray.shape
    if height < 24 or width < max(80, height * 3):
        return []
    channel = float(reference_width_px)
    if not np.isfinite(channel) or channel < 16 or channel > height:
        return []

    smoothed = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.1)

    def band(start: float, stop: float) -> np.ndarray:
        first = max(0, min(height - 1, int(round(height * start))))
        last = max(first + 1, min(height, int(round(height * stop))))
        return np.mean(smoothed[first:last], axis=0)

    centre = band(23 / 53, 30 / 53)
    upper = centre - band(11 / 53, 19 / 53)
    lower = centre - band(34 / 53, 43 / 53)
    signal = cv2.GaussianBlur(np.minimum(upper, lower).reshape(1, -1),
                              (0, 0), 1.4).ravel()
    low, high = np.percentile(signal, [15, 85])
    # This branch addresses weak phase images. High-contrast synthetic and
    # ordinary generation images retain the established contour detector.
    if low >= -0.5 or not 2.0 <= high <= 12.0:
        return []

    def runs(mask: np.ndarray) -> list[tuple[int, int]]:
        transitions = np.diff(np.r_[0, mask.astype(np.int8), 0])
        return [(int(a), int(b)) for a, b in zip(
            np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1))]

    seeds = [(a, b) for a, b in runs(signal > 0.7)
             if b - a >= max(12, int(round(channel * 0.56)))
             and np.percentile(signal[a:b], 80) >= 2.0]
    supports = runs(signal > -1.5)
    extension = max(2, int(round(channel * 0.23)))
    neighbour_gap = max(2, int(round(channel * 0.38)))
    spacing = max(1, int(round(channel * 0.09)))
    accepted: list[tuple[int, int]] = []
    for a, b in supports:
        contained = [(sa, sb) for sa, sb in seeds if a <= (sa + sb) / 2 < b]
        for index, (sa, sb) in enumerate(contained):
            previous = contained[index - 1] if index else None
            following = contained[index + 1] if index + 1 < len(contained) else None
            left = sa if a <= spacing else max(a, sa - extension)
            right = min(b, sb + extension)
            if b == width and right >= width - 1:
                # A very weak carrier tail may never reach the -1.5 support
                # cutoff before the ROI ends. Accept its visible right
                # meniscus only when the paired signal has crossed zero and
                # remained negative for several measured columns. A body
                # still positive at the image edge stays incomplete.
                crossing = np.flatnonzero(signal[sb:] <= 0.0)
                if crossing.size:
                    turn = sb + int(crossing[0])
                    tail = signal[turn:]
                    if len(tail) >= max(7, int(round(channel * 0.13))) \
                            and float(np.mean(tail)) <= -0.5:
                        right = min(width - 1, turn + max(2, int(round(channel * 0.06))))
            if previous is not None:
                left = max(left, previous[1] + spacing)
            if following is not None:
                right = min(right, following[0] - spacing)
            length = right - left
            separated = ((previous is None or sa - previous[1] >= neighbour_gap)
                         and (following is None or following[0] - sb >= neighbour_gap))
            if (separated and left >= int(round(channel * 0.19))
                    and (left >= int(round(channel * 0.38)) or length >= channel * 2.08)
                    and right < width and channel * 1.4 <= length <= channel * 3.6
                    and np.mean(signal[sa:sb] > 0.7) >= 0.8):
                accepted.append((left, right))
    return accepted


def shoulder_phase_intervals(gray: np.ndarray) -> list[tuple[int, int]] | None:
    """Separate weak capsules by agreement of their two shoulder contrasts."""
    height, width = gray.shape
    if height < 16 or width < height * 3:
        return None
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
    center = np.median(smooth[int(height * .34):int(height * .66)], axis=0)
    upper = np.mean(smooth[max(1, int(height * .09)):int(height * .28)], axis=0) - center
    lower = np.mean(smooth[int(height * .73):int(height * .94)], axis=0) - center
    product = upper * lower
    signal = np.sign(product) * np.sqrt(np.abs(product))
    signal = cv2.GaussianBlur(signal.reshape(1, -1), (0, 0), 1.5).ravel()
    low, high = np.percentile(signal, [15, 85])
    if low > -1.5 or high < 2.0 or high - low < 5.0:
        return None
    mask = (signal > 0).astype(np.uint8).reshape(1, -1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8)).ravel().astype(bool)
    transitions = np.diff(np.r_[False, mask, False].astype(np.int8))
    intervals = [(int(start), int(stop - 1))
            for start, stop in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1))
            if stop - start >= max(4, height // 3)]
    edge_margin = max(3, int(round(height * .25)))
    return [(0 if start < edge_margin else start,
             width - 1 if stop >= width - edge_margin else stop)
            for start, stop in intervals]


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
