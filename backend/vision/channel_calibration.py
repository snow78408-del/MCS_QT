from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class ChannelWidthMeasurement:
    width_px: float | None
    confidence: float
    reason: str
    upper_center_px: float | None = None
    lower_center_px: float | None = None
    upper_slope: float | None = None
    lower_slope: float | None = None


@dataclass(frozen=True)
class ChannelRoiSuggestion:
    x_start_ratio: float
    y_start_ratio: float
    x_end_ratio: float
    y_end_ratio: float
    confidence: float
    reason: str


@dataclass(frozen=True)
class _Line:
    center: float
    slope: float
    length: float
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0


DEFAULT_HOUGH_LINE_PARAMETERS: dict[str, float | int] = {
    "canny_low": 35,
    "canny_high": 100,
    # Zero keeps the previous adaptive value: max(24, image_width * 0.10).
    "hough_threshold": 0,
    "min_line_length_ratio": 0.55,
    "max_line_gap_ratio": 0.16,
    "max_tilt_degrees": 33.0,
    "merge_distance_px": 4.0,
    "max_lines": 32,
}


def normalize_hough_line_parameters(
    parameters: dict[str, float | int] | None,
) -> dict[str, float | int]:
    values = dict(DEFAULT_HOUGH_LINE_PARAMETERS)
    values.update(dict(parameters or {}))
    canny_low = max(0, min(254, int(float(values["canny_low"]))))
    canny_high = max(canny_low + 1, min(255, int(float(values["canny_high"]))))
    return {
        "canny_low": canny_low,
        "canny_high": canny_high,
        "hough_threshold": max(0, min(1000, int(float(values["hough_threshold"])))),
        "min_line_length_ratio": max(
            0.05, min(1.0, float(values["min_line_length_ratio"]))
        ),
        "max_line_gap_ratio": max(
            0.0, min(1.0, float(values["max_line_gap_ratio"]))
        ),
        "max_tilt_degrees": max(0.0, min(89.0, float(values["max_tilt_degrees"]))),
        "merge_distance_px": max(0.0, min(100.0, float(values["merge_distance_px"]))),
        "max_lines": max(2, min(200, int(float(values["max_lines"])))),
    }


def _long_wall_lines(
    gray: np.ndarray,
    hough_parameters: dict[str, float | int] | None = None,
) -> list[_Line]:
    height, width = gray.shape[:2]
    parameters = normalize_hough_line_parameters(hough_parameters)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
    edges = cv2.Canny(
        blurred,
        int(parameters["canny_low"]),
        int(parameters["canny_high"]),
    )
    requested_threshold = int(parameters["hough_threshold"])
    raw = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 360.0,
        threshold=(requested_threshold if requested_threshold > 0 else max(24, int(width * 0.10))),
        minLineLength=max(10, int(width * float(parameters["min_line_length_ratio"]))),
        maxLineGap=max(0, int(width * float(parameters["max_line_gap_ratio"]))),
    )
    if raw is None:
        return []

    center_x = (width - 1) * 0.5
    lines: list[_Line] = []
    for x1, y1, x2, y2 in np.asarray(raw).reshape(-1, 4):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if abs(dx) < 1.0:
            continue
        slope = dy / dx
        # Keep substantially tilted channels selectable. Near-vertical flow is
        # handled separately by the flow-axis rotation path.
        max_abs_slope = math.tan(math.radians(float(parameters["max_tilt_degrees"])))
        if abs(slope) > max_abs_slope:
            continue
        length = math.hypot(dx, dy)
        center = float(y1) + slope * (center_x - float(x1))
        lines.append(
            _Line(
                center=center,
                slope=slope,
                length=length,
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
            )
        )
    return lines


def detect_wall_line_candidates(
    frame: np.ndarray,
    max_lines: int | None = None,
    hough_parameters: dict[str, float | int] | None = None,
) -> list[dict[str, float | int]]:
    if frame is None or getattr(frame, "size", 0) == 0:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else np.asarray(frame)
    height, width = gray.shape[:2]
    parameters = normalize_hough_line_parameters(hough_parameters)
    candidates = _long_wall_lines(gray, parameters)
    candidates.sort(key=lambda line: line.length, reverse=True)
    selected: list[_Line] = []
    for line in candidates:
        if any(
            abs(line.center - prior.center) < float(parameters["merge_distance_px"])
            and abs(line.slope - prior.slope) < 0.03
            for prior in selected
        ):
            continue
        selected.append(line)
        requested_max_lines = int(parameters["max_lines"] if max_lines is None else max_lines)
        if len(selected) >= max(2, requested_max_lines):
            break
    selected.sort(key=lambda line: line.center)
    return [
        {
            "id": index,
            "x1": max(0.0, min(1.0, line.x1 / float(width))),
            "y1": max(0.0, min(1.0, line.y1 / float(height))),
            "x2": max(0.0, min(1.0, line.x2 / float(width))),
            "y2": max(0.0, min(1.0, line.y2 / float(height))),
            "length_ratio": line.length / float(width),
            "slope": line.slope,
        }
        for index, line in enumerate(selected, start=1)
    ]


def estimate_channel_width_px(
    roi_frame: np.ndarray,
    *,
    flow_axis: str = "x",
    hough_parameters: dict[str, float | int] | None = None,
) -> ChannelWidthMeasurement:
    """Measure the inner wall separation in a tightly selected channel ROI.

    The ROI is expected to include both channel walls with only a small margin.
    Work is intentionally bounded and is meant for startup calibration frames,
    not the permanent realtime detection loop.
    """

    if roi_frame is None or getattr(roi_frame, "size", 0) == 0:
        return ChannelWidthMeasurement(None, 0.0, "ROI 没有有效图像")
    gray = (
        cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        if roi_frame.ndim == 3
        else np.asarray(roi_frame)
    )
    if str(flow_axis).strip().lower() == "y":
        gray = np.rot90(gray)
    height, width = gray.shape[:2]
    if height < 40 or width < 100:
        return ChannelWidthMeasurement(None, 0.0, "ROI 太小，无法稳定拟合两侧管壁")

    lines = _long_wall_lines(gray, hough_parameters)
    min_span = width * 0.62
    # A tightly fitted ROI places the two walls in its outer bands. This gate
    # prevents long rows of adjacent droplets from being mistaken for walls.
    upper = [line for line in lines if line.length >= min_span and line.center <= height * 0.28]
    lower = [line for line in lines if line.length >= min_span and line.center >= height * 0.72]
    if not upper or not lower:
        return ChannelWidthMeasurement(
            None,
            0.0,
            "未同时找到两侧长管壁；请让 ROI 完整包含两条内壁并仅留 3–5 px 边缘",
        )

    # The operator deliberately fits the ROI to the two reference walls, so
    # prefer the long edge nearest each ROI boundary. Rows of touching droplets
    # can also produce long Hough segments, but they sit farther inside.
    band = max(6.0, min(12.0, height * 0.06))
    top_edge = min(line.center for line in upper)
    bottom_edge = max(line.center for line in lower)
    top_group = [line for line in upper if line.center <= top_edge + band]
    bottom_group = [line for line in lower if line.center >= bottom_edge - band]
    top = _Line(
        center=float(np.median([line.center for line in top_group])),
        slope=float(np.median([line.slope for line in top_group])),
        length=max(line.length for line in top_group),
    )
    bottom = _Line(
        center=float(np.median([line.center for line in bottom_group])),
        slope=float(np.median([line.slope for line in bottom_group])),
        length=max(line.length for line in bottom_group),
    )
    separation_y = bottom.center - top.center
    mean_slope = 0.5 * (top.slope + bottom.slope)
    separation = separation_y / math.sqrt(1.0 + mean_slope * mean_slope)
    ratio = separation / float(height)
    slope_delta = abs(top.slope - bottom.slope)
    if not (0.62 <= ratio <= 1.02) or slope_delta > 0.16:
        return ChannelWidthMeasurement(None, 0.0, "候选管壁不平行或间距与 ROI 不匹配")
    span_score = min(1.0, min(top.length, bottom.length) / float(width))
    parallel_score = max(0.0, 1.0 - slope_delta / 0.16)
    fill_score = max(0.0, 1.0 - abs(ratio - 0.94) / 0.30)
    confidence = 0.45 * span_score + 0.30 * parallel_score + 0.25 * fill_score
    if confidence < 0.68:
        return ChannelWidthMeasurement(None, confidence, "管壁拟合可信度不足，请收紧 ROI 或改善照明")
    return ChannelWidthMeasurement(
        width_px=float(separation),
        confidence=float(confidence),
        reason="ok",
        upper_center_px=float(top.center),
        lower_center_px=float(bottom.center),
        upper_slope=float(top.slope),
        lower_slope=float(bottom.slope),
    )


def suggest_channel_roi(
    frame: np.ndarray,
    *,
    flow_axis: str = "x",
    hough_parameters: dict[str, float | int] | None = None,
) -> ChannelRoiSuggestion | None:
    """Find the widest plausible pair of long parallel channel walls."""

    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else np.asarray(frame)
    vertical = str(flow_axis).strip().lower() == "y"
    working = np.rot90(gray) if vertical else gray
    height, width = working.shape[:2]
    if height < 40 or width < 100:
        return None
    lines = [
        line
        for line in _long_wall_lines(working, hough_parameters)
        if line.length >= width * 0.62 and height * 0.03 <= line.center <= height * 0.97
    ]
    if len(lines) < 2:
        return None

    lines.sort(key=lambda line: line.center)
    groups: list[list[_Line]] = []
    for line in lines:
        if not groups or line.center - float(np.median([item.center for item in groups[-1]])) > 12.0:
            groups.append([line])
        else:
            groups[-1].append(line)
    representatives = [
        _Line(
            center=float(np.median([line.center for line in group])),
            slope=float(np.median([line.slope for line in group])),
            length=max(line.length for line in group),
        )
        for group in groups
    ]

    best: tuple[float, _Line, _Line] | None = None
    for index, top in enumerate(representatives[:-1]):
        for bottom in representatives[index + 1 :]:
            separation = bottom.center - top.center
            ratio = separation / float(height)
            slope_delta = abs(top.slope - bottom.slope)
            if not (0.15 <= ratio <= 0.90) or slope_delta > 0.16:
                continue
            span = min(top.length, bottom.length) / float(width)
            # The physical channel walls normally form the widest strong,
            # parallel pair; internal droplet rows form narrower pairs.
            score = 0.60 * ratio + 0.25 * min(1.0, span) + 0.15 * max(0.0, 1.0 - slope_delta / 0.16)
            if best is None or score > best[0]:
                best = (score, top, bottom)
    if best is None:
        return None

    score, top, bottom = best
    margin = max(5, int(round(height * 0.012)))
    cross0 = max(0, int(math.floor(top.center)) - margin)
    cross1 = min(height, int(math.ceil(bottom.center)) + margin + 1)
    confidence = max(0.0, min(1.0, score))
    if vertical:
        # np.rot90 maps original x to the reversed working y coordinate.
        original_h, original_w = gray.shape[:2]
        x0 = max(0, original_w - cross1)
        x1 = min(original_w, original_w - cross0)
        return ChannelRoiSuggestion(
            x_start_ratio=x0 / float(original_w),
            y_start_ratio=0.0,
            x_end_ratio=x1 / float(original_w),
            y_end_ratio=1.0,
            confidence=confidence,
            reason="ok",
        )
    return ChannelRoiSuggestion(
        x_start_ratio=0.0,
        y_start_ratio=cross0 / float(height),
        x_end_ratio=1.0,
        y_end_ratio=cross1 / float(height),
        confidence=confidence,
        reason="ok",
    )
