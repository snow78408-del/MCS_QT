from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import cv2
import numpy as np

from .config import ChannelRegionConfig
from .rectified_roi import rectify_channel_frame


@dataclass(frozen=True)
class ChannelRegionResult:
    status: str
    confidence: float
    reason: str
    wall_lines: list[dict[str, float]] = field(default_factory=list)
    high_frequency_map: np.ndarray | None = None
    line_overlay: np.ndarray | None = None
    region_overlay: np.ndarray | None = None
    rectified_frame: np.ndarray | None = None


@dataclass(frozen=True)
class _CandidateLine:
    p1: np.ndarray
    p2: np.ndarray
    direction: np.ndarray
    length: float
    support: float


class ChannelRegionDetector:
    """Calibrate a channel from persistent high-frequency, straight wall signals."""

    def __init__(self, config: ChannelRegionConfig) -> None:
        self.config = config
        self._frames: list[np.ndarray] = []
        status = "collecting" if config.enabled else "skipped"
        reason = "等待管道区域检定帧" if config.enabled else "已跳过管道区域检定"
        self._result = ChannelRegionResult(status, 0.0, reason)

    @property
    def result(self) -> ChannelRegionResult:
        return self._result

    def reset(self) -> None:
        self._frames.clear()
        status = "collecting" if self.config.enabled else "skipped"
        reason = "等待管道区域检定帧" if self.config.enabled else "已跳过管道区域检定"
        self._result = ChannelRegionResult(status, 0.0, reason)

    def add_frame(self, frame: np.ndarray) -> ChannelRegionResult:
        if not self.config.enabled:
            self._result = ChannelRegionResult("skipped", 0.0, "已跳过管道区域检定")
            return self._result
        if self._result.status in {"calibrated", "fallback"}:
            return self._result
        if frame is None or getattr(frame, "size", 0) == 0:
            self._result = ChannelRegionResult("fallback", 0.0, "输入图像为空，已回退整帧识别")
            return self._result
        # Keep bounded grayscale work frames instead of retaining a startup
        # burst of full-resolution color images (12 UHD BGR frames are about
        # 300 MB). Wall coordinates are normalized, so calibration at this
        # bounded resolution still applies to the original frames.
        gray = _gray_u8(np.asarray(frame))
        height, width = gray.shape[:2]
        scale = min(
            1.0,
            float(self.config.work_max_width) / max(1, width),
            float(self.config.work_max_height) / max(1, height),
        )
        if scale < 0.999:
            gray = cv2.resize(
                gray,
                (max(32, int(round(width * scale))), max(32, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        self._frames.append(gray.copy())
        required = max(1, int(self.config.sample_frames))
        if len(self._frames) < required:
            self._result = ChannelRegionResult(
                "collecting",
                0.0,
                f"正在采集管道检定帧 {len(self._frames)}/{required}",
            )
            return self._result
        self._result = detect_channel_region(self._frames, self.config)
        self._frames.clear()
        return self._result


def _gray_u8(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else np.asarray(frame)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return gray


def _line_support(response: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    length = max(2, int(round(float(np.linalg.norm(p2 - p1)))))
    points = np.linspace(p1, p2, length)
    xs = np.clip(np.rint(points[:, 0]).astype(np.int32), 0, response.shape[1] - 1)
    ys = np.clip(np.rint(points[:, 1]).astype(np.int32), 0, response.shape[0] - 1)
    mask = np.zeros(response.shape, dtype=np.uint8)
    mask[ys, xs] = 255
    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8))
    values = response[mask > 0]
    return 0.0 if values.size == 0 else float(np.mean(values) / 255.0)


def _extend_to_bounds(point: np.ndarray, direction: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    values: list[float] = []
    if abs(float(direction[0])) > 1e-8:
        values.extend([(0.0 - float(point[0])) / float(direction[0]), ((width - 1.0) - float(point[0])) / float(direction[0])])
    if abs(float(direction[1])) > 1e-8:
        values.extend([(0.0 - float(point[1])) / float(direction[1]), ((height - 1.0) - float(point[1])) / float(direction[1])])
    intersections = []
    for value in values:
        candidate = point + direction * value
        if -0.5 <= candidate[0] <= width - 0.5 and -0.5 <= candidate[1] <= height - 0.5:
            intersections.append(candidate)
    if len(intersections) < 2:
        return point.copy(), point.copy()
    intersections.sort(key=lambda item: float(np.dot(item - point, direction)))
    return intersections[0], intersections[-1]


def _normalized_line(p1: np.ndarray, p2: np.ndarray, width: int, height: int) -> dict[str, float]:
    return {
        "x1": float(np.clip(p1[0] / width, 0.0, 1.0)),
        "y1": float(np.clip(p1[1] / height, 0.0, 1.0)),
        "x2": float(np.clip(p2[0] / width, 0.0, 1.0)),
        "y2": float(np.clip(p2[1] / height, 0.0, 1.0)),
    }


def detect_channel_region(frames: Sequence[np.ndarray], config: ChannelRegionConfig) -> ChannelRegionResult:
    usable = [np.asarray(frame) for frame in frames if frame is not None and getattr(frame, "size", 0) > 0]
    if not usable:
        return ChannelRegionResult("fallback", 0.0, "没有有效检定帧，已回退整帧识别")
    original_h, original_w = usable[0].shape[:2]
    usable = [frame for frame in usable if frame.shape[:2] == (original_h, original_w)]
    if not usable or original_h < 32 or original_w < 32:
        return ChannelRegionResult("fallback", 0.0, "检定帧尺寸不足或不一致，已回退整帧识别")

    scale = min(1.0, float(config.work_max_width) / original_w, float(config.work_max_height) / original_h)
    work_w = max(32, int(round(original_w * scale)))
    work_h = max(32, int(round(original_h * scale)))
    grays: list[np.ndarray] = []
    responses: list[np.ndarray] = []
    window = max(5, int(round(min(work_w, work_h) * float(config.frequency_window_ratio))))
    if window % 2 == 0:
        window += 1
    # The broad-region filter is always smaller than the narrowest accepted
    # channel, even if a saved profile contains conflicting values.
    region_ratio = min(
        max(0.001, float(config.min_frequency_region_thickness_ratio)),
        max(0.001, float(config.min_width_ratio) * 0.75),
    )
    region_size = max(3, int(round(min(work_w, work_h) * region_ratio)))
    if region_size % 2 == 0:
        region_size += 1
    region_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (region_size, region_size))
    for frame in usable:
        gray = _gray_u8(frame)
        if (work_w, work_h) != (original_w, original_h):
            gray = cv2.resize(gray, (work_w, work_h), interpolation=cv2.INTER_AREA)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        smooth = cv2.GaussianBlur(gray, (5, 5), 0)
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        # A wall edge is only a thin high-gradient line. What identifies the
        # channel is instead a *region* with persistently higher spatial
        # frequency than its surroundings. Convert per-pixel gradients into a
        # local RMS energy map before combining frames.
        local_energy = cv2.boxFilter(magnitude * magnitude, cv2.CV_32F, (window, window))
        local_energy = cv2.sqrt(np.maximum(local_energy, 0.0))
        # Thin edges are removed per frame before temporal support is measured;
        # otherwise a one-frame artifact could survive the later normalization.
        responses.append(cv2.morphologyEx(local_energy, cv2.MORPH_OPEN, region_kernel))
        grays.append(gray)

    aggregate_gray = np.median(np.stack(grays), axis=0).astype(np.uint8)
    energy_stack = np.stack(responses).astype(np.float32)
    # One shared scale preserves relative strength across all startup frames.
    energy_low = float(np.percentile(energy_stack, 10.0))
    energy_high = max(energy_low + 1.0, float(np.percentile(energy_stack, 98.0)))
    normalized_stack = np.clip(
        (energy_stack - energy_low) * (255.0 / (energy_high - energy_low)), 0, 255
    ).astype(np.uint8)
    support_maps: list[np.ndarray] = []
    for response in normalized_stack:
        threshold, _mask = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        support_maps.append(response > threshold)
    frame_support = np.mean(np.stack(support_maps), axis=0)
    persistent = frame_support >= float(np.clip(config.min_frequency_frame_support, 0.0, 1.0))
    high_frequency_float = np.mean(normalized_stack.astype(np.float32), axis=0)
    high_frequency_float[~persistent] = 0.0
    fused_low = float(np.percentile(high_frequency_float, 10.0))
    fused_high = max(fused_low + 1.0, float(np.percentile(high_frequency_float, 98.0)))
    high_frequency = np.clip(
        (high_frequency_float - fused_low) * (255.0 / (fused_high - fused_low)), 0, 255
    ).astype(np.uint8)
    high_frequency = cv2.GaussianBlur(high_frequency, (0, 0), 2.0)

    # Segment the high-frequency region first. Its high/low transition is the
    # only source used to generate boundary-line candidates; raw image edges
    # and visually strong but unrelated structures are deliberately ignored.
    _threshold, high_region_mask = cv2.threshold(
        high_frequency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    morphology_size = max(3, int(round(min(work_w, work_h) * 0.018)))
    if morphology_size % 2 == 0:
        morphology_size += 1
    morphology_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morphology_size, morphology_size)
    )
    high_region_mask = cv2.morphologyEx(high_region_mask, cv2.MORPH_CLOSE, morphology_kernel)
    high_region_mask = cv2.morphologyEx(high_region_mask, cv2.MORPH_OPEN, morphology_kernel)
    boundary_edges = cv2.Canny(
        high_frequency,
        max(0, int(config.canny_low)),
        max(int(config.canny_low) + 1, int(config.canny_high)),
    )
    boundary_edges = cv2.max(
        boundary_edges,
        cv2.morphologyEx(high_region_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)),
    )
    boundary_strength = cv2.magnitude(
        cv2.Sobel(high_frequency, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(high_frequency, cv2.CV_32F, 0, 1, ksize=3),
    )
    boundary_upper = max(1.0, float(np.percentile(boundary_strength, 98.0)))
    boundary_strength_u8 = np.clip(boundary_strength * (255.0 / boundary_upper), 0, 255).astype(np.uint8)

    min_dimension = float(min(work_w, work_h))
    max_dimension = float(max(work_w, work_h))
    raw = cv2.HoughLinesP(
        boundary_edges,
        1,
        np.pi / 360.0,
        threshold=max(8, int(config.hough_threshold)),
        minLineLength=max(12, int(max_dimension * float(config.min_line_length_ratio))),
        maxLineGap=max(2, int(max_dimension * float(config.max_line_gap_ratio))),
    )
    if raw is None:
        overlay = cv2.cvtColor(high_frequency, cv2.COLOR_GRAY2BGR)
        return ChannelRegionResult(
            "fallback",
            0.0,
            "高低频区域之间未形成可拟合的长边界，已回退整帧识别",
            [],
            high_frequency,
            overlay,
            overlay.copy(),
        )

    candidates: list[_CandidateLine] = []
    for x1, y1, x2, y2 in np.asarray(raw).reshape(-1, 4):
        p1 = np.array([x1, y1], dtype=np.float32)
        p2 = np.array([x2, y2], dtype=np.float32)
        vector = p2 - p1
        length = float(np.linalg.norm(vector))
        if length < 12.0:
            continue
        direction = vector / length
        if direction[0] < 0.0 or (abs(float(direction[0])) < 1e-6 and direction[1] < 0.0):
            p1, p2, direction = p2, p1, -direction
        support = _line_support(boundary_strength_u8, p1, p2)
        candidates.append(_CandidateLine(p1, p2, direction, length, support))
    candidates.sort(key=lambda item: item.length * (0.4 + item.support), reverse=True)
    candidates = candidates[: max(4, int(config.max_lines))]

    yy, xx = np.indices(high_frequency.shape, dtype=np.float32)
    frequency_low = float(np.percentile(high_frequency, 10.0))
    frequency_high = float(np.percentile(high_frequency, 90.0))
    frequency_range = max(1.0, frequency_high - frequency_low)
    best: tuple[float, float, _CandidateLine, _CandidateLine] | None = None
    angle_tolerance = math.radians(max(0.5, float(config.parallel_tolerance_degrees)))
    for index, first in enumerate(candidates[:-1]):
        for second in candidates[index + 1 :]:
            dot = float(np.clip(abs(np.dot(first.direction, second.direction)), 0.0, 1.0))
            angle = math.acos(dot)
            if angle > angle_tolerance:
                continue
            direction = first.direction + (
                second.direction if np.dot(first.direction, second.direction) >= 0 else -second.direction
            )
            direction /= max(1e-8, float(np.linalg.norm(direction)))
            normal = np.array([-direction[1], direction[0]], dtype=np.float32)
            first_offset = float(np.dot((first.p1 + first.p2) * 0.5, normal))
            second_offset = float(np.dot((second.p1 + second.p2) * 0.5, normal))
            low_offset, high_offset = sorted((first_offset, second_offset))
            separation = high_offset - low_offset
            width_ratio = separation / min_dimension
            if not float(config.min_width_ratio) <= width_ratio <= float(config.max_width_ratio):
                continue
            first_distances = [float(np.dot(point - first.p1, normal)) for point in (second.p1, second.p2)]
            straightness_error = abs(first_distances[0] - first_distances[1]) / max(separation, 1.0)
            if straightness_error > float(config.max_separation_variation_ratio):
                continue

            signed_distance = xx * normal[0] + yy * normal[1]
            inset = max(2.0, separation * 0.08)
            inside_mask = (signed_distance >= low_offset + inset) & (signed_distance <= high_offset - inset)
            outside_width = max(inset * 2.0, separation * 0.35)
            outside_mask = (
                ((signed_distance >= low_offset - outside_width) & (signed_distance <= low_offset - inset))
                | ((signed_distance >= high_offset + inset) & (signed_distance <= high_offset + outside_width))
            )
            if int(np.count_nonzero(inside_mask)) < 32 or int(np.count_nonzero(outside_mask)) < 32:
                continue
            inside_mean = float(np.mean(high_frequency[inside_mask]))
            outside_mean = float(np.mean(high_frequency[outside_mask]))
            contrast_ratio = (inside_mean - outside_mean) / frequency_range
            if contrast_ratio < float(config.min_region_contrast):
                continue
            inside_coverage = float(np.mean(high_region_mask[inside_mask] > 0))
            outside_coverage = float(np.mean(high_region_mask[outside_mask] > 0))
            coverage_advantage = max(0.0, inside_coverage - outside_coverage)
            if (
                inside_coverage < float(config.min_region_coverage)
                or coverage_advantage < float(config.min_coverage_advantage)
            ):
                continue
            contrast_score = min(1.0, contrast_ratio / max(0.01, float(config.full_region_contrast)))
            boundary_score = min(1.0, 0.5 * (first.support + second.support) * 1.5)
            frequency_score = 0.65 * contrast_score + 0.20 * coverage_advantage + 0.15 * boundary_score

            span_score = min(1.0, min(first.length, second.length) / max_dimension)
            parallel_score = max(0.0, 1.0 - angle / angle_tolerance)
            consistency_score = max(
                0.0,
                1.0 - straightness_error / max(1e-6, float(config.max_separation_variation_ratio)),
            )
            straightness_score = 0.45 * span_score + 0.30 * parallel_score + 0.25 * consistency_score
            geometry_score = min(1.0, width_ratio / 0.30) * min(
                1.0, span_score / max(0.1, float(config.min_line_length_ratio))
            )
            weights = np.maximum(
                np.asarray(
                    [config.high_frequency_weight, config.straightness_weight, config.geometry_weight],
                    dtype=np.float64,
                ),
                0.0,
            )
            weight_sum = float(np.sum(weights))
            score = 0.0 if weight_sum <= 1e-9 else float(
                np.dot(weights, [frequency_score, straightness_score, geometry_score]) / weight_sum
            )
            score = max(0.0, min(1.0, score))
            if best is None or score > best[0]:
                best = (score, separation, first, second)

    line_overlay = cv2.applyColorMap(high_frequency, cv2.COLORMAP_TURBO)
    contours, _hierarchy = cv2.findContours(
        high_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(line_overlay, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
    for candidate in candidates:
        cv2.line(
            line_overlay,
            tuple(np.rint(candidate.p1).astype(int)),
            tuple(np.rint(candidate.p2).astype(int)),
            (255, 150, 0),
            1,
            cv2.LINE_AA,
        )
    if best is None:
        return ChannelRegionResult(
            "fallback",
            0.0,
            "未找到内部高频、外部低频且边界近似平行直线的区域，已回退整帧识别",
            [],
            high_frequency,
            line_overlay,
            line_overlay.copy(),
        )

    score, _separation, first, second = best
    selected: list[dict[str, float]] = []
    region_overlay = usable[-1].copy()
    for candidate in (first, second):
        start, end = _extend_to_bounds(candidate.p1, candidate.direction, work_w, work_h)
        start_original = start / scale
        end_original = end / scale
        selected.append(_normalized_line(start_original, end_original, original_w, original_h))
        cv2.line(region_overlay, tuple(np.rint(start_original).astype(int)), tuple(np.rint(end_original).astype(int)), (0, 140, 255), 3, cv2.LINE_AA)
    rectified = rectify_channel_frame(usable[-1], selected)
    if score < float(config.min_confidence) or rectified is None:
        reason = f"管道区域可信度 {score:.2f} 低于阈值 {float(config.min_confidence):.2f}，已回退整帧识别"
        return ChannelRegionResult("fallback", float(score), reason, [], high_frequency, line_overlay, region_overlay)
    return ChannelRegionResult(
        "calibrated",
        float(score),
        "已由高低频区域界线拟合出两条管壁并确定有效管道区域",
        selected,
        high_frequency,
        line_overlay,
        region_overlay,
        rectified,
    )
