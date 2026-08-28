from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from .config import DebugConfig, DetectorConfig
except ImportError:
    from config import DebugConfig, DetectorConfig


@dataclass
class DetectionResult:
    centers: List[np.ndarray]
    radii: List[float]
    debug_image: np.ndarray
    helper_mask: np.ndarray
    diameter_valid: List[bool] = field(default_factory=list)


@dataclass
class _ScoredCandidate:
    center: np.ndarray
    radius: float
    score: float
    edge_support: float
    ring_contrast: float
    center_contrast: float


class DropletDetector:
    def __init__(self, config: DetectorConfig, debug: DebugConfig) -> None:
        self._config = config
        self._debug = debug
        self._circle_offset_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._frame_index = 0
        self._background: np.ndarray | None = None
        self._runtime_min_radius = max(1.0, float(config.min_radius))
        self._runtime_max_radius = max(self._runtime_min_radius + 1.0, float(config.max_radius))
        configured_preferred = float(config.expected_radius or config.hough_preferred_radius)
        self._has_expected_size = configured_preferred > 0.0
        self._runtime_preferred_radius = (
            configured_preferred
            if configured_preferred > 0.0
            else float(np.sqrt(self._runtime_min_radius * self._runtime_max_radius))
        )
        self._configured_preferred_radius = float(self._runtime_preferred_radius)

    def configure_expected_diameter(self, diameter_um: float, pixel_to_micron: float) -> None:
        """Compatibility hook; the PID target must never alter measurement.

        Detector priors come only from detector configuration or an explicit
        image-based calibration (`calibrate_preferred_radius`).  Keeping this
        method as a no-op avoids silently reintroducing target leakage through
        older callers.
        """
        _ = (diameter_um, pixel_to_micron)

    def reset_adaptive_size(self) -> None:
        self._runtime_preferred_radius = float(self._configured_preferred_radius)
        self._background = None
        self._frame_index = 0

    def runtime_radius_range(self) -> tuple[float, float, float]:
        return (
            float(self._runtime_min_radius),
            float(self._runtime_preferred_radius),
            float(self._runtime_max_radius),
        )

    def calibrate_preferred_radius(self, radii: list[float]) -> float:
        """Calibrate the soft size preference from observed camera detections."""
        values = [float(value) for value in radii if self._runtime_min_radius <= float(value) <= self._runtime_max_radius]
        if not values:
            raise ValueError("没有可用于标定的有效液滴半径样本")
        preferred = float(np.median(np.asarray(values, dtype=np.float32)))
        self._runtime_preferred_radius = preferred
        self._configured_preferred_radius = preferred
        self._has_expected_size = True
        return preferred

    def detect(self, gray_frame: np.ndarray, mode: Optional[str] = None) -> DetectionResult:
        gray = self._ensure_gray(gray_frame)
        normalized = (
            cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            if self._config.enable_intensity_normalization
            else gray.copy()
        )

        blur_size = self._odd(self._config.gaussian_blur_size)
        smoothed = (
            cv2.GaussianBlur(normalized, (blur_size, blur_size), 0)
            if self._config.enable_gaussian_blur and blur_size > 1
            else normalized
        )

        cut_line = int(smoothed.shape[0] * self._config.cut_line_ratio)
        centers, radii = self._detect_hybrid_candidates(smoothed, cut_line)
        centers, radii = self._score_and_suppress_candidates(normalized, centers, radii)
        diameter_valid = [
            self._candidate_diameter_valid(normalized.shape[:2], center, radius)
            for center, radius in zip(centers, radii)
        ]
        helper_mask = self._build_bead_helper_mask(normalized)
        debug_image = np.empty((0, 0, 3), dtype=np.uint8)

        return DetectionResult(
            centers=centers,
            radii=radii,
            debug_image=debug_image,
            helper_mask=helper_mask,
            diameter_valid=diameter_valid,
        )

    def _ensure_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _detect_hybrid_candidates(
        self,
        normalized_gray: np.ndarray,
        cut_line: int,
        trace: dict[str, object] | None = None,
        *,
        force_hough: bool = False,
    ) -> Tuple[List[np.ndarray], List[float]]:
        contour_centers: List[np.ndarray] = []
        contour_radii: List[float] = []
        ambiguous_regions: List[Tuple[int, int, int, int]] = []
        if self._config.enable_contour_candidates:
            contour_centers, contour_radii, ambiguous_regions = self._detect_contour_candidates(
                normalized_gray,
                cut_line,
                trace,
            )

        interval = max(0, int(self._config.hough_refresh_interval))
        periodic_refresh = interval > 0 and self._frame_index % interval == 0
        contour_empty = not contour_centers
        hough_enabled = bool(self._config.enable_hough_candidates)
        run_full_hough = hough_enabled and (
            force_hough or (contour_empty and not ambiguous_regions) or periodic_refresh
        )
        run_local_hough = hough_enabled and bool(ambiguous_regions) and not run_full_hough
        self._frame_index += 1

        hough_centers: List[np.ndarray] = []
        hough_radii: List[float] = []
        hough_trace: dict[str, object] | None = {} if trace is not None else None
        if run_full_hough:
            hough_centers, hough_radii = self._detect_hough_candidates(
                normalized_gray,
                cut_line,
                hough_trace,
            )
        elif run_local_hough:
            hough_centers, hough_radii = self._detect_local_hough_candidates(
                normalized_gray,
                cut_line,
                ambiguous_regions,
            )
        if trace is not None:
            trace["hough_executed"] = run_full_hough or run_local_hough
            trace["hough_scope"] = "full" if run_full_hough else ("local" if run_local_hough else "skipped")
            trace["hough"] = hough_trace or {}
            trace["hybrid_contour_count"] = len(contour_centers)
            trace["hybrid_hough_count"] = len(hough_centers)
            trace["hybrid_hough_centers"] = list(hough_centers)
            trace["hybrid_hough_radii"] = list(hough_radii)
            trace["ambiguous_regions"] = list(ambiguous_regions)

        novel_hough_centers: List[np.ndarray] = []
        novel_hough_radii: List[float] = []
        for hough_center, hough_radius in zip(hough_centers, hough_radii):
            overlaps_verified_contour = any(
                float(np.linalg.norm(hough_center - contour_center))
                < 0.35 * (float(hough_radius) + float(contour_radius))
                for contour_center, contour_radius in zip(contour_centers, contour_radii)
            )
            if not overlaps_verified_contour:
                novel_hough_centers.append(hough_center)
                novel_hough_radii.append(float(hough_radius))

        return (
            [*contour_centers, *novel_hough_centers],
            [*contour_radii, *novel_hough_radii],
        )

    def _detect_contour_candidates(
        self,
        normalized_gray: np.ndarray,
        cut_line: int,
        trace: dict[str, object] | None = None,
    ) -> Tuple[List[np.ndarray], List[float], List[Tuple[int, int, int, int]]]:
        """Segment cheaply at reduced resolution and refine accepted shapes."""
        work_scale = min(1.0, max(0.25, float(self._config.contour_work_scale)))
        if work_scale < 0.999:
            target = (
                max(1, int(round(normalized_gray.shape[1] * work_scale))),
                max(1, int(round(normalized_gray.shape[0] * work_scale))),
            )
            work_gray = cv2.resize(normalized_gray, target, interpolation=cv2.INTER_AREA)
        else:
            work_gray = normalized_gray
            work_scale = 1.0
        scaled_cut_line = int(round(float(cut_line) * work_scale))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(work_gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        block_size = self._odd(self._config.adaptive_threshold_block_size)
        block_size = max(3, block_size)
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            float(self._config.adaptive_threshold_c),
        )
        open_size = self._odd(self._config.morphology_open_kernel)
        close_size = self._odd(self._config.morphology_close_kernel)
        if self._config.enable_morphology and open_size > 1:
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)),
            )
        if self._config.enable_morphology and close_size > 1:
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
            )

        foreground = self._background_foreground_mask(work_gray)
        low = max(1.0, float(self._config.contour_canny_low))
        high = max(low + 1.0, float(self._config.contour_canny_high))
        edges = cv2.Canny(blurred, low, high)
        kernel_size = self._odd(self._config.contour_close_kernel)
        if kernel_size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        else:
            closed = edges
        segmentation = cv2.bitwise_or(binary, foreground)
        segmentation = cv2.bitwise_or(segmentation, closed)
        support_edges = self._build_support_edges(edges)
        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            np.uint8(segmentation > 0),
            connectivity=8,
        )

        raw_candidates: List[Tuple[float, float, float]] = []
        accepted: List[Tuple[float, float, float]] = []
        split_candidates: set[Tuple[float, float, float]] = set()
        ambiguous_regions: List[Tuple[int, int, int, int]] = []
        split_regions = 0
        minimum_component_pixels = max(4, int(round(float(self._config.min_contour_area) * work_scale)))
        for label in range(1, component_count):
            if int(stats[label, cv2.CC_STAT_AREA]) < minimum_component_pixels:
                continue
            component = np.uint8(labels == label) * 255
            contours, _hierarchy = cv2.findContours(
                component,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) < max(1.0, float(self._config.min_contour_area) * work_scale * work_scale):
                continue
            segments: List[np.ndarray] = []
            if self._config.enable_watershed_split and self._contour_needs_split(contour):
                segments = self._split_contour_watershed(work_gray, contour)
            shapes = segments if len(segments) > 1 else [contour]
            if len(segments) > 1:
                split_regions += 1
            region_accepted = 0
            for shape in shapes:
                if len(segments) > 1:
                    (split_cx, split_cy), split_radius = cv2.minEnclosingCircle(shape)
                    fitted = (float(split_cx), float(split_cy), float(split_radius))
                else:
                    fitted = self._fit_contour_candidate(shape)
                if fitted is None:
                    continue
                raw_candidates.append(fitted)
                if self._validate_contour_candidate(shape, fitted, support_edges, scaled_cut_line):
                    accepted.append(fitted)
                    if len(segments) > 1:
                        split_candidates.add(fitted)
                    region_accepted += 1
            if len(segments) > 1 or region_accepted == 0:
                x, y, width, height = cv2.boundingRect(contour)
                ambiguous_regions.append(
                    (
                        int(round(x / work_scale)),
                        int(round(y / work_scale)),
                        int(round(width / work_scale)),
                        int(round(height / work_scale)),
                    )
                )

        accepted = self._filter_expected_size(accepted)
        accepted.sort(key=lambda item: self._candidate_priority(item[2]))
        accepted = accepted[: max(1, int(self._config.contour_max_candidates))]
        refinement_edges = cv2.Canny(
            cv2.GaussianBlur(normalized_gray, (3, 3), 0),
            low,
            high,
        )
        refined: List[Tuple[float, float, float]] = []
        for cx, cy, radius in accepted:
            scaled_candidate = (cx / work_scale, cy / work_scale, radius / work_scale)
            refined.append(
                scaled_candidate
                if (cx, cy, radius) in split_candidates
                else self._refine_contour_candidate(refinement_edges, scaled_candidate)
            )
        if trace is not None:
            trace.update(
                {
                    "contour_work_gray": work_gray,
                    "contour_work_scale": work_scale,
                    "contour_enhanced": enhanced,
                    "contour_binary": binary,
                    "contour_foreground": foreground,
                    "contour_edges": edges,
                    "contour_closed": segmentation,
                    "contour_raw_centers": [
                        np.array([cx / work_scale, cy / work_scale], dtype=np.float32)
                        for cx, cy, _radius in raw_candidates
                    ],
                    "contour_raw_radii": [
                        float(radius / work_scale) for _cx, _cy, radius in raw_candidates
                    ],
                    "contour_centers": [
                        np.array([cx, cy], dtype=np.float32) for cx, cy, _radius in refined
                    ],
                    "contour_radii": [float(radius) for _cx, _cy, radius in refined],
                    "contour_split_regions": split_regions,
                }
            )
        return (
            [np.array([cx, cy], dtype=np.float32) for cx, cy, _radius in refined],
            [float(radius) for _cx, _cy, radius in refined],
            ambiguous_regions[: max(1, int(self._config.local_hough_max_regions))],
        )

    def _background_foreground_mask(self, gray: np.ndarray) -> np.ndarray:
        if not self._config.enable_background_subtraction:
            return np.zeros_like(gray)
        if self._background is None or self._background.shape != gray.shape:
            self._background = gray.astype(np.float32)
            return np.zeros_like(gray)

        background_u8 = cv2.convertScaleAbs(self._background)
        difference = cv2.absdiff(gray, background_u8)
        _threshold, foreground = cv2.threshold(
            difference,
            max(1.0, float(self._config.background_difference_threshold)),
            255,
            cv2.THRESH_BINARY,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

        learning_rate = min(1.0, max(0.0, float(self._config.background_learning_rate)))
        if learning_rate > 0.0:
            cv2.accumulateWeighted(
                gray.astype(np.float32),
                self._background,
                learning_rate,
            )
        return foreground

    def _refine_contour_candidate(
        self,
        edges: np.ndarray,
        candidate: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        cx, cy, radius = candidate
        margin = max(4, int(round(radius * 1.35)))
        x0 = max(0, int(round(cx)) - margin)
        x1 = min(edges.shape[1], int(round(cx)) + margin + 1)
        y0 = max(0, int(round(cy)) - margin)
        y1 = min(edges.shape[0], int(round(cy)) + margin + 1)
        if x1 - x0 < 5 or y1 - y0 < 5:
            return candidate
        edge_y, edge_x = np.nonzero(edges[y0:y1, x0:x1])
        if edge_x.size < 12:
            return candidate
        absolute_x = edge_x.astype(np.float32) + float(x0)
        absolute_y = edge_y.astype(np.float32) + float(y0)
        radial_distance = np.hypot(absolute_x - cx, absolute_y - cy)
        annulus = (radial_distance >= radius * 0.65) & (radial_distance <= radius * 1.35)
        if int(np.count_nonzero(annulus)) < 12:
            return candidate
        points = np.column_stack((absolute_x[annulus], absolute_y[annulus])).astype(np.float32)
        (refined_cx, refined_cy), (axis_a, axis_b), _angle = cv2.fitEllipse(points.reshape(-1, 1, 2))
        refined_radius = float(np.sqrt(max(0.5, axis_a * 0.5) * max(0.5, axis_b * 0.5)))
        center_shift = float(np.hypot(refined_cx - cx, refined_cy - cy))
        radius_ratio = refined_radius / max(1.0, radius)
        if center_shift > max(3.0, radius * 0.35) or not 0.65 <= radius_ratio <= 1.35:
            return candidate
        return float(refined_cx), float(refined_cy), refined_radius

    def _detect_local_hough_candidates(
        self,
        gray: np.ndarray,
        cut_line: int,
        regions: List[Tuple[int, int, int, int]],
    ) -> Tuple[List[np.ndarray], List[float]]:
        centers: List[np.ndarray] = []
        radii: List[float] = []
        padding = max(
            3,
            int(
                round(
                    self._runtime_preferred_radius
                    * float(self._config.local_hough_padding_ratio)
                )
            ),
        )
        for x, y, width, height in regions[: max(1, int(self._config.local_hough_max_regions))]:
            x0 = max(0, int(x) - padding)
            y0 = max(0, int(y) - padding)
            x1 = min(gray.shape[1], int(x + width) + padding)
            y1 = min(gray.shape[0], int(y + height) + padding)
            if x1 - x0 < 2 * self._hough_min_radius() or y1 - y0 < 2 * self._hough_min_radius():
                continue
            local_cut_line = min(y1 - y0, max(0, int(cut_line) - y0))
            local_centers, local_radii = self._detect_hough_candidates(
                gray[y0:y1, x0:x1],
                local_cut_line,
            )
            centers.extend(
                np.array([float(center[0]) + x0, float(center[1]) + y0], dtype=np.float32)
                for center in local_centers
            )
            radii.extend(float(radius) for radius in local_radii)
        return centers, radii

    def _split_contour_watershed(
        self,
        gray: np.ndarray,
        contour: np.ndarray,
    ) -> List[np.ndarray]:
        x, y, width, height = cv2.boundingRect(contour)
        if width < 3 or height < 3:
            return []
        padding = 2
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(gray.shape[1], x + width + padding)
        y1 = min(gray.shape[0], y + height + padding)
        local_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        shifted = contour.astype(np.int32).copy()
        shifted[:, 0, 0] -= x0
        shifted[:, 0, 1] -= y0
        cv2.drawContours(local_mask, [shifted], -1, 255, cv2.FILLED)

        distance = cv2.distanceTransform(local_mask, cv2.DIST_L2, 5)
        maximum = float(distance.max())
        minimum_peak = self._runtime_min_radius * float(
            self._config.watershed_min_peak_radius_ratio
        )
        if maximum < max(1.0, minimum_peak):
            return []
        threshold = max(
            minimum_peak,
            maximum * float(self._config.watershed_peak_ratio),
        )
        peaks = np.uint8(distance >= threshold) * 255
        peaks = cv2.morphologyEx(
            peaks,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        marker_count, seed_labels = cv2.connectedComponents(peaks)
        foreground_markers = marker_count - 1
        if foreground_markers < 2 or foreground_markers > max(
            2,
            int(self._config.watershed_max_markers),
        ):
            return []

        markers = seed_labels.astype(np.int32) + 1
        markers[local_mask == 0] = 1
        markers[(local_mask > 0) & (seed_labels == 0)] = 0
        local_gray = gray[y0:y1, x0:x1]
        watershed_input = cv2.cvtColor(local_gray, cv2.COLOR_GRAY2BGR)
        cv2.watershed(watershed_input, markers)

        segments: List[np.ndarray] = []
        for label in range(2, marker_count + 1):
            region = np.uint8((markers == label) & (local_mask > 0)) * 255
            region_contours, _hierarchy = cv2.findContours(
                region,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            if not region_contours:
                continue
            segment = max(region_contours, key=cv2.contourArea).astype(np.int32)
            segment[:, 0, 0] += x0
            segment[:, 0, 1] += y0
            segments.append(segment)
        return segments

    @staticmethod
    def _contour_needs_split(contour: np.ndarray) -> bool:
        area = abs(float(cv2.contourArea(contour)))
        perimeter = float(cv2.arcLength(contour, True))
        _x, _y, width, height = cv2.boundingRect(contour)
        shorter = max(1.0, float(min(width, height)))
        aspect_ratio = float(max(width, height)) / shorter
        circularity = (
            4.0 * np.pi * area / (perimeter * perimeter)
            if perimeter > 1e-6
            else 0.0
        )
        return aspect_ratio >= 1.35 or circularity < 0.78

    def _fit_contour_candidate(
        self,
        contour: np.ndarray,
    ) -> Tuple[float, float, float] | None:
        if len(contour) >= 5:
            (cx, cy), (axis_a, axis_b), _angle = cv2.fitEllipse(contour)
            semi_a = max(0.5, float(axis_a) * 0.5)
            semi_b = max(0.5, float(axis_b) * 0.5)
            radius = float(np.sqrt(semi_a * semi_b))
        else:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if not np.isfinite([cx, cy, radius]).all():
            return None
        return float(cx), float(cy), float(radius)

    def _validate_contour_candidate(
        self,
        contour: np.ndarray,
        candidate: Tuple[float, float, float],
        support_edges: np.ndarray,
        cut_line: int,
    ) -> bool:
        cx, cy, radius = candidate
        if cy > float(cut_line):
            return False
        if radius < self._runtime_min_radius or radius > self._runtime_max_radius:
            return False
        area = abs(float(cv2.contourArea(contour)))
        perimeter = float(cv2.arcLength(contour, True))
        if area < max(float(self._config.min_contour_area), np.pi * radius * radius * 0.20):
            return False
        if perimeter <= 1e-6:
            return False
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < float(self._config.contour_min_circularity):
            return False

        if len(contour) >= 5:
            _center, (axis_a, axis_b), _angle = cv2.fitEllipse(contour)
            major = max(float(axis_a), float(axis_b))
            minor = min(float(axis_a), float(axis_b))
            axis_ratio = minor / major if major > 1e-6 else 0.0
            ellipse_area = np.pi * max(0.5, axis_a * 0.5) * max(0.5, axis_b * 0.5)
            area_fill = area / ellipse_area if ellipse_area > 1e-6 else 0.0
            if axis_ratio < float(self._config.contour_min_axis_ratio):
                return False
            if area_fill < float(self._config.contour_min_area_fill_ratio):
                return False

        points = contour.reshape(-1, 2)
        height, width = support_edges.shape[:2]
        valid = (
            (points[:, 0] >= 0)
            & (points[:, 0] < width)
            & (points[:, 1] >= 0)
            & (points[:, 1] < height)
        )
        if not np.any(valid):
            return False
        valid_points = points[valid]
        edge_support = float(
            np.count_nonzero(support_edges[valid_points[:, 1], valid_points[:, 0]])
        ) / float(len(valid_points))
        return edge_support >= float(self._config.contour_min_edge_support)

    def _detect_hough_candidates(
        self,
        normalized_gray: np.ndarray,
        cut_line: int,
        trace: dict[str, object] | None = None,
    ) -> Tuple[List[np.ndarray], List[float]]:
        hough_gray, scale = self._prepare_hough_frame(normalized_gray)
        scaled_cut_line = int(float(cut_line) * scale)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(hough_gray) if self._config.enable_hough_clahe else hough_gray
        blurred = cv2.medianBlur(enhanced, 5) if self._config.enable_hough_median_blur else enhanced
        edges = cv2.Canny(blurred, 50, 150)
        support_edges = self._build_support_edges(edges)
        if trace is not None:
            trace.update(
                {
                    "work_gray": hough_gray,
                    "scale": scale,
                    "enhanced": enhanced,
                    "median_blurred": blurred,
                    "edges": edges,
                    "support_edges": support_edges,
                    "raw_centers": [],
                    "raw_radii": [],
                }
            )
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=max(1.0, float(self._config.hough_dp)),
            minDist=max(1.0, self._hough_min_distance() * scale),
            param1=max(1.0, float(self._config.hough_param1)),
            param2=max(1.0, float(self._config.hough_param2)),
            minRadius=max(1, int(round(self._hough_min_radius() * scale))),
            maxRadius=max(1, int(round(self._hough_max_radius() * scale))),
        )
        if circles is None:
            return [], []

        raw_circles = np.round(circles[0]).astype(np.float32)
        if trace is not None:
            trace["raw_centers"] = [
                np.array([float(cx) / scale, float(cy) / scale], dtype=np.float32)
                for cx, cy, _radius in raw_circles
            ]
            trace["raw_radii"] = [float(radius) / scale for _cx, _cy, radius in raw_circles]

        candidates: List[Tuple[float, float, float]] = []
        for cx, cy, radius in raw_circles:
            if float(cy) > float(scaled_cut_line):
                continue
            original_radius = float(radius) / scale
            if original_radius < self._hough_min_radius() or original_radius > self._hough_max_radius():
                continue
            if self._circle_edge_support(support_edges, float(cx), float(cy), float(radius)) < float(
                self._config.hough_edge_support_threshold
            ):
                continue
            candidates.append((float(cx) / scale, float(cy) / scale, original_radius))

        if self._config.reject_multi_droplet_circles:
            candidates = self._remove_multi_droplet_circles(candidates)
        candidates = sorted(candidates, key=lambda item: self._candidate_priority(float(item[2])))
        max_candidates = max(1, int(self._config.hough_max_candidates))
        candidates = candidates[:max_candidates]
        if trace is not None:
            trace["geometric_centers"] = [np.array([cx, cy], dtype=np.float32) for cx, cy, _radius in candidates]
            trace["geometric_radii"] = [float(radius) for _cx, _cy, radius in candidates]

        candidates = self._filter_expected_size(candidates)
        if trace is not None:
            trace["size_centers"] = [np.array([cx, cy], dtype=np.float32) for cx, cy, _radius in candidates]
            trace["size_radii"] = [float(radius) for _cx, _cy, radius in candidates]
        if self._config.enable_edge_ownership_filter and len(candidates) > 1:
            candidates = self._filter_edge_ownership(edges, candidates, scale)
        if trace is not None:
            trace["owned_centers"] = [np.array([cx, cy], dtype=np.float32) for cx, cy, _radius in candidates]
            trace["owned_radii"] = [float(radius) for _cx, _cy, radius in candidates]

        centers: List[np.ndarray] = []
        radii: List[float] = []
        for cx, cy, radius in candidates:
            centers.append(np.array([cx, cy], dtype=np.float32))
            radii.append(radius)
        return centers, radii

    def _prepare_hough_frame(self, normalized_gray: np.ndarray) -> Tuple[np.ndarray, float]:
        height, width = normalized_gray.shape[:2]
        max_width = max(1, int(self._config.hough_work_max_width))
        max_height = max(1, int(self._config.hough_work_max_height))
        scale = min(1.0, float(max_width) / float(max(1, width)), float(max_height) / float(max(1, height)))
        if scale >= 0.999:
            return normalized_gray, 1.0
        target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        return cv2.resize(normalized_gray, target, interpolation=cv2.INTER_AREA), float(scale)

    def _hough_min_radius(self) -> float:
        configured = max(1.0, float(self._config.hough_min_radius))
        return max(configured, self._runtime_min_radius)

    def _hough_max_radius(self) -> float:
        configured = max(self._hough_min_radius(), float(self._config.hough_max_radius))
        return min(configured, self._runtime_max_radius)

    def _hough_min_distance(self) -> float:
        configured = float(self._config.hough_min_distance)
        if configured > 0.0:
            return configured
        return max(
            2.0,
            self._runtime_preferred_radius * float(self._config.min_center_distance_radius_ratio),
        )

    def _build_support_edges(self, edges: np.ndarray) -> np.ndarray:
        neighborhood = max(0, int(self._config.hough_edge_neighborhood))
        if neighborhood <= 0:
            return edges
        size = neighborhood * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        return cv2.dilate(edges, kernel)

    def _remove_multi_droplet_circles(self, candidates: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        kept: List[Tuple[float, float, float]] = []
        min_children = max(1, int(self._config.multi_droplet_child_count))
        child_radius_ratio = float(self._config.multi_droplet_child_radius_ratio)
        child_distance_ratio = float(self._config.multi_droplet_child_distance_ratio)
        for cx, cy, radius in candidates:
            child_count = 0
            for ox, oy, other_radius in candidates:
                if ox == cx and oy == cy and other_radius == radius:
                    continue
                if float(other_radius) > float(radius) * child_radius_ratio:
                    continue
                distance = float(np.hypot(float(cx) - float(ox), float(cy) - float(oy)))
                if distance < float(radius) * child_distance_ratio:
                    child_count += 1
                    if child_count >= min_children:
                        break
            if child_count >= min_children:
                continue
            kept.append((cx, cy, radius))
        return kept

    def _filter_expected_size(
        self,
        candidates: List[Tuple[float, float, float]],
    ) -> List[Tuple[float, float, float]]:
        # A configured target is a preference in soft mode.  Only the
        # explicit hard-gate option is allowed to reject Hough candidates by
        # radius; runtime bounds still protect the absolute detector range.
        if (
            not self._config.enable_expected_size_filter
            or not self._has_expected_size
            or not bool(self._config.expected_size_hard_gate)
        ):
            return candidates
        tolerance = max(0.0, float(self._config.expected_radius_tolerance_ratio))
        preferred = float(self._runtime_preferred_radius)
        minimum = preferred * max(0.0, 1.0 - tolerance)
        maximum = preferred * (1.0 + tolerance)
        return [item for item in candidates if minimum <= float(item[2]) <= maximum]

    def _filter_edge_ownership(
        self,
        edges: np.ndarray,
        candidates: List[Tuple[float, float, float]],
        scale: float,
    ) -> List[Tuple[float, float, float]]:
        """Reject circles whose supporting pixels belong to neighbouring circles.

        Each candidate perimeter is sampled angularly. The nearest real edge
        pixel is assigned to the candidate circle with the smallest radial
        residual. A false circle assembled from several neighbouring arcs has
        little uniquely owned support even when its total edge support is high.
        """
        scaled = [
            (float(cx) * scale, float(cy) * scale, float(radius) * scale)
            for cx, cy, radius in candidates
        ]
        search = max(1, int(round(float(self._config.edge_ownership_search_radius) * scale)))
        margin = max(0.0, float(self._config.edge_ownership_margin) * scale)
        minimum_ratio = min(1.0, max(0.0, float(self._config.edge_ownership_min_ratio)))
        height, width = edges.shape[:2]
        kept: List[Tuple[float, float, float]] = []

        for index, ((cx, cy, radius), original) in enumerate(zip(scaled, candidates)):
            neighbor_indices = [
                other_index
                for other_index, (ox, oy, other_radius) in enumerate(scaled)
                if other_index != index
                and float(np.hypot(cx - ox, cy - oy))
                <= radius + other_radius + float(search * 2)
            ]
            if not neighbor_indices:
                kept.append(original)
                continue
            xs, ys = self._circle_offsets(radius)
            supported = 0
            owned = 0
            for offset_x, offset_y in zip(xs, ys):
                expected_x = int(round(cx + float(offset_x)))
                expected_y = int(round(cy + float(offset_y)))
                x0 = max(0, expected_x - search)
                x1 = min(width, expected_x + search + 1)
                y0 = max(0, expected_y - search)
                y1 = min(height, expected_y + search + 1)
                if x0 >= x1 or y0 >= y1:
                    continue
                edge_y, edge_x = np.nonzero(edges[y0:y1, x0:x1])
                if edge_x.size == 0:
                    continue
                absolute_x = edge_x.astype(np.float32) + float(x0)
                absolute_y = edge_y.astype(np.float32) + float(y0)
                nearest = int(
                    np.argmin((absolute_x - expected_x) ** 2 + (absolute_y - expected_y) ** 2)
                )
                px = float(absolute_x[nearest])
                py = float(absolute_y[nearest])
                supported += 1
                own_residual = abs(float(np.hypot(px - cx, py - cy)) - radius)
                other_residual = min(
                    abs(float(np.hypot(px - scaled[other_index][0], py - scaled[other_index][1])) - scaled[other_index][2])
                    for other_index in neighbor_indices
                )
                if own_residual + margin < other_residual:
                    owned += 1
            ownership_ratio = float(owned) / float(supported) if supported else 0.0
            if ownership_ratio >= minimum_ratio:
                kept.append(original)
        return kept

    def _circle_edge_support(self, edges: np.ndarray, cx: float, cy: float, radius: float) -> float:
        h, w = edges.shape[:2]
        xs, ys = self._circle_offsets(float(radius))
        x = np.rint(float(cx) + xs).astype(np.int32)
        y = np.rint(float(cy) + ys).astype(np.int32)
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        valid_count = int(np.count_nonzero(valid))
        if valid_count <= 0:
            return 0.0
        hits = int(np.count_nonzero(edges[y[valid], x[valid]] > 0))
        return float(hits) / float(valid_count)

    def _circle_offsets(self, radius: float) -> tuple[np.ndarray, np.ndarray]:
        rounded_radius = max(1, int(round(radius)))
        samples = max(48, int(rounded_radius * 4.0))
        key = (rounded_radius, samples)
        cached = self._circle_offset_cache.get(key)
        if cached is not None:
            return cached
        theta = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False, dtype=np.float32)
        xs = np.cos(theta) * float(rounded_radius)
        ys = np.sin(theta) * float(rounded_radius)
        self._circle_offset_cache[key] = (xs, ys)
        return xs, ys

    def _score_and_suppress_candidates(
        self,
        normalized_gray: np.ndarray,
        centers: List[np.ndarray],
        radii: List[float],
    ) -> Tuple[List[np.ndarray], List[float]]:
        if not centers:
            return [], []

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized_gray)
        blurred = cv2.medianBlur(enhanced, 5)
        edges = cv2.Canny(blurred, 50, 150)
        support_edges = self._build_support_edges(edges)

        min_edge = max(0.0, float(self._config.candidate_min_edge_support))
        # Negative thresholds disable illumination-polarity filtering. Droplet
        # interiors may be brighter or darker depending on exposure and chip.
        min_contrast = float(self._config.candidate_min_ring_contrast)
        min_center_contrast = float(self._config.candidate_min_center_contrast)
        minimum_visible = min(
            1.0,
            max(0.0, float(self._config.candidate_min_visible_circle_ratio)),
        )
        height, width = normalized_gray.shape[:2]
        scored: List[_ScoredCandidate] = []

        for center, radius_value in zip(centers, radii):
            radius = float(radius_value)
            if radius < self._runtime_min_radius or radius > self._runtime_max_radius:
                continue
            cx = float(center[0])
            cy = float(center[1])
            if self._circle_visible_ratio((height, width), cx, cy, radius) < minimum_visible:
                continue

            edge_support = self._circle_edge_support(support_edges, cx, cy, radius)
            ring_contrast = self._ring_contrast(enhanced, cx, cy, radius)
            center_contrast = self._center_contrast(enhanced, cx, cy, radius)
            # Strong edges may survive locally weak illumination contrast, but
            # weak candidates must satisfy both tests.
            if edge_support < min_edge:
                continue
            if center_contrast < min_center_contrast:
                continue
            if ring_contrast < min_contrast and edge_support < (min_edge * 1.35):
                continue

            radius_span = max(
                1.0,
                self._runtime_max_radius - self._runtime_min_radius,
            )
            radius_score = max(
                0.0,
                1.0 - abs(radius - self._runtime_preferred_radius) / radius_span,
            )
            contrast_score = min(1.0, max(0.0, ring_contrast) / 0.10)
            center_score = min(1.0, max(0.0, center_contrast) / 0.25)
            score = (
                0.52 * edge_support
                + 0.22 * center_score
                + 0.16 * contrast_score
                + 0.10 * radius_score
            )
            scored.append(
                _ScoredCandidate(
                    center=np.asarray(center, dtype=np.float32),
                    radius=radius,
                    score=float(score),
                    edge_support=float(edge_support),
                    ring_contrast=float(ring_contrast),
                    center_contrast=float(center_contrast),
                )
            )

        scored.sort(key=lambda item: (-item.score, *self._candidate_priority(item.radius)))
        kept: List[_ScoredCandidate] = []
        overlap_ratio = max(0.0, float(self._config.candidate_nms_overlap_ratio))
        for candidate in scored:
            duplicate = False
            for existing in kept:
                distance = float(np.linalg.norm(candidate.center - existing.center))
                scale_distance = overlap_ratio * (candidate.radius + existing.radius)
                duplicate_distance = max(
                    scale_distance,
                    self._duplicate_distance(candidate.radius, existing.radius),
                )
                if distance < duplicate_distance:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)

        self._learn_preferred_radius(kept)
        return (
            [item.center for item in kept],
            [float(item.radius) for item in kept],
        )

    def _circle_visible_ratio(
        self,
        shape: tuple[int, int],
        cx: float,
        cy: float,
        radius: float,
    ) -> float:
        height, width = shape
        xs, ys = self._circle_offsets(radius)
        x = np.rint(float(cx) + xs).astype(np.int32)
        y = np.rint(float(cy) + ys).astype(np.int32)
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        return float(np.count_nonzero(valid)) / float(max(1, len(valid)))

    def _candidate_diameter_valid(
        self,
        shape: tuple[int, int],
        center: np.ndarray,
        radius: float,
    ) -> bool:
        height, width = shape
        margin = float(radius) * max(0.0, float(self._config.candidate_full_circle_ratio))
        cx = float(center[0])
        cy = float(center[1])
        return (
            cx >= margin
            and cy >= margin
            and cx < float(width) - margin
            and cy < float(height) - margin
        )

    def _learn_preferred_radius(self, candidates: List[_ScoredCandidate]) -> None:
        min_candidates = max(1, int(self._config.adaptive_radius_min_candidates))
        if len(candidates) < min_candidates:
            return

        scores = np.asarray([item.score for item in candidates], dtype=np.float32)
        score_floor = float(np.percentile(scores, 55.0))
        trusted_radii = [
            float(item.radius)
            for item in candidates
            if float(item.score) >= score_floor
        ]
        if len(trusted_radii) < min_candidates:
            return

        observed = float(np.median(np.asarray(trusted_radii, dtype=np.float32)))
        observed = min(self._runtime_max_radius, max(self._runtime_min_radius, observed))
        alpha = min(1.0, max(0.0, float(self._config.adaptive_radius_learning_rate)))
        self._runtime_preferred_radius = (
            (1.0 - alpha) * self._runtime_preferred_radius
            + alpha * observed
        )

    def _ring_contrast(
        self,
        image: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
    ) -> float:
        edge_mean = self._circle_sample_mean(image, cx, cy, radius)
        inner_mean = self._circle_sample_mean(image, cx, cy, radius * 0.70)
        outer_mean = self._circle_sample_mean(image, cx, cy, radius * 1.22)
        if edge_mean is None or inner_mean is None or outer_mean is None:
            return 0.0
        return float(((inner_mean + outer_mean) * 0.5 - edge_mean) / 255.0)

    def _center_contrast(
        self,
        image: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
    ) -> float:
        edge_mean = self._circle_sample_mean(image, cx, cy, radius)
        center_mean = self._circle_sample_mean(image, cx, cy, radius * 0.25)
        if edge_mean is None or center_mean is None:
            return 0.0
        return float((center_mean - edge_mean) / 255.0)

    def _circle_sample_mean(
        self,
        image: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
    ) -> float | None:
        xs, ys = self._circle_offsets(max(1.0, float(radius)))
        x = np.rint(float(cx) + xs).astype(np.int32)
        y = np.rint(float(cy) + ys).astype(np.int32)
        h, w = image.shape[:2]
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        if int(np.count_nonzero(valid)) < max(12, int(len(x) * 0.75)):
            return None
        return float(np.mean(image[y[valid], x[valid]]))

    def _deduplicate(
        self,
        centers: List[np.ndarray],
        radii: List[float],
    ) -> Tuple[List[np.ndarray], List[float]]:
        kept_centers: List[np.ndarray] = []
        kept_radii: List[float] = []

        candidates = sorted(zip(centers, radii), key=lambda item: self._candidate_priority(float(item[1])))
        for center, radius in candidates:
            duplicate = False
            for existing, existing_radius in zip(kept_centers, kept_radii):
                duplicate_distance = self._duplicate_distance(float(radius), float(existing_radius))
                if float(np.linalg.norm(center - existing)) < duplicate_distance:
                    duplicate = True
                    break
            if duplicate:
                continue
            kept_centers.append(center)
            kept_radii.append(float(radius))

        return kept_centers, kept_radii

    def _candidate_priority(self, radius: float) -> Tuple[float, float]:
        preferred = float(self._runtime_preferred_radius)
        return (abs(float(radius) - preferred), -float(radius))

    def _duplicate_distance(self, radius_a: float, radius_b: float) -> float:
        smaller = min(float(radius_a), float(radius_b))
        larger = max(float(radius_a), float(radius_b))
        radius_limit = smaller * float(self._config.deduplicate_distance_ratio)
        radius_limit = max(float(self._config.deduplicate_min_distance), radius_limit)
        if smaller > 0.0 and larger >= smaller * 1.45:
            radius_limit = max(radius_limit, larger * float(self._config.deduplicate_contained_ratio))
        return max(float(self._config.deduplicate_min_distance), radius_limit)

    def _build_bead_helper_mask(self, normalized_gray: np.ndarray) -> np.ndarray:
        threshold_value = float(np.percentile(normalized_gray, 15.0))
        _, helper = cv2.threshold(normalized_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        helper = cv2.morphologyEx(helper, cv2.MORPH_OPEN, kernel)
        return helper

    def _odd(self, value: int) -> int:
        value = max(1, int(value))
        return value if value % 2 == 1 else value + 1
