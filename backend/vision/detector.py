from __future__ import annotations

from dataclasses import dataclass
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
        self._runtime_min_radius = max(1.0, float(config.min_radius))
        self._runtime_max_radius = max(self._runtime_min_radius + 1.0, float(config.max_radius))
        configured_preferred = float(config.expected_radius or config.hough_preferred_radius)
        self._runtime_preferred_radius = (
            configured_preferred
            if configured_preferred > 0.0
            else float(np.sqrt(self._runtime_min_radius * self._runtime_max_radius))
        )
        self._configured_preferred_radius = float(self._runtime_preferred_radius)

    def configure_expected_diameter(self, diameter_um: float, pixel_to_micron: float) -> None:
        """Adapt pixel-domain detector limits to the configured physical size."""
        diameter = float(diameter_um)
        scale = float(pixel_to_micron)
        if diameter <= 0.0 or scale <= 0.0:
            return

        expected = diameter / scale / 2.0
        absolute_min = max(1.0, float(self._config.min_radius))
        absolute_max = max(absolute_min + 1.0, float(self._config.max_radius))
        if bool(self._config.expected_size_hard_gate):
            runtime_min = max(absolute_min, expected * float(self._config.adaptive_min_radius_ratio))
            runtime_max = min(absolute_max, expected * float(self._config.adaptive_max_radius_ratio))
            if runtime_max <= runtime_min:
                runtime_min = max(absolute_min, min(expected * 0.5, absolute_max - 1.0))
                runtime_max = min(absolute_max, max(expected * 1.6, runtime_min + 1.0))
        else:
            runtime_min = absolute_min
            runtime_max = absolute_max

        self._runtime_min_radius = runtime_min
        self._runtime_max_radius = runtime_max
        self._runtime_preferred_radius = min(runtime_max, max(runtime_min, expected))
        self._configured_preferred_radius = float(self._runtime_preferred_radius)

    def reset_adaptive_size(self) -> None:
        self._runtime_preferred_radius = float(self._configured_preferred_radius)

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
        return preferred

    def detect(self, gray_frame: np.ndarray, mode: Optional[str] = None) -> DetectionResult:
        gray = self._ensure_gray(gray_frame)
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        blur_size = self._odd(self._config.gaussian_blur_size)
        smoothed = cv2.GaussianBlur(normalized, (blur_size, blur_size), 0) if blur_size > 1 else normalized

        _, binary = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = self._morphology_clean(binary)

        detect_mode = mode or self._config.detection_mode
        cut_line = int(binary.shape[0] * self._config.cut_line_ratio)

        if detect_mode == "no_split":
            centers, radii = self._detect_no_split(binary, cut_line)
        else:
            centers, radii = self._detect_split_connected(binary, cut_line)

        if self._config.hough_fallback_only:
            centers, radii = self._score_and_suppress_candidates(normalized, centers, radii)
            if not centers and self._config.enable_hough_candidates:
                centers, radii = self._detect_hough_candidates(normalized, cut_line)
                centers, radii = self._score_and_suppress_candidates(normalized, centers, radii)
            if not centers and (
                self._config.enable_intensity_peak_candidates
                or self._config.enable_intensity_peak_fallback
            ):
                centers, radii = self._detect_intensity_peak_candidates(normalized, cut_line)
                centers, radii = self._score_and_suppress_candidates(normalized, centers, radii)
        else:
            if self._config.enable_hough_candidates:
                hough_centers, hough_radii = self._detect_hough_candidates(normalized, cut_line)
                centers.extend(hough_centers)
                radii.extend(hough_radii)
            if self._config.enable_intensity_peak_candidates or (
                self._config.enable_intensity_peak_fallback and not centers
            ):
                peak_centers, peak_radii = self._detect_intensity_peak_candidates(normalized, cut_line)
                centers.extend(peak_centers)
                radii.extend(peak_radii)
            centers, radii = self._score_and_suppress_candidates(normalized, centers, radii)
        helper_mask = self._build_bead_helper_mask(normalized)
        debug_image = np.empty((0, 0, 3), dtype=np.uint8)

        return DetectionResult(centers=centers, radii=radii, debug_image=debug_image, helper_mask=helper_mask)

    def _ensure_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _morphology_clean(self, binary: np.ndarray) -> np.ndarray:
        open_k = self._odd(self._config.morphology_open_kernel)
        close_k = self._odd(self._config.morphology_close_kernel)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
        return cleaned

    def _detect_no_split(self, binary: np.ndarray, cut_line: int) -> Tuple[List[np.ndarray], List[float]]:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers: List[np.ndarray] = []
        radii: List[float] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self._config.min_contour_area:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self._config.circularity_threshold:
                continue

            radius = float(np.sqrt(area / np.pi))
            if radius < self._runtime_min_radius or radius > self._runtime_max_radius:
                continue

            moments = cv2.moments(contour)
            if abs(moments["m00"]) <= 1e-6:
                continue

            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
            if cy > cut_line:
                continue

            centers.append(np.array([cx, cy], dtype=np.float32))
            radii.append(radius)

        return centers, radii

    def _detect_intensity_peak_candidates(
        self,
        normalized_gray: np.ndarray,
        cut_line: int,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Seed partially occluded droplets from bright interior maxima."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized_gray)
        blurred = cv2.GaussianBlur(enhanced, (9, 9), 0)
        peak_kernel_size = self._odd(
            max(
                5,
                int(
                    round(
                        self._runtime_preferred_radius
                        * float(self._config.intensity_peak_kernel_radius_ratio)
                    )
                ),
            )
        )
        peak_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (peak_kernel_size, peak_kernel_size),
        )
        dilated = cv2.dilate(blurred, peak_kernel)
        percentile = min(99.0, max(1.0, float(self._config.intensity_peak_percentile)))
        brightness_floor = float(np.percentile(blurred, percentile))
        maxima = (
            (blurred >= (dilated - 1.0))
            & (blurred >= brightness_floor)
        ).astype(np.uint8)
        count, _, _, centroids = cv2.connectedComponentsWithStats(
            maxima,
            8,
            cv2.CV_32S,
        )

        edges = cv2.Canny(cv2.medianBlur(enhanced, 5), 50, 150)
        support_edges = self._build_support_edges(edges)
        min_radius = max(
            self._runtime_min_radius,
            self._runtime_preferred_radius
            * float(self._config.intensity_peak_min_radius_ratio),
        )
        max_radius = min(
            self._runtime_max_radius,
            self._runtime_preferred_radius
            * float(self._config.intensity_peak_max_radius_ratio),
        )
        if max_radius < min_radius:
            return [], []

        margin = max(2.0, min_radius)
        height, width = normalized_gray.shape[:2]
        seeds: list[tuple[float, np.ndarray, float]] = []
        for label in range(1, count):
            cx, cy = (float(value) for value in centroids[label])
            if cy > float(cut_line):
                continue
            if cx < margin or cy < margin or cx >= width - margin or cy >= height - margin:
                continue

            best: tuple[float, float] | None = None
            for radius in np.arange(min_radius, max_radius + 0.5, 1.0):
                edge_support = self._circle_edge_support(
                    support_edges,
                    cx,
                    cy,
                    float(radius),
                )
                if edge_support < float(self._config.intensity_peak_min_edge_support):
                    continue
                center_contrast = self._center_contrast(
                    enhanced,
                    cx,
                    cy,
                    float(radius),
                )
                if center_contrast < float(self._config.candidate_min_center_contrast):
                    continue
                ring_contrast = self._ring_contrast(
                    enhanced,
                    cx,
                    cy,
                    float(radius),
                )
                radius_score = max(
                    0.0,
                    1.0
                    - abs(float(radius) - self._runtime_preferred_radius)
                    / max(1.0, max_radius - min_radius),
                )
                score = (
                    0.50 * edge_support
                    + 0.20 * min(1.0, center_contrast / 0.25)
                    + 0.15 * min(1.0, max(0.0, ring_contrast) / 0.10)
                    + 0.15 * radius_score
                )
                if best is None or score > best[0]:
                    best = (float(score), float(radius))
            if best is not None:
                seeds.append(
                    (
                        best[0],
                        np.array([cx, cy], dtype=np.float32),
                        best[1],
                    )
                )

        seeds.sort(key=lambda item: item[0], reverse=True)
        seeds = seeds[: max(1, int(self._config.intensity_peak_max_candidates))]
        return [item[1] for item in seeds], [item[2] for item in seeds]

    def _detect_split_connected(self, binary: np.ndarray, cut_line: int) -> Tuple[List[np.ndarray], List[float]]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8, cv2.CV_32S)

        single_max_area = np.pi * (self._runtime_max_radius ** 2) * self._config.split_large_area_ratio
        centers: List[np.ndarray] = []
        radii: List[float] = []

        for label in range(1, num_labels):
            area = float(stats[label, cv2.CC_STAT_AREA])
            component_mask = (labels == label).astype(np.uint8) * 255

            if area <= single_max_area:
                c_centers, c_radii = self._detect_no_split(component_mask, cut_line)
            else:
                c_centers, c_radii = self._split_component(component_mask, cut_line)

            centers.extend(c_centers)
            radii.extend(c_radii)

        return centers, radii

    def _split_component(self, component_mask: np.ndarray, cut_line: int) -> Tuple[List[np.ndarray], List[float]]:
        dist = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
        kernel_size = self._odd(max(3, int(self._runtime_preferred_radius * 0.9)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(dist, kernel)

        peak_threshold = max(
            1.0,
            self._runtime_min_radius * self._config.split_peak_threshold_ratio,
        )
        local_peaks = ((dist >= (dilated - 1e-6)) & (dist > peak_threshold)).astype(np.uint8)

        num_labels, _, _, centroids = cv2.connectedComponentsWithStats(local_peaks, 8, cv2.CV_32S)
        centers: List[np.ndarray] = []
        radii: List[float] = []

        for label in range(1, num_labels):
            cx, cy = centroids[label]
            x = int(round(cx))
            y = int(round(cy))
            if y < 0 or y >= dist.shape[0] or x < 0 or x >= dist.shape[1]:
                continue

            radius = float(dist[y, x])
            split_min = self._runtime_min_radius * float(self._config.split_min_radius_ratio)
            if radius < split_min or radius > self._runtime_max_radius:
                continue
            if y > cut_line:
                continue

            centers.append(np.array([cx, cy], dtype=np.float32))
            radii.append(radius)

        if centers:
            return centers, radii
        return self._detect_no_split(component_mask, cut_line)

    def _detect_hough_candidates(self, normalized_gray: np.ndarray, cut_line: int) -> Tuple[List[np.ndarray], List[float]]:
        hough_gray, scale = self._prepare_hough_frame(normalized_gray)
        scaled_cut_line = int(float(cut_line) * scale)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(hough_gray)
        blurred = cv2.medianBlur(enhanced, 5)
        edges = cv2.Canny(blurred, 50, 150)
        support_edges = self._build_support_edges(edges)
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

        candidates: List[Tuple[float, float, float]] = []
        for cx, cy, radius in np.round(circles[0]).astype(np.float32):
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
        full_circle_ratio = max(0.0, float(self._config.candidate_full_circle_ratio))
        height, width = normalized_gray.shape[:2]
        scored: List[_ScoredCandidate] = []

        for center, radius_value in zip(centers, radii):
            radius = float(radius_value)
            if radius < self._runtime_min_radius or radius > self._runtime_max_radius:
                continue
            cx = float(center[0])
            cy = float(center[1])
            margin = radius * full_circle_ratio
            if cx < margin or cy < margin or cx >= float(width) - margin or cy >= float(height) - margin:
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
