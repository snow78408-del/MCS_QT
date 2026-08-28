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


class DropletDetector:
    """Detect circular droplets with one full-frame Hough transform.

    The detector intentionally has no contour, connected-component, background
    subtraction, Watershed, candidate fusion, or intensity-scoring branch.
    Hough results only pass inexpensive geometric filters: configured radius,
    cut line, visible perimeter, edge support, optional calibrated-size gate,
    and center-distance de-duplication.
    """

    def __init__(self, config: DetectorConfig, debug: DebugConfig) -> None:
        self._config = config
        self._debug = debug
        self._circle_offset_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
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
        """Keep the legacy hook without leaking the PID target into detection."""
        _ = (diameter_um, pixel_to_micron)

    def reset_adaptive_size(self) -> None:
        self._runtime_preferred_radius = float(self._configured_preferred_radius)

    def runtime_radius_range(self) -> tuple[float, float, float]:
        return (
            float(self._runtime_min_radius),
            float(self._runtime_preferred_radius),
            float(self._runtime_max_radius),
        )

    def calibrate_preferred_radius(self, radii: list[float]) -> float:
        values = [
            float(value)
            for value in radii
            if self._runtime_min_radius <= float(value) <= self._runtime_max_radius
        ]
        if not values:
            raise ValueError("没有可用于标定的有效液滴半径样本")
        preferred = float(np.median(np.asarray(values, dtype=np.float32)))
        self._runtime_preferred_radius = preferred
        self._configured_preferred_radius = preferred
        self._has_expected_size = True
        return preferred

    def detect(self, gray_frame: np.ndarray, mode: Optional[str] = None) -> DetectionResult:
        _ = mode
        gray = self._ensure_gray(gray_frame)
        normalized = self._normalize(gray)
        smoothed = self._smooth(normalized)
        cut_line = int(smoothed.shape[0] * float(self._config.cut_line_ratio))
        centers, radii = self._detect_hough_candidates(smoothed, cut_line)
        diameter_valid = [
            self._candidate_diameter_valid(normalized.shape[:2], center, radius)
            for center, radius in zip(centers, radii)
        ]
        return DetectionResult(
            centers=centers,
            radii=radii,
            debug_image=np.empty((0, 0, 3), dtype=np.uint8),
            helper_mask=self._build_bead_helper_mask(normalized),
            diameter_valid=diameter_valid,
        )

    def _ensure_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("液滴检测输入图像为空")
        if frame.ndim == 2:
            return np.asarray(frame, dtype=np.uint8)
        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        raise ValueError(f"不支持的液滴检测图像形状：{frame.shape}")

    def _normalize(self, gray: np.ndarray) -> np.ndarray:
        if not self._config.enable_intensity_normalization:
            return gray.copy()
        return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    def _smooth(self, gray: np.ndarray) -> np.ndarray:
        blur_size = self._odd(self._config.gaussian_blur_size)
        if not self._config.enable_gaussian_blur or blur_size <= 1:
            return gray.copy()
        return cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    def _detect_hough_candidates(
        self,
        gray: np.ndarray,
        cut_line: int,
        trace: dict[str, object] | None = None,
    ) -> Tuple[List[np.ndarray], List[float]]:
        work_gray, scale = self._prepare_hough_frame(gray)
        enhanced = self._hough_preprocess(work_gray)
        edges = cv2.Canny(
            enhanced,
            max(1.0, float(self._config.hough_param1) * 0.5),
            max(2.0, float(self._config.hough_param1)),
        )
        support_edges = self._build_support_edges(edges)
        if trace is not None:
            trace.update(
                {
                    "work_gray": work_gray,
                    "scale": scale,
                    "enhanced": enhanced,
                    "edges": edges,
                    "support_edges": support_edges,
                    "raw_centers": [],
                    "raw_radii": [],
                    "filtered_centers": [],
                    "filtered_radii": [],
                }
            )

        if not bool(self._config.enable_hough_candidates):
            return [], []

        circles = cv2.HoughCircles(
            enhanced,
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

        raw = np.asarray(circles[0], dtype=np.float32)
        raw_candidates = [
            (float(cx) / scale, float(cy) / scale, float(radius) / scale)
            for cx, cy, radius in raw
        ]
        if trace is not None:
            trace["raw_centers"] = [
                np.array([cx, cy], dtype=np.float32) for cx, cy, _radius in raw_candidates
            ]
            trace["raw_radii"] = [radius for _cx, _cy, radius in raw_candidates]

        minimum_visible = min(
            1.0,
            max(0.0, float(self._config.candidate_min_visible_circle_ratio)),
        )
        minimum_edge_support = max(0.0, float(self._config.hough_edge_support_threshold))
        scaled_cut_line = float(cut_line) * scale
        candidates: list[tuple[float, float, float]] = []
        for cx, cy, radius in raw:
            original = (float(cx) / scale, float(cy) / scale, float(radius) / scale)
            original_cx, original_cy, original_radius = original
            if float(cy) > scaled_cut_line:
                continue
            if not self._hough_min_radius() <= original_radius <= self._hough_max_radius():
                continue
            if self._circle_visible_ratio(gray.shape[:2], original_cx, original_cy, original_radius) < minimum_visible:
                continue
            if self._circle_edge_support(support_edges, float(cx), float(cy), float(radius)) < minimum_edge_support:
                continue
            candidates.append(original)

        candidates = self._filter_expected_size(candidates)
        if self._has_expected_size:
            candidates.sort(key=lambda item: self._candidate_priority(item[2]))
        centers = [np.array([cx, cy], dtype=np.float32) for cx, cy, _radius in candidates]
        radii = [float(radius) for _cx, _cy, radius in candidates]
        centers, radii = self._deduplicate(centers, radii)
        maximum = max(1, int(self._config.hough_max_candidates))
        centers, radii = centers[:maximum], radii[:maximum]
        if trace is not None:
            trace["filtered_centers"] = list(centers)
            trace["filtered_radii"] = list(radii)
        return centers, radii

    def _prepare_hough_frame(self, gray: np.ndarray) -> Tuple[np.ndarray, float]:
        height, width = gray.shape[:2]
        max_width = max(1, int(self._config.hough_work_max_width))
        max_height = max(1, int(self._config.hough_work_max_height))
        scale = min(
            1.0,
            float(max_width) / float(max(1, width)),
            float(max_height) / float(max(1, height)),
        )
        if scale >= 0.999:
            return gray, 1.0
        target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        return cv2.resize(gray, target, interpolation=cv2.INTER_AREA), float(scale)

    def _hough_preprocess(self, gray: np.ndarray) -> np.ndarray:
        output = gray
        if self._config.enable_hough_clahe:
            output = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(output)
        if self._config.enable_hough_median_blur:
            output = cv2.medianBlur(output, 5)
        return output

    def _hough_min_radius(self) -> float:
        return max(1.0, float(self._config.hough_min_radius), self._runtime_min_radius)

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

    def _filter_expected_size(
        self,
        candidates: list[tuple[float, float, float]],
    ) -> list[tuple[float, float, float]]:
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

    def _build_support_edges(self, edges: np.ndarray) -> np.ndarray:
        neighborhood = max(0, int(self._config.hough_edge_neighborhood))
        if neighborhood <= 0:
            return edges
        size = neighborhood * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        return cv2.dilate(edges, kernel)

    def _circle_edge_support(self, edges: np.ndarray, cx: float, cy: float, radius: float) -> float:
        height, width = edges.shape[:2]
        xs, ys = self._circle_offsets(radius)
        x = np.rint(cx + xs).astype(np.int32)
        y = np.rint(cy + ys).astype(np.int32)
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        count = int(np.count_nonzero(valid))
        if count == 0:
            return 0.0
        return float(np.count_nonzero(edges[y[valid], x[valid]] > 0)) / float(count)

    def _circle_visible_ratio(
        self,
        shape: tuple[int, int],
        cx: float,
        cy: float,
        radius: float,
    ) -> float:
        height, width = shape
        xs, ys = self._circle_offsets(radius)
        x = np.rint(cx + xs).astype(np.int32)
        y = np.rint(cy + ys).astype(np.int32)
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        return float(np.count_nonzero(valid)) / float(max(1, len(valid)))

    def _circle_offsets(self, radius: float) -> tuple[np.ndarray, np.ndarray]:
        rounded_radius = max(1, int(round(radius)))
        samples = max(48, int(rounded_radius * 4.0))
        key = (rounded_radius, samples)
        cached = self._circle_offset_cache.get(key)
        if cached is not None:
            return cached
        theta = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False, dtype=np.float32)
        offsets = (np.cos(theta) * rounded_radius, np.sin(theta) * rounded_radius)
        self._circle_offset_cache[key] = offsets
        return offsets

    def _deduplicate(
        self,
        centers: List[np.ndarray],
        radii: List[float],
    ) -> Tuple[List[np.ndarray], List[float]]:
        kept_centers: List[np.ndarray] = []
        kept_radii: List[float] = []
        for center, radius in zip(centers, radii):
            if any(
                float(np.linalg.norm(center - existing))
                < self._duplicate_distance(float(radius), existing_radius)
                for existing, existing_radius in zip(kept_centers, kept_radii)
            ):
                continue
            kept_centers.append(center)
            kept_radii.append(float(radius))
        return kept_centers, kept_radii

    def _candidate_priority(self, radius: float) -> Tuple[float, float]:
        return (abs(float(radius) - self._runtime_preferred_radius), -float(radius))

    def _duplicate_distance(self, radius_a: float, radius_b: float) -> float:
        smaller = min(float(radius_a), float(radius_b))
        ratio_distance = smaller * float(self._config.deduplicate_distance_ratio)
        overlap_distance = (float(radius_a) + float(radius_b)) * float(
            self._config.candidate_nms_overlap_ratio
        )
        return max(
            float(self._config.deduplicate_min_distance),
            ratio_distance,
            overlap_distance,
        )

    def _candidate_diameter_valid(
        self,
        shape: tuple[int, int],
        center: np.ndarray,
        radius: float,
    ) -> bool:
        height, width = shape
        margin = float(radius) * max(0.0, float(self._config.candidate_full_circle_ratio))
        cx, cy = float(center[0]), float(center[1])
        return cx >= margin and cy >= margin and cx < width - margin and cy < height - margin

    # These two measurements are retained for camera auto-calibration reports;
    # they do not accept or reject Hough detections.
    def _ring_contrast(self, image: np.ndarray, cx: float, cy: float, radius: float) -> float:
        edge = self._circle_sample_mean(image, cx, cy, radius)
        inner = self._circle_sample_mean(image, cx, cy, radius * 0.70)
        outer = self._circle_sample_mean(image, cx, cy, radius * 1.22)
        if edge is None or inner is None or outer is None:
            return 0.0
        return float(((inner + outer) * 0.5 - edge) / 255.0)

    def _center_contrast(self, image: np.ndarray, cx: float, cy: float, radius: float) -> float:
        edge = self._circle_sample_mean(image, cx, cy, radius)
        center = self._circle_sample_mean(image, cx, cy, radius * 0.25)
        if edge is None or center is None:
            return 0.0
        return float((center - edge) / 255.0)

    def _circle_sample_mean(
        self,
        image: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
    ) -> float | None:
        xs, ys = self._circle_offsets(max(1.0, radius))
        x = np.rint(cx + xs).astype(np.int32)
        y = np.rint(cy + ys).astype(np.int32)
        height, width = image.shape[:2]
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if int(np.count_nonzero(valid)) < max(12, int(len(x) * 0.75)):
            return None
        return float(np.mean(image[y[valid], x[valid]]))

    def _build_bead_helper_mask(self, normalized_gray: np.ndarray) -> np.ndarray:
        threshold_value = float(np.percentile(normalized_gray, 15.0))
        _, helper = cv2.threshold(normalized_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(helper, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def _odd(value: int) -> int:
        value = max(1, int(value))
        return value if value % 2 == 1 else value + 1
