from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

try:
    from .config import DebugConfig, DetectorConfig
except ImportError:
    from config import DebugConfig, DetectorConfig


@dataclass(frozen=True)
class CircleDetection:
    """One circular droplet candidate in original-image coordinates."""

    center: np.ndarray
    radius: float


@dataclass
class DetectionResult:
    centers: list[np.ndarray]
    radii: list[float]
    debug_image: np.ndarray
    helper_mask: np.ndarray
    diameter_valid: list[bool] = field(default_factory=list)


class DropletDetector:
    """Detect circular droplets from EdgeDrawing closed-edge records."""

    def __init__(self, config: DetectorConfig, debug: DebugConfig) -> None:
        self._config = config
        self._debug = debug
        self._runtime_min_radius = max(1.0, float(config.min_radius))
        self._runtime_max_radius = max(self._runtime_min_radius + 1.0, float(config.max_radius))
        self._has_expected_size = float(config.expected_radius) > 0.0
        self._runtime_preferred_radius = (
            float(config.expected_radius)
            if self._has_expected_size
            else float(np.sqrt(self._runtime_min_radius * self._runtime_max_radius))
        )
        self._configured_preferred_radius = self._runtime_preferred_radius

    def configure_expected_diameter(self, diameter_um: float, pixel_to_micron: float) -> None:
        """Keep the control-facing hook; PID targets must not alter detection gates."""
        _ = (diameter_um, pixel_to_micron)

    def reset_adaptive_size(self) -> None:
        self._runtime_preferred_radius = self._configured_preferred_radius

    def runtime_radius_range(self) -> tuple[float, float, float]:
        return (
            self._runtime_min_radius,
            self._runtime_preferred_radius,
            self._runtime_max_radius,
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
        circles = self._detect_candidates(smoothed, cut_line)
        return DetectionResult(
            centers=[item.center.copy() for item in circles],
            radii=[item.radius for item in circles],
            debug_image=np.empty((0, 0, 3), dtype=np.uint8),
            helper_mask=self._build_bead_helper_mask(normalized),
            diameter_valid=[
                self._circle_fully_visible(normalized.shape[:2], item) for item in circles
            ],
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
        size = self._odd(self._config.gaussian_blur_size)
        if not self._config.enable_gaussian_blur or size <= 1:
            return gray.copy()
        return cv2.GaussianBlur(gray, (size, size), 0)

    def _detect_candidates(
        self,
        gray: np.ndarray,
        cut_line: int,
        trace: dict[str, object] | None = None,
    ) -> list[CircleDetection]:
        work_gray, scale = self._prepare_frame(gray)
        enhanced = self._preprocess(work_gray)
        edge_image, work_circles = self._run_edge_drawing(enhanced)
        support_edges = self._expand_edges(edge_image)
        raw_circles = [self._rescale_circle(item, 1.0 / scale) for item in work_circles]

        minimum_support = max(0.0, min(1.0, float(self._config.edge_min_support_ratio)))
        minimum_visible = max(0.0, min(1.0, float(self._config.edge_min_visible_ratio)))
        candidates: list[CircleDetection] = []
        for original, working in zip(raw_circles, work_circles):
            if float(original.center[1]) > float(cut_line):
                continue
            if not self._runtime_min_radius <= original.radius <= self._runtime_max_radius:
                continue
            if self._circle_visible_ratio(gray.shape[:2], original) < minimum_visible:
                continue
            if self._circle_edge_support(support_edges, working) < minimum_support:
                continue
            if not self._expected_size_valid(original.radius):
                continue
            candidates.append(original)

        if self._has_expected_size:
            candidates.sort(key=lambda item: abs(item.radius - self._runtime_preferred_radius))
        else:
            candidates.sort(key=lambda item: item.radius, reverse=True)
        candidates = self._deduplicate(candidates)
        candidates = candidates[: max(1, int(self._config.edge_max_candidates))]

        if trace is not None:
            trace.update(
                {
                    "work_gray": work_gray,
                    "scale": scale,
                    "enhanced": enhanced,
                    "edges": edge_image,
                    "support_edges": support_edges,
                    "raw_circles": raw_circles,
                    "filtered_circles": candidates,
                }
            )
        return candidates

    def _run_edge_drawing(self, gray: np.ndarray) -> tuple[np.ndarray, list[CircleDetection]]:
        ximgproc = getattr(cv2, "ximgproc", None)
        create = getattr(ximgproc, "createEdgeDrawing", None)
        params_type = getattr(getattr(ximgproc, "EdgeDrawing", None), "Params", None)
        if not callable(create) or params_type is None:
            raise RuntimeError("当前 OpenCV 不包含 EdgeDrawing，请安装 opencv-contrib-python")

        params = params_type()
        params.EdgeDetectionOperator = max(0, min(3, int(self._config.edge_operator)))
        params.GradientThresholdValue = max(1, int(self._config.edge_gradient_threshold))
        params.AnchorThresholdValue = max(0, int(self._config.edge_anchor_threshold))
        params.ScanInterval = max(1, int(self._config.edge_scan_interval))
        params.MinPathLength = max(2, int(self._config.edge_min_path_length))
        params.MinLineLength = int(self._config.edge_min_line_length)
        params.Sigma = max(0.01, float(self._config.edge_sigma))
        params.LineFitErrorThreshold = max(0.01, float(self._config.edge_line_fit_error))
        params.MaxDistanceBetweenTwoLines = max(0.0, float(self._config.edge_max_line_gap))
        params.MaxErrorThreshold = max(0.01, float(self._config.edge_max_error))
        params.NFAValidation = bool(self._config.edge_nfa_validation)
        params.PFmode = bool(self._config.edge_pf_mode)

        detector = create()
        detector.setParams(params)
        detector.detectEdges(gray)
        edge_image = np.asarray(detector.getEdgeImage(), dtype=np.uint8)
        # EdgeDrawing exposes circles and other closed conics through this one API.
        records = detector.detectEllipses()
        return edge_image, self._parse_circle_records(records)

    def _parse_circle_records(self, records: np.ndarray | None) -> list[CircleDetection]:
        output: list[CircleDetection] = []
        minimum_circle_ratio = max(
            0.0,
            min(1.0, float(self._config.edge_min_circle_ratio)),
        )
        for record in np.asarray(records if records is not None else []):
            values = np.asarray(record, dtype=np.float64).reshape(-1)
            if values.size < 3 or not np.isfinite(values).all():
                continue
            cx, cy = float(values[0]), float(values[1])
            if values.size >= 5 and values[3] > 0.0 and values[4] > 0.0:
                first_radius, second_radius = float(values[3]), float(values[4])
                larger = max(first_radius, second_radius)
                smaller = min(first_radius, second_radius)
                if smaller / larger < minimum_circle_ratio:
                    continue
                radius = (first_radius + second_radius) * 0.5
            else:
                radius = float(values[2])
            if radius <= 0.0:
                continue
            output.append(
                CircleDetection(
                    center=np.array([cx, cy], dtype=np.float32),
                    radius=radius,
                )
            )
        return output

    def _prepare_frame(self, gray: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = gray.shape[:2]
        scale = min(
            1.0,
            float(max(1, int(self._config.edge_work_max_width))) / max(1, width),
            float(max(1, int(self._config.edge_work_max_height))) / max(1, height),
        )
        if scale >= 0.999:
            return gray, 1.0
        size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        return cv2.resize(gray, size, interpolation=cv2.INTER_AREA), scale

    def _preprocess(self, gray: np.ndarray) -> np.ndarray:
        output = gray
        if self._config.enable_edge_clahe:
            output = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(output)
        if self._config.enable_edge_median_blur:
            output = cv2.medianBlur(output, 5)
        return output

    @staticmethod
    def _rescale_circle(item: CircleDetection, factor: float) -> CircleDetection:
        return CircleDetection(
            center=np.asarray(item.center * factor, dtype=np.float32),
            radius=float(item.radius * factor),
        )

    def _expected_size_valid(self, radius: float) -> bool:
        if not (
            self._config.enable_expected_size_filter
            and self._config.expected_size_hard_gate
            and self._has_expected_size
        ):
            return True
        tolerance = max(0.0, float(self._config.expected_radius_tolerance_ratio))
        preferred = self._runtime_preferred_radius
        return preferred * max(0.0, 1.0 - tolerance) <= radius <= preferred * (1.0 + tolerance)

    def _expand_edges(self, edges: np.ndarray) -> np.ndarray:
        neighborhood = max(0, int(self._config.edge_support_neighborhood))
        if neighborhood == 0:
            return edges
        size = neighborhood * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        return cv2.dilate(edges, kernel)

    @staticmethod
    def _circle_points(item: CircleDetection) -> tuple[np.ndarray, np.ndarray]:
        samples = max(48, int(round(2.0 * np.pi * item.radius)))
        theta = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False, dtype=np.float32)
        xs = item.center[0] + item.radius * np.cos(theta)
        ys = item.center[1] + item.radius * np.sin(theta)
        return xs, ys

    def _circle_edge_support(self, edges: np.ndarray, item: CircleDetection) -> float:
        xs, ys = self._circle_points(item)
        x = np.rint(xs).astype(np.int32)
        y = np.rint(ys).astype(np.int32)
        height, width = edges.shape[:2]
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        count = int(np.count_nonzero(valid))
        if count == 0:
            return 0.0
        return float(np.count_nonzero(edges[y[valid], x[valid]] > 0)) / count

    def _circle_visible_ratio(self, shape: tuple[int, int], item: CircleDetection) -> float:
        xs, ys = self._circle_points(item)
        height, width = shape
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        return float(np.count_nonzero(valid)) / max(1, len(valid))

    def _circle_fully_visible(self, shape: tuple[int, int], item: CircleDetection) -> bool:
        required = max(0.0, min(1.0, float(self._config.diameter_min_visible_ratio)))
        return self._circle_visible_ratio(shape, item) >= required

    def _deduplicate(self, candidates: list[CircleDetection]) -> list[CircleDetection]:
        kept: list[CircleDetection] = []
        for candidate in candidates:
            duplicate = False
            for existing in kept:
                threshold = max(
                    float(self._config.min_center_distance),
                    float(self._config.deduplicate_min_distance),
                    min(candidate.radius, existing.radius)
                    * float(self._config.deduplicate_distance_ratio),
                    (candidate.radius + existing.radius)
                    * float(self._config.deduplicate_overlap_ratio),
                )
                if float(np.linalg.norm(candidate.center - existing.center)) < threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        return kept

    @staticmethod
    def _build_bead_helper_mask(normalized_gray: np.ndarray) -> np.ndarray:
        threshold_value = float(np.percentile(normalized_gray, 15.0))
        _, helper = cv2.threshold(normalized_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(helper, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def _odd(value: int) -> int:
        value = max(1, int(value))
        return value if value % 2 == 1 else value + 1
