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
    """Detect droplets using illumination correction and one Hough transform."""

    def __init__(self, config: DetectorConfig, debug: DebugConfig) -> None:
        self._config = config
        self._debug = debug
        self._circle_offset_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._runtime_min_radius = max(1.0, float(config.min_radius))
        self._runtime_max_radius = max(self._runtime_min_radius, float(config.max_radius))
        configured_preferred = float(config.expected_radius or config.hough_preferred_radius)
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
        return preferred

    def detect(self, frame: np.ndarray, mode: Optional[str] = None) -> DetectionResult:
        _ = mode
        gray = self._ensure_gray(frame)
        corrected = self._preprocess(gray)
        centers, radii = self._detect_hough_candidates(corrected)
        diameter_valid = [
            self._candidate_diameter_valid(gray.shape[:2], center, radius)
            for center, radius in zip(centers, radii)
        ]
        return DetectionResult(
            centers=centers,
            radii=radii,
            debug_image=np.empty((0, 0, 3), dtype=np.uint8),
            helper_mask=self._build_bead_helper_mask(gray),
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

    def _preprocess(
        self,
        gray: np.ndarray,
        trace: dict[str, object] | None = None,
    ) -> np.ndarray:
        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
        illumination_corrected = cv2.addWeighted(gray, 1.0, background, -1.0, 128)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
            illumination_corrected
        )
        corrected = cv2.GaussianBlur(clahe, (7, 7), 1.4)
        if trace is not None:
            trace.update(
                {
                    "background": background,
                    "illumination_corrected": illumination_corrected,
                    "clahe": clahe,
                    "corrected": corrected,
                }
            )
        return corrected

    def _detect_hough_candidates(
        self,
        corrected: np.ndarray,
        trace: dict[str, object] | None = None,
    ) -> Tuple[List[np.ndarray], List[float]]:
        if trace is not None:
            trace.update({"raw_centers": [], "raw_radii": []})
        if not bool(self._config.enable_hough_candidates):
            return [], []

        minimum = int(round(float(self._config.min_radius)))
        maximum = int(round(float(self._config.max_radius)))
        sensitivity = float(self._config.sensitivity)
        radius_adjustment_percent = float(self._config.radius_adjustment_percent)
        if minimum <= 0 or maximum < minimum:
            raise ValueError("液滴检测半径范围无效")
        if not 0.0 <= sensitivity <= 1.0:
            raise ValueError("液滴检测敏感度必须在 0 到 1 之间")
        if not -20.0 <= radius_adjustment_percent <= 20.0:
            raise ValueError("液滴整体尺寸调节必须在 -20% 到 20% 之间")

        circles = cv2.HoughCircles(
            corrected,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(1.0, float(self._config.min_center_distance)),
            param1=75,
            param2=45.0 - 25.0 * sensitivity,
            minRadius=minimum,
            maxRadius=maximum,
        )
        if circles is None:
            return [], []

        result = np.rint(circles[0]).astype(np.int32)
        result = result[np.lexsort((result[:, 0], result[:, 1]))]
        centers = [np.asarray((x, y), dtype=np.float32) for x, y, _radius in result]
        raw_radii = [float(radius) for _x, _y, radius in result]
        if trace is not None:
            trace["raw_centers"] = list(centers)
            trace["raw_radii"] = list(raw_radii)
        scale = 1.0 + radius_adjustment_percent / 100.0
        radii = [radius * scale for radius in raw_radii]
        return centers, radii

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

    # Retained for camera auto-calibration reports; these do not filter circles.
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

    def _build_bead_helper_mask(self, gray: np.ndarray) -> np.ndarray:
        threshold_value = float(np.percentile(gray, 15.0))
        _, helper = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(helper, cv2.MORPH_OPEN, kernel)
