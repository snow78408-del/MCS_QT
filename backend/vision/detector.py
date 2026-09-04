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
    plug_lengths_px: List[float] = field(default_factory=list)
    equivalent_diameters_px: List[float] = field(default_factory=list)


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
        self._pixel_to_micron = 1.0

    def configure_expected_diameter(self, diameter_um: float, pixel_to_micron: float) -> None:
        """Configure physical scale without leaking the PID target into detection."""
        _ = diameter_um
        scale = float(pixel_to_micron)
        if np.isfinite(scale) and scale > 0.0:
            self._pixel_to_micron = scale

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
        gray = self._ensure_gray(frame)
        selected_mode = str(mode or self._config.measurement_mode).strip().lower()
        if selected_mode == "generation_plug":
            return self._detect_generation_plugs(gray)
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

    def _detect_generation_plugs(
        self,
        gray: np.ndarray,
        trace: dict[str, object] | None = None,
    ) -> DetectionResult:
        """Measure detached C-regime plugs from paired menisci.

        The rectified channel is reduced to a robust one-dimensional centre
        profile.  Adjacent, strong meniscus peaks are paired only when their
        separation is compatible with a detached plug.  Each accepted length
        is converted to volume using the square-channel formula from
        van Steijn et al. (Scientific Reports, 2017), then reported as an
        equivalent-sphere diameter so the existing controller keeps one clear
        physical setpoint.
        """
        height, width = gray.shape[:2]
        flow_axis = "x" if width >= height else "y"
        working = gray if flow_axis == "x" else gray.T
        cross_size, axial_size = working.shape
        band_ratio = min(0.90, max(0.20, float(self._config.generation_center_band_ratio)))
        band_size = max(3, int(round(cross_size * band_ratio)))
        band_start = max(0, (cross_size - band_size) // 2)
        band = working[band_start : band_start + band_size]
        corrected = self._preprocess(working)
        corrected_band = corrected[band_start : band_start + band_size]
        profile = np.median(corrected_band.astype(np.float32), axis=0)
        profile = cv2.GaussianBlur(profile.reshape(1, -1), (0, 0), 1.2).reshape(-1)
        gradient = np.abs(np.gradient(profile))
        median_energy = float(np.median(gradient))
        mad = float(np.median(np.abs(gradient - median_energy)))
        robust_sigma = max(0.25, 1.4826 * mad)
        threshold = median_energy + float(self._config.generation_edge_mad_multiplier) * robust_sigma
        corrected_float = corrected.astype(np.float32)
        edge_2d = np.abs(np.gradient(corrected_float, axis=1))

        scale = max(1e-9, float(self._pixel_to_micron))
        channel_h_px = float(self._config.generation_channel_height_um) / scale
        channel_w_px = float(self._config.generation_channel_width_um) / scale
        reference_width_px = max(2.0, min(channel_h_px, channel_w_px))
        min_peak_gap = max(
            2,
            int(round(reference_width_px * float(self._config.generation_min_edge_separation_ratio))),
        )
        peak_indices = self._local_profile_peaks(gradient, threshold, min_peak_gap)
        min_length = reference_width_px * float(self._config.generation_min_length_ratio)
        max_length = reference_width_px * float(self._config.generation_max_length_ratio)
        profile_sigma = max(1.0, float(np.std(profile)))
        transverse_gradient = np.abs(np.gradient(corrected_float, axis=0))
        transverse_margin = max(
            2,
            min(
                max(1, cross_size // 4),
                max(
                    int(round(reference_width_px * 0.08)),
                    int(round(cross_size * 0.12)),
                ),
            ),
        )
        transverse_inner = transverse_gradient[
            transverse_margin : max(transverse_margin + 1, cross_size - transverse_margin)
        ]
        transverse_threshold = max(
            1.0,
            float(np.percentile(transverse_inner, 80.0)),
        )
        minimum_outline_ratio = min(
            1.0,
            max(0.0, float(self._config.generation_min_capsule_outline_ratio)),
        )

        candidates: list[tuple[float, int, int, float, float]] = []
        for left, right in zip(peak_indices, peak_indices[1:]):
            length_px = float(right - left)
            if length_px < min_length or length_px > max_length:
                continue
            pad = max(2, int(round(reference_width_px * 0.20)))
            inside = profile[left + 1 : right]
            outside_parts = []
            if left - pad >= 0:
                outside_parts.append(profile[left - pad : left])
            if right + pad <= axial_size:
                outside_parts.append(profile[right : right + pad])
            if inside.size < 3 or not outside_parts:
                continue
            outside = np.concatenate(outside_parts)
            signed_contrast = float(np.median(inside)) - float(np.median(outside))
            polarity = str(self._config.generation_polarity).strip().lower()
            contrast = abs(signed_contrast)
            threshold_contrast = (
                float(self._config.generation_min_profile_contrast_sigma) * profile_sigma
            )
            polarity_valid = (
                signed_contrast >= threshold_contrast
                if polarity == "brighter"
                else signed_contrast <= -threshold_contrast
                if polarity == "darker"
                else contrast >= threshold_contrast
            )
            outline_support = self._capsule_outline_support(
                transverse_gradient,
                left=left,
                right=right,
                row_margin=transverse_margin,
                edge_threshold=transverse_threshold,
                reference_width_px=reference_width_px,
            )
            if outline_support < minimum_outline_ratio:
                continue
            # Phase-contrast halos can reverse or flatten the median intensity
            # inside a complete capsule, so polarity is a score preference and
            # never a substitute for the required two-dimensional outline.
            appearance_score = contrast if polarity_valid else 0.0
            support_values: list[float] = []
            for edge_index in (left, right):
                edge_start = max(0, edge_index - 2)
                edge_stop = min(axial_size, edge_index + 3)
                local_edge = np.max(edge_2d[:, edge_start:edge_stop], axis=1)
                support_values.append(
                    float(np.count_nonzero(local_edge >= threshold * 0.50))
                    / float(max(1, cross_size))
                )
            if min(support_values) < float(
                self._config.generation_min_meniscus_support_ratio
            ):
                continue
            edge_score = float(gradient[left] + gradient[right])
            candidates.append(
                (
                    edge_score
                    + appearance_score
                    + outline_support * transverse_threshold,
                    left,
                    right,
                    length_px,
                    outline_support,
                )
            )

        # Intervals share a meniscus when the bright/dark phase assignment is
        # ambiguous.  Keep the stronger non-overlapping interval.
        selected: list[tuple[int, int, float]] = []
        selected_outline_support: list[float] = []
        occupied: set[int] = set()
        for _score, left, right, length_px, outline_support in sorted(candidates, reverse=True):
            if left in occupied or right in occupied:
                continue
            occupied.update((left, right))
            selected.append((left, right, length_px))
            selected_outline_support.append(outline_support)
        selected_with_support = sorted(
            zip(selected, selected_outline_support),
            key=lambda item: item[0][0],
        )
        selected = [item for item, _support in selected_with_support]
        selected_outline_support = [support for _item, support in selected_with_support]

        if trace is not None:
            trace.update(
                {
                    "flow_axis": flow_axis,
                    "working": working,
                    "corrected": corrected,
                    "band_start": band_start,
                    "band_size": band_size,
                    "profile": profile,
                    "gradient": gradient,
                    "gradient_threshold": threshold,
                    "peak_indices": peak_indices,
                    "selected_intervals": list(selected),
                    "reference_width_px": reference_width_px,
                    "minimum_length_px": min_length,
                    "maximum_length_px": max_length,
                    "transverse_gradient": transverse_gradient,
                    "transverse_gradient_threshold": transverse_threshold,
                    "selected_outline_support": selected_outline_support,
                }
            )

        centers: list[np.ndarray] = []
        radii: list[float] = []
        lengths: list[float] = []
        diameters: list[float] = []
        valid: list[bool] = []
        for left, right, length_px in selected:
            equivalent_px = self._plug_equivalent_diameter_px(
                length_px,
                channel_h_px,
                channel_w_px,
            )
            if equivalent_px is None:
                continue
            axial_center = (float(left) + float(right)) * 0.5
            center = (
                np.asarray((axial_center, cross_size * 0.5), dtype=np.float32)
                if flow_axis == "x"
                else np.asarray((cross_size * 0.5, axial_center), dtype=np.float32)
            )
            full = left > 0 and right < axial_size - 1
            centers.append(center)
            radii.append(equivalent_px * 0.5)
            lengths.append(length_px)
            diameters.append(equivalent_px)
            valid.append(full)

        helper = self._build_bead_helper_mask(gray)
        return DetectionResult(
            centers=centers,
            radii=radii,
            debug_image=np.empty((0, 0, 3), dtype=np.uint8),
            helper_mask=helper,
            diameter_valid=valid,
            plug_lengths_px=lengths,
            equivalent_diameters_px=diameters,
        )

    @staticmethod
    def _capsule_outline_support(
        transverse_gradient: np.ndarray,
        *,
        left: int,
        right: int,
        row_margin: int,
        edge_threshold: float,
        reference_width_px: float,
    ) -> float:
        """Return axial coverage having separated upper/lower capsule edges."""
        height, width = transverse_gradient.shape[:2]
        trim = max(1, int(round(reference_width_px * 0.08)))
        start = max(0, int(left) + trim)
        stop = min(width, int(right) - trim)
        row_start = max(0, int(row_margin))
        row_stop = min(height, height - int(row_margin))
        if stop <= start or row_stop - row_start < 3:
            return 0.0

        minimum_separation = max(2, int(round(reference_width_px * 0.18)))
        maximum_separation = max(
            minimum_separation,
            int(round(reference_width_px * 1.25)),
        )
        supported = 0
        for column in range(start, stop):
            edge_rows = np.flatnonzero(
                transverse_gradient[row_start:row_stop, column] >= float(edge_threshold)
            )
            if edge_rows.size < 2:
                continue
            separation = int(edge_rows[-1] - edge_rows[0])
            if minimum_separation <= separation <= maximum_separation:
                supported += 1
        return float(supported) / float(max(1, stop - start))

    @staticmethod
    def _local_profile_peaks(values: np.ndarray, threshold: float, minimum_gap: int) -> list[int]:
        raw = [
            index
            for index in range(1, max(1, len(values) - 1))
            if float(values[index]) >= float(threshold)
            and float(values[index]) >= float(values[index - 1])
            and float(values[index]) >= float(values[index + 1])
        ]
        selected: list[int] = []
        for index in sorted(raw, key=lambda item: float(values[item]), reverse=True):
            if all(abs(index - previous) >= int(minimum_gap) for previous in selected):
                selected.append(index)
        return sorted(selected)

    def _plug_equivalent_diameter_px(
        self,
        length_px: float,
        height_px: float,
        width_px: float,
    ) -> float | None:
        if min(length_px, height_px, width_px) <= 0.0:
            return None
        inverse_sum = (2.0 / height_px) + (2.0 / width_px)
        effective_area = height_px * width_px - (4.0 - np.pi) * inverse_sum ** -2
        volume_px3 = (
            float(self._config.generation_volume_correction)
            * effective_area
            * (length_px - width_px / 3.0)
        )
        if not np.isfinite(volume_px3) or volume_px3 <= 0.0:
            return None
        return float(np.cbrt(6.0 * volume_px3 / np.pi))

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
