from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from .config import DebugConfig, DetectorConfig
    from .capsule_profile import (bounded_shoulder_intervals, capsule_intervals,
                                  dark_body_intervals, raw_outline_contrast,
                                  transverse_body_intervals,
                                  shoulder_phase_intervals)
    from .plug_geometry import equivalent_sphere_diameter_px
except ImportError:
    from config import DebugConfig, DetectorConfig
    from capsule_profile import (bounded_shoulder_intervals, capsule_intervals,
                                 dark_body_intervals, raw_outline_contrast,
                                 transverse_body_intervals,
                                 shoulder_phase_intervals)
    from plug_geometry import equivalent_sphere_diameter_px


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
        if not np.isfinite(config.generation_min_raw_outline_contrast) or not 0 < config.generation_min_raw_outline_contrast <= 255:
            raise ValueError("generation_min_raw_outline_contrast must be in (0, 255]")
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
        # 由 declare_pixel_cross_section 记录：本帧像素截面是否被显式声明，以及来源。
        self._pixel_cross_section_declaration: dict | None = None

    def configure_expected_diameter(self, diameter_um: float, pixel_to_micron: float) -> None:
        """Configure physical scale without leaking the PID target into detection."""
        _ = diameter_um
        scale = float(pixel_to_micron)
        if np.isfinite(scale) and scale > 0.0:
            self._pixel_to_micron = scale

    @property
    def config(self) -> DetectorConfig:
        """当前生效的检测配置对象。追溯需要读它来记录**实际**参数，而不是抄一份常量。"""
        return self._config

    def declare_pixel_cross_section(self, cross_px: float, *, reason: str) -> None:
        """把名义通道截面设为**本帧观测到的像素截面**，用于像素域诊断。

        为什么需要它：``detect`` 的长度门槛与 ``duct_geometry_px`` 的名义几何都来自
        ``generation_channel_*_um / pixel_to_micron``。若这两个字段是保存配置里的旧值，
        它们与当前画面不符，测量会被 ``detector_duct_geometry_mismatch`` 拒掉——诊断
        就看不到候选。本方法让名义几何与画面自洽。

        **这不是标尺**：写入的两个 "µm" 数值等于像素数（隐含 1 px = 1 µm）。它只
        消除名义几何与画面的偏差，不产生任何 µm 结论——物理单位仍由调用方的
        ``ScaleEvidence`` 与深度闸门决定。``reason`` 会被记录，便于审计区分
        声明来源（例如 ``"diagnostic_pixel_domain"``）。
        """
        value = float(cross_px)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("像素截面必须是正的有限值")
        self._config.generation_channel_height_um = value
        self._config.generation_channel_width_um = value
        self._pixel_to_micron = 1.0
        self._pixel_cross_section_declaration = {"cross_px": value, "reason": str(reason)}

    def reset_adaptive_size(self) -> None:
        self._runtime_preferred_radius = float(self._configured_preferred_radius)

    @property
    def duct_geometry_px(self) -> tuple[float, float]:
        """生成区模式下 detector 内部实际使用的通道截面 (height_px, width_px)。

        体积公式和长度门槛都用这两个数；它们由 ``generation_channel_*_um`` 除以当前
        ``pixel_to_micron`` 得到。调用方可以把它们和真正测量的那张图里的通道像素数对比，
        从而发现“标尺与图像分辨率不一致”这类错误，而不是让它隐式通过。
        """
        scale = max(1e-9, float(self._pixel_to_micron))
        return (
            float(self._config.generation_channel_height_um) / scale,
            float(self._config.generation_channel_width_um) / scale,
        )

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

    def detect(self, frame: np.ndarray, mode: Optional[str] = None, *,
               channel_width_px: float | None = None,
               flow_axis: str | None = None,
               outer_observation: np.ndarray | None = None,
               trace: dict | None = None) -> DetectionResult:
        """Detect droplets or generation-zone plugs.

        ``channel_width_px`` 是**本帧**扶正后的横向**索引跨度**（像素个数 − 1），
        必须由调用方用 :func:`backend.vision.plug_geometry.rectified_axes` 取得。
        它是长度门槛与边缘间隔门槛的参考宽度；``None`` 表示退回配置里的名义几何
        （``generation_channel_*_um / pixel_to_micron``），只应在没有扶正几何的
        场景使用——名义值与真实画面不一致时门槛会失真。

        ``flow_axis`` 是流向所在的轴。扶正图**必须**显式传 ``"x"``（扶正变换把管壁
        方向映到 x）；``None`` 表示调用方不知道，此时按「较长边为流向」的启发式推断，
        并在 trace 里把来源标为 ``longer_side_heuristic``——那是启发式，不是契约。

        ``trace`` 为可选输出字典：给出时写入本次检测的中间量（参考宽度及其来源、
        轴向来源、候选区间、轮廓检查等），供诊断与审计使用。

        ``outer_observation`` 是与扶正内壁图同轴、同宽的原灰度外侧观察带。它只
        用于弱边界液柱的轴向端点判定；物理截面和等效直径仍来自内壁图与配置。
        """
        gray = self._ensure_gray(frame)
        selected_mode = str(mode or self._config.measurement_mode).strip().lower()
        if selected_mode == "generation_plug":
            return self._detect_generation_plugs(gray, trace, channel_width_px=channel_width_px,
                                                flow_axis=flow_axis,
                                                outer_observation=outer_observation)
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
        *,
        channel_width_px: float | None = None,
        flow_axis: str | None = None,
        outer_observation: np.ndarray | None = None,
    ) -> DetectionResult:
        """Measure detached C-regime plugs from paired menisci.

        A transverse outline envelope separates capsule bodies from carrier
        gaps. A centre profile provides fallback meniscus candidates when the
        envelope has insufficient contrast. Each accepted length
        is converted to volume using the square-channel formula from
        van Steijn et al. (Scientific Reports, 2017), then reported as an
        equivalent-sphere diameter so the existing controller keeps one clear
        physical setpoint.
        """
        height, width = gray.shape[:2]
        if flow_axis is None:
            # 调用方不知道轴向：按较长边推断，并把来源记为启发式。扶正图**必须**显式传轴，
            # 否则 80×40 这类「短而宽」的扶正图会被静默转轴。
            flow_axis = "x" if width >= height else "y"
            flow_axis_source = "longer_side_heuristic"
        else:
            flow_axis = str(flow_axis).strip().lower()
            if flow_axis not in {"x", "y"}:
                raise ValueError(f"flow_axis 只能是 'x' 或 'y'，得到 {flow_axis!r}")
            flow_axis_source = "explicit"
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
        # 参考宽度的来源必须可追溯：名义配置派生，还是本帧扶正几何。二者混杂会让
        # 长度门槛跟着名义 50 µm/默认倍率走，而不是跟着真实画面走。
        reference_width_source = "config_nominal_geometry"
        if channel_width_px is not None:
            if not np.isfinite(channel_width_px) or not 2 <= channel_width_px <= cross_size:
                raise ValueError("channel_width_px must fit the rectified cross-section")
            reference_width_px = float(channel_width_px)
            reference_width_source = "frame_rectified_geometry"
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

        phase_intervals = shoulder_phase_intervals(working)
        body_intervals = capsule_intervals(working) if phase_intervals is None else phase_intervals
        raw_smoothed = cv2.GaussianBlur(working.astype(np.float32), (0, 0), 0.8)
        raw_minimum = float(self._config.generation_min_raw_outline_contrast)
        if phase_intervals is not None:
            noise = float(np.median(np.abs(working.astype(np.float32) - raw_smoothed)))
            raw_minimum = min(raw_minimum, max(3.0, 6.0 * noise))
        raw_contrast_checks: list[tuple[int, int, float]] = []
        if body_intervals is not None:
            # Enhancement can erase a broad, weak boundary or amplify the
            # background enough to reject it. Check envelope candidates in the
            # same original intensity domain from which they were obtained.
            transverse_gradient = np.abs(np.gradient(working.astype(np.float32), axis=0))
            transverse_inner = transverse_gradient[
                transverse_margin : max(transverse_margin + 1, cross_size - transverse_margin)
            ]
            transverse_threshold = max(1.0, float(np.percentile(transverse_inner, 80.0)))
        # Centre-band peaks alone can disappear at a weak meniscus or pair
        # across the carrier gap. Independent transverse outlines provide both
        # body membership and missing endpoint candidates.
        interval_pairs = (
            list(zip(peak_indices, peak_indices[1:]))
            if body_intervals is None else body_intervals
        )
        candidates: list[tuple[float, int, int, float, float]] = []
        for interval_index, (left, right) in enumerate(interval_pairs):
            if left <= 0 or right >= axial_size - 1:
                continue
            length_px = float(right - left)
            if length_px < min_length or length_px > max_length:
                continue
            # Envelope intervals are ordered and disjoint. Exclude all nearby
            # bodies before size filtering so even partial neighbours cannot
            # contaminate the carrier-phase reference of a complete droplet.
            background_start, background_stop = 0, axial_size
            if body_intervals is not None:
                if interval_index > 0:
                    background_start = interval_pairs[interval_index - 1][1] + 1
                if interval_index + 1 < len(interval_pairs):
                    background_stop = interval_pairs[interval_index + 1][0]
            raw_contrast = raw_outline_contrast(
                raw_smoothed, left, right,
                background_start=background_start,
                background_stop=background_stop,
            )
            if trace is not None:
                raw_contrast_checks.append((left, right, raw_contrast))
            if raw_contrast < raw_minimum:
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
                row_window_ratio=float(self._config.generation_outline_row_window_ratio),
                gap_min_ratio=float(self._config.generation_outline_gap_min_ratio),
                gap_max_ratio=float(self._config.generation_outline_gap_max_ratio),
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
            if body_intervals is None and min(support_values) < float(
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

        # In weak phase-contrast video the corrected-image contour gates can
        # erase every real plug. A pair of raw shoulders supplies independent
        # body evidence and bounds each endpoint near its strong seed. Use it
        # only when it actually resolves complete intervals; other imagery
        # keeps the established generation detector behavior.
        shoulder_image = working
        shoulder_width = reference_width_px
        if outer_observation is not None:
            observation = self._ensure_gray(outer_observation)
            shoulder_image = observation if flow_axis == "x" else observation.T
            if shoulder_image.shape[1] != axial_size:
                raise ValueError("outer observation must share the rectified axial coordinates")
            shoulder_width = float(shoulder_image.shape[0] - 1)
        shoulder_selected = bounded_shoulder_intervals(shoulder_image, shoulder_width)
        if shoulder_selected:
            selected = [(left, right, float(right - left))
                        for left, right in shoulder_selected]
            selected_outline_support = []
        # This raw centre-line branch distinguishes dark bodies from the
        # narrow bright carrier gaps that the outline path can mistake for a
        # plug in low-contrast camera recordings.
        # Only the rectified measurement route supplies both an explicit flow
        # axis and a frame-derived channel width. Other detector callers keep
        # their established polarity-independent contour behavior.
        dark_selected = (dark_body_intervals(working, reference_width_px)
                         if flow_axis_source == "explicit" and channel_width_px is not None
                         else [])
        if dark_selected:
            selected = [(left, right, float(right - left))
                        for left, right in dark_selected]
            selected_outline_support = []
        transverse_selected = (transverse_body_intervals(working, reference_width_px)
                               if not dark_selected and flow_axis_source == "explicit"
                               and channel_width_px is not None else [])
        if transverse_selected:
            selected = [(left, right, float(right - left))
                        for left, right in transverse_selected]
            selected_outline_support = []

        if trace is not None:
            trace.update(
                {
                    "flow_axis": flow_axis,
                    "flow_axis_source": flow_axis_source,
                    "working": working,
                    "corrected": corrected,
                    "band_start": band_start,
                    "band_size": band_size,
                    "profile": profile,
                    "gradient": gradient,
                    "gradient_threshold": threshold,
                    "peak_indices": peak_indices,
                    "selected_intervals": list(selected),
                    "interval_source": ("dark_body_between_carrier_gaps" if dark_selected
                                        else "transverse_body_between_carrier_gaps"
                                        if transverse_selected
                                        else "bounded_raw_shoulders" if shoulder_selected
                                        else "generation_outline"),
                    "dark_body_intervals": dark_selected,
                    "outer_observation_used": bool(outer_observation is not None),
                    "reference_width_px": reference_width_px,
                    "reference_width_source": reference_width_source,
                    "reference_width_limits_px": (2.0, float(cross_size)),
                    "config_channel_px": (float(channel_h_px), float(channel_w_px)),
                    "pixel_cross_section_declaration": self._pixel_cross_section_declaration,
                    "minimum_length_px": min_length,
                    "maximum_length_px": max_length,
                    "transverse_gradient": transverse_gradient,
                    "transverse_gradient_threshold": transverse_threshold,
                    "selected_outline_support": selected_outline_support,
                    "body_intervals": body_intervals,
                    "shoulder_phase_intervals": phase_intervals,
                    "raw_outline_contrast_checks": raw_contrast_checks,
                    "minimum_raw_outline_contrast": raw_minimum,
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
            # Raw-shoulder intervals already require a carrier column beyond
            # each endpoint. An exclusive right endpoint at width-1 is still
            # complete; the older contour path keeps its stricter gate.
            full = (left > 0 and right < axial_size
                    if shoulder_selected else left > 0 and right < axial_size - 1)
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
        row_window_ratio: float = 0.70,
        gap_min_ratio: float = 0.55,
        gap_max_ratio: float = 1.45,
    ) -> float:
        """Return axial coverage carrying a pair of separated capsule edges.

        A column supports a capsule when its two strongest transverse-gradient
        rows sit about one duct width apart. Taking the outermost rows of the
        thresholded set instead follows the fixed walls and the background
        shading rather than the capsule: on the 2026-09-18 recording their
        separation is ~53 px against a 27 px duct, so every real capsule failed
        the outline gate and almost no droplet was ever reported.
        """
        height, width = transverse_gradient.shape[:2]
        trim = max(1, int(round(reference_width_px * 0.08)))
        start = max(0, int(left) + trim)
        stop = min(width, int(right) - trim)
        centre = height * 0.5
        window = int(round(float(row_window_ratio) * reference_width_px))
        row_start = max(0, int(row_margin), int(round(centre - window)))
        row_stop = min(height, height - int(row_margin), int(round(centre + window)))
        if stop <= start or row_stop - row_start < 4:
            return 0.0

        minimum_gap = max(2, int(round(reference_width_px * float(gap_min_ratio))))
        maximum_gap = max(
            minimum_gap,
            int(round(reference_width_px * float(gap_max_ratio))),
        )
        column = transverse_gradient[row_start:row_stop, start:stop]
        if column.shape[0] <= minimum_gap:
            return 0.0

        ranked = np.argsort(column, axis=0)[::-1]
        columns = np.arange(column.shape[1], dtype=np.intp)
        strongest = ranked[0]
        strong_enough = column[strongest, columns] >= float(edge_threshold)
        # The partner must be a distinct edge, not the shoulder of the first.
        suppression = max(1, minimum_gap // 2)
        distinct = np.abs(ranked - strongest[None, :]) >= suppression
        paired = distinct.any(axis=0)
        partner = np.where(paired, ranked[np.argmax(distinct, axis=0), columns], -1)
        separation = np.abs(strongest - partner)
        supported = (
            strong_enough
            & paired
            & (separation >= minimum_gap)
            & (separation <= maximum_gap)
        )
        return float(np.count_nonzero(supported)) / float(max(1, stop - start))

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
        """Volume-equivalent sphere diameter of one plug.

        The formula lives in :mod:`backend.vision.plug_geometry` because the
        controller, the tuner and the calibration records all quote the same
        number; the detector must not be a second source of truth for it.
        """
        if min(length_px, height_px, width_px) <= 0.0:
            return None
        try:
            return equivalent_sphere_diameter_px(
                length_px,
                height_px,
                width_px,
                float(self._config.generation_volume_correction),
            )
        except ValueError:
            return None

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
        # A box of this width covers the same +-75 px as the sigma=25 Gaussian it
        # replaces, but costs O(1) per pixel instead of 151 taps.  On the real
        # 83x720 generation ROI that is 11.9 ms -> 0.2 ms per frame, and the
        # detected count and median plug length are unchanged: the axial profile
        # the measurement is built from differences this slowly varying term out
        # again, so the extra accuracy of the Gaussian buys nothing here.
        background = cv2.boxFilter(gray, -1, (151, 151))
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
