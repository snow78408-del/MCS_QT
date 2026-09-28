"""生成区柱塞测量的单一路径：定位管壁 → 扶正 → 全分辨率检测 → 带标尺元数据的尺寸。

存在的理由：2026-09-21 的现场链在**被缩放到 640×41 的 JPEG 预览**上检测，
却乘以原始分辨率才适用的 1.725 μm/px，同时让 detector 内部按
``50 μm / 1.725 = 28.99 px`` 建模通道截面，而该图里通道实际占 41 px。
两个方向的偏差在体积公式里部分抵消，因此“看起来量级正常”，错误被隐藏了。

本模块把五件事钉在一起：

1. **只在扶正后的全分辨率图上测量**。任何显示/传输用的缩放都发生在测量之后，
   缩放比例不得进入物理尺寸。
2. **标尺必须显式声明来源和是否已验证**。未知或未验证的标尺返回明确无效状态；
   像素域结果仍然保留，供离线诊断使用（项目既有决定：缺标尺不填 µm）。
3. **通道截面分两部分**：图内横向宽度由扶正几何实测；芯片深度**不可从图像推断**，
   必须由调用方声明。深度未知时拒绝输出任何体积等效尺寸，只给轴向长度。
4. **帧身份与定位证据必须绑定**：帧号必须为正且一致，管壁定位所用的帧号必须等于
   被测量帧号；时间来源要声明，主机时钟代理会被标记而不是当成采集时刻。
5. **复用几何必须在当前画面上核验位置、方向、间距与 ROI 覆盖**，而不只是间距。

尺寸定义：``plug_lengths`` 是两弯月面之间的轴向长度；``equivalent_diameters`` 是
体积等效球直径（见 :mod:`backend.vision.plug_geometry`）。两者不是同一个量，本模块
分别存储，不互相替代。``valid`` 只表示测量链完整；是否存在完整柱塞另由
``has_complete_plug`` 表示。
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import math

import cv2
import numpy as np

from .plug_geometry import (
    RectifiedAxes,
    equivalent_sphere_diameter_px,
    rectified_axes,
)
from .rectified_roi import (rectify_channel_frame, rectify_lower_outer_band,
                            wall_line_quad, wall_separation_px)

SCALE_SOURCES = frozenset({
    "scale_bar",                  # 同一成像配置下的实物标尺，独立测量
    "channel_width_reference",    # 由“框选内通道宽度 = 已知值”反推
    "configured_optical",         # 只是配置里的光学比例，无本次测量证据
})

TIME_SOURCES = frozenset({
    "camera_frame_timestamp",     # 相机帧自带时间戳
    "host_clock_proxy",           # 主机时钟代理，不能当作该帧的采集时刻
})

DEPTH_SOURCES = frozenset({
    "declared_chip_geometry",     # 调用方声明的芯片结构尺寸
    "measured_depth",             # 独立测得的深度（如共聚焦/截面成像）
    "unknown",
})

DUCT_GEOMETRY_TOLERANCE = 0.10
"""detector 通道像素数与扶正后截面像素数的允许相对差。

取 10% 而不是 2%：detector 是按名义 ROI 几何配置一次的，而扶正高度来自**逐帧定位**，
逐帧之间有几个百分点的自然波动（实测同一场景 39–41 px）。2% 会把正常帧判成几何不一致。
10% 仍然拦得住 2026-09-21 的现场错误（28.99 px 对 41 px，差 29%）。"""

MID_OFFSET_TOLERANCE_PX = 6.0
"""复用几何与当前画面中线位置的允许绝对偏差（像素）。"""

TRACE_SUMMARY_KEYS = (
    "flow_axis",
    "reference_width_px",
    "reference_width_source",
    "reference_width_limits_px",
    "config_channel_px",
    "pixel_cross_section_declaration",
    "minimum_length_px",
    "maximum_length_px",
    "selected_intervals",
    "body_intervals",
    "raw_outline_contrast_checks",
    "minimum_raw_outline_contrast",
)
"""可从 trace 落盘的判据量。其余键含 numpy 数组，只供进程内诊断。"""


def trace_summary(trace: dict | None) -> dict:
    """trace 的 JSON 安全子集。

    完整 trace 里含 numpy 数组（工作图、剖面、梯度），无法直接落盘；这里只取标量与
    列表形式的判据量。``PlugMeasurement.detection_trace_summary`` 与本函数同源。
    """
    if not isinstance(trace, dict):
        return {}
    summary: dict = {}
    for key in TRACE_SUMMARY_KEYS:
        if key not in trace:
            continue
        value = trace[key]
        if isinstance(value, tuple):
            value = [list(item) if isinstance(item, tuple) else item for item in value]
        summary[key] = value
    return summary


def detect_rectified_generation_plugs(
    gray_rectified: np.ndarray,
    *,
    detector,
    trace: dict | None = None,
    outer_observation: np.ndarray | None = None,
):
    """**扶正图检测的唯一入口**：生产路径与诊断工具都必须走这里。

    参考宽度由**本帧扶正几何**显式给出（:func:`rectified_axes` 的
    ``transverse_index_span``，即横向像素个数 − 1），轴向由扶正契约显式定为 ``"x"``。
    不猜方向、不用名义 50 µm、不用默认倍率。跨度为非法尺寸时直接抛错，不回退到名义
    几何——回退会让门槛与画面不一致而无人察觉。

    返回 ``(DetectionResult, trace)``；``trace`` 为传入的同一字典，已含
    ``reference_width_source="frame_rectified_geometry"`` 等判据量。
    """
    if gray_rectified is None or getattr(gray_rectified, "size", 0) == 0:
        raise ValueError("扶正图为空，无法检测")
    axes = rectified_axes(gray_rectified.shape[:2])
    if trace is None:
        trace = {}
    trace["axes"] = axes.to_dict()
    result = detector.detect(
        gray_rectified,
        mode="generation_plug",
        # 参考宽 = 横向**索引跨度**（像素个数 − 1），与实测长度（索引差）同单位。
        channel_width_px=float(axes.transverse_index_span),
        # 轴向由扶正契约定为 x，显式传入；不让检测器按长边猜。
        flow_axis=axes.flow_axis,
        outer_observation=outer_observation,
        trace=trace,
    )
    return result, trace


@dataclass(frozen=True)
class ScaleEvidence:
    """μm/px 的取值、来源和验证状态。``validated=False`` 一律不产生 µm 结论。"""

    um_per_px: float | None
    source: str
    validated: bool
    reference_um: float | None = None
    detail: str = ""

    @property
    def usable(self) -> bool:
        return (
            self.um_per_px is not None
            and math.isfinite(float(self.um_per_px))
            and float(self.um_per_px) > 0.0
            and self.validated
            and self.source in SCALE_SOURCES
        )

    def to_dict(self) -> dict:
        return {
            "um_per_px": self.um_per_px,
            "source": self.source,
            "validated": self.validated,
            "reference_um": self.reference_um,
            "detail": self.detail,
            "usable": self.usable,
        }


@dataclass(frozen=True)
class FrameEvidence:
    """被测量图像的来源与身份证据。

    帧号必须为正：``0`` 是“未初始化”的哨兵值，不能证明时间属于这张图。
    ``localization_frame_id`` 是管壁定位所用的帧号，必须等于 ``frame_id``，
    否则定位证据与图像不匹配。
    """

    frame_id: int
    hardware_frame_id: int
    capture_monotonic: float
    localization_frame_id: int
    time_source: str = "host_clock_proxy"
    coordinate_space: str = "rectified_full_resolution"

    @property
    def time_is_proxy(self) -> bool:
        return self.time_source != "camera_frame_timestamp"

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "hardware_frame_id": self.hardware_frame_id,
            "capture_monotonic": self.capture_monotonic,
            "localization_frame_id": self.localization_frame_id,
            "time_source": self.time_source,
            "time_is_proxy": self.time_is_proxy,
            "coordinate_space": self.coordinate_space,
        }


@dataclass(frozen=True)
class WallGeometryEvidence:
    lines: tuple[dict[str, float], ...]
    source: str
    frame_shape: tuple[int, int]
    separation_px: float | None
    tilt_ratio: float | None
    mid_y_px: float | None = None
    consistency: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "lines": [dict(line) for line in self.lines],
            "source": self.source,
            "frame_shape": list(self.frame_shape),
            "separation_px": self.separation_px,
            "tilt_ratio": self.tilt_ratio,
            "mid_y_px": self.mid_y_px,
            "consistency": dict(self.consistency),
        }


@dataclass(frozen=True)
class DuctGeometryEvidence:
    """通道截面：横向宽度来自图像，深度来自声明。

    深度是**芯片结构尺寸**，图像里看不到；把图高当成深度会静默改写体积模型。
    """

    width_px: float | None
    width_um: float | None
    width_source: str
    depth_px: float | None
    depth_um: float | None
    depth_source: str
    depth_validated: bool
    detector_duct_geometry_px: tuple[float, float] | None = None
    detector_geometry_consistent: bool | None = None
    detector_width_axis_consistent: bool | None = None
    detector_depth_axis_consistent: bool | None = None

    @property
    def usable_for_volume(self) -> bool:
        return (
            self.width_px is not None
            and self.depth_px is not None
            and float(self.depth_px) > 0.0
            and self.depth_source in {"declared_chip_geometry", "measured_depth"}
        )

    def to_dict(self) -> dict:
        return {
            "width_px": self.width_px,
            "width_um": self.width_um,
            "width_source": self.width_source,
            "depth_px": self.depth_px,
            "depth_um": self.depth_um,
            "depth_source": self.depth_source,
            "depth_validated": self.depth_validated,
            "detector_duct_geometry_px": (
                None if self.detector_duct_geometry_px is None
                else list(self.detector_duct_geometry_px)
            ),
            "detector_geometry_consistent": self.detector_geometry_consistent,
            "detector_width_axis_consistent": self.detector_width_axis_consistent,
            "detector_depth_axis_consistent": self.detector_depth_axis_consistent,
            "usable_for_volume": self.usable_for_volume,
        }


@dataclass(frozen=True)
class PlugMeasurement:
    """一次生成区测量的全部元数据。

    ``input_image_shape`` 是送进来的原始帧尺寸（宽, 高）；``rectified_shape`` 是真正
    测量的扶正图尺寸（宽, 高），物理尺寸只在后者上定义。两者分开就是为了让“有没有被
    重新缩放”可核对，而不是靠约定。
    """

    valid: bool
    reason: str
    frame: FrameEvidence
    scale: ScaleEvidence
    input_image_shape: tuple[int, int]
    walls: WallGeometryEvidence | None = None
    duct: DuctGeometryEvidence | None = None
    rectified_shape: tuple[int, int] | None = None
    localization: dict | None = None
    axes: RectifiedAxes | None = None
    plug_lengths_px: tuple[float, ...] = ()
    plug_lengths_um: tuple[float, ...] = ()
    equivalent_diameters_px: tuple[float, ...] = ()
    equivalent_diameters_um: tuple[float, ...] = ()
    complete_plug_flags: tuple[bool, ...] = ()
    rectified_preview: np.ndarray | None = field(default=None, repr=False, compare=False)
    detection_trace: dict | None = field(default=None, repr=False, compare=False)

    @property
    def plug_count(self) -> int:
        return len(self.plug_lengths_px)

    @property
    def complete_plug_count(self) -> int:
        return sum(1 for flag in self.complete_plug_flags if flag)

    @property
    def has_complete_plug(self) -> bool:
        """是否存在可用于下游的完整柱塞。与 ``valid`` 是两件事。"""
        return self.complete_plug_count > 0

    def detection_trace_summary(self) -> dict:
        """trace 的 JSON 安全子集（见模块级 :func:`trace_summary`）。"""
        return trace_summary(self.detection_trace)

    def to_dict(self) -> dict:
        payload: dict = {}
        for item in fields(self):
            if item.name in {"rectified_preview", "detection_trace"}:
                continue
            value = getattr(self, item.name)
            if hasattr(value, "to_dict"):
                value = value.to_dict()
            elif isinstance(value, tuple):
                value = list(value)
            payload[item.name] = value
        payload["plug_count"] = self.plug_count
        payload["complete_plug_count"] = self.complete_plug_count
        payload["has_complete_plug"] = self.has_complete_plug
        payload["detection_trace"] = self.detection_trace_summary()
        return payload


def _line_y_at(lines: list[dict[str, float]], x_px: float, width: int, height: int) -> list[float]:
    """各条线在给定 x（像素）处的 y（像素）。"""
    values = []
    for line in lines:
        x1 = float(line["x1"]) * width
        x2 = float(line["x2"]) * width
        y1 = float(line["y1"]) * height
        y2 = float(line["y2"]) * height
        if abs(x2 - x1) < 1e-6:
            values.append(0.5 * (y1 + y2))
            continue
        slope = (y2 - y1) / (x2 - x1)
        values.append(y1 + slope * (x_px - x1))
    return values


def _band_mid_y(lines: list[dict[str, float]], x_px: float, width: int,
                height: int) -> float | None:
    values = _line_y_at(lines, x_px, width, height)
    if len(values) != 2:
        return None
    return float(0.5 * (min(values) + max(values)))


def _tilt_ratio(lines: list[dict[str, float]], width: int, height: int) -> float | None:
    """管壁平均倾角 dy/dx，单位为像素/像素（与估计器给出的 slope 同量纲）。"""
    ratios = []
    for line in lines:
        dx = (float(line["x2"]) - float(line["x1"])) * width
        dy = (float(line["y2"]) - float(line["y1"])) * height
        if abs(dx) < 1.0:
            return None
        ratios.append(dy / dx)
    return float(np.mean(ratios)) if ratios else None


def _full_frame_parallel_pair(
    frame: np.ndarray,
    target_separation_px: float,
    *,
    separation_tolerance_ratio: float = 0.4,
    slope_tolerance: float = 0.05,
    max_lines: int = 32,
) -> dict | None:
    """在全幅上找一对间距与倾角匹配的长线对，作为无偏的当前画面参照。

    为什么不用 ``estimate_channel_width_px`` 跑一遍复用 ROI：把画面裁到某个带里之后，
    估计器会锁定**裁剪边界**本身（它偏好离 ROI 边界最近的长边），于是任何平移过的
    复用几何都会“自证一致”。全幅搜索不受这个影响，因此只有它才配当参照。
    """
    from .channel_calibration import detect_wall_line_candidates

    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    height, width = frame.shape[:2]
    candidates = detect_wall_line_candidates(frame, max_lines=max_lines)
    if len(candidates) < 2:
        return None
    center_x = (width - 1) * 0.5

    def y_at_center(line: dict) -> float:
        x1 = float(line["x1"]) * width
        x2 = float(line["x2"]) * width
        y1 = float(line["y1"]) * height
        y2 = float(line["y2"]) * height
        if abs(x2 - x1) < 1e-6:
            return 0.5 * (y1 + y2)
        return y1 + (y2 - y1) / (x2 - x1) * (center_x - x1)

    best = None
    for index, first in enumerate(candidates):
        for second in candidates[index + 1:]:
            separation = abs(y_at_center(first) - y_at_center(second))
            if abs(separation - float(target_separation_px)) / max(float(target_separation_px), 1.0) \
                    > separation_tolerance_ratio:
                continue
            if abs(float(first["slope"]) - float(second["slope"])) > slope_tolerance:
                continue
            span = min(float(first["length_ratio"]), float(second["length_ratio"]))
            if best is None or span > best["span_ratio"]:
                bag = sorted([y_at_center(first), y_at_center(second)])
                best = {
                    "span_ratio": float(span),
                    "upper_y_px": float(bag[0]),
                    "lower_y_px": float(bag[1]),
                    "mid_y_px": float(0.5 * (bag[0] + bag[1])),
                    "separation_px": float(separation),
                    "tilt_ratio": float(0.5 * (float(first["slope"]) + float(second["slope"]))),
                    "wall_slope_delta": float(abs(float(first["slope"]) - float(second["slope"]))),
                    "line_ids": [int(first["id"]), int(second["id"])],
                }
    return best


def _estimate_in_area(frame: np.ndarray, rows: tuple[int, int], columns: tuple[int, int],
                      min_confidence: float) -> dict | None:
    """在一块 ROI 内用项目既有估计器重测管壁，并把结果映回全图坐标。"""
    from .channel_calibration import estimate_channel_width_px

    y0, y1 = rows
    x0, x1 = columns
    if y1 - y0 < 40 or x1 - x0 < 100:
        return None
    found = estimate_channel_width_px(frame[y0:y1, x0:x1], flow_axis="x")
    if found.width_px is None or found.confidence < min_confidence:
        return None
    if found.upper_center_px is None or found.lower_center_px is None:
        return None
    # 估计器的 center 是相对 ROI 的 y（在该 ROI 水平中心处），映回全图只需加 y0。
    upper_y = float(y0) + float(found.upper_center_px)
    lower_y = float(y0) + float(found.lower_center_px)
    return {
        "upper_y_px": upper_y,
        "lower_y_px": lower_y,
        "mid_y_px": 0.5 * (upper_y + lower_y),
        "separation_px": float(found.width_px),
        "tilt_ratio": 0.5 * (float(found.upper_slope or 0.0) + float(found.lower_slope or 0.0)),
        "wall_slope_delta": abs(float(found.upper_slope or 0.0) - float(found.lower_slope or 0.0)),
        "confidence": float(found.confidence),
    }


def verify_reused_walls(
    frame: np.ndarray,
    wall_lines: list[dict[str, float]],
    *,
    frame_id: int = 0,
    separation_tolerance_ratio: float = 0.25,
    tilt_tolerance: float = 0.05,
    mid_offset_tolerance_px: float = MID_OFFSET_TOLERANCE_PX,
    min_confidence: float = 0.68,
) -> dict:
    """在**当前画面**上独立核验要复用的管壁几何。

    只比较间距和斜率是不够的：整体平移过的 ROI 会有相同的间距与倾角，却完全不在
    通道上。本函数把当前画面测到的两壁映回全图，比较**中线位置、方向、间距、
    ROI 覆盖与沿程一致性**，五项都通过才返回 ``verified``。

    定位证据绑定帧号：返回的 ``verified_frame_id`` 必须等于调用方随后用来测量的
    帧号，:func:`measure_generation_plugs` 会核对这一点。

    返回 ``status`` ∈ {``verified``, ``mismatch``, ``unverifiable``, ``invalid``}。
    """
    result: dict = {
        "verified_frame_id": int(frame_id),
        "status": "invalid",
        "consistent": False,
        "reason": "",
        "rejection_reasons": [],
    }
    if frame is None or getattr(frame, "size", 0) == 0:
        result["reason"] = "当前画面为空"
        return result
    if len(wall_lines) != 2:
        result["reason"] = "需要恰好两条管壁线"
        return result

    height, width = frame.shape[:2]
    separation = wall_separation_px(width, height, wall_lines)
    tilt = _tilt_ratio(list(wall_lines), width, height)
    center_x = (width - 1) * 0.5
    reused_mid = _band_mid_y(list(wall_lines), center_x, width, height)
    result.update(
        reused_separation_px=None if separation is None else float(separation),
        reused_tilt_ratio=tilt,
        reused_mid_y_px=reused_mid,
    )
    if separation is None or separation <= 1.0 or tilt is None or reused_mid is None:
        result["reason"] = "复用几何本身不可用"
        return result

    from .rectified_roi import wall_lines_bbox

    bbox = wall_lines_bbox(wall_lines)
    bbox_rows: tuple[int, int] | None = None
    cross_check = None
    if bbox is not None and bbox["y_end_ratio"] > bbox["y_start_ratio"]:
        bbox_rows = (max(0, int(math.floor(bbox["y_start_ratio"] * height))),
                     max(1, min(height, int(math.ceil(bbox["y_end_ratio"] * height)))))
        columns = (max(0, int(math.floor(bbox["x_start_ratio"] * width))),
                   max(1, min(width, int(math.ceil(bbox["x_end_ratio"] * width)))))
        cross_check = _estimate_in_area(frame, bbox_rows, columns, min_confidence)

    # 参照只取全幅无偏搜索；ROI 内估计会因为锁在裁剪边界上而“自证一致”，
    # 只能当作交叉检查，不能当证据。
    reference = _full_frame_parallel_pair(frame, float(separation))
    if reference is None:
        result.update(
            status="unverifiable",
            reason="当前画面无法独立重测管壁；不得退回默认几何",
            current_frame_separation_px=None,
            current_frame_confidence=0.0,
            measured_via="none",
        )
        if cross_check is not None:
            result["reused_roi_estimate"] = cross_check
        return result

    result["measured_via"] = "prior_seeded_full_frame_search"
    result["method_note"] = (
        "这是**带先验的核验**：候选线对按复用间距 ±40% 筛选，因此只能在“已知大致间距”的"
        "前提下判断旧几何是否仍然成立，不能用来在没有旧坐标时定位管道。无先验定位见 "
        "backend/vision/parallel_walls.py"
    )
    result["full_frame_estimate"] = reference
    if cross_check is not None:
        result["reused_roi_estimate"] = cross_check

    reasons: list[str] = []
    if cross_check is not None:
        disagreement = abs(float(cross_check["mid_y_px"]) - float(reference["mid_y_px"]))
        result["roi_vs_full_frame_mid_offset_px"] = float(disagreement)
        if disagreement > mid_offset_tolerance_px:
            reasons.append(
                f"复用 ROI 内找到的结构与全幅搜索不是同一条通道（差 {disagreement:.1f} px）"
            )

    current = reference
    mid_offset = abs(float(current["mid_y_px"]) - float(reused_mid))
    separation_ratio = (abs(float(current["separation_px"]) - float(separation))
                        / max(float(separation), 1.0))
    tilt_delta = abs(float(current["tilt_ratio"]) - float(tilt))
    roi_covers = None
    if bbox_rows is not None:
        roi_covers = bool(bbox_rows[0] <= current["upper_y_px"] <= bbox_rows[1]
                          and bbox_rows[0] <= current["lower_y_px"] <= bbox_rows[1])

    if separation_ratio > separation_tolerance_ratio:
        reasons.append(f"间距不符（相对差 {separation_ratio:.3f}）")
    if tilt_delta > tilt_tolerance:
        reasons.append(f"倾角不符（差 {tilt_delta:.4f}）")
    if mid_offset > mid_offset_tolerance_px:
        reasons.append(f"中线位置不符（差 {mid_offset:.1f} px）")
    if roi_covers is False:
        reasons.append("复用 ROI 没有包住当前画面的管壁")
    if float(current["wall_slope_delta"]) > tilt_tolerance:
        reasons.append("当前画面两条壁不平行（沿程不一致）")

    result.update(
        current_frame_separation_px=float(current["separation_px"]),
        current_frame_mid_y_px=float(current["mid_y_px"]),
        current_frame_tilt_ratio=float(current["tilt_ratio"]),
        current_frame_confidence=float(current.get("confidence", current["span_ratio"])),
        current_frame_upper_y_px=float(current["upper_y_px"]),
        current_frame_lower_y_px=float(current["lower_y_px"]),
        current_frame_wall_slope_delta=float(current["wall_slope_delta"]),
        reused_roi_covers_current_walls=roi_covers,
        mid_offset_px=float(mid_offset),
        mid_offset_tolerance_px=float(mid_offset_tolerance_px),
        separation_relative_delta=float(separation_ratio),
        tilt_delta=float(tilt_delta),
        rejection_reasons=reasons,
    )
    result["status"] = "verified" if not reasons else "mismatch"
    result["consistent"] = not reasons
    result["reason"] = "当前画面与复用几何一致" if not reasons else "；".join(reasons)
    return result


def display_scales(
    rectified_shape: tuple[int, int],
    display_shape: tuple[int, int],
    scale_um_per_px: float,
) -> dict:
    """扶正图被缩放到显示尺寸后，**轴向与横向各自**的 μm/px。

    显示缩放不是各向同性的：``cv2.resize(718×47 → 640×41)`` 的横向因子是
    640/718 = 0.89136、纵向因子是 41/47 = 0.87234，两者因整数取整而不同。
    因此沿流向的柱塞长度不能用纵向倍率换算，横向通道宽度也不能用横向倍率，
    必须分别给出。任一输入缺失或非法时两个值都为 None，调用方不得填 µm。
    """
    invalid = {"axial_um_per_px": None, "transverse_um_per_px": None,
               "anisotropic": None, "reason": ""}
    if not rectified_shape or not display_shape:
        invalid["reason"] = "缺少扶正图或显示图尺寸"
        return invalid
    if scale_um_per_px is None or not math.isfinite(float(scale_um_per_px)) \
            or float(scale_um_per_px) <= 0.0:
        invalid["reason"] = "标尺未知或非法"
        return invalid
    rect_w, rect_h = (int(rectified_shape[0]), int(rectified_shape[1]))
    disp_w, disp_h = (int(display_shape[0]), int(display_shape[1]))
    if min(rect_w, rect_h, disp_w, disp_h) <= 0:
        invalid["reason"] = "图像尺寸必须为正"
        return invalid
    axial = float(scale_um_per_px) * rect_w / float(disp_w)
    transverse = float(scale_um_per_px) * rect_h / float(disp_h)
    return {
        "axial_um_per_px": axial,
        "transverse_um_per_px": transverse,
        "anisotropic": abs(axial - transverse) / max(axial, transverse) > 1e-6,
        "axial_display_factor": disp_w / float(rect_w),
        "transverse_display_factor": disp_h / float(rect_h),
        "reason": "",
    }


def _duct_evidence(
    detector,
    *,
    scale: ScaleEvidence,
    width_px: float,
    duct_depth_um: float | None,
    duct_depth_source: str,
    duct_depth_validated: bool,
) -> tuple[DuctGeometryEvidence, float | None]:
    """构造通道截面证据，并给出 detector 一致性判定。返回 (证据, 实际使用的深度像素)。"""
    depth_um = None if duct_depth_um is None else float(duct_depth_um)
    source = str(duct_depth_source) if duct_depth_source in DEPTH_SOURCES else "unknown"
    if depth_um is None or not math.isfinite(depth_um) or depth_um <= 0.0:
        depth_um = None
        source = "unknown"
    depth_px = None
    if depth_um is not None and scale.um_per_px:
        depth_px = depth_um / float(scale.um_per_px)
    try:
        detector_duct = tuple(float(v) for v in detector.duct_geometry_px)
    except Exception:
        detector_duct = None
    consistent = None
    width_consistent = None
    depth_consistent = None
    if detector_duct is not None:
        # 宽度轴总是可比：图内实测的通道宽度必须与 detector 实际使用的宽度一致。
        width_consistent = bool(
            abs(detector_duct[1] - float(width_px)) / max(float(width_px), 1e-9)
            <= DUCT_GEOMETRY_TOLERANCE
        )
        # 深度轴只有声明了深度才可比；未声明时不把这一轴算作通过。
        if depth_px is not None:
            depth_consistent = bool(
                abs(detector_duct[0] - depth_px) / max(depth_px, 1e-9) <= DUCT_GEOMETRY_TOLERANCE
            )
        consistent = bool(width_consistent and depth_consistent is not False)
    width_um = None if not scale.um_per_px else float(width_px) * float(scale.um_per_px)
    return DuctGeometryEvidence(
        width_px=float(width_px),
        width_um=width_um,
        width_source="rectified_image_measurement",
        depth_px=depth_px,
        depth_um=depth_um,
        depth_source=source,
        depth_validated=bool(duct_depth_validated),
        detector_duct_geometry_px=detector_duct,
        detector_geometry_consistent=consistent,
        detector_width_axis_consistent=width_consistent,
        detector_depth_axis_consistent=depth_consistent,
    ), depth_px


def measure_generation_plugs(
    image: np.ndarray,
    *,
    detector,
    wall_lines: list[dict[str, float]] | None = None,
    scale: ScaleEvidence,
    frame_evidence: FrameEvidence,
    duct_depth_um: float | None,
    duct_depth_source: str = "declared_chip_geometry",
    duct_depth_validated: bool = False,
    duct_width_reference_um: float | None = None,
    wall_source: str = "current_frame",
    wall_consistency: dict | None = None,
    localization=None,
    trace: dict | None = None,
) -> PlugMeasurement:
    """在扶正后的**全分辨率**图上测量生成区柱塞，并附带完整标尺与几何元数据。

    ``detector`` 必须先由调用方配置好；本函数**不**改它的配置，只校验它与扶正几何
    是否一致，不一致就判无效——否则体积公式会用一个和图像不符的截面。

    ``duct_depth_um`` 是芯片深度，必须显式给出；``None`` 表示未知，此时拒绝输出任何
    体积等效尺寸（只给轴向长度），不把图高当成深度。

    ``frame_evidence`` 是帧身份证据，必须提供：软件序号为正、**硬件帧号为正**（两者来自
    不同计数器，不要求相等——实测软件 0 对应硬件 74）、定位帧号等于被测帧号、时间来源已声明。

    管壁有两种来源，必须显式选一种：

    * 传 ``localization``（:class:`backend.vision.parallel_walls.WallLocalization`）：
      只有 ``status == "localized"`` 才可用，壁线取自它，并核对其 ``frame_id``、
      ``image_shape`` 与被测帧一致；歧义/待确认/过期一律拒绝。
    * 或直接给 ``wall_lines``：属于低层原语，``wall_source != "current_frame"`` 时
      必须附 ``wall_consistency``（``consistent=True`` 且 ``verified_frame_id`` 等于被测帧号）。

    ``trace`` 为可选输出字典。给出时保存检测中间量（参考宽度及其来源、候选区间、
    轮廓检查），并经 ``detection_trace`` / ``to_dict()["detection_trace"]`` 落盘。
    参考宽度一律取自**本帧扶正几何**（``reference_width_source`` 为
    ``frame_rectified_geometry``），不取名义 50 µm、也不取默认倍率。
    **生产路径与诊断工具都必须经本函数**，这样候选、门槛与拒绝理由不会分叉。
    """
    localization_payload = None
    if localization is not None:
        localization_payload = localization.to_dict()
        if not getattr(localization, "usable", False):
            shape = (0, 0) if image is None else tuple(int(v) for v in np.asarray(image).shape[:2])
            return PlugMeasurement(
                valid=False,
                reason=f"localization_{getattr(localization, 'status', 'invalid')}",
                frame=frame_evidence, scale=scale, input_image_shape=(shape[1], shape[0]),
                localization=localization_payload)
        localization_frame = int(localization.geometry.get("frame_id", 0) or 0)
        if localization_frame != int(frame_evidence.localization_frame_id):
            shape = tuple(int(v) for v in np.asarray(image).shape[:2])
            return PlugMeasurement(
                valid=False, reason="localization_frame_mismatch",
                frame=frame_evidence, scale=scale, input_image_shape=(shape[1], shape[0]),
                localization=localization_payload)
        wall_lines = localization.wall_lines
        wall_source = "localized"

    shape = (0, 0) if image is None else tuple(int(v) for v in np.asarray(image).shape[:2])
    input_shape = (shape[1], shape[0])
    base = dict(frame=frame_evidence, scale=scale, input_image_shape=input_shape,
                localization=localization_payload)

    if localization_payload is not None:
        declared = tuple(int(v) for v in (localization_payload.get("image_shape") or [0, 0]))
        if declared != input_shape:
            return PlugMeasurement(valid=False, reason="localization_image_shape_mismatch", **base)

    if image is None or getattr(image, "size", 0) == 0:
        return PlugMeasurement(valid=False, reason="frame_empty", **base)
    if not (isinstance(frame_evidence.frame_id, int) and frame_evidence.frame_id > 0):
        return PlugMeasurement(valid=False, reason="frame_id_invalid", **base)
    # 硬件帧号是**附加**溯源信息，不是软件序号的别名：实机上两者来自不同计数器，
    # 本来就不同（本仓库实测：软件序号 0 对应硬件帧号 74，末尾相差 317）。
    # 因此**不要求二者相等**——要求相等只会逼调用方伪造其中一个。
    # 但硬件帧号必须存在且为正：缺失时这条测量没有硬件级溯源，按证据不足拒测。
    if not (isinstance(frame_evidence.hardware_frame_id, int)
            and not isinstance(frame_evidence.hardware_frame_id, bool)
            and frame_evidence.hardware_frame_id > 0):
        return PlugMeasurement(valid=False, reason="hardware_frame_id_invalid", **base)
    if int(frame_evidence.localization_frame_id) != int(frame_evidence.frame_id):
        return PlugMeasurement(valid=False, reason="wall_localization_frame_mismatch", **base)
    if frame_evidence.time_source not in TIME_SOURCES:
        return PlugMeasurement(valid=False, reason="time_source_undeclared", **base)
    if not (math.isfinite(float(frame_evidence.capture_monotonic))
            and float(frame_evidence.capture_monotonic) > 0.0):
        return PlugMeasurement(valid=False, reason="capture_time_invalid", **base)
    if len(wall_lines or []) != 2:
        return PlugMeasurement(valid=False, reason="wall_geometry_missing", **base)

    height, width = image.shape[:2]
    geometry = wall_line_quad(width, height, wall_lines)
    if geometry is None:
        return PlugMeasurement(valid=False, reason="wall_geometry_unusable", **base)
    _, _, rectified_height = geometry
    separation = wall_separation_px(width, height, wall_lines)
    evidence = WallGeometryEvidence(
        lines=tuple(dict(line) for line in wall_lines),
        source=str(wall_source),
        frame_shape=(width, height),
        separation_px=None if separation is None else float(separation),
        tilt_ratio=_tilt_ratio(list(wall_lines), width, height),
        mid_y_px=_band_mid_y(list(wall_lines), (width - 1) * 0.5, width, height),
        consistency=dict(wall_consistency or {}),
    )
    base["walls"] = evidence

    if str(wall_source) == "reused":
        consistency = dict(wall_consistency or {})
        if consistency.get("consistent") is not True:
            reason = ("wall_geometry_unverified" if consistency.get("status") == "unverifiable"
                      else "wall_geometry_stale")
            return PlugMeasurement(valid=False, reason=reason, **base)
        if int(consistency.get("verified_frame_id", -1)) != int(frame_evidence.frame_id):
            return PlugMeasurement(valid=False, reason="wall_verification_frame_mismatch", **base)

    duct, depth_px = _duct_evidence(
        detector, scale=scale, width_px=float(separation) if separation else 0.0,
        duct_depth_um=duct_depth_um, duct_depth_source=duct_depth_source,
        duct_depth_validated=duct_depth_validated,
    )
    base["duct"] = duct

    rectified = rectify_channel_frame(image, wall_lines)
    if rectified is None:
        return PlugMeasurement(valid=False, reason="rectification_failed", **base)
    base["rectified_shape"] = (int(rectified.shape[1]), int(rectified.shape[0]))
    gray = rectified if rectified.ndim == 2 else cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    outer_band = rectify_lower_outer_band(image, wall_lines)
    outer_gray = (outer_band if outer_band is None or outer_band.ndim == 2
                  else cv2.cvtColor(outer_band, cv2.COLOR_BGR2GRAY))
    # 扶正轴与参考宽度只在这里取一次：来自本帧扶正几何，不来自名义配置。
    axes = rectified_axes(gray.shape[:2])
    base["axes"] = axes

    detected, trace = detect_rectified_generation_plugs(
        gray, detector=detector, trace=trace, outer_observation=outer_gray)
    base["detection_trace"] = trace
    lengths = tuple(float(v) for v in detected.plug_lengths_px)
    flags = tuple(bool(v) for v in detected.diameter_valid)
    base.update(plug_lengths_px=lengths, complete_plug_flags=flags,
                rectified_preview=rectified)

    if duct.detector_geometry_consistent is False:
        # 检测已经跑完，像素域候选保留在 base 里。几何不一致作废的是**物理结论**，
        # 不是像素域证据：诊断需要看到候选与门槛，但这条测量不得用于控制或物理标定。
        return PlugMeasurement(valid=False, reason="detector_duct_geometry_mismatch", **base)

    # 等效直径用真实的矩形截面（图内宽度 + 声明深度）重算，不把图高当深度。
    diameters: tuple[float, ...] = ()
    if duct.usable_for_volume and depth_px is not None and separation is not None:
        values = []
        for length in lengths:
            try:
                values.append(equivalent_sphere_diameter_px(
                    length, float(depth_px), float(separation)))
            except ValueError:
                continue
        diameters = tuple(values)
    base["equivalent_diameters_px"] = diameters

    if not scale.usable:
        reason = ("scale_source_unknown" if scale.source not in SCALE_SOURCES
                  else ("scale_unknown" if scale.um_per_px is None else "scale_unvalidated"))
        return PlugMeasurement(valid=False, reason=reason, **base)

    factor = float(scale.um_per_px)
    if (duct_width_reference_um is not None and scale.source == "channel_width_reference"
            and separation):
        expected = float(duct_width_reference_um) / factor
        if abs(expected - float(rectified_height)) / float(rectified_height) > DUCT_GEOMETRY_TOLERANCE:
            return PlugMeasurement(valid=False, reason="scale_geometry_inconsistent", **base)

    base["plug_lengths_um"] = tuple(length * factor for length in lengths)
    if not duct.usable_for_volume:
        # 轴向长度不需要深度；体积等效尺寸需要，深度未知就不给。
        return PlugMeasurement(valid=False, reason="duct_depth_unknown", **base)
    base["equivalent_diameters_um"] = tuple(value * factor for value in diameters)
    return PlugMeasurement(
        valid=True,
        reason="ok" if lengths else "no_complete_plug",
        **base,
    )
