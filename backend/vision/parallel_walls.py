"""平行管壁定位：为生成区下游的直管测量段定位一对平行内壁。

与 :mod:`backend.vision.rectified_measurement` 的分工：本模块只负责**几何定位**，
输出一对壁线的原图端点、共见测量范围与证据；标尺（μm/px）与芯片深度是另外两类
证据，定位成功**不**意味着物理尺寸已标定。

设计要点（对应任务书 §3）：

1. 候选、支持边缘、有效可见长度与坐标空间一起保留；``work_scale`` 允许降采样工作图，
   坐标映射显式可逆（默认 1.0，即工作图就是原图）。所有输出端点都是**原图像素**。
2. 两壁间距一律用**法向距离**，不用垂直 y 差；只在两壁的**共见轴向范围**内评估，
   不把短边外推到全图。
3. 图上若有多条同样可信的管道，报告候选竞争；不按最长线静默选中。带内出现与壁面
   同向、长度相当的长边时报告歧义——那可能是液柱边缘而不是内壁。
4. 跨帧证据用有界缓冲（``deque(maxlen=...)``）聚合；不等待、不排队、不阻塞。
5. 运动界面与静止壁边的差异作为辅助证据；**静止不是充分条件**——停泵液柱、静止气泡、
   固定纹理同样静止。没有有效运动时状态为 ``pending_motion``，不允许据此认证内壁。
6. 沿测量段多点核验上下壁与管内覆盖，用两壁方程与实际四边形，不用中心点加外接矩形。
7. 只有唯一可信线对才生成扶正变换；歧义、截断、单壁、低对比、证据过期一律给明确
   拒绝理由，禁止落回旧 ROI。

阈值集中在 :class:`LocalizationThresholds`，来源记录在 :data:`THRESHOLD_PROVENANCE`。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, fields
import hashlib
import math
import threading

import cv2
import numpy as np

GEOMETRY_VERSION = "parallel-walls-20260922"


@dataclass(frozen=True)
class LocalizationThresholds:
    """定位阈值。改动前先看 :data:`THRESHOLD_PROVENANCE` 里对应条目的来源。"""

    min_visible_length_ratio: float = 0.25
    min_common_range_ratio: float = 0.30
    min_common_range_px: float = 120.0
    max_slope_difference: float = 0.05
    max_separation_cv: float = 0.15
    min_separation_px: float = 12.0
    max_separation_ratio: float = 0.60
    separation_sample_count: int = 24
    segment_count: int = 5
    min_supported_segments: int = 4
    max_slope_for_any_candidate: float = 0.35
    min_edge_support: float = 0.45
    edge_support_sigma_multiple: float = 3.0
    edge_support_min_gradient: float = 1.0
    min_mean_gradient_sigma_multiple: float = 5.0
    segment_min_edge_support: float = 0.40
    use_supplementary_candidates: bool = True
    competition_score_ratio: float = 0.90
    competition_separation_ratio: float = 0.15
    competition_mid_offset_px: float = 10.0
    interior_competitor_length_ratio: float = 0.55
    required_support_frames: int = 3
    evidence_max_age_s: float = 20.0
    max_buffer_frames: int = 24
    cluster_separation_ratio: float = 0.20
    cluster_mid_offset_px: float = 12.0
    motion_min_interior_change: float = 1.5
    motion_change_sigma_multiple: float = 3.0
    motion_noise_floor_gray: float = 0.25
    motion_min_axial_shift_px: float = 1.0
    motion_min_shift_correlation: float = 0.5
    motion_min_axial_structure_gray: float = 1.0
    motion_max_geometry_change_px: float = 3.0
    motion_max_tilt_change: float = 0.02
    motion_max_axial_change_px: float = 10.0
    coverage_edge_margin_px: float = 2.0
    coverage_min_contrast: float = 1.0


THRESHOLD_PROVENANCE: dict[str, str] = {
    "min_visible_length_ratio": "工程下限，尚未由带标注样本验证；不得为了通过停泵标定帧而放宽",
    "min_common_range_ratio": "不到半幅宽的共见范围不足以支撑稳定扶正",
    "min_common_range_px": "保证沿程多点核验有足够采样",
    "max_slope_difference": "沿用 channel_calibration 平行判据 0.16 的约 1/3",
    "max_separation_cv": "工程上限；真实平行管道间距沿程变化应远小于此，尚未由样本标定",
    "min_separation_px": "低于 12 px 的扶正带无法分辨柱塞端部：相衬亮带自身的两条边相距只有几个像素，"
                         "实测停泵帧上就出现过 6.03 px 的伪线对（由分辨率下限推出，不依赖 50 µm 假设）",
    "max_separation_ratio": "超过 60% 画幅高的“两条壁”通常是把不同结构连起来",
    "separation_sample_count": "采样数，兼顾精度与开销",
    "segment_count": "沿程核验位置数；5 段可分辨局部截断",
    "min_supported_segments": "允许 1 段缺失（遮挡/低对比），其余必须支持",
    "max_slope_for_any_candidate": "沿用 channel_calibration 的 max_tilt_degrees=33° 量级",
    "min_edge_support": "沿线边缘命中比例下限；未经样本标定，不得为通过停泵帧而放宽",
    "edge_support_sigma_multiple": "阈值 = max(min_gradient, k×稳健噪声尺度)；k=3 为常规检测下限",
    "edge_support_min_gradient": "绝对下限（灰阶/px），防止极低噪声图像把纹理噪声判成边缘",
    "min_mean_gradient_sigma_multiple": "候选线沿线**平均**法向梯度也要超过 k×噪声尺度："
                                       "逐点命中比例会被局部噪声刷高，平均幅度才反映这条边是否真实",
    "segment_min_edge_support": "单段边缘命中下限，低于此该段记为不支持",
    "use_supplementary_candidates": "是否在对比度归一化图上补一路候选；固定 Canny 阈值会让中等对比长边时有时无",
    "competition_score_ratio": "次优线对得分达到最优的 90% 即视为竞争",
    "competition_separation_ratio": "间距相差 <15% 且中线相近者视为同一管道的竞争解释",
    "competition_mid_offset_px": "同上，位置判据",
    "interior_competitor_length_ratio": "带内同向长边达到最短壁长的 55% 时无法区分壁面与液柱边缘",
    "required_support_frames": "工程下限；按内容签名去重后的独立帧数下限",
    "evidence_max_age_s": "证据过期时间；超过则必须重新定位",
    "max_buffer_frames": "有界缓冲容量，防止无界增长",
    "cluster_separation_ratio": "跨帧聚类：间距相对差小于该值归入同一簇",
    "cluster_mid_offset_px": "跨帧聚类：中线偏移小于该值归入同一簇",
    "motion_min_interior_change": "平均变化的参考幅度，仅作诊断；判定改用按噪声归一的显著变化比例",
    "motion_change_sigma_multiple": "单点显著变化阈值 = k×帧差稳健噪声尺度",
    "motion_noise_floor_gray": "噪声尺度的绝对下限：帧差完全均匀（例如整体加常数）时 MAD 为 0，"
                               "没有下限会让阈值退化成 0、把所有点都判成显著变化",
    "motion_min_axial_shift_px": "沿轴互相关峰值对应的位移下限：亮度漂移、全局闪烁、静态噪声都"
                                 "不产生位移，只有空间位移才可能是流动",
    "motion_min_shift_correlation": "沿轴互相关峰值的相关度下限，低于此视为无法判定位移",
    "motion_min_axial_structure_gray": "沿轴签名的 90-10 分位差下限：管内没有沿轴结构"
                                       "（例如无柱塞的均匀管道）时不试图测位移",
    "motion_max_geometry_change_px": "静止壁面的位置允许变化上限；超过则认为该线在移动",
    "motion_max_tilt_change": "静止壁面的倾角允许变化上限",
    "motion_max_axial_change_px": "静止壁面的轴向可见区间允许变化上限；柱塞肩部的长边在 y 上不动，"
                                  "但会沿轴平移，必须靠这一项才能与固定壁面区分",
    "coverage_edge_margin_px": "壁线到画幅边缘的最小留白",
    "coverage_min_contrast": "带内相对带外的最小对比；低于此认为该位置没有液柱证据",
}


@dataclass(frozen=True)
class WallCandidate:
    """一条全幅长边候选，端点为原图像素。"""

    id: int
    x1: float
    y1: float
    x2: float
    y2: float
    slope: float
    length_px: float
    length_ratio: float
    edge_support: float
    mean_normal_gradient: float = 0.0
    coordinate_space: str = "original_frame"

    @property
    def direction(self) -> np.ndarray:
        vector = np.array([self.x2 - self.x1, self.y2 - self.y1], dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-9 else np.array([1.0, 0.0])

    @property
    def normal(self) -> np.ndarray:
        direction = self.direction
        return np.array([-direction[1], direction[0]])

    def y_at(self, x: float) -> float:
        if abs(self.x2 - self.x1) < 1e-9:
            return 0.5 * (self.y1 + self.y2)
        return self.y1 + self.slope * (x - self.x1)

    def point_at(self, x: float) -> np.ndarray:
        return np.array([x, self.y_at(x)], dtype=np.float64)

    @property
    def x_range(self) -> tuple[float, float]:
        return (min(self.x1, self.x2), max(self.x1, self.x2))

    def with_endpoints(self, x1: float, y1: float, x2: float, y2: float) -> "WallCandidate":
        return WallCandidate(
            id=self.id, x1=x1, y1=y1, x2=x2, y2=y2, slope=self.slope,
            length_px=math.hypot(x2 - x1, y2 - y1),
            length_ratio=self.length_ratio, edge_support=self.edge_support,
            mean_normal_gradient=self.mean_normal_gradient,
            coordinate_space=self.coordinate_space)

    def to_dict(self) -> dict:
        return {
            "id": int(self.id),
            "endpoints_px": [round(self.x1, 3), round(self.y1, 3), round(self.x2, 3), round(self.y2, 3)],
            "slope": round(float(self.slope), 6),
            "length_px": round(float(self.length_px), 3),
            "length_ratio": round(float(self.length_ratio), 4),
            "edge_support": round(float(self.edge_support), 4),
            "mean_normal_gradient": round(float(self.mean_normal_gradient), 4),
            "coordinate_space": self.coordinate_space,
        }


@dataclass(frozen=True)
class WallPair:
    """一对候选壁，间距按共见范围内的法向距离计算。"""

    first: WallCandidate
    second: WallCandidate
    common_x_range: tuple[float, float]
    separation_px: float
    separation_std_px: float
    separation_cv: float
    slope_difference: float
    tilt_ratio: float
    mid_y_px: float
    segment_support: tuple[bool, ...]
    score: float
    rejection: str = ""

    @property
    def supported_segments(self) -> int:
        return sum(1 for flag in self.segment_support if flag)

    @property
    def ids(self) -> tuple[int, int]:
        return (int(self.first.id), int(self.second.id))

    def wall_lines(self, width: int, height: int) -> list[dict[str, float]]:
        """转成下游扶正用的归一化壁线（与 channel_calibration 的约定一致）。"""
        low, high = self.common_x_range
        upper = self.first if self.first.y_at(low) < self.second.y_at(low) else self.second
        lower = self.second if upper is self.first else self.first

        def norm_y(candidate: WallCandidate, x: float) -> float:
            return max(0.0, min(1.0, candidate.y_at(x) / float(height)))

        return [
            {"x1": low / float(width), "y1": norm_y(upper, low),
             "x2": high / float(width), "y2": norm_y(upper, high)},
            {"x1": low / float(width), "y1": norm_y(lower, low),
             "x2": high / float(width), "y2": norm_y(lower, high)},
        ]

    def to_dict(self) -> dict:
        return {
            "candidate_ids": list(self.ids),
            "common_x_range_px": [round(self.common_x_range[0], 3), round(self.common_x_range[1], 3)],
            "separation_px": round(float(self.separation_px), 4),
            "separation_std_px": round(float(self.separation_std_px), 4),
            "separation_cv": round(float(self.separation_cv), 4),
            "slope_difference": round(float(self.slope_difference), 6),
            "tilt_ratio": round(float(self.tilt_ratio), 6),
            "mid_y_px": round(float(self.mid_y_px), 3),
            "supported_segments": int(self.supported_segments),
            "segment_support": list(self.segment_support),
            "score": round(float(self.score), 4),
            "rejection": self.rejection,
        }


@dataclass(frozen=True)
class FrameCandidates:
    """一帧的候选与它们所在坐标系的配套数据。"""

    candidates: tuple[WallCandidate, ...]
    edges: EdgeSupportMap = field(repr=False, compare=False)
    gray: np.ndarray = field(repr=False, compare=False)
    coordinate_mapping: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeSupportMap:
    """方向梯度支持图。用对比度自适应的阈值，不用固定 Canny 阈值。

    真实相衬图的壁面梯度只有几个灰阶；固定的 Canny(40,110) 在这种图上完全没有输出
    （实测 24 张停泵标定帧上非零边缘比例为 0）。因此这里用 Sobel 幅值 + 稳健噪声尺度
    （1.4826×MAD）作阈值，并要求梯度方向**横跨**候选线，而不是沿线。
    """

    gx: np.ndarray = field(repr=False, compare=False)
    gy: np.ndarray = field(repr=False, compare=False)
    magnitude: np.ndarray = field(repr=False, compare=False)
    sigma: float

    def support(self, candidate: "WallCandidate", *, sigma_multiple: float,
                min_absolute_gradient: float, perpendicularity: float = 0.6) -> tuple[float, float]:
        """返回 (命中比例, 沿线平均法向梯度幅值)。"""
        height, width = self.magnitude.shape[:2]
        count = max(8, int(round(candidate.length_px / 2.0)))
        xs = np.linspace(candidate.x1, candidate.x2, count)
        ys = np.linspace(candidate.y1, candidate.y2, count)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if int(valid.sum()) < 4:
            return 0.0, 0.0
        xi = xs[valid].astype(np.int32)
        yi = ys[valid].astype(np.int32)
        magnitude = self.magnitude[yi, xi]
        threshold = max(float(min_absolute_gradient), sigma_multiple * float(self.sigma))
        normal_x, normal_y = candidate.normal
        projected = np.abs(self.gx[yi, xi] * normal_x + self.gy[yi, xi] * normal_y)
        perpendicular = projected / np.maximum(magnitude, 1e-6)
        hits = (magnitude >= threshold) & (perpendicular >= perpendicularity)
        return float(np.mean(hits)), float(np.mean(projected))


def gradient_support_map(gray: np.ndarray) -> EdgeSupportMap:
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    median = float(np.median(magnitude))
    sigma = max(1e-6, 1.4826 * float(np.median(np.abs(magnitude - median))))
    return EdgeSupportMap(gx=gx, gy=gy, magnitude=magnitude, sigma=sigma)


def _edge_support(edges: EdgeSupportMap, candidate: WallCandidate,
                  limits: "LocalizationThresholds") -> tuple[float, float]:
    return edges.support(candidate, sigma_multiple=limits.edge_support_sigma_multiple,
                         min_absolute_gradient=limits.edge_support_min_gradient)


def _credible_edge(edges: EdgeSupportMap, candidate: WallCandidate, limits: LocalizationThresholds,
                   support: float, mean_gradient: float) -> bool:
    """命中比例与平均幅度都要过线：只看比例会被局部噪声刷高。"""
    if support < limits.min_edge_support:
        return False
    required = max(limits.edge_support_min_gradient,
                   limits.min_mean_gradient_sigma_multiple * float(edges.sigma))
    return bool(mean_gradient >= required)


def _supplementary_candidates(gray: np.ndarray, *, thresholds: LocalizationThresholds,
                              edges: EdgeSupportMap, max_lines: int) -> list[WallCandidate]:
    """在对比度归一化的边缘图上再跑一次长边检测，补上固定 Canny 阈值漏掉的候选。

    为什么需要第二路：``channel_calibration.detect_wall_line_candidates`` 用的是固定的
    Canny(35,100)。真实相衬图里管壁与带内界面的梯度只有几个到十几个灰阶，中等对比的
    长边会随噪声时有时无（同一条带内亮边在三帧里只被检出一次）。这里先把灰度做
    min-max 拉伸再取边缘，使检测对绝对对比度不敏感；端点与筛选条件与原路一致。
    """
    height, width = gray.shape[:2]
    parameters = dict(DEFAULT_SUPPLEMENTARY_HOUGH)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
    edges_map = cv2.Canny(blurred, int(parameters["canny_low"]), int(parameters["canny_high"]))
    raw = cv2.HoughLinesP(
        edges_map, 1, np.pi / 360.0,
        threshold=max(24, int(width * parameters["hough_width_ratio"])),
        minLineLength=max(10, int(width * parameters["min_line_length_ratio"])),
        maxLineGap=max(0, int(width * parameters["max_line_gap_ratio"])),
    )
    if raw is None:
        return []
    found: list[WallCandidate] = []
    for index, (x1, y1, x2, y2) in enumerate(np.asarray(raw).reshape(-1, 4)):
        dx = float(x2 - x1)
        if abs(dx) < 1.0:
            continue
        slope = float(y2 - y1) / dx
        if abs(slope) > thresholds.max_slope_for_any_candidate:
            continue
        length = math.hypot(dx, float(y2 - y1))
        if length / float(width) < thresholds.min_visible_length_ratio:
            continue
        seed = WallCandidate(id=1000 + index, x1=float(x1), y1=float(y1), x2=float(x2),
                             y2=float(y2), slope=slope, length_px=length,
                             length_ratio=length / float(width), edge_support=0.0)
        support, mean_gradient = _edge_support(edges, seed, thresholds)
        if not _credible_edge(edges, seed, thresholds, support, mean_gradient):
            continue
        found.append(WallCandidate(
            id=seed.id, x1=seed.x1, y1=seed.y1, x2=seed.x2, y2=seed.y2, slope=seed.slope,
            length_px=seed.length_px, length_ratio=seed.length_ratio, edge_support=support,
            mean_normal_gradient=mean_gradient))
    found.sort(key=lambda candidate: candidate.length_px, reverse=True)
    return found[:max_lines]


DEFAULT_SUPPLEMENTARY_HOUGH: dict[str, float] = {
    "canny_low": 35.0,
    "canny_high": 100.0,
    "hough_width_ratio": 0.10,
    "min_line_length_ratio": 0.55,
    "max_line_gap_ratio": 0.16,
}


def _merge_candidates(primary: list[WallCandidate], secondary: list[WallCandidate],
                      *, merge_distance_px: float = 4.0,
                      merge_slope: float = 0.03) -> list[WallCandidate]:
    """合并两路候选并按位置去重（先保留主路的 id，便于与既有工具对照）。"""
    merged: list[WallCandidate] = []
    for candidate in list(primary) + list(secondary):
        duplicate = False
        for kept in merged:
            center_x = 0.5 * (candidate.x1 + candidate.x2)
            if abs(candidate.y_at(center_x) - kept.y_at(center_x)) < merge_distance_px \
                    and abs(candidate.slope - kept.slope) < merge_slope:
                duplicate = True
                break
        if not duplicate:
            merged.append(candidate)
    merged.sort(key=lambda candidate: candidate.length_px, reverse=True)
    for position, candidate in enumerate(merged, start=1):
        merged[position - 1] = WallCandidate(
            id=position, x1=candidate.x1, y1=candidate.y1, x2=candidate.x2, y2=candidate.y2,
            slope=candidate.slope, length_px=candidate.length_px,
            length_ratio=candidate.length_ratio, edge_support=candidate.edge_support,
            mean_normal_gradient=candidate.mean_normal_gradient)
    return merged


def prepare_frame(frame: np.ndarray, *, thresholds: LocalizationThresholds | None = None,
                  work_scale: float = 1.0, max_lines: int = 32,
                  contrast_enhance: bool = False) -> FrameCandidates:
    """生成全幅长边候选（端点映回原图），并返回配套的边缘图与坐标映射。"""
    from .channel_calibration import detect_wall_line_candidates

    limits = thresholds or LocalizationThresholds()
    if frame is None or getattr(frame, "size", 0) == 0:
        return FrameCandidates((), gradient_support_map(np.zeros((8, 8), np.uint8)),
                               np.zeros((0, 0), np.uint8),
                               {"invertible": False, "reason": "frame_empty"})
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    original_height, original_width = gray.shape[:2]
    if not (work_scale > 0.0 and math.isfinite(work_scale)):
        raise ValueError("work_scale 必须是正的有限值")
    if abs(work_scale - 1.0) < 1e-9:
        work = gray
    else:
        work = cv2.resize(gray, (max(8, int(round(original_width * work_scale))),
                                 max(8, int(round(original_height * work_scale)))),
                          interpolation=cv2.INTER_AREA)
    work_height, work_width = work.shape[:2]
    # CLAHE is used only to propose edges. Motion and coverage still inspect the
    # original gray frame, so enhancement cannot invent fluid movement.
    candidate_image = (cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(work)
                       if contrast_enhance else work)
    edge_image = (cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
                  if contrast_enhance else gray)
    mapping = {
        "work_scale": float(work_scale),
        "work_shape": [int(work_width), int(work_height)],
        "original_shape": [int(original_width), int(original_height)],
        "offset_px": [0.0, 0.0],
        "invertible": True,
        "forward": "x_work = x_original * work_scale + offset_px[0]",
        "inverse": "x_original = x_work / work_scale - offset_px[0] / work_scale",
    }
    # 边缘支持在原图分辨率上量：候选端点已经是原图坐标，两者必须同系。
    edges = gradient_support_map(edge_image)
    scale_x = original_width / float(work_width)
    scale_y = original_height / float(work_height)

    candidates: list[WallCandidate] = []
    for raw in detect_wall_line_candidates(candidate_image, max_lines=max_lines):
        if abs(float(raw["slope"])) > limits.max_slope_for_any_candidate:
            continue
        if float(raw["length_ratio"]) < limits.min_visible_length_ratio:
            continue
        x1 = float(raw["x1"]) * work_width * scale_x
        y1 = float(raw["y1"]) * work_height * scale_y
        x2 = float(raw["x2"]) * work_width * scale_x
        y2 = float(raw["y2"]) * work_height * scale_y
        seed = WallCandidate(
            id=int(raw["id"]), x1=x1, y1=y1, x2=x2, y2=y2, slope=float(raw["slope"]),
            length_px=math.hypot(x2 - x1, y2 - y1), length_ratio=float(raw["length_ratio"]),
            edge_support=0.0)
        support, mean_gradient = _edge_support(edges, seed, limits)
        if not _credible_edge(edges, seed, limits, support, mean_gradient):
            continue
        candidates.append(WallCandidate(
            id=seed.id, x1=seed.x1, y1=seed.y1, x2=seed.x2, y2=seed.y2, slope=seed.slope,
            length_px=seed.length_px, length_ratio=seed.length_ratio, edge_support=support,
            mean_normal_gradient=mean_gradient))
    if limits.use_supplementary_candidates:
        candidates = _merge_candidates(
            candidates,
            _supplementary_candidates(edge_image, thresholds=limits, edges=edges, max_lines=max_lines))
    mapping["candidate_count"] = len(candidates)
    mapping["gradient_noise_sigma"] = round(float(edges.sigma), 6)
    mapping["supplementary_candidates_enabled"] = bool(limits.use_supplementary_candidates)
    mapping["contrast_enhance"] = bool(contrast_enhance)
    return FrameCandidates(tuple(candidates), edges, gray, mapping)


def _separation_profile(first: WallCandidate, second: WallCandidate,
                        common: tuple[float, float], count: int) -> np.ndarray:
    """在共见范围内按同一轴向坐标取两点，算法向距离。"""
    positions = np.linspace(common[0], common[1], max(4, count))
    normal = first.normal
    return np.asarray([abs(float(normal @ (second.point_at(x) - first.point_at(x))))
                       for x in positions], dtype=np.float64)


def _segment_support(first: WallCandidate, second: WallCandidate, edges: EdgeSupportMap,
                     common: tuple[float, float], segments: int, limits: LocalizationThresholds
                     ) -> tuple[bool, ...]:
    """沿共见范围分段，检查两壁在每一段都有边缘支持。"""
    bounds = np.linspace(common[0], common[1], segments + 1)
    flags = []
    for index in range(segments):
        left, right = float(bounds[index]), float(bounds[index + 1])
        if right - left < 6.0:
            flags.append(False)
            continue
        first_strip = first.with_endpoints(left, first.y_at(left), right, first.y_at(right))
        second_strip = second.with_endpoints(left, second.y_at(left), right, second.y_at(right))
        first_support, _ = _edge_support(edges, first_strip, limits)
        second_support, _ = _edge_support(edges, second_strip, limits)
        flags.append(bool(first_support >= limits.segment_min_edge_support
                          and second_support >= limits.segment_min_edge_support))
    return tuple(flags)


def pair_candidates(frame_candidates: FrameCandidates, frame_shape: tuple[int, int],
                    *, thresholds: LocalizationThresholds | None = None) -> list[WallPair]:
    """枚举候选对，按法向间距、共见范围、平行度与沿程一致性给出可用线对。"""
    limits = thresholds or LocalizationThresholds()
    height, width = frame_shape
    candidates = list(frame_candidates.candidates)
    edges = frame_candidates.edges
    if len(candidates) < 2:
        return []

    pairs: list[WallPair] = []
    for index, first in enumerate(candidates):
        for second in candidates[index + 1:]:
            slope_difference = abs(first.slope - second.slope)
            low = max(first.x_range[0], second.x_range[0])
            high = min(first.x_range[1], second.x_range[1])
            if slope_difference > limits.max_slope_difference:
                pairs.append(WallPair(first, second, (low, high), 0.0, 0.0, 0.0,
                                      slope_difference, 0.0, 0.0, (), 0.0,
                                      rejection="not_parallel"))
                continue
            if high - low < max(limits.min_common_range_px,
                                limits.min_common_range_ratio * width):
                pairs.append(WallPair(first, second, (low, high), 0.0, 0.0, 0.0,
                                      slope_difference, 0.0, 0.0, (), 0.0,
                                      rejection="insufficient_common_range"))
                continue
            profile = _separation_profile(first, second, (low, high),
                                          limits.separation_sample_count)
            separation = float(np.mean(profile))
            deviation = float(np.std(profile))
            cv_value = deviation / max(separation, 1e-6)
            if separation < limits.min_separation_px:
                rejection = "separation_too_small"
            elif separation > limits.max_separation_ratio * height:
                rejection = "separation_too_large"
            elif cv_value > limits.max_separation_cv:
                rejection = "separation_inconsistent"
            else:
                rejection = ""
            support = _segment_support(first, second, edges, (low, high),
                                       limits.segment_count, limits)
            mid_x = 0.5 * (low + high)
            mid_y = 0.5 * (first.y_at(mid_x) + second.y_at(mid_x))
            score = 0.0
            if not rejection:
                score = (min(first.length_ratio, second.length_ratio)
                         * (1.0 - min(1.0, cv_value / max(limits.max_separation_cv, 1e-6)))
                         * (1.0 - min(1.0, slope_difference / max(limits.max_slope_difference, 1e-6)))
                         * min(first.edge_support, second.edge_support))
                if sum(1 for flag in support if flag) < limits.min_supported_segments:
                    rejection = "insufficient_along_track_support"
                    score = 0.0
            pairs.append(WallPair(
                first=first, second=second, common_x_range=(low, high),
                separation_px=separation, separation_std_px=deviation, separation_cv=cv_value,
                slope_difference=slope_difference,
                tilt_ratio=0.5 * (first.slope + second.slope), mid_y_px=mid_y,
                segment_support=support, score=score, rejection=rejection))
    return pairs


def interior_parallel_competitors(pair: WallPair, candidates: list[WallCandidate], width: int,
                                  *, thresholds: LocalizationThresholds | None = None
                                  ) -> list[WallCandidate]:
    """带内与壁面同向、长度相当的长边。它们可能是液柱边缘，不是内壁。"""
    limits = thresholds or LocalizationThresholds()
    low, high = pair.common_x_range
    mid_x = 0.5 * (low + high)
    first_y = pair.first.y_at(mid_x)
    second_y = pair.second.y_at(mid_x)
    top, bottom = min(first_y, second_y), max(first_y, second_y)
    shortest = min(pair.first.length_px, pair.second.length_px)
    found = []
    for candidate in candidates:
        if candidate.id in pair.ids:
            continue
        if abs(candidate.slope - pair.tilt_ratio) > limits.max_slope_difference:
            continue
        if candidate.length_px < limits.interior_competitor_length_ratio * shortest:
            continue
        overlap = min(candidate.x_range[1], high) - max(candidate.x_range[0], low)
        if overlap < limits.min_common_range_ratio * width:
            continue
        y = candidate.y_at(0.5 * (max(candidate.x_range[0], low) + min(candidate.x_range[1], high)))
        if top + 1.0 < y < bottom - 1.0:
            found.append(candidate)
    return found


def assess_coverage(gray: np.ndarray, pair: WallPair, *,
                    thresholds: LocalizationThresholds | None = None) -> dict:
    """沿测量段多点核验：两壁在画幅内、带内连续、且该位置有管内对比证据。"""
    limits = thresholds or LocalizationThresholds()
    height, width = gray.shape[:2]
    low, high = pair.common_x_range
    if high - low < 8.0:
        return {"ok": False, "reason": "measurement_segment_too_short", "positions": []}
    rows = []
    for x in np.linspace(low, high, limits.segment_count):
        first_y = pair.first.y_at(x)
        second_y = pair.second.y_at(x)
        top, bottom = min(first_y, second_y), max(first_y, second_y)
        inside = bool(limits.coverage_edge_margin_px <= top
                      and bottom <= height - 1 - limits.coverage_edge_margin_px
                      and 0 <= x <= width - 1)
        column = int(round(x))
        column = min(max(column, 0), width - 1)
        band = gray[max(0, int(round(top))):max(1, int(round(bottom))), column]
        band_mean = float(np.mean(band)) if band.size else float("nan")
        outside_parts = [
            gray[max(0, int(round(top)) - 8):max(1, int(round(top)) - 3), column],
            gray[min(height, int(round(bottom)) + 3):min(height, int(round(bottom)) + 8), column],
        ]
        outside_values = np.concatenate([part for part in outside_parts if part.size]) \
            if any(part.size for part in outside_parts) else np.zeros(0)
        outside_mean = float(np.mean(outside_values)) if outside_values.size else float("nan")
        contrast = abs(band_mean - outside_mean) if np.isfinite(outside_mean) else 0.0
        rows.append({
            "x_px": round(float(x), 3),
            "upper_y_px": round(float(top), 3),
            "lower_y_px": round(float(bottom), 3),
            "band_inside_frame": inside,
            "band_mean_gray": None if not np.isfinite(band_mean) else round(band_mean, 3),
            "outside_mean_gray": None if not np.isfinite(outside_mean) else round(outside_mean, 3),
            "interior_contrast": round(float(contrast), 3),
            "has_interior_evidence": bool(contrast >= limits.coverage_min_contrast),
        })
    covered = sum(1 for row in rows if row["band_inside_frame"] and row["has_interior_evidence"])
    return {
        "ok": bool(covered == len(rows)),
        "reason": "" if covered == len(rows) else "along_track_coverage_incomplete",
        "positions_checked": len(rows),
        "positions_covered": int(covered),
        "positions": rows,
    }


def _band_mask(shape: tuple[int, int], pair: WallPair) -> np.ndarray:
    """带内像素掩码（用两壁方程，不是外接矩形）。"""
    height, width = shape
    low, high = pair.common_x_range
    xs = np.arange(width)
    upper = np.array([pair.first.y_at(x) for x in xs])
    lower = np.array([pair.second.y_at(x) for x in xs])
    top = np.minimum(upper, lower)
    bottom = np.maximum(upper, lower)
    rows = np.arange(height)[:, None]
    inside = (rows >= top[None, :]) & (rows <= bottom[None, :])
    inside[:, :max(0, int(low))] = False
    inside[:, min(width, int(high) + 1):] = False
    return inside


def _photometric_correction(previous: np.ndarray, current: np.ndarray,
                            background: np.ndarray) -> tuple[float, float, float]:
    """用**管外背景**估计两帧之间的增益与偏置，返回 (gain, offset, 残差)。

    照明漂移、曝光变化与增益变化对全画幅是共同的，用带外的同一区域拟合就能把它们
    从带内差异中解掉；只用带内像素无法区分“整体变亮”和“内容移动”。
    """
    before = previous[background].astype(np.float64)
    after = current[background].astype(np.float64)
    if before.size < 32 or after.size != before.size:
        return 1.0, 0.0, float("inf")
    center_before = float(np.median(before))
    center_after = float(np.median(after))
    # 稳健尺度不能只用 MAD：量化后的背景（例如只有 19/20 两个取值）MAD 会是 0，
    # 那样按中心±2·MAD 取子集会把样本塌成单一取值，比值退化。取几个估计量的上界。
    mad = float(np.median(np.abs(before - center_before))) * 1.4826
    quartile = (float(np.percentile(before, 75)) - float(np.percentile(before, 25))) / 1.3490
    spread = max(mad, quartile, 0.5)
    keep = np.abs(before - center_before) <= 2.0 * spread
    if int(keep.sum()) < 16:
        keep = np.ones_like(before, dtype=bool)
    # 稳健的斜率：用四分位差之比，避免个别亮点拉偏
    before_q = float(np.percentile(before[keep], 75) - np.percentile(before[keep], 25))
    after_q = float(np.percentile(after[keep], 75) - np.percentile(after[keep], 25))
    gain = 1.0 if before_q < 0.5 else max(0.2, min(5.0, after_q / before_q))
    offset = center_after - gain * center_before
    corrected = gain * before + offset
    residual = float(np.median(np.abs(after - corrected)))
    return gain, offset, residual


def _axial_profiles(image: np.ndarray, pair: WallPair, samples: int) -> dict[str, np.ndarray]:
    """带内沿轴向的两条签名。

    只看一条会漏：柱塞与载液只差灰度时（平坦柱塞）签名藏在**列均值**里；柱塞靠弯月面
    亮边成像时（相衬）签名藏在**横向对比度**里。两条都算，调用方按轴向结构强弱择一。
    """
    low, high = pair.common_x_range
    positions = np.linspace(low, high, max(8, samples)).astype(int)
    positions = positions[(positions >= 0) & (positions < image.shape[1])]
    means, contrasts = [], []
    for x in positions:
        top = int(round(min(pair.first.y_at(x), pair.second.y_at(x))))
        bottom = int(round(max(pair.first.y_at(x), pair.second.y_at(x))))
        inner, outer = max(0, top + 2), min(image.shape[0], bottom - 1)
        if outer - inner < 4:
            means.append(np.nan)
            contrasts.append(np.nan)
            continue
        column = image[inner:outer, x].astype(np.float64)
        means.append(float(np.median(column)))
        contrasts.append(float(np.percentile(column, 95) - np.percentile(column, 5)))
    return {"column_mean": np.asarray(means, dtype=np.float64),
            "transverse_contrast": np.asarray(contrasts, dtype=np.float64)}


def _axial_structure(profile: np.ndarray) -> float:
    """沿轴结构的强弱：去掉端点效应后的 90-10 分位差。"""
    values = profile[np.isfinite(profile)]
    if values.size < 8:
        return 0.0
    return float(np.percentile(values, 90) - np.percentile(values, 10))


def _best_axial_shift(before: np.ndarray, after: np.ndarray, maximum: int) -> tuple[int, float]:
    """沿轴互相关求位移与峰值相关度；返回 (位移像素, 相关度)。

    符号约定：返回值 shift 表示 ``after[i] ≈ before[i + shift]``。判据只用 |shift|。
    """
    if before.size < 8 or after.size != before.size:
        return 0, 0.0
    left = before - np.nanmean(before)
    right = after - np.nanmean(after)
    left = np.nan_to_num(left)
    right = np.nan_to_num(right)
    norm = float(np.linalg.norm(left) * np.linalg.norm(right))
    if norm <= 1e-9:
        return 0, 0.0
    best = (0, -1.0)
    bound = max(1, min(int(maximum), left.size // 3))
    for shift in range(-bound, bound + 1):
        if shift >= 0:
            a, b = left[shift:], right[:right.size - shift] if shift else right
        else:
            a, b = left[:left.size + shift], right[-shift:]
        if a.size < 4:
            continue
        value = float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        if value > best[1]:
            best = (shift, value)
    return best


def assess_motion(previous_gray: np.ndarray | None, gray: np.ndarray, pair: WallPair,
                  *, previous_pair: WallPair | None = None,
                  thresholds: LocalizationThresholds | None = None) -> dict:
    """管内是否有**位移**证据。静止**不**构成真管壁的充分条件。

    判据分三步，缺一不可：

    1. **用管外背景去掉照明/曝光/增益变化**：对带外同一区域拟合增益与偏置，把它从带内
       差异中解掉。整体变亮、全局闪烁、局部光斑在解掉之后不留下带内差异。
    2. **管内变化要超过壁带变化**：壁带跟着变说明该线在动或光照在变，不能算“静止结构 + 流动内容”。
    3. **要能测到沿轴位移**：对带内轴向剖面做互相关，要求峰值位移 ≥ 阈值且相关度足够。
       亮度漂移、闪烁、静态噪声都不产生位移；只有空间位移才可能是流动。

    停泵液柱、静止气泡、固定纹理仍然判为无运动证据（位移为 0）。
    """
    limits = thresholds or LocalizationThresholds()
    if previous_gray is None or previous_gray.shape != gray.shape:
        return {"available": False, "reason": "no_previous_frame",
                "interior_change": None, "wall_change": None, "geometry_change": None}
    band = _band_mask(gray.shape[:2], pair)
    background = ~band
    if int(background.sum()) < 64 or int(band.sum()) < 64:
        return {"available": False, "reason": "insufficient_pixels",
                "interior_change": None, "wall_change": None, "geometry_change": None}

    gain, offset, residual = _photometric_correction(previous_gray, gray, background)
    corrected_previous = np.clip(gain * previous_gray.astype(np.float64) + offset, 0, 255)

    difference = np.abs(gray.astype(np.float64) - corrected_previous)
    interior_change = float(np.mean(difference[band]))
    background_change = float(np.mean(difference[background]))
    noise_median = float(np.median(difference[background]))
    noise_sigma = max(limits.motion_noise_floor_gray,
                      1.4826 * float(np.median(np.abs(difference[background] - noise_median))))
    significant = float(limits.motion_change_sigma_multiple) * noise_sigma
    changed_fraction = float(np.mean(difference[band] > significant))

    wall_values_before, wall_values_after = [], []
    for x in np.linspace(*pair.common_x_range, max(4, limits.separation_sample_count)).astype(int):
        for y in (pair.first.y_at(x), pair.second.y_at(x)):
            row = max(0, int(round(y)) - 1)
            strip_after = gray[row:int(round(y)) + 2, x]
            strip_before = corrected_previous[row:int(round(y)) + 2, x]
            if strip_after.size and strip_before.size == strip_after.size:
                wall_values_after.append(float(np.mean(strip_after)))
                wall_values_before.append(float(np.mean(strip_before)))
    wall_change = (float(np.mean(np.abs(np.asarray(wall_values_after)
                                        - np.asarray(wall_values_before))))
                   if wall_values_after else None)

    profiles_before = _axial_profiles(corrected_previous, pair, limits.separation_sample_count * 4)
    profiles_after = _axial_profiles(gray, pair, limits.separation_sample_count * 4)
    structure_before = {name: _axial_structure(values)
                        for name, values in profiles_before.items()}
    structure_after = {name: _axial_structure(values) for name, values in profiles_after.items()}
    chosen_signature = max(structure_before,
                           key=lambda name: min(structure_before[name], structure_after[name]))
    axial_structure = min(structure_before[chosen_signature], structure_after[chosen_signature])
    maximum_shift = max(2, int(round(0.25 * profiles_after[chosen_signature].size)))
    axial_shift, shift_correlation = _best_axial_shift(
        profiles_before[chosen_signature], profiles_after[chosen_signature], maximum_shift)

    geometry_change = None
    stable = None
    if previous_pair is not None:
        separation_change = abs(pair.separation_px - previous_pair.separation_px)
        mid_change = abs(pair.mid_y_px - previous_pair.mid_y_px)
        tilt_change = abs(pair.tilt_ratio - previous_pair.tilt_ratio)
        geometry_change = {
            "separation_px": round(float(separation_change), 4),
            "mid_y_px": round(float(mid_change), 4),
            "tilt_ratio": round(float(tilt_change), 6),
        }
        stable = bool(separation_change <= limits.motion_max_geometry_change_px
                      and mid_change <= limits.motion_max_geometry_change_px
                      and tilt_change <= limits.motion_max_tilt_change)

    checks = {
        "photometric_residual_small": bool(
            residual <= max(limits.motion_noise_floor_gray, significant)),
        "interior_change_sufficient": bool(interior_change >= limits.motion_min_interior_change),
        "axial_structure_sufficient": bool(axial_structure >= limits.motion_min_axial_structure_gray),
        "axial_shift_measured": bool(abs(axial_shift) >= limits.motion_min_axial_shift_px
                                     and shift_correlation >= limits.motion_min_shift_correlation),
        "line_pair_stable": stable,
    }
    if geometry_change is None:
        reason = "no_previous_pair"
    elif not stable:
        reason = "line_pair_not_stable"
    elif not checks["interior_change_sufficient"]:
        reason = "no_interior_change"
    elif not checks["axial_structure_sufficient"]:
        reason = "no_axial_structure"
    elif not checks["axial_shift_measured"]:
        reason = "no_axial_displacement"
    else:
        reason = ""
    available = bool(reason == "" and all(value for value in checks.values() if value is not None))
    interpretation = {
        "no_previous_pair": "没有上一帧的线对，无法判断位移",
        "line_pair_not_stable": "线对本身在移动，可能是随流体移动的界面而不是固定壁面",
        "no_interior_change": "去掉照明/增益变化后管内没有可分辨的变化，无法证明有流动",
        "no_axial_structure": "管内没有沿轴的结构可供配准，无法测位移",
        "no_axial_displacement": "管内变化没有对应的沿轴位移，更像亮度漂移或闪烁而不是流动",
        "": "去掉照明变化后管内出现一致的沿轴位移，且线对位置稳定",
    }[reason]
    return {
        "available": available,
        "reason": reason,
        "interior_change": round(interior_change, 4),
        "background_change": round(background_change, 4),
        # 壁带变化只作诊断：真实器件里液柱贴壁，壁带本身就叠着液柱亮边，
        # 因此它不能当作“干净参照”，光照抑制由管外背景的光度校正负责。
        "wall_change_diagnostic": None if wall_change is None else round(wall_change, 4),
        "photometric_gain": round(float(gain), 5),
        "photometric_offset": round(float(offset), 5),
        "photometric_residual": round(float(residual), 4),
        "frame_noise_sigma": round(noise_sigma, 4),
        "significant_change_threshold": round(significant, 4),
        "changed_fraction": round(changed_fraction, 4),
        "axial_signature": chosen_signature,
        "axial_structure_gray": round(float(axial_structure), 4),
        "axial_structure_candidates": {name: round(float(value), 4)
                                      for name, value in structure_after.items()},
        "axial_shift_px": int(axial_shift),
        "shift_correlation": round(float(shift_correlation), 4),
        "checks": checks,
        "geometry_change": geometry_change,
        "line_pair_stable": stable,
        "interpretation": interpretation,
    }


def _signature(gray: np.ndarray) -> str:
    small = cv2.resize(gray, (32, 24), interpolation=cv2.INTER_AREA).astype(np.uint8)
    return hashlib.sha1(np.ascontiguousarray(small).tobytes()).hexdigest()[:16]


class ParallelWallLocalizer:
    """有界、无阻塞的跨帧平行管壁定位器。只做计算，不等待、不排队。"""

    def __init__(self, *, thresholds: LocalizationThresholds | None = None,
                 work_scale: float = 1.0, max_lines: int = 32,
                 contrast_enhance: bool = False) -> None:
        self._limits = thresholds or LocalizationThresholds()
        self._work_scale = float(work_scale)
        self._max_lines = int(max_lines)
        self._contrast_enhance = bool(contrast_enhance)
        self._observations: deque[dict] = deque(maxlen=self._limits.max_buffer_frames)
        self._signatures: deque[str] = deque(maxlen=self._limits.max_buffer_frames)
        self._last_gray: np.ndarray | None = None
        self._last_pair: WallPair | None = None
        self._last_interior: list[dict] = []
        # 只保护内部状态的短临界区：不等待其它线程、不排队、不在持锁期间做 I/O。
        self._lock = threading.RLock()

    @property
    def thresholds(self) -> LocalizationThresholds:
        return self._limits

    def observe(self, frame: np.ndarray, *, frame_id: int, capture_monotonic: float,
                region=None) -> dict:
        """加入一帧观测，返回该帧的候选、选择与逐帧证据。线程安全，不阻塞等待。"""
        with self._lock:
            return self._observe_locked(frame, frame_id=frame_id,
                                        capture_monotonic=capture_monotonic, region=region)

    def _observe_locked(self, frame: np.ndarray, *, frame_id: int, capture_monotonic: float,
                        region=None) -> dict:
        prepared = prepare_frame(frame, thresholds=self._limits, work_scale=self._work_scale,
                                max_lines=self._max_lines,
                                contrast_enhance=self._contrast_enhance)
        gray = prepared.gray
        if gray.size == 0:
            return {"frame_id": int(frame_id), "capture_monotonic": float(capture_monotonic),
                    "rejected": "frame_empty"}
        height, width = gray.shape[:2]
        pairs = pair_candidates(prepared, (height, width), thresholds=self._limits)
        usable = sorted((pair for pair in pairs if not pair.rejection and pair.score > 0.0),
                        key=lambda pair: pair.score, reverse=True)
        selected = usable[0] if usable else None
        interior = (interior_parallel_competitors(selected, list(prepared.candidates), width,
                                                  thresholds=self._limits)
                    if selected else [])
        # 带内同向长边只有在**静止**时才无法与壁面区分；随流体移动的是液柱边缘。
        # 判据必须同时看位置与轴向可见区间：柱塞肩部的长边在 y 上不动，却会沿轴平移。
        band_mid_x = (0.5 * (selected.common_x_range[0] + selected.common_x_range[1])
                      if selected else 0.5 * width)
        interior_detail = []
        for candidate in interior:
            y_now = float(candidate.y_at(band_mid_x))
            matched = False
            for previous in self._last_interior:
                same_height = abs(previous["y_at_band_mid"] - y_now) \
                    <= self._limits.motion_max_geometry_change_px
                same_slope = abs(previous["slope"] - candidate.slope) \
                    <= self._limits.motion_max_tilt_change
                same_span = (abs(previous["x1"] - candidate.x1)
                             <= self._limits.motion_max_axial_change_px
                             and abs(previous["x2"] - candidate.x2)
                             <= self._limits.motion_max_axial_change_px)
                if same_height and same_slope and same_span:
                    matched = True
                    break
            interior_detail.append({
                "id": int(candidate.id),
                "y_at_band_mid": y_now,
                "slope": float(candidate.slope),
                "x1": float(candidate.x1),
                "x2": float(candidate.x2),
                "edge_support": float(candidate.edge_support),
                # 没有上一帧时是“未知”，不能当成“静止”——否则静/动判据会被第一帧一票否决。
                "static_across_frames": True if matched else (None if not self._last_interior else False),
            })
        self._last_interior = [dict(item) for item in interior_detail]
        motion = (assess_motion(self._last_gray, gray, selected, previous_pair=self._last_pair,
                               thresholds=self._limits)
                  if selected else {"available": False, "reason": "no_selected_pair",
                                    "interior_change": None, "wall_change": None,
                                    "geometry_change": None})
        coverage = assess_coverage(gray, selected, thresholds=self._limits) if selected else None
        signature = _signature(gray)
        region_hint = None
        if region is not None and getattr(region, "wall_lines", None):
            region_hint = {"status": getattr(region, "status", ""),
                           "confidence": float(getattr(region, "confidence", 0.0) or 0.0),
                           "wall_lines": [dict(line) for line in region.wall_lines],
                           "used_as": "recorded_only_not_used_for_decision",
                           "why": "区域统计聚合的是高频区域，不证明区域内亮边就是内壁，"
                                  "因此只记录不作判据"}

        payload = {
            "frame_id": int(frame_id),
            "capture_monotonic": float(capture_monotonic),
            "image_shape": [int(width), int(height)],
            "candidates": [candidate.to_dict() for candidate in prepared.candidates],
            "coordinate_mapping": prepared.coordinate_mapping,
            "selected_pair": None if selected is None else selected.to_dict(),
            "selected_wall_lines": None if selected is None else selected.wall_lines(width, height),
            "pair_scores": [pair.to_dict() for pair in pairs],
            "interior_competitors": interior_detail,
            "interior_competitor_ids": [item["id"] for item in interior_detail],
            "motion": motion,
            "coverage": coverage,
            "region_hint": region_hint,
            "signature": signature,
            "repeated_frame": signature in self._signatures,
        }
        self._observations.append(payload)
        self._signatures.append(signature)
        self._last_gray = gray
        self._last_pair = selected
        return {key: value for key, value in payload.items() if key != "signature"}

    def reset(self) -> None:
        with self._lock:
            self._observations.clear()
            self._signatures.clear()
            self._last_gray = None
            self._last_pair = None
            self._last_interior = []

    @property
    def buffered_frames(self) -> int:
        with self._lock:
            return len(self._observations)

    @property
    def thresholds(self) -> LocalizationThresholds:
        """本定位器**实际使用**的阈值对象。

        追溯需要序列化真正的生效参数；新建一个默认实例来「代表」它是不诚实的——
        调用方可以构造时传入自己的阈值。只读返回，不复制。
        """
        return self._limits

    # ------------------------------------------------------------------ 聚合

    def _fresh(self, reference_now: float) -> list[dict]:
        return [item for item in self._observations
                if reference_now - float(item["capture_monotonic"])
                <= self._limits.evidence_max_age_s]

    def _clusters(self, fresh: list[dict]) -> list[dict]:
        limits = self._limits
        clusters: list[dict] = []
        for item in fresh:
            pair = item["selected_pair"]
            if pair is None:
                continue
            target = None
            for cluster in clusters:
                separation_ratio = (abs(pair["separation_px"] - cluster["separation_median"])
                                    / max(cluster["separation_median"], 1e-6))
                if separation_ratio <= limits.cluster_separation_ratio \
                        and abs(pair["mid_y_px"] - cluster["mid_median"]) <= limits.cluster_mid_offset_px:
                    target = cluster
                    break
            if target is None:
                target = {"members": [], "separation_median": float(pair["separation_px"]),
                          "mid_median": float(pair["mid_y_px"])}
                clusters.append(target)
            target["members"].append(item)
            target["separation_median"] = float(np.median(
                [member["selected_pair"]["separation_px"] for member in target["members"]]))
            target["mid_median"] = float(np.median(
                [member["selected_pair"]["mid_y_px"] for member in target["members"]]))
        return clusters

    def localize(self, *, now_monotonic: float | None = None,
                 exclude_regions: list[dict] | None = None) -> "WallLocalization":
        """聚合当前缓冲，给出版本化定位结果或明确拒绝理由。"""
        with self._lock:
            return self._localize_locked(now_monotonic=now_monotonic,
                                        exclude_regions=exclude_regions)

    def _localize_locked(self, *, now_monotonic: float | None = None,
                         exclude_regions: list[dict] | None = None) -> "WallLocalization":
        limits = self._limits
        ordered = list(self._observations)
        if not ordered:
            return WallLocalization.rejected("no_frames", limits, self._work_scale)
        newest = float(ordered[-1]["capture_monotonic"])
        reference_now = newest if now_monotonic is None else float(now_monotonic)
        frame_id = int(ordered[-1]["frame_id"])
        shape = tuple(ordered[-1]["image_shape"])
        if reference_now - newest > limits.evidence_max_age_s:
            return WallLocalization.rejected("evidence_expired", limits, self._work_scale,
                                             frame_id=frame_id, image_shape=shape)
        fresh = self._fresh(reference_now)
        if not fresh:
            return WallLocalization.rejected("evidence_expired", limits, self._work_scale,
                                             frame_id=frame_id, image_shape=shape)

        with_pair = [item for item in fresh if item["selected_pair"] is not None]
        # 重复相同帧不增加独立支持数：按内容签名去重。
        independent = len({item["signature"] for item in with_pair if not item["repeated_frame"]})
        support = {
            "frames_examined": len(fresh),
            "frames_with_pair": len(with_pair),
            "independent_frames": int(independent),
            "frame_ids": sorted(int(item["frame_id"]) for item in with_pair),
            "time_range_monotonic": [round(float(fresh[0]["capture_monotonic"]), 4),
                                     round(float(fresh[-1]["capture_monotonic"]), 4)],
            "repeated_frame_count": sum(1 for item in fresh if item.get("repeated_frame")),
        }

        newest_observation = ordered[-1]
        current_frame = {
            "frame_id": int(newest_observation["frame_id"]),
            "capture_monotonic": round(float(newest_observation["capture_monotonic"]), 4),
            "has_pair": newest_observation["selected_pair"] is not None,
            "candidate_count": len(newest_observation["candidates"]),
            "pair_rejections": _rejection_histogram(newest_observation["pair_scores"]),
            "coverage_ok": (newest_observation["coverage"] or {}).get("ok"),
            "motion_available": newest_observation["motion"].get("available"),
            "motion_reason": newest_observation["motion"].get("reason"),
            "interior_competitor_ids": list(newest_observation["interior_competitor_ids"]),
        }
        if with_pair:
            support["time_range_monotonic"] = [
                round(float(min(item["capture_monotonic"] for item in with_pair)), 4),
                round(float(max(item["capture_monotonic"] for item in with_pair)), 4),
            ]
        if not with_pair:
            return WallLocalization.rejected(
                "no_credible_pair", limits, self._work_scale,
                frame_id=int(newest_observation["frame_id"]), image_shape=shape,
                support=support, observations=ordered, current_frame=current_frame)
        # 最新帧必须自己给出线对：历史几何可以作先验，但不能冒充“当前帧已定位”，
        # 否则下游“定位帧号 == 测量帧号”的检查会被绕过。
        if newest_observation["selected_pair"] is None:
            return WallLocalization.rejected(
                "no_current_frame_evidence", limits, self._work_scale,
                frame_id=int(newest_observation["frame_id"]), image_shape=shape,
                support=support, observations=ordered, current_frame=current_frame,
                detail="最新帧没有可用线对；历史几何只能作先验，不能作为当前帧的定位结果")

        clusters = self._clusters(fresh)
        current_cluster = next((cluster for cluster in clusters
                                if any(member is newest_observation
                                       for member in cluster["members"])), None)
        if current_cluster is None:
            return WallLocalization.rejected(
                "no_current_frame_evidence", limits, self._work_scale,
                frame_id=int(newest_observation["frame_id"]), image_shape=shape,
                support=support, observations=ordered, current_frame=current_frame,
                detail="最新帧的线对没有进入任何跨帧簇")
        clusters.sort(key=lambda cluster: (
            len({member.get("signature", member["frame_id"]) for member in cluster["members"]}),
            max(member["selected_pair"]["score"] for member in cluster["members"]),
        ), reverse=True)
        # 目标轨道固定为“含最新帧的那一簇”；其它簇只作为竞争者出现，
        # 这样“突然换到另一条通道”会走歧义而不是静默切换。
        best = current_cluster
        members = best["members"]
        best_pair = max((member["selected_pair"] for member in members),
                        key=lambda pair: pair["score"])
        separations = np.array([member["selected_pair"]["separation_px"] for member in members])
        mids = np.array([member["selected_pair"]["mid_y_px"] for member in members])
        tilts = np.array([member["selected_pair"]["tilt_ratio"] for member in members])
        interior_all_ids = sorted({int(item["id"]) for member in members
                                   for item in member.get("interior_competitors", [])})
        # 只有静止的带内长边才真的无法与壁面区分；随流体移动的是液柱边缘。
        interior_static_ids = sorted({int(item["id"]) for member in members
                                      for item in member.get("interior_competitors", [])
                                      if item.get("static_across_frames") is True})
        # 运动证据：最新帧自己必须给出位移证据，且簇内所有可评估帧一致。
        evaluable = [member["motion"] for member in members
                     if member["motion"].get("geometry_change") is not None]
        newest_motion = newest_observation["motion"]
        motion_available = bool(
            newest_motion.get("geometry_change") is not None
            and newest_motion.get("available")
            and evaluable
            and all(item["available"] for item in evaluable))
        # 沿程覆盖取**所有成员都要通过**：取“最好的那一帧”会漏掉只在部分帧出现的截断。
        coverages = [member["coverage"] for member in members if member["coverage"]]
        coverage = coverages[-1] if coverages else None
        coverage_all_ok = bool(coverages) and all(item["ok"] for item in coverages)
        if coverage is not None:
            coverage = dict(coverage)
            coverage["frames_evaluated"] = len(coverages)
            coverage["frames_ok"] = sum(1 for item in coverages if item["ok"])
        competition = _competition(clusters, limits, best)
        support.update({
            "cluster_count": len(clusters),
            "cluster_members": len(members),
            "separation_px": _distribution(separations),
            "mid_y_px": _distribution(mids),
            "tilt_ratio": _distribution(tilts),
            "appearance_ratio": round(len(members) / max(1, len(fresh)), 4),
            "supported_segments": int(best_pair["supported_segments"]),
            "region_hint": next((member["region_hint"] for member in reversed(members)
                                 if member["region_hint"]), None),
        })
        geometry = {
            "status": "",
            "reason": "",
            # frame_id / capture_monotonic 描述的是**提供几何的那一帧**，即最新帧；
            # 它在上面已被要求必须有线对并进入目标簇。
            "frame_id": int(newest_observation["frame_id"]),
            "capture_monotonic": round(float(newest_observation["capture_monotonic"]), 4),
            "evaluated_at_monotonic": round(float(reference_now), 4),
            "evidence_age_s": round(float(reference_now)
                                    - float(newest_observation["capture_monotonic"]), 4),
            "image_shape": list(shape),
            "coordinate_mapping": newest_observation["coordinate_mapping"],
            "measurement_segment_px": list(best_pair["common_x_range_px"]),
            "wall_lines": [dict(line) for line in (newest_observation["selected_wall_lines"] or [])],
            "separation_px": round(float(newest_observation["selected_pair"]["separation_px"]), 4),
            "tilt_ratio": round(float(newest_observation["selected_pair"]["tilt_ratio"]), 6),
            "mid_y_px": round(float(newest_observation["selected_pair"]["mid_y_px"]), 4),
            "track_separation_median_px": round(float(np.median(separations)), 4),
            "track_tilt_median": round(float(np.median(tilts)), 6),
            "candidate_ids": list(best_pair["candidate_ids"]),
            "pair_score": float(best_pair["score"]),
            "competition": competition,
            "current_frame": current_frame,
            "interior_parallel_competitor_ids": interior_all_ids,
            "static_interior_competitor_ids": interior_static_ids,
            "support": support,
            "motion": {"available": motion_available,
                       "current_frame": newest_motion,
                       "per_frame": [member["motion"] for member in members]},
            "coverage": coverage,
            "candidates": newest_observation["candidates"],
            "pair_rejections": _rejection_histogram(newest_observation["pair_scores"]),
            "thresholds": {item.name: getattr(limits, item.name) for item in fields(limits)},
            "threshold_provenance": dict(THRESHOLD_PROVENANCE),
            "geometry_version": GEOMETRY_VERSION,
            "exclude_regions": list(exclude_regions or []),
            "exclude_regions_declared": bool(exclude_regions),
            "generation_zone_exclusion": (
                "调用方未声明生成区位置，因此“下游直管段”只按共见范围与沿程支持约束"
                if not exclude_regions else "已按调用方声明的区域排除生成区"),
        }

        if competition["ambiguous"]:
            return WallLocalization.from_geometry(geometry, "ambiguous", competition["reason"])
        if interior_static_ids:
            detail = ("带内存在**静止**的同向长边，无法区分壁面与液柱边缘"
                      if not motion_available else
                      "带内存在静止的同向长边，即使管内有流动也需人工确认哪条是内壁")
            return WallLocalization.from_geometry(
                geometry, "ambiguous", f"{detail}；候选 id={interior_static_ids}")
        if int(independent) < limits.required_support_frames:
            return WallLocalization.from_geometry(
                geometry, "insufficient_support",
                f"独立支持帧 {int(independent)} < 要求 {limits.required_support_frames}")
        if not motion_available:
            current_motion_reason = newest_motion.get("reason") or "unknown"
            return WallLocalization.from_geometry(
                geometry, "pending_motion",
                "当前帧没有可判定的位移证据，无法证明这两条是内壁而非停泵液柱或固定纹理"
                f"（当前帧原因：{current_motion_reason}；"
                f"轴向位移 {newest_motion.get('axial_shift_px')} px，"
                f"相关度 {newest_motion.get('shift_correlation')}）")
        if coverage is None or not coverage_all_ok:
            return WallLocalization.from_geometry(geometry, "coverage_incomplete",
                                                  "沿测量段的覆盖核验未通过")
        return WallLocalization.from_geometry(geometry, "localized", "")


def _distribution(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"median": None, "mad": None, "min": None, "max": None}
    median = float(np.median(values))
    return {"median": round(median, 4),
            "mad": round(float(np.median(np.abs(values - median))), 4),
            "min": round(float(np.min(values)), 4),
            "max": round(float(np.max(values)), 4)}


def _competition(clusters: list[dict], limits: LocalizationThresholds, best: dict) -> dict:
    best_score = max(member["selected_pair"]["score"] for member in best["members"])
    rivals = []
    for cluster in clusters:
        if cluster is best:
            continue
        score = max(member["selected_pair"]["score"] for member in cluster["members"])
        separation_ratio = abs(cluster["separation_median"] - best["separation_median"]) \
            / max(best["separation_median"], 1e-6)
        mid_offset = abs(cluster["mid_median"] - best["mid_median"])
        similar = bool(score >= limits.competition_score_ratio * best_score
                       and (separation_ratio <= limits.competition_separation_ratio
                            or mid_offset <= limits.competition_mid_offset_px))
        rivals.append({
            "separation_median_px": round(float(cluster["separation_median"]), 3),
            "mid_median_px": round(float(cluster["mid_median"]), 3),
            "score": round(float(score), 4),
            "members": len(cluster["members"]),
            "treated_as_competing": similar,
        })
    competing = [rival for rival in rivals if rival["treated_as_competing"]]
    return {
        "ambiguous": bool(competing),
        "reason": "" if not competing else f"有 {len(competing)} 组同样可信的线对，无法唯一确定目标管道",
        "best_separation_px": round(float(best["separation_median"]), 3),
        "best_score": round(float(best_score), 4),
        "rivals": rivals,
    }


def _rejection_histogram(pair_scores: list[dict]) -> dict:
    histogram: dict[str, int] = {}
    for pair in pair_scores:
        if pair["rejection"]:
            histogram[pair["rejection"]] = histogram.get(pair["rejection"], 0) + 1
    return histogram


@dataclass(frozen=True)
class WallLocalization:
    """一次定位的完整结果。``status == "localized"`` 才允许生成扶正变换。"""

    status: str
    reason: str
    geometry: dict = field(default_factory=dict)

    @classmethod
    def from_geometry(cls, geometry: dict, status: str, reason: str) -> "WallLocalization":
        payload = dict(geometry)
        payload["status"] = status
        payload["reason"] = reason
        return cls(status=status, reason=reason, geometry=payload)

    @classmethod
    def rejected(cls, reason: str, limits: LocalizationThresholds, work_scale: float,
                 *, frame_id: int = 0, image_shape: tuple[int, int] = (0, 0),
                 support: dict | None = None,
                 observations: list[dict] | None = None,
                 current_frame: dict | None = None,
                 detail: str = "") -> "WallLocalization":
        geometry = {
            "status": "rejected",
            "reason": reason,
            "detail": detail,
            "frame_id": int(frame_id),
            "capture_monotonic": None,
            "image_shape": list(image_shape),
            "coordinate_mapping": {"work_scale": float(work_scale), "invertible": True},
            "measurement_segment_px": None,
            "wall_lines": [],
            "separation_px": None,
            "tilt_ratio": None,
            "mid_y_px": None,
            "candidate_ids": [],
            "pair_score": None,
            "competition": {"ambiguous": False, "rivals": []},
            "current_frame": current_frame,
            "interior_parallel_competitor_ids": [],
            "static_interior_competitor_ids": [],
            "support": support or {},
            "motion": {"available": False, "per_frame": []},
            "coverage": None,
            "candidates": (observations[-1]["candidates"] if observations
                           else (current_frame or {}).get("candidates", [])),
            "pair_rejections": (current_frame or {}).get("pair_rejections", {}),
            "thresholds": {item.name: getattr(limits, item.name) for item in fields(limits)},
            "threshold_provenance": dict(THRESHOLD_PROVENANCE),
            "geometry_version": GEOMETRY_VERSION,
            "generation_zone_exclusion": "未评估",
        }
        return cls(status="rejected", reason=reason, geometry=geometry)

    @property
    def usable(self) -> bool:
        return self.status == "localized" and len(self.geometry.get("wall_lines", [])) == 2

    @property
    def wall_lines(self) -> list[dict[str, float]]:
        """仅当 ``usable`` 为真时返回两条壁线，否则为空——禁止落回旧 ROI。"""
        if not self.usable:
            return []
        return [dict(line) for line in self.geometry.get("wall_lines", [])]

    @property
    def frame_id(self) -> int:
        return int(self.geometry.get("frame_id", 0) or 0)

    @property
    def image_shape(self) -> tuple[int, int]:
        shape = self.geometry.get("image_shape") or [0, 0]
        return (int(shape[0]), int(shape[1]))

    def to_dict(self) -> dict:
        return dict(self.geometry)
