from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .config import DetectorConfig, DebugConfig
from .detector import DetectionResult, DropletDetector


@dataclass(frozen=True)
class TuningFrame:
    index: int
    image: np.ndarray


@dataclass(frozen=True)
class TuningEvaluation:
    score: float
    mean_count: float
    count_std: float
    mean_radius: float | None
    radius_cv: float | None
    processed_frames: int


@dataclass(frozen=True)
class SearchResult:
    score: float
    parameters: dict[str, float | int | bool]
    evaluation: TuningEvaluation


@dataclass(frozen=True)
class PipelineStage:
    name: str
    description: str
    image: np.ndarray
    parameters: str = ""
    statistics: str = ""


SEARCHABLE_FIELDS = (
    "min_radius",
    "max_radius",
    "gaussian_blur_size",
    "hough_dp",
    "hough_min_distance",
    "hough_param1",
    "hough_param2",
    "hough_min_radius",
    "hough_max_radius",
    "hough_edge_support_threshold",
    "expected_radius",
    "expected_radius_tolerance_ratio",
    "edge_ownership_search_radius",
    "edge_ownership_margin",
    "edge_ownership_min_ratio",
    "candidate_min_edge_support",
    "candidate_full_circle_ratio",
)


def read_video_frames(path: str | Path, max_frames: int = 80, stride: int = 1) -> list[TuningFrame]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频：{path}")
    frames: list[TuningFrame] = []
    frame_index = 0
    stride = max(1, int(stride))
    try:
        while len(frames) < max(1, int(max_frames)):
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                frames.append(TuningFrame(frame_index, frame))
            frame_index += 1
    finally:
        capture.release()
    if not frames:
        raise ValueError("视频中没有可读取的帧")
    return frames


def _candidate_overlay(frame: np.ndarray, centers: list[np.ndarray], radii: list[float], color=(0, 180, 255)) -> np.ndarray:
    output = frame.copy()
    for center, radius in zip(centers, radii):
        point = tuple(int(round(float(value))) for value in center)
        cv2.circle(output, point, max(1, int(round(radius))), color, 2)
    return output


def inspect_frame(frame: np.ndarray, config: DetectorConfig) -> tuple[DetectionResult, list[PipelineStage]]:
    """Run the detector pipeline while preserving its important intermediate images."""
    detector = DropletDetector(config, DebugConfig(enabled=False))
    gray = detector._ensure_gray(frame)
    normalized = (
        cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        if config.enable_intensity_normalization
        else gray.copy()
    )
    blur_size = detector._odd(config.gaussian_blur_size)
    smoothed = (
        cv2.GaussianBlur(normalized, (blur_size, blur_size), 0)
        if config.enable_gaussian_blur and blur_size > 1
        else normalized
    )
    cut_line = int(smoothed.shape[0] * config.cut_line_ratio)
    hough_trace: dict[str, object] = {}
    candidate_centers, candidate_radii = detector._detect_hough_candidates(
        smoothed,
        cut_line,
        hough_trace,
    )
    centers, radii = detector._score_and_suppress_candidates(
        normalized,
        candidate_centers,
        candidate_radii,
    )
    raw_centers = hough_trace.get("raw_centers", [])
    raw_radii = hough_trace.get("raw_radii", [])
    geometric_centers = hough_trace.get("geometric_centers", [])
    geometric_radii = hough_trace.get("geometric_radii", [])
    size_centers = hough_trace.get("size_centers", [])
    size_radii = hough_trace.get("size_radii", [])
    enhanced = np.asarray(hough_trace.get("enhanced", smoothed))
    median_blurred = np.asarray(hough_trace.get("median_blurred", smoothed))
    edges = np.asarray(hough_trace.get("edges", np.zeros_like(smoothed)))

    helper_mask = detector._build_bead_helper_mask(normalized)
    result = DetectionResult(centers, radii, np.empty((0, 0, 3), dtype=np.uint8), helper_mask)
    stages = [
        PipelineStage("1. 原始图像", "检测器收到的当前视频帧。", frame.copy(), statistics=f"尺寸 {frame.shape[1]}×{frame.shape[0]}"),
        PipelineStage("2. 灰度转换", "去除颜色信息，为梯度圆检测准备单通道图像。", gray, parameters="BGR → Gray", statistics=f"亮度范围 {int(gray.min())}–{int(gray.max())}"),
        PipelineStage("3. 对比度归一化", "拉伸灰度范围，减轻不同曝光条件对梯度强度的影响。", normalized, parameters="NORM_MINMAX: 0–255" if config.enable_intensity_normalization else "已跳过", statistics=f"均值 {float(normalized.mean()):.1f}"),
        PipelineStage("4. 输入高斯平滑", "在进入 Hough 分支前抑制原始高频噪声。", smoothed, parameters=f"核大小 {blur_size}×{blur_size}" if config.enable_gaussian_blur else "已跳过"),
        PipelineStage("5. Hough 局部对比度增强", "使用 CLAHE 强化不同亮度区域中的圆形边缘。", enhanced, parameters="CLAHE clipLimit=2.0；网格 8×8" if config.enable_hough_clahe else "已跳过", statistics=f"工作缩放 {float(hough_trace.get('scale', 1.0)):.3f}"),
        PipelineStage("6. Hough 中值滤波", "在圆变换前抑制孤立噪声，同时尽量保留边缘。", median_blurred, parameters="medianBlur 5×5" if config.enable_hough_median_blur else "已跳过"),
        PipelineStage("7. Canny 梯度边缘", "展示 Hough 圆变换用于寻找圆周的梯度边缘。", edges, parameters=f"低阈值 50；高阈值 150；param1={config.hough_param1:g}", statistics=f"边缘像素 {100.0 * np.count_nonzero(edges) / edges.size:.1f}%"),
        PipelineStage("8. Hough 原始圆", "展示 HoughCircles 尚未经过几何和边缘支撑过滤的全部圆。", _candidate_overlay(frame, raw_centers, raw_radii), parameters=f"dp={config.hough_dp:g}；param2={config.hough_param2:g}；半径 {detector._hough_min_radius():g}–{detector._hough_max_radius():g}", statistics=f"原始圆 {len(raw_centers)} 个"),
        PipelineStage("9. Hough 几何过滤", "按截断线、绝对半径、圆周边缘支撑和多液滴包围关系过滤。", _candidate_overlay(frame, geometric_centers, geometric_radii), parameters=f"Hough 边缘支撑 ≥ {config.hough_edge_support_threshold:g}", statistics=f"几何过滤后 {len(geometric_centers)} 个"),
        PipelineStage("10. 目标尺寸过滤", "利用可靠的目标半径拒绝尺寸异常圆；红色为进入本步骤的圆，绿色为保留圆。", _candidate_overlay(_candidate_overlay(frame, geometric_centers, geometric_radii, (0, 0, 255)), size_centers, size_radii, (0, 220, 80)), parameters=(f"期望半径 {detector.runtime_radius_range()[1]:.1f}；容差 ±{config.expected_radius_tolerance_ratio * 100:.0f}%" if detector._has_expected_size and config.enable_expected_size_filter else "未设置目标尺寸，已跳过"), statistics=f"拒绝 {len(geometric_centers) - len(size_centers)} 个；保留 {len(size_centers)} 个"),
        PipelineStage("11. 圆周边缘归属", "将实际边缘像素归属给径向残差最小的候选，拒绝借用相邻液滴边缘拼成的圆；红色为进入本步骤的圆，绿色为保留圆。", _candidate_overlay(_candidate_overlay(frame, size_centers, size_radii, (0, 0, 255)), candidate_centers, candidate_radii, (0, 220, 80)), parameters=(f"搜索半径 {config.edge_ownership_search_radius}px；最小独占比例 {config.edge_ownership_min_ratio:g}；归属间隔 {config.edge_ownership_margin:g}px" if config.enable_edge_ownership_filter else "已跳过"), statistics=f"拒绝 {len(size_centers) - len(candidate_centers)} 个；保留 {len(candidate_centers)} 个"),
        PipelineStage("12. 最终评分与抑制", "按边缘支撑、环形对比度等评分，并抑制重复圆。", annotate_frame(frame, result), parameters=f"候选边缘支撑 ≥ {config.candidate_min_edge_support:g}；完整圆比例 ≥ {config.candidate_full_circle_ratio:g}", statistics=f"最终液滴 {len(centers)} 个"),
    ]
    return result, stages


def detect_frame(frame: np.ndarray, config: DetectorConfig) -> DetectionResult:
    return inspect_frame(frame, config)[0]


def annotate_frame(frame: np.ndarray, result: DetectionResult) -> np.ndarray:
    output = frame.copy()
    for center, radius in zip(result.centers, result.radii):
        point = tuple(int(round(float(value))) for value in center)
        cv2.circle(output, point, max(1, int(round(radius))), (0, 220, 80), 2)
        cv2.circle(output, point, 2, (0, 80, 255), -1)
        cv2.putText(
            output,
            f"r={radius:.1f}",
            (point[0] + 4, point[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 220, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        output,
        f"droplets: {len(result.centers)}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def evaluate_config(
    frames: Iterable[TuningFrame],
    config: DetectorConfig,
    expected_count: float = 1.0,
) -> TuningEvaluation:
    counts: list[float] = []
    radii: list[float] = []
    processed = 0
    for item in frames:
        result = detect_frame(item.image, config)
        counts.append(float(len(result.centers)))
        radii.extend(float(value) for value in result.radii)
        processed += 1
    if not counts:
        raise ValueError("没有可评估的帧")

    count_array = np.asarray(counts, dtype=np.float32)
    mean_count = float(np.mean(count_array))
    count_std = float(np.std(count_array))
    mean_radius = float(np.mean(radii)) if radii else None
    radius_cv = None
    if radii and mean_radius and mean_radius > 1e-6:
        radius_cv = float(np.std(np.asarray(radii, dtype=np.float32)) / mean_radius)

    target = max(0.0, float(expected_count))
    count_score = 1.0 if target <= 0 else float(np.exp(-abs(mean_count - target) / max(1.0, target)))
    stability_score = float(1.0 / (1.0 + count_std))
    radius_score = 1.0 if radius_cv is None else float(1.0 / (1.0 + 4.0 * radius_cv))
    non_empty_score = 1.0 if mean_count > 0 else 0.0
    score = 0.45 * count_score + 0.25 * stability_score + 0.20 * radius_score + 0.10 * non_empty_score
    return TuningEvaluation(score, mean_count, count_std, mean_radius, radius_cv, processed)


def grid_search(
    frames: list[TuningFrame],
    base_config: DetectorConfig,
    parameter_grid: dict[str, list[float | int | bool]],
    expected_count: float = 1.0,
) -> list[SearchResult]:
    unknown = set(parameter_grid) - set(SEARCHABLE_FIELDS)
    if unknown:
        raise ValueError(f"不支持自动搜索的参数：{', '.join(sorted(unknown))}")
    keys = list(parameter_grid)
    values = [parameter_grid[key] for key in keys]
    results: list[SearchResult] = []
    for combination in product(*values):
        config = DetectorConfig(**asdict(base_config))
        parameters = dict(zip(keys, combination))
        for key, value in parameters.items():
            setattr(config, key, value)
        evaluation = evaluate_config(frames, config, expected_count)
        results.append(SearchResult(evaluation.score, parameters, evaluation))
    return sorted(results, key=lambda item: item.score, reverse=True)
