from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .channel_region import ChannelRegionResult, detect_channel_region
from .config import ChannelRegionConfig, DetectorConfig, DebugConfig
from .detector import CircleDetection, DetectionResult, DropletDetector


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
    "edge_gradient_threshold",
    "edge_anchor_threshold",
    "edge_scan_interval",
    "edge_min_path_length",
    "edge_sigma",
    "edge_min_circle_ratio",
    "edge_min_support_ratio",
    "edge_support_neighborhood",
    "expected_radius",
    "expected_radius_tolerance_ratio",
    "edge_min_visible_ratio",
    "diameter_min_visible_ratio",
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


def _circle_overlay(
    frame: np.ndarray,
    circles: list[CircleDetection],
    color: tuple[int, int, int] = (0, 180, 255),
) -> np.ndarray:
    output = frame.copy()
    for item in circles:
        center = tuple(int(round(float(value))) for value in item.center)
        cv2.circle(output, center, max(1, int(round(item.radius))), color, 2)
    return output


def inspect_frame(
    frame: np.ndarray,
    config: DetectorConfig,
    *,
    channel_config: ChannelRegionConfig | None = None,
    channel_frames: Iterable[np.ndarray] | None = None,
) -> tuple[DetectionResult, list[PipelineStage]]:
    """Inspect optional channel calibration followed by EdgeDrawing detection."""
    channel_stages: list[PipelineStage] = []
    detection_frame = frame
    if channel_config is not None:
        samples = list(channel_frames or [frame])
        channel_result = (
            detect_channel_region(samples[: max(1, int(channel_config.sample_frames))], channel_config)
            if channel_config.enabled
            else ChannelRegionResult("skipped", 0.0, "已跳过管道区域检定")
        )
        high_frequency = channel_result.high_frequency_map
        if high_frequency is None:
            high_frequency = np.zeros(frame.shape[:2], dtype=np.uint8)
        line_overlay = channel_result.line_overlay if channel_result.line_overlay is not None else frame.copy()
        region_overlay = channel_result.region_overlay if channel_result.region_overlay is not None else frame.copy()
        if channel_result.status == "calibrated" and channel_result.rectified_frame is not None:
            detection_frame = channel_result.rectified_frame
        channel_stages = [
            PipelineStage(
                "A1. 管道检定 · 原始大图",
                "启动时从原始大图采集多帧；手动 ROI 优先，也可关闭并跳过本步骤。",
                frame.copy(),
                parameters=f"采样 {min(len(samples), max(1, int(channel_config.sample_frames)))}/{max(1, int(channel_config.sample_frames))} 帧" if channel_config.enabled else "已跳过",
                statistics=f"尺寸 {frame.shape[1]}×{frame.shape[0]}",
            ),
            PipelineStage(
                "A2. 管道检定 · 高频信号",
                "计算每帧局部空间高频能量并进行多帧融合；管内高频、管外低频应形成连续区域和明显界线。",
                high_frequency,
                parameters=(
                    f"局部窗口 {channel_config.frequency_window_ratio:g}；"
                    f"持续帧比例 ≥ {channel_config.min_frequency_frame_support:g}；"
                    f"界线 Canny {channel_config.canny_low}–{channel_config.canny_high}"
                ),
                statistics=f"平均响应 {float(high_frequency.mean()):.1f}",
            ),
            PipelineStage(
                "A3. 管道检定 · 直线性质",
                "只从高低频区域的界线上生成候选，再将两侧边界拟合为直线并验证跨度、平行度和间距稳定性。",
                line_overlay,
                parameters=(
                    f"最短线 {channel_config.min_line_length_ratio:g}；"
                    f"平行容差 {channel_config.parallel_tolerance_degrees:g}°"
                ),
            ),
            PipelineStage(
                "A4. 管道检定 · 有效区域",
                "两条可信直线管壁围成的区域将被透视摆正，再交给液滴识别；失败时安全回退整帧。",
                region_overlay,
                parameters=f"最低可信度 {channel_config.min_confidence:g}",
                statistics=(
                    f"状态 {channel_result.status}；可信度 {channel_result.confidence:.2f}；{channel_result.reason}"
                ),
            ),
        ]

    detector = DropletDetector(config, DebugConfig(enabled=False))
    gray = detector._ensure_gray(detection_frame)
    normalized = detector._normalize(gray)
    smoothed = detector._smooth(normalized)
    cut_line = int(smoothed.shape[0] * config.cut_line_ratio)
    trace: dict[str, object] = {}
    circles = detector._detect_candidates(smoothed, cut_line, trace)
    centers = [item.center.copy() for item in circles]
    radii = [item.radius for item in circles]
    diameter_valid = [
        detector._circle_fully_visible(normalized.shape[:2], item)
        for item in circles
    ]
    result = DetectionResult(
        centers=centers,
        radii=radii,
        debug_image=np.empty((0, 0, 3), dtype=np.uint8),
        helper_mask=detector._build_bead_helper_mask(normalized),
        diameter_valid=diameter_valid,
    )
    work_gray = np.asarray(trace.get("work_gray", smoothed))
    enhanced = np.asarray(trace.get("enhanced", work_gray))
    edges = np.asarray(trace.get("edges", np.zeros_like(work_gray)))
    raw_circles = list(trace.get("raw_circles", []))
    scale = float(trace.get("scale", 1.0))
    blur_size = detector._odd(config.gaussian_blur_size)
    stages = [
        PipelineStage(
            "1. 原始图像",
            "检测器收到的当前视频帧。",
            detection_frame.copy(),
            statistics=f"尺寸 {detection_frame.shape[1]}×{detection_frame.shape[0]}",
        ),
        PipelineStage(
            "2. 灰度转换",
            "将输入统一为单通道灰度图。",
            gray,
            parameters="BGR → Gray" if frame.ndim == 3 else "输入已是灰度图",
            statistics=f"亮度范围 {int(gray.min())}–{int(gray.max())}",
        ),
        PipelineStage(
            "3. 对比度归一化",
            "拉伸灰度范围，降低曝光变化对 EdgeDrawing 的影响。",
            normalized,
            parameters="NORM_MINMAX: 0–255" if config.enable_intensity_normalization else "已跳过",
            statistics=f"均值 {float(normalized.mean()):.1f}",
        ),
        PipelineStage(
            "4. 高斯平滑",
            "在 EdgeDrawing 前抑制高频噪声。",
            smoothed,
            parameters=f"核大小 {blur_size}×{blur_size}" if config.enable_gaussian_blur else "已跳过",
        ),
        PipelineStage(
            "5. EdgeDrawing 输入预处理",
            "按工作尺寸缩放，并可选执行 CLAHE 与中值滤波。",
            enhanced,
            parameters=(
                f"缩放 {scale:.2f}；CLAHE {'开' if config.enable_edge_clahe else '关'}；"
                f"中值滤波 {'开' if config.enable_edge_median_blur else '关'}"
            ),
        ),
        PipelineStage(
            "6. EdgeDrawing 边缘图",
            "生成 EdgeDrawing 提取的连接边缘，用于验证候选圆的边缘覆盖率。",
            edges,
            parameters=(
                f"梯度阈值 {config.edge_gradient_threshold}；"
                f"边缘邻域 {config.edge_support_neighborhood}px"
            ),
        ),
        PipelineStage(
            "7. EdgeDrawing 原始圆候选",
            "从 EdgeDrawing 闭合边缘结果中仅保留原生圆和达到圆度阈值的近圆记录。",
            _circle_overlay(detection_frame, raw_circles, (255, 120, 0)),
            parameters=(
                f"近圆比 ≥ {config.edge_min_circle_ratio:g}；"
                f"半径 {config.min_radius:g}–{config.max_radius:g}px"
            ),
            statistics=f"原始圆候选 {len(raw_circles)} 个",
        ),
        PipelineStage(
            "8. 简单过滤结果",
            "按半径、近圆比、截断线、可见圆周、边缘支撑和中心距离过滤。",
            annotate_frame(detection_frame, result),
            parameters=(
                f"边缘支撑 ≥ {config.edge_min_support_ratio:g}；"
                f"可见圆周 ≥ {config.edge_min_visible_ratio:g}；"
                f"直径完整度 ≥ {config.diameter_min_visible_ratio:g}"
            ),
            statistics=f"最终液滴 {len(centers)} 个",
        ),
    ]
    if channel_stages:
        stages = channel_stages + [
            PipelineStage(
                f"B{index}. {stage.name.split('. ', 1)[-1]}",
                stage.description,
                stage.image,
                stage.parameters,
                stage.statistics,
            )
            for index, stage in enumerate(stages, start=1)
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
        label = f"r={radius:.1f}"
        cv2.putText(
            output,
            label,
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
    detector = DropletDetector(config, DebugConfig(enabled=False))
    for item in frames:
        result = detector.detect(item.image)
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
