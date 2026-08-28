from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .config import DetectorConfig, DebugConfig
from .algorithms import get_algorithm
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
    "contour_work_scale",
    "adaptive_threshold_block_size",
    "adaptive_threshold_c",
    "background_difference_threshold",
    "contour_canny_low",
    "contour_canny_high",
    "contour_min_circularity",
    "contour_min_axis_ratio",
    "contour_min_edge_support",
    "watershed_peak_ratio",
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
    "candidate_min_visible_circle_ratio",
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


def inspect_hybrid_frame(frame: np.ndarray, config: DetectorConfig) -> tuple[DetectionResult, list[PipelineStage]]:
    """Inspect the built-in hybrid detector and preserve its intermediate images."""
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
    hybrid_trace: dict[str, object] = {}
    candidate_centers, candidate_radii = detector._detect_hybrid_candidates(
        smoothed,
        cut_line,
        hybrid_trace,
    )
    centers, radii = detector._score_and_suppress_candidates(
        normalized,
        candidate_centers,
        candidate_radii,
    )
    contour_raw_centers = hybrid_trace.get("contour_raw_centers", [])
    contour_raw_radii = hybrid_trace.get("contour_raw_radii", [])
    contour_centers = hybrid_trace.get("contour_centers", [])
    contour_radii = hybrid_trace.get("contour_radii", [])
    contour_work = np.asarray(hybrid_trace.get("contour_work_gray", smoothed))
    contour_binary = np.asarray(hybrid_trace.get("contour_binary", np.zeros_like(contour_work)))
    contour_foreground = np.asarray(hybrid_trace.get("contour_foreground", np.zeros_like(contour_work)))
    contour_edges = np.asarray(hybrid_trace.get("contour_edges", np.zeros_like(contour_work)))
    contour_mask = np.asarray(hybrid_trace.get("contour_closed", np.zeros_like(contour_work)))
    hough_centers = hybrid_trace.get("hybrid_hough_centers", [])
    hough_radii = hybrid_trace.get("hybrid_hough_radii", [])
    hough_scope = str(hybrid_trace.get("hough_scope", "skipped"))

    helper_mask = detector._build_bead_helper_mask(normalized)
    result = DetectionResult(centers, radii, np.empty((0, 0, 3), dtype=np.uint8), helper_mask)
    stages = [
        PipelineStage(
            "1. 原始图像",
            "检测器收到的当前视频帧。",
            frame.copy(),
            statistics=f"尺寸 {frame.shape[1]}×{frame.shape[0]}",
        ),
        PipelineStage(
            "2. 灰度转换",
            "去除颜色信息，为分割和梯度检测准备单通道图像。",
            gray,
            parameters="BGR → Gray",
            statistics=f"亮度范围 {int(gray.min())}–{int(gray.max())}",
        ),
        PipelineStage(
            "3. 对比度归一化",
            "拉伸灰度范围，减轻不同曝光条件对梯度强度的影响。",
            normalized,
            parameters="NORM_MINMAX: 0–255" if config.enable_intensity_normalization else "已跳过",
            statistics=f"均值 {float(normalized.mean()):.1f}",
        ),
        PipelineStage(
            "4. 输入高斯平滑",
            "抑制高频噪声，为快速分割和候选精修准备图像。",
            smoothed,
            parameters=f"核大小 {blur_size}×{blur_size}" if config.enable_gaussian_blur else "已跳过",
        ),
        PipelineStage(
            "5. 缩放检测图",
            "在低分辨率图像上寻找候选，随后回到原图精修中心和尺寸。",
            contour_work,
            parameters=f"工作缩放 {float(hybrid_trace.get('contour_work_scale', 1.0)):.2f}",
        ),
        PipelineStage(
            "6. 自适应二值分割",
            "提取相对局部背景更暗的液滴边界。",
            contour_binary,
            parameters=f"块大小 {config.adaptive_threshold_block_size}；C={config.adaptive_threshold_c:g}",
            statistics=(
                f"前景像素 {100.0 * np.count_nonzero(contour_binary) / max(1, contour_binary.size):.1f}%"
            ),
        ),
        PipelineStage(
            "7. 背景运动分割",
            "固定相机下补充弱边缘但发生运动的液滴区域。",
            contour_foreground,
            parameters=(
                f"差分阈值 {config.background_difference_threshold:g}；"
                f"学习率 {config.background_learning_rate:g}"
                if config.enable_background_subtraction
                else "已跳过"
            ),
            statistics=(
                f"运动像素 {100.0 * np.count_nonzero(contour_foreground) / max(1, contour_foreground.size):.1f}%"
            ),
        ),
        PipelineStage(
            "8. 连通域与边缘",
            "融合二值、背景差分和 Canny 边缘后生成局部连通区域。",
            cv2.bitwise_or(contour_edges, contour_mask),
            parameters=f"Canny {config.contour_canny_low:g}–{config.contour_canny_high:g}",
            statistics=(
                f"候选区域像素 {100.0 * np.count_nonzero(contour_mask) / max(1, contour_mask.size):.1f}%"
            ),
        ),
        PipelineStage(
            "9. 轮廓初始候选",
            "对连通区域拟合椭圆，并按面积、圆度和长短轴比初筛。",
            _candidate_overlay(frame, contour_raw_centers, contour_raw_radii),
            parameters=(
                f"圆度 ≥ {config.contour_min_circularity:g}；"
                f"轴比 ≥ {config.contour_min_axis_ratio:g}"
            ),
            statistics=f"初始候选 {len(contour_raw_centers)} 个",
        ),
        PipelineStage(
            "10. 局部 Watershed",
            "只拆分包含多个距离峰的粘连区域，规则液滴不承担此开销。",
            _candidate_overlay(frame, contour_centers, contour_radii, (0, 220, 80)),
            parameters=(
                f"峰值比例 {config.watershed_peak_ratio:g}"
                if config.enable_watershed_split
                else "已跳过"
            ),
            statistics=(
                f"拆分区域 {int(hybrid_trace.get('contour_split_regions', 0))} 个；"
                f"轮廓通过 {len(contour_centers)} 个"
            ),
        ),
        PipelineStage(
            "11. 局部 Hough 验证",
            "只在粘连或低置信局部区域运行 Hough；无候选时才全帧兜底。",
            _candidate_overlay(frame, hough_centers, hough_radii, (255, 120, 0)),
            parameters=f"范围 {hough_scope}；param2={config.hough_param2:g}",
            statistics=(
                f"Hough 补充 {len(hough_centers)} 个；"
                f"疑难区域 {len(hybrid_trace.get('ambiguous_regions', []))} 个"
            ),
        ),
        PipelineStage(
            "12. 融合评分与抑制",
            "融合轮廓与 Hough 候选，按边缘、内外亮度、尺寸和重叠关系输出。",
            annotate_frame(frame, result),
            parameters=(
                f"边缘支撑 ≥ {config.candidate_min_edge_support:g}；"
                f"跟踪可见圆周 ≥ {config.candidate_min_visible_circle_ratio:g}；"
                f"直径完整度 ≥ {config.candidate_full_circle_ratio:g}"
            ),
            statistics=f"最终液滴 {len(centers)} 个",
        ),
    ]
    return result, stages


def inspect_frame(
    frame: np.ndarray,
    config: Any,
    algorithm_id: str = "hybrid_v1",
) -> tuple[DetectionResult, list[PipelineStage]]:
    """Inspect one registered algorithm without coupling the workbench to its implementation."""
    return get_algorithm(algorithm_id).inspector(frame, config)


def detect_frame(frame: np.ndarray, config: Any, algorithm_id: str = "hybrid_v1") -> DetectionResult:
    return inspect_frame(frame, config, algorithm_id)[0]


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
    config: Any,
    expected_count: float = 1.0,
    algorithm_id: str = "hybrid_v1",
) -> TuningEvaluation:
    counts: list[float] = []
    radii: list[float] = []
    processed = 0
    detector = get_algorithm(algorithm_id).detector_factory(config, DebugConfig(enabled=False))
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
