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
    "circularity_threshold",
    "gaussian_blur_size",
    "morphology_open_kernel",
    "morphology_close_kernel",
    "candidate_min_edge_support",
    "candidate_full_circle_ratio",
    "hough_param2",
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
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    blur_size = detector._odd(config.gaussian_blur_size)
    smoothed = cv2.GaussianBlur(normalized, (blur_size, blur_size), 0) if blur_size > 1 else normalized
    otsu_threshold, raw_binary = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cleaned = detector._morphology_clean(raw_binary)
    cut_line = int(cleaned.shape[0] * config.cut_line_ratio)
    detect_mode = config.detection_mode
    if detect_mode == "no_split":
        connected_centers, connected_radii = detector._detect_no_split(cleaned, cut_line)
    else:
        connected_centers, connected_radii = detector._detect_split_connected(cleaned, cut_line)

    centers, radii = list(connected_centers), list(connected_radii)
    source_parts = [f"连通域 {len(centers)}"]
    candidate_centers, candidate_radii = list(centers), list(radii)
    if config.hough_fallback_only:
        centers, radii = detector._score_and_suppress_candidates(normalized, centers, radii)
        if not centers and config.enable_hough_candidates:
            candidate_centers, candidate_radii = detector._detect_hough_candidates(normalized, cut_line)
            source_parts.append(f"Hough {len(candidate_centers)}")
            centers, radii = detector._score_and_suppress_candidates(normalized, candidate_centers, candidate_radii)
        if not centers and (config.enable_intensity_peak_candidates or config.enable_intensity_peak_fallback):
            candidate_centers, candidate_radii = detector._detect_intensity_peak_candidates(normalized, cut_line)
            source_parts.append(f"亮度峰值 {len(candidate_centers)}")
            centers, radii = detector._score_and_suppress_candidates(normalized, candidate_centers, candidate_radii)
    else:
        if config.enable_hough_candidates:
            extra_centers, extra_radii = detector._detect_hough_candidates(normalized, cut_line)
            source_parts.append(f"Hough {len(extra_centers)}")
            candidate_centers.extend(extra_centers); candidate_radii.extend(extra_radii)
        if config.enable_intensity_peak_candidates or (config.enable_intensity_peak_fallback and not candidate_centers):
            extra_centers, extra_radii = detector._detect_intensity_peak_candidates(normalized, cut_line)
            source_parts.append(f"亮度峰值 {len(extra_centers)}")
            candidate_centers.extend(extra_centers); candidate_radii.extend(extra_radii)
        centers, radii = detector._score_and_suppress_candidates(normalized, candidate_centers, candidate_radii)

    helper_mask = detector._build_bead_helper_mask(normalized)
    result = DetectionResult(centers, radii, np.empty((0, 0, 3), dtype=np.uint8), helper_mask)
    stages = [
        PipelineStage("1. 原始图像", "检测器收到的当前视频帧。", frame.copy(), statistics=f"尺寸 {frame.shape[1]}×{frame.shape[0]}"),
        PipelineStage("2. 灰度转换", "去除颜色信息，保留像素亮度。", gray, parameters="BGR → Gray", statistics=f"亮度范围 {int(gray.min())}–{int(gray.max())}"),
        PipelineStage("3. 对比度归一化", "将亮度拉伸到完整范围，减轻曝光差异。", normalized, parameters="NORM_MINMAX: 0–255", statistics=f"均值 {float(normalized.mean()):.1f}"),
        PipelineStage("4. 高斯平滑", "抑制高频噪声，避免产生破碎轮廓。", smoothed, parameters=f"核大小 {blur_size}×{blur_size}"),
        PipelineStage("5. Otsu 二值化", "自动计算阈值，将暗色液滴区域转为白色。", raw_binary, parameters="THRESH_BINARY_INV + OTSU", statistics=f"阈值 {otsu_threshold:.1f}；前景 {100.0 * np.count_nonzero(raw_binary) / raw_binary.size:.1f}%"),
        PipelineStage("6. 形态学清理", "先开运算去除噪点，再闭运算填补缺口。", cleaned, parameters=f"开核 {detector._odd(config.morphology_open_kernel)}；闭核 {detector._odd(config.morphology_close_kernel)}", statistics=f"前景 {100.0 * np.count_nonzero(cleaned) / cleaned.size:.1f}%"),
        PipelineStage("7. 连通域/分裂", "提取轮廓，并按面积、圆度、半径和截断线初筛。", _candidate_overlay(frame, connected_centers, connected_radii), parameters=f"模式 {detect_mode}；圆度 ≥ {config.circularity_threshold:g}；半径 {config.min_radius:g}–{config.max_radius:g}", statistics=f"初筛候选 {len(connected_centers)} 个"),
        PipelineStage("8. 候选生成", "按配置合并或回退到 Hough 圆与亮度峰值候选。", _candidate_overlay(frame, candidate_centers, candidate_radii), parameters="；".join(source_parts), statistics=f"进入评分候选 {len(candidate_centers)} 个"),
        PipelineStage("9. 评分与抑制", "按边缘支撑、环形对比度等评分，并抑制重复圆。", annotate_frame(frame, result), parameters=f"边缘支撑 ≥ {config.candidate_min_edge_support:g}", statistics=f"最终液滴 {len(centers)} 个"),
        PipelineStage("10. 磁珠辅助掩膜", "检测器输出的辅助掩膜；不在本工作台执行磁珠识别。", helper_mask, statistics=f"有效像素 {100.0 * np.count_nonzero(helper_mask) / helper_mask.size:.1f}%"),
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
