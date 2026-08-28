from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np

from .detector import DropletDetector
from .tuning import TuningFrame


@dataclass(frozen=True)
class LabeledDroplet:
    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class LabeledFrame:
    index: int
    droplets: tuple[LabeledDroplet, ...]


@dataclass(frozen=True)
class DetectionBenchmark:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_radius_error_px: float | None
    mean_frame_ms: float
    p95_frame_ms: float
    processed_frames: int

    def meets_targets(
        self,
        *,
        minimum_precision: float = 0.98,
        minimum_recall: float = 0.95,
        maximum_radius_error_px: float = 3.0,
    ) -> bool:
        radius_ok = (
            self.mean_radius_error_px is not None
            and self.mean_radius_error_px <= float(maximum_radius_error_px)
        )
        return (
            self.precision >= float(minimum_precision)
            and self.recall >= float(minimum_recall)
            and radius_ok
        )


def load_labeled_frames(path: str | Path) -> list[LabeledFrame]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_frames = payload.get("frames", []) if isinstance(payload, dict) else []
    if not isinstance(raw_frames, list):
        raise ValueError("标注文件的 frames 必须是列表")
    frames: list[LabeledFrame] = []
    seen_indices: set[int] = set()
    for raw_frame in raw_frames:
        if not isinstance(raw_frame, dict):
            raise ValueError("每个标注帧必须是对象")
        index = int(raw_frame.get("index", -1))
        if index < 0 or index in seen_indices:
            raise ValueError(f"无效或重复的标注帧索引：{index}")
        raw_droplets = raw_frame.get("droplets", [])
        if not isinstance(raw_droplets, list):
            raise ValueError(f"第 {index} 帧的 droplets 必须是列表")
        droplets = tuple(
            LabeledDroplet(
                x=float(item["x"]),
                y=float(item["y"]),
                radius=float(item["radius"]),
            )
            for item in raw_droplets
        )
        if any(item.radius <= 0.0 for item in droplets):
            raise ValueError(f"第 {index} 帧包含非正半径")
        seen_indices.add(index)
        frames.append(LabeledFrame(index=index, droplets=droplets))
    if not frames:
        raise ValueError("标注文件中没有帧")
    return sorted(frames, key=lambda item: item.index)


def match_frame(
    detected_centers: Sequence[np.ndarray],
    detected_radii: Sequence[float],
    labels: Sequence[LabeledDroplet],
    *,
    center_tolerance_radius_ratio: float = 0.60,
    radius_tolerance_ratio: float = 0.45,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    candidates: list[tuple[float, int, int]] = []
    for detected_index, (center, detected_radius) in enumerate(
        zip(detected_centers, detected_radii)
    ):
        for label_index, label in enumerate(labels):
            label_radius = max(1.0, float(label.radius))
            distance = float(
                np.hypot(float(center[0]) - float(label.x), float(center[1]) - float(label.y))
            )
            radius_error = abs(float(detected_radius) - label_radius) / label_radius
            if distance > max(3.0, label_radius * float(center_tolerance_radius_ratio)):
                continue
            if radius_error > float(radius_tolerance_ratio):
                continue
            cost = distance / label_radius + radius_error
            candidates.append((cost, detected_index, label_index))

    adjacency: dict[int, list[tuple[float, int]]] = {
        index: [] for index in range(len(detected_centers))
    }
    for cost, detected_index, label_index in candidates:
        adjacency[detected_index].append((cost, label_index))
    for edges in adjacency.values():
        edges.sort(key=lambda item: item[0])

    label_match: dict[int, int] = {}

    def augment(detected_index: int, seen_labels: set[int]) -> bool:
        for _cost, label_index in adjacency[detected_index]:
            if label_index in seen_labels:
                continue
            seen_labels.add(label_index)
            previous = label_match.get(label_index)
            if previous is None or augment(previous, seen_labels):
                label_match[label_index] = detected_index
                return True
        return False

    detection_order = sorted(
        adjacency,
        key=lambda index: (
            len(adjacency[index]),
            adjacency[index][0][0] if adjacency[index] else float("inf"),
        ),
    )
    for detected_index in detection_order:
        augment(detected_index, set())
    matches = [(detected_index, label_index) for label_index, detected_index in label_match.items()]
    used_detections = {detected_index for detected_index, _label_index in matches}
    used_labels = {label_index for _detected_index, label_index in matches}
    unmatched_detections = [
        index for index in range(len(detected_centers)) if index not in used_detections
    ]
    unmatched_labels = [index for index in range(len(labels)) if index not in used_labels]
    return matches, unmatched_detections, unmatched_labels


def evaluate_labeled_frames(
    detector: DropletDetector,
    frames: Iterable[TuningFrame],
    labels: Iterable[LabeledFrame],
) -> DetectionBenchmark:
    labels_by_index = {int(item.index): item for item in labels}
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    radius_errors: list[float] = []
    timings_ms: list[float] = []
    processed = 0

    for frame in frames:
        labeled = labels_by_index.get(int(frame.index))
        if labeled is None:
            continue
        started = perf_counter()
        result = detector.detect(frame.image)
        timings_ms.append((perf_counter() - started) * 1000.0)
        matches, unmatched_detections, unmatched_labels = match_frame(
            result.centers,
            result.radii,
            labeled.droplets,
        )
        true_positives += len(matches)
        false_positives += len(unmatched_detections)
        false_negatives += len(unmatched_labels)
        radius_errors.extend(
            abs(float(result.radii[detected_index]) - float(labeled.droplets[label_index].radius))
            for detected_index, label_index in matches
        )
        processed += 1

    if processed <= 0:
        raise ValueError("没有同时具备图像和人工标注的帧")
    precision_denom = true_positives + false_positives
    recall_denom = true_positives + false_negatives
    precision = true_positives / precision_denom if precision_denom else 1.0
    recall = true_positives / recall_denom if recall_denom else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return DetectionBenchmark(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        mean_radius_error_px=float(np.mean(radius_errors)) if radius_errors else None,
        mean_frame_ms=float(np.mean(timings_ms)),
        p95_frame_ms=float(np.percentile(timings_ms, 95.0)),
        processed_frames=processed,
    )
