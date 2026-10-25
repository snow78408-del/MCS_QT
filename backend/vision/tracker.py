from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class DropletTrack:
    id: int
    position: np.ndarray
    radius: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    predicted_position: Optional[np.ndarray] = None
    unmatched_frames: int = 0
    age: int = 1
    is_active: bool = True
    is_confirmed: bool = True
    observation_history: List[bool] = field(default_factory=list)
    metadata: Dict[str, float] = field(default_factory=dict)
    last_timestamp: float | None = None

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float32)
        self.velocity = np.asarray(self.velocity, dtype=np.float32)
        if self.predicted_position is not None:
            self.predicted_position = np.asarray(self.predicted_position, dtype=np.float32)


@dataclass
class TrackingResult:
    active_tracks: List[DropletTrack]
    matched_pairs: List[Tuple[int, int]]
    new_track_ids: List[int]
    removed_track_ids: List[int]
    total_count: int


class BaseTracker(ABC):
    @abstractmethod
    def update(
        self,
        detections: Sequence[np.ndarray],
        radii: Optional[Sequence[float]] = None,
        timestamp: float | None = None,
    ) -> TrackingResult:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_active_tracks(self) -> List[DropletTrack]:
        raise NotImplementedError


def as_points(detections: Sequence[np.ndarray]) -> List[np.ndarray]:
    return [np.asarray(det, dtype=np.float32) for det in detections]


def greedy_match(
    track_positions: Sequence[np.ndarray],
    detections: Sequence[np.ndarray],
    max_distance: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if not track_positions or not detections:
        return [], list(range(len(track_positions))), list(range(len(detections)))

    candidates: List[Tuple[float, int, int]] = []
    for track_idx, track_pos in enumerate(track_positions):
        for det_idx, det in enumerate(detections):
            dist = float(np.linalg.norm(det - track_pos))
            if dist <= max_distance:
                candidates.append((dist, track_idx, det_idx))

    candidates.sort(key=lambda item: item[0])

    used_tracks = set()
    used_dets = set()
    matches: List[Tuple[int, int]] = []
    for _, track_idx, det_idx in candidates:
        if track_idx in used_tracks or det_idx in used_dets:
            continue
        used_tracks.add(track_idx)
        used_dets.add(det_idx)
        matches.append((track_idx, det_idx))

    unmatched_tracks = [idx for idx in range(len(track_positions)) if idx not in used_tracks]
    unmatched_dets = [idx for idx in range(len(detections)) if idx not in used_dets]
    return matches, unmatched_tracks, unmatched_dets


def radius_gated_match(
    track_positions: Sequence[np.ndarray],
    track_radii: Sequence[float],
    detections: Sequence[np.ndarray],
    detection_radii: Sequence[float],
    *,
    max_distance: float,
    min_distance: float,
    distance_radius_ratio: float,
    radius_match_ratio: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Maximize gated matches, then minimize position/size cost globally.

    Dummy columns allow missed observations without ever matching across a
    distance/size gate. Uncontested pairs bypass the assignment solver.
    """
    if not track_positions or not detections:
        return [], list(range(len(track_positions))), list(range(len(detections)))

    positions = np.asarray(track_positions, dtype=np.float64)
    points = np.asarray(detections, dtype=np.float64)
    raw_track_radii = np.asarray(track_radii, dtype=np.float64)[:, None]
    raw_det_radii = np.asarray(detection_radii, dtype=np.float64)[None, :]
    track_radius = np.maximum(1.0, raw_track_radii)
    det_radius = np.maximum(1.0, raw_det_radii)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        radius_delta = np.abs(track_radius - det_radius) / np.maximum(track_radius, det_radius)
        distance_limit = np.minimum(
            float(max_distance),
            np.maximum(float(min_distance), (track_radius + det_radius) * 0.5 * float(distance_radius_ratio)),
        )
        distance = np.linalg.norm(points[None, :, :] - positions[:, None, :], axis=2)
        costs = distance + radius_delta * distance_limit * 0.35
    allowed = (
        (radius_delta <= float(radius_match_ratio))
        & (distance <= distance_limit)
        & np.isfinite(costs)
        & np.isfinite(raw_track_radii)
        & np.isfinite(raw_det_radii)
    )
    rows, cols = np.nonzero(allowed)
    if len(rows) and (np.any(allowed.sum(axis=0) > 1) or np.any(allowed.sum(axis=1) > 1)):
        count = len(track_positions)
        # All valid normalized costs are <= 1. Losing one match costs more
        # than the largest possible change in total real-edge cost.
        penalty = float(count + 1)
        assignment = np.full((count, len(detections) + count), penalty)
        assignment[:, :len(detections)] = np.inf
        assignment[rows, cols] = costs[rows, cols] / max(1.0, float(costs[rows, cols].max()))
        rows, cols = linear_sum_assignment(assignment)
        real = cols < len(detections)
        rows, cols = rows[real], cols[real]
    matches = [(int(row), int(col)) for row, col in zip(rows, cols)]
    used_tracks = set(rows.tolist())
    used_dets = set(cols.tolist())

    unmatched_tracks = [idx for idx in range(len(track_positions)) if idx not in used_tracks]
    unmatched_dets = [idx for idx in range(len(detections)) if idx not in used_dets]
    return matches, unmatched_tracks, unmatched_dets
