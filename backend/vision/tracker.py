from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


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
    metadata: Dict[str, float] = field(default_factory=dict)

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
    """Match by position while rejecting implausible size jumps."""
    if not track_positions or not detections:
        return [], list(range(len(track_positions))), list(range(len(detections)))

    candidates: List[Tuple[float, int, int]] = []
    for track_idx, track_pos in enumerate(track_positions):
        track_radius = max(1.0, float(track_radii[track_idx]))
        for det_idx, det in enumerate(detections):
            det_radius = max(1.0, float(detection_radii[det_idx]))
            larger_radius = max(track_radius, det_radius)
            radius_delta = abs(track_radius - det_radius) / larger_radius
            if radius_delta > float(radius_match_ratio):
                continue

            pair_radius = (track_radius + det_radius) * 0.5
            distance_limit = max(
                float(min_distance),
                pair_radius * float(distance_radius_ratio),
            )
            distance_limit = min(float(max_distance), distance_limit)
            distance = float(np.linalg.norm(det - track_pos))
            if distance > distance_limit:
                continue

            # Prefer a slightly farther detection with a consistent radius over
            # a nearby fragment whose apparent radius changes abruptly.
            cost = distance + radius_delta * distance_limit * 0.35
            candidates.append((cost, track_idx, det_idx))

    candidates.sort(key=lambda item: item[0])
    used_tracks: set[int] = set()
    used_dets: set[int] = set()
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
