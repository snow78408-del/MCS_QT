from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

try:
    from .config import TrackerConfig
    from .tracker import BaseTracker, DropletTrack, TrackingResult, as_points, radius_gated_match
except ImportError:
    from config import TrackerConfig
    from tracker import BaseTracker, DropletTrack, TrackingResult, as_points, radius_gated_match


class NearestTracker(BaseTracker):
    def __init__(self, config: TrackerConfig) -> None:
        self._config = config
        self._tracks: List[DropletTrack] = []
        self._next_id = 1
        self._last_timestamp: float | None = None

    def update(
        self,
        detections: Sequence[np.ndarray],
        radii: Optional[Sequence[float]] = None,
        timestamp: float | None = None,
    ) -> TrackingResult:
        default_dt = max(1e-3, float(self._config.kalman.dt))
        if timestamp is None or self._last_timestamp is None:
            dt = default_dt
        else:
            dt = min(5.0, max(1e-3, float(timestamp) - self._last_timestamp))
        if timestamp is not None:
            self._last_timestamp = float(timestamp)
        points = as_points(detections)
        radius_values = list(radii) if radii is not None else [0.0] * len(points)
        if len(radius_values) < len(points):
            radius_values += [0.0] * (len(points) - len(radius_values))

        track_positions = [track.position for track in self._tracks]
        matches, unmatched_tracks, unmatched_dets = radius_gated_match(
            track_positions,
            [float(track.radius) for track in self._tracks],
            points,
            radius_values,
            max_distance=float(self._config.match_distance),
            min_distance=float(self._config.min_match_distance),
            distance_radius_ratio=float(self._config.match_distance_radius_ratio),
            radius_match_ratio=float(self._config.radius_match_ratio),
        )

        matched_pairs = []
        for track_idx, det_idx in matches:
            track = self._tracks[track_idx]
            prev = track.position.copy()
            track.position = points[det_idx]
            track.predicted_position = points[det_idx].copy()
            track.velocity = (track.position - prev) / dt
            track.last_timestamp = float(timestamp) if timestamp is not None else track.last_timestamp
            alpha = min(1.0, max(0.0, float(self._config.radius_smoothing_alpha)))
            track.radius = (1.0 - alpha) * float(track.radius) + alpha * float(radius_values[det_idx])
            track.unmatched_frames = 0
            track.age += 1
            matched_pairs.append((track.id, det_idx))

        removed_track_ids: List[int] = []
        for track_idx in unmatched_tracks:
            track = self._tracks[track_idx]
            track.position = track.position + track.velocity * dt
            track.predicted_position = track.position.copy()
            track.unmatched_frames += 1
            track.age += 1
            if track.unmatched_frames > self._config.max_unmatched_frames:
                track.is_active = False
                removed_track_ids.append(track.id)

        self._tracks = [track for track in self._tracks if track.is_active]

        new_track_ids: List[int] = []
        for det_idx in unmatched_dets:
            track = DropletTrack(
                id=self._next_id,
                position=points[det_idx],
                radius=float(radius_values[det_idx]),
                last_timestamp=float(timestamp) if timestamp is not None else None,
            )
            track.predicted_position = track.position.copy()
            self._tracks.append(track)
            new_track_ids.append(track.id)
            matched_pairs.append((track.id, det_idx))
            self._next_id += 1

        return TrackingResult(
            active_tracks=[track for track in self._tracks if track.is_active],
            matched_pairs=matched_pairs,
            new_track_ids=new_track_ids,
            removed_track_ids=removed_track_ids,
            total_count=self._next_id - 1,
        )

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1
        self._last_timestamp = None

    def get_active_tracks(self) -> List[DropletTrack]:
        return [track for track in self._tracks if track.is_active]
