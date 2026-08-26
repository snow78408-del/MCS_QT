from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from .config import TrackerConfig
    from .tracker import BaseTracker, DropletTrack, TrackingResult, as_points, radius_gated_match
except ImportError:
    from config import TrackerConfig
    from tracker import BaseTracker, DropletTrack, TrackingResult, as_points, radius_gated_match


@dataclass
class _KalmanTrack:
    id: int
    state: np.ndarray
    covariance: np.ndarray
    radius: float
    unmatched_frames: int = 0
    age: int = 1


class KalmanTracker(BaseTracker):
    def __init__(self, config: TrackerConfig) -> None:
        self._config = config
        self._tracks: List[_KalmanTrack] = []
        self._next_id = 1
        self._last_timestamp: float | None = None

        self._f = self._transition(float(self._config.kalman.dt))
        self._h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        self._q = np.eye(4, dtype=np.float32) * float(self._config.kalman.process_noise)
        self._r = np.eye(2, dtype=np.float32) * float(self._config.kalman.measurement_noise)
        self._p0 = np.eye(4, dtype=np.float32) * float(self._config.kalman.initial_covariance)

    def _predict(self, track: _KalmanTrack) -> np.ndarray:
        track.state = self._f @ track.state
        track.covariance = self._f @ track.covariance @ self._f.T + self._q
        return track.state[:2, 0].copy()

    @staticmethod
    def _transition(dt: float) -> np.ndarray:
        return np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def _update_with_measurement(self, track: _KalmanTrack, measurement: np.ndarray) -> None:
        z = measurement.reshape(2, 1)
        innovation = z - (self._h @ track.state)
        s = self._h @ track.covariance @ self._h.T + self._r
        k = track.covariance @ self._h.T @ np.linalg.inv(s)
        track.state = track.state + (k @ innovation)
        identity = np.eye(4, dtype=np.float32)
        track.covariance = (identity - (k @ self._h)) @ track.covariance

    def _to_public_track(self, track: _KalmanTrack, predicted_position: np.ndarray) -> DropletTrack:
        position = track.state[:2, 0].astype(np.float32)
        velocity = track.state[2:, 0].astype(np.float32)
        return DropletTrack(
            id=track.id,
            position=position,
            radius=float(track.radius),
            velocity=velocity,
            predicted_position=predicted_position.astype(np.float32),
            unmatched_frames=track.unmatched_frames,
            age=track.age,
            is_active=True,
        )

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
        self._f = self._transition(dt)
        points = as_points(detections)
        radius_values = list(radii) if radii is not None else [0.0] * len(points)
        if len(radius_values) < len(points):
            radius_values += [0.0] * (len(points) - len(radius_values))

        predicted_positions: List[np.ndarray] = []
        for track in self._tracks:
            predicted_positions.append(self._predict(track))

        matches, unmatched_tracks, unmatched_dets = radius_gated_match(
            predicted_positions,
            [float(track.radius) for track in self._tracks],
            points,
            radius_values,
            max_distance=float(self._config.match_distance),
            min_distance=float(self._config.min_match_distance),
            distance_radius_ratio=float(self._config.match_distance_radius_ratio),
            radius_match_ratio=float(self._config.radius_match_ratio),
        )

        matched_pairs: List[Tuple[int, int]] = []
        for track_idx, det_idx in matches:
            track = self._tracks[track_idx]
            self._update_with_measurement(track, points[det_idx])
            alpha = min(1.0, max(0.0, float(self._config.radius_smoothing_alpha)))
            track.radius = (1.0 - alpha) * float(track.radius) + alpha * float(radius_values[det_idx])
            track.unmatched_frames = 0
            track.age += 1
            matched_pairs.append((track.id, det_idx))

        removed_track_ids: List[int] = []
        for track_idx in unmatched_tracks:
            track = self._tracks[track_idx]
            track.unmatched_frames += 1
            track.age += 1
            if track.unmatched_frames > self._config.max_unmatched_frames:
                removed_track_ids.append(track.id)

        removed_set = set(removed_track_ids)
        self._tracks = [track for track in self._tracks if track.id not in removed_set]

        new_track_ids: List[int] = []
        for det_idx in unmatched_dets:
            point = points[det_idx]
            state = np.array([[point[0]], [point[1]], [0.0], [0.0]], dtype=np.float32)
            track = _KalmanTrack(
                id=self._next_id,
                state=state,
                covariance=self._p0.copy(),
                radius=float(radius_values[det_idx]),
            )
            self._tracks.append(track)
            new_track_ids.append(track.id)
            matched_pairs.append((track.id, det_idx))
            self._next_id += 1

        predicted_lookup = {track.id: self._f @ track.state for track in self._tracks}
        active_tracks = [
            self._to_public_track(track, predicted_lookup[track.id][:2, 0])
            for track in self._tracks
        ]

        return TrackingResult(
            active_tracks=active_tracks,
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
        tracks: List[DropletTrack] = []
        for track in self._tracks:
            pred = self._f @ track.state
            tracks.append(self._to_public_track(track, pred[:2, 0]))
        return tracks
