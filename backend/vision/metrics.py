from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict

import numpy as np

try:
    from .bead_counter import BeadResult
    from .config import MetricsConfig
    from .tracker import DropletTrack, TrackingResult
except ImportError:
    from bead_counter import BeadResult
    from config import MetricsConfig
    from tracker import DropletTrack, TrackingResult


@dataclass
class ControlMetrics:
    average_diameter: float | None
    current_active_droplets: int
    sample_size: int
    valid_for_control: bool
    reason: str
    frame_droplet_count: int
    total_droplet_count: int
    new_crossing_count: int
    frame_single_cell_count: int
    frame_diameters: list[float]
    frame_diameter_sum: float
    frame_avg_diameter: float | None
    frame_single_cell_rate: float | None
    frame_diameter_std: float | None
    frame_diameter_cv: float | None
    uniformity_valid: bool
    uniformity_status: str
    uniformity_reason: str


@dataclass
class AnalysisMetrics:
    total_droplets: int
    average_diameter: float | None
    current_valid_droplets: int
    single_bead_count: int
    single_bead_rate: float
    empty_count: int
    empty_rate: float
    multi_bead_count: int
    multi_bead_rate: float
    frame_droplet_count: int
    new_crossing_count: int
    frame_single_cell_count: int
    frame_diameters: list[float]
    frame_diameter_sum: float
    frame_avg_diameter: float | None
    frame_single_cell_rate: float | None
    frame_diameter_std: float | None
    frame_diameter_cv: float | None
    uniformity_valid: bool
    uniformity_status: str
    uniformity_reason: str


@dataclass
class MetricsResult:
    control: ControlMetrics
    analysis: AnalysisMetrics


@dataclass
class _TrackState:
    start_y: float
    last_y: float
    counted: bool = False


class MetricsCalculator:
    def __init__(self, config: MetricsConfig, logger: Callable[[str], None] | None = None) -> None:
        self._config = config
        self._log = logger or (lambda _msg: None)
        self._diameter_history: Deque[float] = deque(maxlen=max(1, config.rolling_window))
        self._realtime_window: Deque[tuple[float, list[float], int, int]] = deque()
        self._realtime_window_start: float | None = None
        self._last_realtime_summary: tuple[list[float], int, int] = ([], 0, 0)
        self._track_bead_max: Dict[int, int] = {}
        self._track_state: Dict[int, _TrackState] = {}
        self._counted_track_ids: set[int] = set()

    @staticmethod
    def _did_cross_line(prev_y: float, cur_y: float, line_y: float) -> bool:
        if prev_y == cur_y:
            return False
        return (prev_y <= line_y < cur_y) or (prev_y >= line_y > cur_y)

    def _update_crossing_count(self, track: DropletTrack, line_y: float) -> bool:
        cur_y = float(track.position[1])
        state = self._track_state.get(track.id)
        if state is None:
            self._track_state[track.id] = _TrackState(start_y=cur_y, last_y=cur_y, counted=False)
            return False

        prev_y = state.last_y
        state.last_y = cur_y
        if state.counted:
            return False

        crossed = self._did_cross_line(prev_y, cur_y, line_y)
        displacement = abs(cur_y - state.start_y)
        age_ok = int(track.age) >= int(self._config.min_track_age_for_count)
        disp_ok = displacement >= float(self._config.min_track_displacement_for_count)
        if crossed and age_ok and disp_ok:
            state.counted = True
            self._counted_track_ids.add(track.id)
            self._log(
                f"[VISION][COUNT] new real droplet count: track_id={track.id}, "
                f"total={len(self._counted_track_ids)}"
            )
            return True
        return False

    def update(self, tracking: TrackingResult, beads: BeadResult, frame_height: int) -> MetricsResult:
        line_y = float(max(1, frame_height) * float(self._config.count_line_ratio))

        # Only tracks observed in this exact frame are allowed into current-frame statistics.
        observed_track_ids: set[int] = set(int(track_id) for track_id, _ in tracking.matched_pairs)
        observed_track_ids.update(int(track_id) for track_id in tracking.new_track_ids)

        valid_tracks: list[DropletTrack] = [
            track
            for track in tracking.active_tracks
            if int(track.id) in observed_track_ids
            and int(track.age) >= int(self._config.min_track_age_for_count)
        ]
        valid_track_ids = {int(track.id) for track in valid_tracks}
        frame_droplet_count = len(valid_tracks)

        frame_diameters: list[float] = []
        new_crossing_count = 0
        for track in valid_tracks:
            if float(track.radius) > 0.0:
                frame_diameters.append(float(track.radius) * 2.0)
            if self._update_crossing_count(track, line_y):
                new_crossing_count += 1

        for track_id in tracking.removed_track_ids:
            self._track_state.pop(int(track_id), None)
            self._track_bead_max.pop(int(track_id), None)

        if frame_diameters:
            self._diameter_history.extend(frame_diameters)
        else:
            self._log("[VISION][NO_DROPLET] current frame has no valid droplets")

        frame_bead_counts: dict[int, int] = {}
        for droplet in beads.droplets:
            droplet_id = int(droplet.droplet_id)
            bead_count = int(droplet.bead_count)
            if droplet_id in valid_track_ids:
                frame_bead_counts[droplet_id] = bead_count
            prev_max = self._track_bead_max.get(droplet_id, 0)
            if bead_count > prev_max:
                self._track_bead_max[droplet_id] = bead_count

        counted_ids = sorted(self._counted_track_ids)
        bead_counts = [self._track_bead_max.get(track_id, 0) for track_id in counted_ids]
        total_droplets = len(counted_ids)

        empty_count = sum(1 for value in bead_counts if value == 0)
        single_count = sum(1 for value in bead_counts if value == 1)
        multi_count = sum(1 for value in bead_counts if value >= 2)

        frame_single_cell_count = sum(1 for value in frame_bead_counts.values() if int(value) == 1)
        window_diameters, window_single_cell_count, window_droplet_count = self._update_realtime_window(
            frame_diameters=frame_diameters,
            frame_single_cell_count=frame_single_cell_count,
            frame_droplet_count=frame_droplet_count,
        )
        frame_diameters = window_diameters
        frame_droplet_count = window_droplet_count
        frame_single_cell_count = window_single_cell_count
        frame_single_cell_rate = (
            (float(frame_single_cell_count) / float(frame_droplet_count)) * 100.0
            if frame_droplet_count > 0
            else None
        )

        frame_diameter_sum = float(sum(frame_diameters)) if frame_diameters else 0.0
        frame_avg_diameter = float(np.mean(frame_diameters)) if frame_diameters else None
        frame_diameter_std = float(np.std(frame_diameters, ddof=0)) if frame_diameters else None
        if frame_avg_diameter is not None and frame_avg_diameter > 0.0 and frame_diameter_std is not None:
            frame_diameter_cv = float(frame_diameter_std / frame_avg_diameter * 100.0)
        else:
            frame_diameter_cv = None

        uniformity_valid = False
        uniformity_status = "样本不足"
        uniformity_reason = ""
        if frame_droplet_count <= 0:
            uniformity_status = "当前无液滴"
            uniformity_reason = "当前帧无有效液滴"
        elif len(frame_diameters) <= 1:
            uniformity_status = "样本不足"
            uniformity_reason = "当前帧仅有一个液滴，均匀程度样本不足"
        elif frame_diameter_cv is None:
            uniformity_status = "样本不足"
            uniformity_reason = "当前帧平均直径无效，无法计算均匀程度"
        else:
            uniformity_valid = True
            if frame_diameter_cv <= float(self._config.uniformity_good_threshold):
                uniformity_status = "均匀"
            elif frame_diameter_cv <= float(self._config.uniformity_normal_threshold):
                uniformity_status = "一般"
            else:
                uniformity_status = "波动较大"

        average_diameter = frame_avg_diameter
        sample_size = len(frame_diameters)
        valid_for_control = True
        reason = "ok"
        if frame_droplet_count == 0:
            valid_for_control = False
            reason = "当前帧无有效液滴"
        elif average_diameter is None or average_diameter <= 0.0:
            valid_for_control = False
            reason = "当前帧平均直径无效"
        elif frame_droplet_count < int(self._config.min_active_for_control):
            valid_for_control = False
            reason = "当前帧有效液滴数量不足"
        elif sample_size < int(self._config.min_samples_for_control):
            valid_for_control = False
            reason = "当前帧控制样本数不足"

        denom = float(total_droplets) if total_droplets > 0 else 1.0
        analysis = AnalysisMetrics(
            total_droplets=total_droplets,
            average_diameter=average_diameter,
            current_valid_droplets=frame_droplet_count,
            single_bead_count=single_count,
            single_bead_rate=frame_single_cell_rate if frame_single_cell_rate is not None else 0.0,
            empty_count=empty_count,
            empty_rate=(empty_count / denom) * 100.0 if total_droplets else 0.0,
            multi_bead_count=multi_count,
            multi_bead_rate=(multi_count / denom) * 100.0 if total_droplets else 0.0,
            frame_droplet_count=frame_droplet_count,
            new_crossing_count=new_crossing_count,
            frame_single_cell_count=frame_single_cell_count,
            frame_diameters=list(frame_diameters),
            frame_diameter_sum=frame_diameter_sum,
            frame_avg_diameter=frame_avg_diameter,
            frame_single_cell_rate=frame_single_cell_rate,
            frame_diameter_std=frame_diameter_std,
            frame_diameter_cv=frame_diameter_cv,
            uniformity_valid=uniformity_valid,
            uniformity_status=uniformity_status,
            uniformity_reason=uniformity_reason,
        )

        control = ControlMetrics(
            average_diameter=average_diameter,
            current_active_droplets=frame_droplet_count,
            sample_size=sample_size,
            valid_for_control=valid_for_control,
            reason=reason,
            frame_droplet_count=frame_droplet_count,
            total_droplet_count=total_droplets,
            new_crossing_count=new_crossing_count,
            frame_single_cell_count=frame_single_cell_count,
            frame_diameters=list(frame_diameters),
            frame_diameter_sum=frame_diameter_sum,
            frame_avg_diameter=frame_avg_diameter,
            frame_single_cell_rate=frame_single_cell_rate,
            frame_diameter_std=frame_diameter_std,
            frame_diameter_cv=frame_diameter_cv,
            uniformity_valid=uniformity_valid,
            uniformity_status=uniformity_status,
            uniformity_reason=uniformity_reason,
        )

        return MetricsResult(control=control, analysis=analysis)

    def _update_realtime_window(
        self,
        *,
        frame_diameters: list[float],
        frame_single_cell_count: int,
        frame_droplet_count: int,
    ) -> tuple[list[float], int, int]:
        now = time.monotonic()
        window_s = max(0.001, float(self._config.realtime_window_ms) / 1000.0)
        if self._realtime_window_start is None:
            self._realtime_window_start = now
        self._realtime_window.append(
            (
                now,
                [float(value) for value in frame_diameters],
                int(frame_single_cell_count),
                int(frame_droplet_count),
            )
        )
        if now - self._realtime_window_start < window_s:
            diameters, single_count, droplet_count = self._last_realtime_summary
            return list(diameters), int(single_count), int(droplet_count)

        diameters: list[float] = []
        single_count = 0
        droplet_count = 0
        for _, entry_diameters, entry_single_count, entry_droplet_count in self._realtime_window:
            diameters.extend(entry_diameters)
            single_count += int(entry_single_count)
            droplet_count += int(entry_droplet_count)
        self._last_realtime_summary = (list(diameters), int(single_count), int(droplet_count))
        self._realtime_window.clear()
        self._realtime_window_start = None
        return list(diameters), int(single_count), int(droplet_count)

    def reset(self) -> None:
        self._diameter_history.clear()
        self._realtime_window.clear()
        self._realtime_window_start = None
        self._last_realtime_summary = ([], 0, 0)
        self._track_bead_max.clear()
        self._track_state.clear()
        self._counted_track_ids.clear()
