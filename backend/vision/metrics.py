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
    period_id: int
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
    start_coord: float
    last_coord: float
    counted: bool = False


class MetricsCalculator:
    def __init__(self, config: MetricsConfig, logger: Callable[[str], None] | None = None) -> None:
        self._config = config
        self._log = logger or (lambda _msg: None)
        self._diameter_history: Deque[float] = deque(maxlen=max(1, config.rolling_window))
        self._period_start: float | None = None
        self._period_diameters: dict[int, list[float]] = {}
        self._period_bead_max: dict[int, int] = {}
        self._period_crossing_count = 0
        self._completed_period_id = 0
        self._completed_period: tuple[list[float], int, int, int, int] = ([], 0, 0, 0, 0)
        self._track_bead_max: Dict[int, int] = {}
        self._track_state: Dict[int, _TrackState] = {}
        self._counted_track_ids: set[int] = set()
        self._last_no_droplet_log_time = 0.0

    @staticmethod
    def _did_cross_line(prev_y: float, cur_y: float, line_y: float) -> bool:
        if prev_y == cur_y:
            return False
        return (prev_y <= line_y < cur_y) or (prev_y >= line_y > cur_y)

    def _update_crossing_count(self, track: DropletTrack, line_y: float) -> bool:
        axis = str(self._config.flow_axis).strip().lower()
        axis_index = 1 if axis == "y" else 0
        cur_coord = float(track.position[axis_index])
        state = self._track_state.get(track.id)
        if state is None:
            self._track_state[track.id] = _TrackState(
                start_coord=cur_coord,
                last_coord=cur_coord,
                counted=False,
            )
            return False

        prev_coord = state.last_coord
        state.last_coord = cur_coord
        if state.counted:
            return False

        crossed = self._did_cross_line(prev_coord, cur_coord, line_y)
        displacement = abs(cur_coord - state.start_coord)
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

    def update(
        self,
        tracking: TrackingResult,
        beads: BeadResult,
        frame_height: int,
        frame_width: int | None = None,
        timestamp: float | None = None,
    ) -> MetricsResult:
        axis = str(self._config.flow_axis).strip().lower()
        line_extent = frame_height if axis == "y" else int(frame_width or frame_height)
        line_y = float(max(1, line_extent) * float(self._config.count_line_ratio))

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
        frame_track_diameters: dict[int, float] = {}
        new_crossing_count = 0
        for track in valid_tracks:
            if float(track.radius) > 0.0:
                diameter = float(track.radius) * 2.0
                frame_diameters.append(diameter)
                frame_track_diameters[int(track.id)] = diameter
            if self._update_crossing_count(track, line_y):
                new_crossing_count += 1

        for track_id in tracking.removed_track_ids:
            self._track_state.pop(int(track_id), None)
            self._track_bead_max.pop(int(track_id), None)

        if frame_diameters:
            self._diameter_history.extend(frame_diameters)
        else:
            log_now = time.monotonic()
            interval = max(0.1, float(self._config.no_droplet_log_interval_s))
            if log_now - self._last_no_droplet_log_time >= interval:
                self._last_no_droplet_log_time = log_now
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
        window_diameters, window_single_cell_count, window_droplet_count, new_crossing_count, period_id = self._update_realtime_window(
            frame_track_diameters=frame_track_diameters,
            frame_track_bead_counts=frame_bead_counts,
            frame_crossing_count=new_crossing_count,
            timestamp=timestamp,
        )
        # Every distinct detected track in the control period contributes one
        # median diameter. Do not discard period droplets as statistical
        # outliers before computing the average used by PID.
        frame_diameters = list(window_diameters)
        # Public "frame_*" fields are retained for API compatibility, but the
        # control/monitoring values represent distinct tracks observed during
        # the configured control-period window.
        frame_droplet_count = window_droplet_count
        frame_single_cell_count = window_single_cell_count
        frame_single_cell_rate = (
            (float(window_single_cell_count) / float(window_droplet_count)) * 100.0
            if window_droplet_count > 0
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
            uniformity_reason = "控制周期内无有效液滴"
        elif len(frame_diameters) <= 1:
            uniformity_status = "样本不足"
            uniformity_reason = "控制周期内仅有一个液滴，均匀程度样本不足"
        elif frame_diameter_cv is None:
            uniformity_status = "样本不足"
            uniformity_reason = "控制周期平均直径无效，无法计算均匀程度"
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
            reason = "控制周期内无有效液滴"
        elif average_diameter is None or average_diameter <= 0.0:
            valid_for_control = False
            reason = "控制周期平均直径无效"
        elif frame_droplet_count < int(self._config.min_active_for_control):
            valid_for_control = False
            reason = "控制周期有效液滴数量不足"
        elif sample_size < int(self._config.min_samples_for_control):
            valid_for_control = False
            reason = "控制周期样本数不足"

        if (
            valid_for_control
            and frame_diameter_cv is not None
            and frame_diameter_cv > float(self._config.max_diameter_cv_for_control)
        ):
            valid_for_control = False
            reason = (
                "diameter measurements are unstable: "
                f"cv={frame_diameter_cv:.2f}% > "
                f"{float(self._config.max_diameter_cv_for_control):.2f}%"
            )

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
            period_id=period_id,
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
        frame_track_diameters: dict[int, float],
        frame_track_bead_counts: dict[int, int],
        frame_crossing_count: int = 0,
        timestamp: float | None = None,
    ) -> tuple[list[float], int, int, int, int]:
        now = float(timestamp) if timestamp is not None else time.monotonic()
        period_s = max(0.001, float(self._config.realtime_window_ms) / 1000.0)
        if self._period_start is None or now < self._period_start:
            self._period_start = now
            self._period_diameters.clear()
            self._period_bead_max.clear()
            self._period_crossing_count = 0

        if now >= self._period_start + period_s:
            completed_diameters = [
                float(np.median(np.asarray(values, dtype=np.float32)))
                for values in self._period_diameters.values()
                if values
            ]
            completed_single = sum(
                1 for track_id in self._period_diameters if self._period_bead_max.get(track_id, 0) == 1
            )
            self._completed_period_id += 1
            self._completed_period = (
                completed_diameters,
                int(completed_single),
                len(self._period_diameters),
                int(self._period_crossing_count),
                int(self._completed_period_id),
            )
            elapsed_periods = max(1, int((now - self._period_start) // period_s))
            self._period_start += elapsed_periods * period_s
            self._period_diameters.clear()
            self._period_bead_max.clear()
            self._period_crossing_count = 0

        for track_id, diameter in frame_track_diameters.items():
            self._period_diameters.setdefault(int(track_id), []).append(float(diameter))
        for track_id, bead_count in frame_track_bead_counts.items():
            key = int(track_id)
            self._period_bead_max[key] = max(self._period_bead_max.get(key, 0), int(bead_count))
        self._period_crossing_count += max(0, int(frame_crossing_count))
        return self._completed_period

    def _robust_diameter_samples(self, values: list[float]) -> list[float]:
        samples = np.asarray(
            [float(value) for value in values if np.isfinite(value) and float(value) > 0.0],
            dtype=np.float32,
        )
        if samples.size < 5:
            return samples.astype(float).tolist()

        median = float(np.median(samples))
        deviations = np.abs(samples - median)
        mad = float(np.median(deviations))
        if mad <= 1e-6:
            return samples.astype(float).tolist()

        robust_sigma = 1.4826 * mad
        limit = max(1.0, float(self._config.robust_mad_multiplier) * robust_sigma)
        filtered = samples[deviations <= limit]
        if filtered.size < int(self._config.min_samples_for_control):
            return samples.astype(float).tolist()
        return filtered.astype(float).tolist()

    def reset(self) -> None:
        self._diameter_history.clear()
        self._period_start = None
        self._period_diameters.clear()
        self._period_bead_max.clear()
        self._period_crossing_count = 0
        self._completed_period_id = 0
        self._completed_period = ([], 0, 0, 0, 0)
        self._track_bead_max.clear()
        self._track_state.clear()
        self._counted_track_ids.clear()
        self._last_no_droplet_log_time = 0.0
