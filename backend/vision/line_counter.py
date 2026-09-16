from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading

import cv2
import numpy as np

from .rectified_roi import wall_line_quad


@dataclass(frozen=True)
class FrequencyWindow:
    count: int = 0
    rate_hz: float | None = None
    valid: bool = False
    reason: str = "waiting for complete window"


class ContinuousLineCounter:
    """Small fixed gate for unidirectional plugs, independent of size batches.

    Events are completed body passages, not tracker IDs. Thresholds are learned
    during startup and frozen until reset. This is a passage-frequency monitor:
    reverse flow, coalescence and unresolved gaps require separate validation.
    """

    def __init__(self, *, history_limit: int = 20000, warmup_frames: int = 32,
                 minimum_phase_contrast: float = 12.0) -> None:
        if history_limit < 2 or warmup_frames < 4:
            raise ValueError("history_limit >= 2 and warmup_frames >= 4 required")
        if not np.isfinite(minimum_phase_contrast) or not 0 < minimum_phase_contrast <= 255:
            raise ValueError("minimum_phase_contrast must be in (0, 255]")
        self._lock = threading.RLock()
        self._limit = history_limit
        self._warmup = warmup_frames
        self._minimum_phase_contrast = float(minimum_phase_contrast)
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._key = None
            self._maps = None
            self._levels: deque[float] = deque(maxlen=self._warmup)
            self._thresholds: tuple[float, float] | None = None
            self._occupied: bool | None = None
            self._armed = False
            self._synchronized = False
            self._last_id: int | None = None
            self._last_time: float | None = None
            self._samples: deque[tuple[float, bool]] = deque(maxlen=self._limit)
            self._events: deque[float] = deque(maxlen=self._limit)
            self._cached_window = None

    def observe_frame(self, frame: np.ndarray, frame_id: int, timestamp: float,
                      wall_lines: list[dict[str, float]], *, line_ratio: float = 0.6) -> None:
        with self._lock:
            key = (frame.shape[:2], line_ratio, tuple(tuple(line.get(k, 0.0) for k in
                   ("x1", "y1", "x2", "y2")) for line in wall_lines))
            if key != self._key:
                self.reset()
                self._key = key
                geometry = wall_line_quad(frame.shape[1], frame.shape[0], wall_lines)
                if geometry is not None:
                    source, width, height = geometry
                    dest = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
                    inverse = cv2.getPerspectiveTransform(dest, source)
                    # Only five columns and at most 64 transverse samples.
                    x, y = np.meshgrid(np.arange(5) + width * line_ratio - 2,
                                       np.linspace(height * .12, height * .88, min(64, height)))
                    points = np.stack((x, y), axis=-1).astype(np.float32)
                    mapped = cv2.perspectiveTransform(points.reshape(-1, 1, 2), inverse).reshape(points.shape)
                    self._maps = (mapped[:, :, 0].copy(), mapped[:, :, 1].copy())
            if self._maps is None:
                return
            strip = cv2.remap(frame, *self._maps, interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if strip.ndim == 3 else strip
            profile = np.mean(gray, axis=1)
            score = float(np.percentile(profile, 95) - np.percentile(profile, 5))
            self.observe_score(score, frame_id, timestamp)

    def observe_score(self, score: float, frame_id: int, timestamp: float) -> None:
        with self._lock:
            if not np.isfinite(score) or not np.isfinite(timestamp):
                self.reset()
                return
            if self._last_time is not None and timestamp <= self._last_time:
                self.reset()
            continuous = self._last_id is None or frame_id == self._last_id + 1
            if not continuous:
                self._occupied = None
                self._armed = False
                self._synchronized = False
            self._last_id, self._last_time = frame_id, timestamp
            ready = self._thresholds is not None
            if not ready:
                self._levels.append(score)
                if len(self._levels) == self._warmup:
                    low, high = np.percentile(self._levels, [10, 90])
                    # Do not learn two "phases" from low-amplitude texture or
                    # compression flicker, even if their relative ratio is high.
                    if high - low >= self._minimum_phase_contrast and high >= max(6.0, low * 1.5):
                        self._thresholds = (float(low + .3*(high-low)), float(low + .65*(high-low)))
            if self._thresholds is not None:
                lower, upper = self._thresholds
                if score <= lower:
                    if self._occupied and self._armed and ready and continuous:
                        self._events.append(timestamp)
                    self._occupied = False
                    self._armed = False
                    self._synchronized = True
                elif score >= upper:
                    if self._occupied is False:
                        self._armed = True
                    self._occupied = True
            self._samples.append((timestamp, ready and continuous and self._synchronized and np.isfinite(score)))

    def window(self, start: float | None, end: float | None) -> FrequencyWindow:
        with self._lock:
            if start is None or end is None or end <= start:
                return FrequencyWindow()
            if self._cached_window is not None and self._cached_window[:2] == (start, end):
                return self._cached_window[2]
            samples = list(self._samples)
            before = [i for i, (t, _) in enumerate(samples) if t <= start]
            after = next((i for i, (t, _) in enumerate(samples) if t >= end), None)
            if not before or after is None:
                return FrequencyWindow(reason="window not covered or history expired")
            if not all(valid for _, valid in samples[before[-1]:after+1]):
                return FrequencyWindow(reason="frame gap or gate calibration incomplete")
            count = sum(start <= t < end for t in self._events)
            result = FrequencyWindow(count, count / (end-start), True, "")
            self._cached_window = (start, end, result)
            return result

    def passage_times(self) -> tuple[float, ...]:
        """Bounded event evidence for offline, event-by-event review."""
        with self._lock:
            return tuple(self._events)
