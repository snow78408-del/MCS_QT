"""Separate spatial occupancy from successful candidate detection.

These counters are frame statistics, never independent droplet counts or
measurement acceptance. Empty phase output does not establish physical absence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class CandidateFrameStatistics:
    frames: int = 0
    candidate_frames: int = 0
    window_occupied_frames: int = 0
    candidates_outside_window_frames: int = 0
    phase_unresolved_frames: int = 0

    def observe(self, intervals: list, *, width: int,
                center_window: tuple[float, float] = (.35, .65)) -> str:
        low, high = center_window
        if width < 2 or not (0 <= low < high <= 1):
            raise ValueError("invalid frame width or centre window")
        if any(len(v) < 2 or not all(math.isfinite(float(x)) for x in v[:2])
               or not 0 <= v[0] < v[1] < width for v in intervals):
            raise ValueError("invalid candidate endpoints")
        self.frames += 1
        if not intervals:
            self.phase_unresolved_frames += 1
            return "phase_unresolved"
        self.candidate_frames += 1
        if any(low * (width - 1) <= (v[0] + v[1]) / 2 <= high * (width - 1)
               for v in intervals):
            self.window_occupied_frames += 1
            return "window_occupied"
        self.candidates_outside_window_frames += 1
        return "candidates_outside_window"

    def to_dict(self) -> dict:
        return {"frames": self.frames, "candidate_frames": self.candidate_frames,
                "window_occupied_frames": self.window_occupied_frames,
                "candidates_outside_window_frames": self.candidates_outside_window_frames,
                "phase_unresolved_frames": self.phase_unresolved_frames,
                "measurement_accepted": False, "absence_confirmed": False,
                "independent_droplet_count": None}
