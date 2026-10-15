from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.vision.capsule_profile import calibrated_phase_intervals
from backend.vision.config import DebugConfig, DetectorConfig
from backend.vision.detector import DropletDetector
from backend.vision.rectified_measurement import detect_rectified_generation_plugs, trace_summary
from backend.vision.measurement_statistics import CandidateFrameStatistics

ROOT = Path(__file__).parent / "fixtures" / "phase_registration"


def detect(name: str) -> dict:
    frame = cv2.imread(str(ROOT / name), cv2.IMREAD_GRAYSCALE)
    assert frame is not None
    detector = DropletDetector(DetectorConfig(
        measurement_mode="generation_plug", generation_phase_mode="transverse_bounded"), DebugConfig())
    return detect_rectified_generation_plugs(frame, detector=detector)[1]


def test_recent_a6_registration_shifts_keep_lengths() -> None:
    traces = [detect(f"A6_{shift}.png") for shift in [-2, 0, 2]]
    assert all(t["selected_intervals"] for t in traces)
    # Match the same downstream object. Frame medians change when an extra
    # upstream/edge object becomes observable; they are not endpoint pairs.
    matched = [[v for v in t["selected_intervals"] if v[0] < 285 < v[1]] for t in traces]
    assert all(len(v) == 1 for v in matched)
    for endpoint in (0, 1):
        positions = [v[0][endpoint] for v in matched]
        assert max(positions) - min(positions) <= 2
    assert "phase_diagnostics" in trace_summary(traces[0])


def test_v2_two_geometry_estimates_both_resolve_phase() -> None:
    assert detect("V2_tail.png")["selected_intervals"]
    assert detect("V2_reference.png")["selected_intervals"]


@pytest.mark.parametrize("shift", [-2, 0, 2])
def test_recent_long_structure_is_not_rescued_as_droplet(shift: int) -> None:
    assert detect(f"A7_{shift}.png")["selected_intervals"] == []


def test_weak_texture_is_not_contrast_amplified() -> None:
    rng = np.random.default_rng(123)
    frame = rng.integers(120, 126, size=(32, 500), dtype=np.uint8)
    diagnostics = {}
    assert calibrated_phase_intervals(frame, 31, diagnostics=diagnostics) == []
    assert diagnostics["status"] == "phase_contrast_unresolved"


def test_window_occupancy_is_not_detection_failure() -> None:
    stats = CandidateFrameStatistics()
    assert stats.observe([(20, 80, 60)], width=500) == "candidates_outside_window"
    assert stats.observe([(220, 280, 60)], width=500) == "window_occupied"
    assert stats.observe([], width=500) == "phase_unresolved"
    result = stats.to_dict()
    assert result["candidate_frames"] == 2
    assert result["phase_unresolved_frames"] == 1
    assert result["candidates_outside_window_frames"] == 1
    assert not result["absence_confirmed"]
    assert result["independent_droplet_count"] is None


def test_invalid_candidate_does_not_increment_statistics() -> None:
    stats = CandidateFrameStatistics()
    with pytest.raises(ValueError):
        stats.observe([(30, float("nan"))], width=500)
    assert stats.frames == 0
