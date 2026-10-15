from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from backend.vision.config import DebugConfig, DetectorConfig
from backend.vision.detector import DropletDetector
from backend.vision.rectified_measurement import detect_rectified_generation_plugs


def test_user_corrected_boundaries_and_negative_frames() -> None:
    """Calibration regression, not a claim of held-out accuracy."""
    root = Path(__file__).parent / "fixtures" / "user_phase_calibration"
    records = json.loads((root / "labels.json").read_text(encoding="utf-8"))
    detectors = {}
    relative_errors = []
    for record in records:
        q1 = record["q1"]
        if q1 not in detectors:
            detectors[q1] = DropletDetector(DetectorConfig(
                measurement_mode="generation_plug",
                generation_phase_mode="transverse_bounded"), DebugConfig())
        frame = cv2.imread(str(root / f"{record['id']}.png"), cv2.IMREAD_GRAYSCALE)
        _, trace = detect_rectified_generation_plugs(frame, detector=detectors[q1])
        predictions = trace["selected_intervals"]
        if record["negative_frame"]:
            assert predictions == [], record["id"]
            continue
        used = set()
        for a, b, length in predictions:
            scores = [
                (max(0, min(b, d["right"]) - max(a, d["left"])) /
                 (max(b, d["right"]) - min(a, d["left"])), j)
                for j, d in enumerate(record["droplets"]) if j not in used]
            score, index = max(scores, default=(0, -1))
            assert score > .3, (record["id"], a, b)
            used.add(index)
            truth = record["droplets"][index]["length_px"]
            relative_errors.append(abs(length - truth) / truth)
    # Preserve recall while fixing the earlier median 36% length error.
    assert len(relative_errors) >= 150
    assert np.median(relative_errors) < .10
