from __future__ import annotations

import cv2
import numpy as np

from backend.vision.benchmark import LabeledDroplet, LabeledFrame, evaluate_labeled_frames
from backend.vision.config import default_config
from backend.vision.detector import DropletDetector
from backend.vision.tuning import TuningFrame


def test_labeled_benchmark_reports_accuracy_radius_error_and_runtime() -> None:
    config = default_config()
    config.detector.enable_hough_candidates = False
    frames: list[TuningFrame] = []
    labels: list[LabeledFrame] = []
    for index in range(4):
        image = np.full((180, 320), 190, dtype=np.uint8)
        center = (90 + 3 * index, 90)
        cv2.circle(image, center, 24, 45, 3)
        frames.append(TuningFrame(index, image))
        labels.append(LabeledFrame(index, (LabeledDroplet(*center, 24.0),)))

    benchmark = evaluate_labeled_frames(
        DropletDetector(config.detector, config.debug),
        frames,
        labels,
    )

    assert benchmark.precision == 1.0
    assert benchmark.recall == 1.0
    assert benchmark.f1 == 1.0
    assert benchmark.mean_radius_error_px is not None
    assert benchmark.mean_radius_error_px <= 3.0
    assert benchmark.mean_frame_ms > 0.0
    assert benchmark.p95_frame_ms >= benchmark.mean_frame_ms * 0.5
    assert benchmark.meets_targets()
