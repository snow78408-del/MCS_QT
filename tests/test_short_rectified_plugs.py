import cv2
import numpy as np
import pytest

from backend.vision.config import DetectorConfig, DebugConfig
from backend.vision.detector import DropletDetector


def test_short_complete_outline_is_not_rejected_by_old_long_plug_limit():
    frame = np.full((40, 300), 100, np.uint8)
    cv2.rectangle(frame, (40, 7), (70, 32), 180, 2)
    cv2.rectangle(frame, (130, 7), (240, 32), 180, 2)
    detector = DropletDetector(DetectorConfig(measurement_mode="generation_plug"), DebugConfig())
    trace = {}
    detector._detect_generation_plugs(frame, trace, channel_width_px=33)
    assert [s[2] for s in trace["selected_intervals"]] == pytest.approx([34, 114], abs=3)
    assert trace["reference_width_px"] == 33


def test_pixel_channel_reference_does_not_admit_uniform_background():
    detector = DropletDetector(DetectorConfig(measurement_mode="generation_plug"), DebugConfig())
    trace = {}
    detector._detect_generation_plugs(np.full((35, 587), 80, np.uint8), trace, channel_width_px=35)
    assert trace["selected_intervals"] == []
    with pytest.raises(ValueError):
        detector._detect_generation_plugs(np.zeros((35, 587), np.uint8), channel_width_px=50)
