from unittest.mock import Mock

import numpy as np
import pytest

from backend.vision.config import default_config
from backend.vision.pipeline import VisionPipeline


def _walls(top: float):
    return [{"x1": 0.0, "y1": y / 100.0, "x2": 1.0, "y2": y / 100.0}
            for y in (top, top + 20.0)]


def test_rejected_current_frame_skips_detector_even_with_saved_manual_walls():
    config = default_config()
    config.roi.enabled = True
    config.roi.wall_lines = _walls(20.0)
    pipeline = VisionPipeline(config)
    pipeline.detector.detect = Mock(side_effect=AssertionError("detector must not run"))

    result = pipeline.process_frame(np.full((100, 200), 40, np.uint8), timestamp=1.0,
                                    current_wall_lines=[], wall_rejection_reason="no_current_frame_evidence")

    assert result.channel_region.status == "rejected"
    assert result.detections.centers == []
    assert result.metrics.control.valid_for_control is False
    assert result.metrics.control.reason == "no_current_frame_evidence"


def test_current_walls_override_stale_saved_walls_before_detection():
    config = default_config()
    config.roi.enabled = True
    config.roi.wall_lines = _walls(20.0)
    pipeline = VisionPipeline(config)
    seen = []
    received = []
    original = pipeline.detector.detect

    def detect(gray, mode=None, **kwargs):
        seen.append(gray.copy())
        received.append(dict(kwargs))
        return original(gray, mode, **kwargs)

    pipeline.detector.detect = detect
    frame = np.full((100, 200), 20, np.uint8)
    frame[60:80] = 90

    result = pipeline.process_frame(frame, timestamp=1.0, current_wall_lines=_walls(60.0))

    assert result.channel_region.status == "localized"
    assert seen and seen[0].mean() > 75
    # 扶正图必须把**本帧**几何显式交给 detector，不让它按名义配置或长边猜轴向。
    assert received[0]["channel_width_px"] == pytest.approx(
        float(result.detection_geometry["channel_width_px"]))
    assert received[0]["flow_axis"] == "x"
    assert result.detection_geometry["reference_source"] == "frame_rectified_geometry"
    assert result.detection_geometry["rectified"] is True
