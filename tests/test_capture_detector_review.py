"""诊断工具必须走生产测量入口，而不是直接调用 detector 的私有方法。

2026-09-24 修改前：``review_frame`` 自己扶正后调 ``detector._detect_generation_plugs``
并传 ``channel_width_px=min(gray.shape)``，与生产路径的参考宽度来源不一致。
现在两者都经 ``measure_generation_plugs``。
"""
from unittest.mock import Mock

import numpy as np

from backend.vision.config import DebugConfig, DetectorConfig
from backend.vision.detector import DropletDetector
from backend.vision.parallel_walls import WallLocalization
from tools.capture_detector_review import review_frame


def _walls(y_top: float = 0.6, y_bottom: float = 0.8):
    return [{"x1": 0.0, "y1": y_top, "x2": 1.0, "y2": y_top},
            {"x1": 0.0, "y1": y_bottom, "x2": 1.0, "y2": y_bottom}]


def _localization(status: str, reason: str, wall_lines):
    return WallLocalization.from_geometry(
        {"frame_id": 42, "image_shape": [200, 100], "wall_lines": list(wall_lines)},
        status, reason)


def _detector() -> DropletDetector:
    config = DetectorConfig(measurement_mode="generation_plug")
    config.generation_min_length_ratio = 0.5
    config.generation_min_raw_outline_contrast = 3.0
    config.generation_min_capsule_outline_ratio = 0.12
    return DropletDetector(config, DebugConfig())


def test_rejected_current_localization_never_calls_detector():
    localizer = Mock()
    localizer.localize.return_value = _localization("rejected", "no_current_frame_evidence", [])
    detector = Mock()
    record, roi, overlay = review_frame(np.zeros((100, 200), np.uint8), detector,
                                        localizer, frame_id=42, timestamp=12.0)
    assert record["status"] == "not_measured"
    assert record["selected"] is None
    assert roi is None and overlay is None
    # 被拒的定位不得触发任何检测，也不得声明几何。
    detector.detect.assert_not_called()
    detector.declare_pixel_cross_section.assert_not_called()
    assert localizer.observe.call_args.kwargs == {"frame_id": 42, "capture_monotonic": 12.0}


def test_rectification_uses_current_localized_walls_and_pixel_units():
    """扶正用当前帧定位的管壁；像素域诊断保留原始像素值，且参考宽度来自本帧几何。"""
    localizer = Mock()
    localizer.localize.return_value = _localization("localized", "", _walls())
    detector = _detector()
    raw = np.zeros((100, 200), np.uint8)
    raw[55:85] = 90

    record, roi, _ = review_frame(raw, detector, localizer, frame_id=42, timestamp=12.0)

    assert record["status"] == "measured_pixels"
    assert np.min(roi) == 90, "扶正不得改变像素值（显示缩放必须发生在测量之后）"
    assert record["physical_scale_validated"] is False
    # 参考宽度来自本帧扶正几何，不是名义 50 µm、也不是 min(shape) 的隐式推导。
    assert record["reference_width_source"] == "frame_rectified_geometry"
    assert record["reference_width_px"] == float(record["axes"]["transverse_index_span"])
    # 声明用的是**像素个数**（= wall_separation_px），不是索引跨度。
    assert record["valid"] is False
    assert record["reason"] == "scale_unknown"
    assert detector._pixel_cross_section_declaration == {
        "cross_px": float(record["axes"]["transverse_pixel_count"]),
        "reason": "diagnostic_pixel_domain",
    }


def test_review_frame_reference_width_follows_frame_not_nominal_config():
    """同一画面、同一 detector 配置：改名义通道 µm 不得改变参考宽度。"""
    localizer = Mock()
    localizer.localize.return_value = _localization("localized", "", _walls())
    raw = np.zeros((100, 200), np.uint8)
    raw[55:85] = 90

    narrow = _detector()
    narrow._config.generation_channel_width_um = 11.0
    narrow._config.generation_channel_height_um = 11.0
    wide = _detector()
    wide._config.generation_channel_width_um = 90.0
    wide._config.generation_channel_height_um = 90.0

    a, _, _ = review_frame(raw, narrow, localizer, frame_id=42, timestamp=12.0)
    b, _, _ = review_frame(raw, wide, localizer, frame_id=42, timestamp=12.0)

    assert a["reference_width_px"] == b["reference_width_px"] == float(a["axes"]["transverse_index_span"])
    assert a["selected"] == b["selected"]
