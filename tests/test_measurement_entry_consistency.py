"""生产与诊断共用同一测量入口的回归。

覆盖任务书 §2A 要求的：短液柱、几何变化、未知/非法宽度、未验证标尺，以及
「同一帧、同一生效配置、同一几何经生产入口与诊断入口得到一致的候选与门槛」。
用**真实 detector**，不用桩证明接线。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.vision.config import DebugConfig, DetectorConfig
from backend.vision.detector import DropletDetector
from backend.vision.plug_geometry import rectified_axes
from backend.vision.rectified_measurement import (
    FrameEvidence,
    ScaleEvidence,
    detect_rectified_generation_plugs,
    measure_generation_plugs,
)
from backend.vision.rectified_roi import rectify_channel_frame

WIDTH, HEIGHT = 720, 540
CHANNEL_UM = 50.0
OFFSET = 60.0
TOOLS = ("run_channel_validation_live.py", "capture_detector_review.py",
         "review_current_wall_pair.py", "preflight_current_channel.py")


def _frame(separation: float, plugs=((130, 300),), *, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = int(round(separation))
    canvas = np.full((HEIGHT, WIDTH), 20.0, np.float32) + rng.normal(0.0, 0.6, (HEIGHT, WIDTH))
    y0 = int(round(OFFSET))
    canvas[y0:y0 + rows, :] = 40.0
    for left, right in plugs:
        for x in range(left, right):
            profile = np.full(rows, 48.0, np.float32)
            outer = max(1, int(round(rows * 0.22)))
            profile[:outer] += 9.0
            profile[rows - outer:] += 9.0
            profile[outer:rows - outer] -= 9.0
            canvas[y0:y0 + rows, x] = profile + rng.normal(0.0, 0.6, rows)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _walls(separation: float):
    def norm(y: float) -> float:
        return float(y) / float(HEIGHT)
    return [
        {"x1": 0.0, "y1": norm(OFFSET), "x2": 1.0, "y2": norm(OFFSET)},
        {"x1": 0.0, "y1": norm(OFFSET + separation), "x2": 1.0, "y2": norm(OFFSET + separation)},
    ]


def _detector(nominal_px: float, *, scale: float = 1.0) -> DropletDetector:
    config = DetectorConfig(measurement_mode="generation_plug")
    config.generation_channel_height_um = nominal_px
    config.generation_channel_width_um = nominal_px
    config.generation_min_raw_outline_contrast = 3.0
    config.generation_min_capsule_outline_ratio = 0.12
    detector = DropletDetector(config, DebugConfig())
    detector.configure_expected_diameter(0.0, scale)
    return detector


def _scale(separation: float, *, validated: bool = True) -> ScaleEvidence:
    if not validated:
        return ScaleEvidence(um_per_px=None, source="configured_optical", validated=False)
    return ScaleEvidence(um_per_px=CHANNEL_UM / separation, source="channel_width_reference",
                         validated=True, reference_um=CHANNEL_UM)


def _evidence(frame_id: int = 7) -> FrameEvidence:
    return FrameEvidence(frame_id=frame_id, hardware_frame_id=frame_id, capture_monotonic=1234.5,
                         localization_frame_id=frame_id, time_source="camera_frame_timestamp")


def _measure(separation: float, detector: DropletDetector, *, plugs=((130, 300),),
             scale: ScaleEvidence | None = None, trace: dict | None = None, depth_um=CHANNEL_UM):
    return measure_generation_plugs(
        _frame(separation, plugs), detector=detector, wall_lines=_walls(separation),
        scale=scale if scale is not None else _scale(separation),
        frame_evidence=_evidence(), duct_depth_um=depth_um, trace=trace)


def _rectified_gray(separation: float, plugs=((130, 300),)) -> np.ndarray:
    rectified = rectify_channel_frame(_frame(separation, plugs), _walls(separation))
    return cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY) if rectified.ndim == 3 else rectified


# ------------------------------------------------------------------ 轴与跨度定义

def test_rectified_axes_takes_the_axis_from_the_contract_not_the_longer_side():
    """轴向由扶正变换契约定为 x；跨度区分像素个数与索引跨度。"""
    wide = rectified_axes((35, 587))
    assert (wide.output_height, wide.output_width) == (35, 587)
    assert wide.flow_axis == "x"
    assert wide.transverse_pixel_count == 35
    assert wide.transverse_index_span == 34          # 索引跨度 = 个数 - 1
    assert wide.axial_pixel_count == 587
    assert wide.axial_index_span == 586
    assert wide.to_dict()["span_units"] == "index_span"
    # 「高而窄」的扶正图：不按长边把轴向翻成 y，轴向仍由契约定为 x。
    tall = rectified_axes((587, 35))
    assert tall.flow_axis == "x", "轴向不得按长边推断"
    assert tall.transverse_pixel_count == 587
    assert tall.axial_index_span == 34
    assert tall.to_dict()["wider_than_tall"] is False


def test_illegal_rectified_size_is_rejected_not_clamped():
    with pytest.raises(ValueError):
        rectified_axes((0, 100))
    with pytest.raises(ValueError):
        rectified_axes((5, 5, 5))
    with pytest.raises(ValueError):
        detect_rectified_generation_plugs(np.zeros((1, 300), np.uint8), detector=_detector(40.0))
    with pytest.raises(ValueError):
        detect_rectified_generation_plugs(np.zeros((0, 0), np.uint8), detector=_detector(40.0))


def test_transverse_span_below_two_pixels_is_refused_by_the_detector():
    detector = _detector(40.0)
    with pytest.raises(ValueError):
        detect_rectified_generation_plugs(np.zeros((1, 300), np.uint8), detector=detector)


# ------------------------------------------------ 生产入口与检测内核一致

def test_production_and_detection_core_agree_on_candidates_and_thresholds():
    detector = _detector(42.0)
    trace_entry: dict = {}
    measurement = _measure(42.0, detector, trace=trace_entry)
    trace_core: dict = {}
    detect_rectified_generation_plugs(_rectified_gray(42.0), detector=detector, trace=trace_core)

    core_lengths = tuple(float(item[2]) for item in trace_core["selected_intervals"])
    assert measurement.plug_lengths_px == core_lengths
    assert trace_entry["selected_intervals"] == trace_core["selected_intervals"]
    assert trace_entry["minimum_length_px"] == trace_core["minimum_length_px"]
    assert trace_entry["maximum_length_px"] == trace_core["maximum_length_px"]
    assert trace_entry["reference_width_px"] == trace_core["reference_width_px"]
    assert trace_entry["reference_width_source"] == "frame_rectified_geometry"


# --------------------------------------------------- 门槛跟着本帧几何，不跟名义配置

def test_short_plug_decision_follows_frame_geometry_not_nominal_config():
    """同一画面：名义通道 40 px 与 42 px 的两次配置，必须给出相同候选。"""
    frame_separation = 42.0
    tight = _measure(frame_separation, _detector(40.0))
    loose = _measure(frame_separation, _detector(42.0))
    assert tight.plug_lengths_px == loose.plug_lengths_px
    assert tight.reason == loose.reason == "ok"


def test_reference_width_reported_is_the_frame_span_not_the_nominal_one():
    detector = _detector(40.0)
    trace: dict = {}
    _measure(42.0, detector, trace=trace)
    # 扶正横向 42 像素 → 索引跨度 41；门槛参考宽用索引跨度，与实测长度同单位。
    assert trace["reference_width_px"] == pytest.approx(41.0)
    assert trace["reference_width_source"] == "frame_rectified_geometry"
    assert trace["flow_axis"] == "x"
    assert trace["flow_axis_source"] == "explicit", "扶正图必须显式给轴向，不得用长边启发式"
    assert trace["config_channel_px"] == pytest.approx((40.0, 40.0))


def test_short_plug_near_threshold_is_decided_by_the_frame_reference():
    """短柱扫掠：门槛取本帧跨度后，判定不再随名义配置漂移。"""
    decisions = {}
    for nominal in (40.0, 42.0):
        detector = _detector(nominal)
        trace: dict = {}
        _measure(42.0, detector, plugs=((130, 151),), trace=trace)
        decisions[nominal] = bool(trace["selected_intervals"])
    assert len(set(decisions.values())) == 1, decisions


# ------------------------------------------------------------ 几何变化与未验证标尺

def test_geometry_mismatch_is_reported_but_pixel_candidates_survive():
    """几何不一致：作废物理结论（valid=False、同因），但不隐藏像素域候选。"""
    measurement = _measure(40.0, _detector(29.0))
    assert measurement.valid is False
    assert measurement.reason == "detector_duct_geometry_mismatch"
    assert measurement.duct.detector_geometry_consistent is False
    # 关键：候选与门槛仍在，诊断能看见——只是不得用于控制或物理标定。
    assert measurement.detection_trace["selected_intervals"]
    assert measurement.plug_lengths_px
    assert measurement.plug_lengths_um == ()
    assert measurement.equivalent_diameters_um == ()


def test_geometry_change_between_frames_shifts_the_threshold():
    detector = _detector(40.0)
    narrow = _measure(30.0, detector)
    wide = _measure(60.0, detector)
    assert narrow.rectified_shape[1] != wide.rectified_shape[1]
    assert (narrow.detection_trace["reference_width_px"]
            != wide.detection_trace["reference_width_px"])


def test_unvalidated_scale_blocks_physical_units_but_keeps_pixels():
    measurement = _measure(42.0, _detector(42.0), scale=_scale(42.0, validated=False))
    assert measurement.valid is False
    assert measurement.reason == "scale_unknown"
    assert measurement.plug_lengths_px
    assert measurement.plug_lengths_um == ()
    assert measurement.equivalent_diameters_px == ()


# ------------------------------------------------------------------ 工具侧约束

def test_diagnostic_tools_do_not_use_private_detector_or_min_shape():
    """源码守卫：工具不得调 detector 私有方法，也不得用 min(shape) 猜参考宽度。

    用 AST 只看可执行代码，避免被文档字符串里的说明文字误伤。
    """
    import ast

    tools = Path(__file__).resolve().parents[1] / "tools"
    for name in TOOLS:
        tree = ast.parse((tools / name).read_text(encoding="utf-8"), filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr != "_detect_generation_plugs", f"{name}:{node.lineno}"
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "min" and node.args
                    and isinstance(node.args[0], ast.Attribute)
                    and node.args[0].attr == "shape"):
                raise AssertionError(f"{name}:{node.lineno} 用 min(shape) 猜参考宽度")


def test_tools_pass_the_frame_geometry_into_the_detector():
    """源码守卫：扶正测量必须把本帧几何（参考宽 + 轴向）交给 detector。"""
    import ast

    tools = Path(__file__).resolve().parents[1] / "tools"
    for name in ("capture_detector_review.py", "review_current_wall_pair.py",
                 "preflight_current_channel.py"):
        source = (tools / name).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=name)
        call = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"measure_generation_plugs",
                                    "detect_rectified_generation_plugs"}:
                    call = node
        assert call is not None, f"{name} 未走测量入口"
        assert "rectified_axes" in source, f"{name} 未用 rectified_axes 取本帧几何"


# ------------------------------------------- 正路径：定位成功且软/硬件帧号不同

def test_positive_path_accepts_differing_software_and_hardware_frame_ids():
    """几何一致的合成序列上，定位成功时硬件帧号与软件序号**不同**也必须通过测量。

    回放里定位全部失败，这个分支从未运行；这里用合成序列把它跑到，并显式让两个
    身份不同（软件 11，硬件 511），确认测量仍然有效且两个身份都被记录。
    """
    from test_generation_measurement_wiring import (  # noqa: PLC0415
        DUCT_PX,
        SCALE,
        _localization_sequence,
    )
    from backend.vision.parallel_walls import ParallelWallLocalizer

    localizer = ParallelWallLocalizer(contrast_enhance=True)
    frames = _localization_sequence()
    localization = None
    for index, frame in enumerate(frames):
        localizer.observe(frame, frame_id=11 + index, capture_monotonic=100.0 + index)
        localization = localizer.localize(now_monotonic=100.0 + index)
    assert localization.usable, localization.reason

    detector = _detector(DUCT_PX)
    measurement = measure_generation_plugs(
        frames[-1], detector=detector, localization=localization,
        scale=ScaleEvidence(um_per_px=SCALE, source="channel_width_reference",
                            validated=True, reference_um=CHANNEL_UM),
        frame_evidence=FrameEvidence(frame_id=13, hardware_frame_id=511,
                                     capture_monotonic=102.0, localization_frame_id=13,
                                     time_source="camera_frame_timestamp"),
        duct_depth_um=CHANNEL_UM, trace={})

    assert measurement.valid is True, measurement.reason
    assert measurement.reason == "ok"
    assert measurement.frame.frame_id == 13
    assert measurement.frame.hardware_frame_id == 511
    assert measurement.plug_lengths_px
    assert measurement.plug_lengths_um
    assert measurement.axes is not None
    assert measurement.axes.flow_axis == "x"
    assert measurement.axes.to_dict()["span_units"] == "index_span"
