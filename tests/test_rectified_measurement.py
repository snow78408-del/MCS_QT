"""生成区测量链的回归测试：几何、标尺、帧身份、通道截面、尺寸定义。

合成图只证明软件行为，不证明真实物理精度：它带上相衬式的横向剖面是为了让
detector 的间隔分割器有信号，不代表真实对比度、噪声或弯月面形状。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.vision.config import DebugConfig, DetectorConfig
from backend.vision.detector import DropletDetector
from backend.vision.plug_geometry import equivalent_sphere_diameter_px
from backend.vision.rectified_measurement import (
    FrameEvidence,
    ScaleEvidence,
    display_scales,
    measure_generation_plugs,
    verify_reused_walls,
)
from backend.vision.rectified_roi import rectify_channel_frame

WIDTH = 600
HEIGHT = 200
CHANNEL_UM = 50.0
OUTSIDE_GRAY = 20.0
DUCT_GRAY = 40.0
PLUG_BODY_GRAY = 48.0
SHOULDER_GAIN = 9.0


def _transverse_profile(rows: int, rng: np.random.Generator, contrast: float) -> np.ndarray:
    """柱塞列的横向剖面：外缘亮、中间暗，模拟弯月面附近的相衬亮带。"""
    profile = np.full(rows, PLUG_BODY_GRAY, np.float32)
    outer = max(1, int(round(rows * 0.22)))
    profile[:outer] += contrast
    profile[rows - outer:] += contrast
    profile[outer:rows - outer] -= contrast
    return profile + rng.normal(0.0, 0.6, rows)


def _channel_frame(*, tilt: float, offset: float, separation: float, plugs,
                   contrast: float = SHOULDER_GAIN, width: int = WIDTH,
                   height: int = HEIGHT, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = int(round(separation))
    canvas = np.full((height, width), OUTSIDE_GRAY, np.float32) + rng.normal(0.0, 0.6, (height, width))
    tops = offset + tilt * np.arange(width)
    for x in range(width):
        y0 = int(round(tops[x]))
        canvas[y0:y0 + rows, x] = DUCT_GRAY + rng.normal(0.0, 0.6, rows)
    for left, right in plugs:
        for x in range(left, right):
            y0 = int(round(tops[x]))
            canvas[y0:y0 + rows, x] = _transverse_profile(rows, rng, contrast)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _textured_frame(*, bands, width: int = WIDTH, height: int = HEIGHT,
                    seed: int = 5) -> np.ndarray:
    """只有横向亮带纹理、没有真实通道的画面，用于“背景纹理被当成管壁”的回归。"""
    rng = np.random.default_rng(seed)
    canvas = np.full((height, width), DUCT_GRAY, np.float32) + rng.normal(0.0, 1.0, (height, width))
    for center, thickness, gain in bands:
        top = int(round(center - thickness / 2))
        canvas[top:top + thickness, :] += gain
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _walls(tilt: float, offset: float, separation: float, *, width: int = WIDTH,
           height: int = HEIGHT):
    def norm(y: float) -> float:
        return float(y) / float(height)

    return [
        {"x1": 0.0, "y1": norm(offset), "x2": 1.0, "y2": norm(offset + tilt * width)},
        {"x1": 0.0, "y1": norm(offset + separation),
         "x2": 1.0, "y2": norm(offset + separation + tilt * width)},
    ]


def _detector_for(separation_px: float, *, channel_um: float = CHANNEL_UM) -> DropletDetector:
    """按扶正几何配置方形通道 detector：内部通道像素数等于扶正截面像素数。"""
    return _detector_rect(channel_um, channel_um, channel_um / separation_px)


def _detector_rect(depth_um: float, width_um: float, scale: float) -> DropletDetector:
    config = DetectorConfig()
    config.measurement_mode = "generation_plug"
    config.generation_channel_height_um = depth_um
    config.generation_channel_width_um = width_um
    config.generation_min_raw_outline_contrast = 3.0
    config.generation_min_capsule_outline_ratio = 0.12
    detector = DropletDetector(config, DebugConfig())
    detector.configure_expected_diameter(0.0, scale)
    return detector


def _scale_for(separation_px: float, *, validated: bool = True,
               source: str = "channel_width_reference") -> ScaleEvidence:
    return ScaleEvidence(um_per_px=CHANNEL_UM / separation_px, source=source,
                         validated=validated, reference_um=CHANNEL_UM)


def _evidence(frame_id: int = 7, *, hardware_frame_id: int | None = None,
              localization_frame_id: int | None = None, capture_monotonic: float = 1234.5,
              time_source: str = "camera_frame_timestamp") -> FrameEvidence:
    return FrameEvidence(
        frame_id=frame_id,
        hardware_frame_id=frame_id if hardware_frame_id is None else hardware_frame_id,
        capture_monotonic=capture_monotonic,
        localization_frame_id=frame_id if localization_frame_id is None else localization_frame_id,
        time_source=time_source,
    )


def _measure(image, walls, *, separation_px, detector=None, scale=None, depth_um=CHANNEL_UM,
             evidence=None, **kwargs):
    return measure_generation_plugs(
        image,
        detector=detector or _detector_for(separation_px),
        wall_lines=walls,
        scale=scale or _scale_for(separation_px),
        frame_evidence=evidence or _evidence(),
        duct_depth_um=depth_um,
        duct_width_reference_um=CHANNEL_UM,
        **kwargs,
    )


PLUGS = [(80, 200), (240, 360), (400, 560)]


# ---------------------------------------------------------------- 管道平移 / 旋转

@pytest.mark.parametrize("tilt", (-0.05, 0.0, 0.05))
@pytest.mark.parametrize("offset", (40.0, 60.0, 80.0))
def test_translation_and_tilt_do_not_change_the_physical_size(tilt: float, offset: float) -> None:
    frame = _channel_frame(tilt=tilt, offset=offset, separation=40.0, plugs=PLUGS)
    result = _measure(frame, _walls(tilt, offset, 40.0), separation_px=40.0)
    assert result.valid, result.reason
    assert result.frame.coordinate_space == "rectified_full_resolution"
    assert result.input_image_shape == (WIDTH, HEIGHT)
    assert result.rectified_shape is not None
    assert result.rectified_shape[0] == pytest.approx(WIDTH, abs=3)
    assert result.rectified_shape[1] == pytest.approx(40, abs=1)
    assert result.duct.width_px == pytest.approx(40.0, abs=0.5)
    assert result.duct.detector_geometry_consistent is True
    assert result.plug_count == 3
    assert result.has_complete_plug is True
    # 三个柱塞 120/120/160 px，通道 50 μm/40 px ⇒ 1.25 μm/px
    assert result.equivalent_diameters_um[0] == pytest.approx(
        equivalent_sphere_diameter_px(120.0, 40.0, 40.0) * 1.25, abs=1.0
    )


def test_tilted_and_translated_frames_agree_with_each_other() -> None:
    baseline = _measure(_channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS),
                        _walls(0.0, 60.0, 40.0), separation_px=40.0)
    for tilt, offset in ((0.05, 40.0), (-0.05, 80.0), (0.05, 80.0)):
        shifted = _measure(_channel_frame(tilt=tilt, offset=offset, separation=40.0, plugs=PLUGS),
                           _walls(tilt, offset, 40.0), separation_px=40.0)
        assert shifted.valid
        assert shifted.equivalent_diameters_um[0] == pytest.approx(
            baseline.equivalent_diameters_um[0], rel=0.02)


# ---------------------------------------------------------------- 采样分辨率 / 显示缩放

def test_resampling_the_frame_does_not_change_physical_sizes() -> None:
    """同一物理场景在 1× 与 2× 采样下必须给出相同的 μm，像素长度则翻倍。"""
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    single = _measure(frame, _walls(0.0, 60.0, 40.0), separation_px=40.0)

    doubled = cv2.resize(frame, (WIDTH * 2, HEIGHT * 2), interpolation=cv2.INTER_NEAREST)
    double_result = _measure(doubled, _walls(0.0, 60.0, 40.0), separation_px=80.0)

    assert single.valid and double_result.valid
    assert double_result.duct.width_px == pytest.approx(80.0, abs=1.0)
    assert double_result.plug_lengths_px[0] == pytest.approx(2.0 * single.plug_lengths_px[0], rel=0.03)
    assert double_result.equivalent_diameters_um[0] == pytest.approx(
        single.equivalent_diameters_um[0], rel=0.05)


def test_measurement_image_is_the_full_resolution_rectified_frame() -> None:
    """测量的图像必须是扶正后的全分辨率图，不能被任何显示用缩放替换。"""
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    result = _measure(frame, _walls(0.0, 60.0, 40.0), separation_px=40.0)
    expected = rectify_channel_frame(frame, _walls(0.0, 60.0, 40.0))
    assert result.rectified_preview is not None
    assert result.rectified_preview.shape == expected.shape
    assert result.rectified_preview.shape[1] == WIDTH


def test_display_scales_are_reported_per_axis_and_never_defaulted() -> None:
    """718×47 缩到 640×41：横向因子 0.89136、纵向 0.87234，两者不同。"""
    scales = display_scales((718, 47), (640, 41), 50.0 / 47.0)
    assert scales["axial_display_factor"] == pytest.approx(640 / 718, rel=1e-9)
    assert scales["transverse_display_factor"] == pytest.approx(41 / 47, rel=1e-9)
    assert scales["axial_um_per_px"] == pytest.approx((50.0 / 47.0) * 718 / 640, rel=1e-9)
    assert scales["transverse_um_per_px"] == pytest.approx(50.0 / 41.0, rel=1e-9)
    assert scales["anisotropic"] is True
    assert scales["axial_um_per_px"] != pytest.approx(scales["transverse_um_per_px"], rel=1e-3)

    uniform = display_scales((100, 50), (50, 25), 2.0)
    assert uniform["anisotropic"] is False
    assert uniform["axial_um_per_px"] == pytest.approx(4.0, rel=1e-9)
    assert uniform["transverse_um_per_px"] == pytest.approx(4.0, rel=1e-9)

    assert display_scales((718, 47), (640, 0), 1.0)["axial_um_per_px"] is None
    assert display_scales(None, (640, 41), 1.0)["axial_um_per_px"] is None
    assert display_scales((718, 47), (640, 41), 0.0)["axial_um_per_px"] is None
    assert display_scales((718, 47), (640, 41), float("nan"))["axial_um_per_px"] is None


# ---------------------------------------------------------------- 标尺

def test_unknown_scale_keeps_pixels_and_refuses_microns() -> None:
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    result = _measure(frame, _walls(0.0, 60.0, 40.0), separation_px=40.0,
                      scale=ScaleEvidence(um_per_px=None, source="channel_width_reference",
                                          validated=True))
    assert result.valid is False
    assert result.reason == "scale_unknown"
    assert len(result.plug_lengths_px) == 3
    assert result.plug_lengths_um == ()
    assert result.equivalent_diameters_um == ()


def test_unvalidated_scale_refuses_microns() -> None:
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    result = _measure(frame, _walls(0.0, 60.0, 40.0), separation_px=40.0,
                      scale=_scale_for(40.0, validated=False))
    assert result.valid is False
    assert result.reason == "scale_unvalidated"
    assert len(result.plug_lengths_px) == 3


def test_unknown_scale_source_refuses_microns() -> None:
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    result = _measure(frame, _walls(0.0, 60.0, 40.0), separation_px=40.0,
                      scale=_scale_for(40.0, source="unknown"))
    assert result.valid is False
    assert result.reason == "scale_source_unknown"


def test_scale_reference_inconsistent_with_rectified_geometry_is_rejected() -> None:
    """标尺声明“参考特征 35 μm”，但该特征在图上量到 40 px × 1.25 = 50 μm —— 必须拒。"""
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    detector = _detector_rect(100.0, 50.0, CHANNEL_UM / 40.0)   # 截面 (80, 40) px，与图一致
    inconsistent = ScaleEvidence(um_per_px=CHANNEL_UM / 40.0, source="channel_width_reference",
                                validated=True, reference_um=35.0)
    result = measure_generation_plugs(
        frame, detector=detector, wall_lines=_walls(0.0, 60.0, 40.0), scale=inconsistent,
        frame_evidence=_evidence(), duct_depth_um=100.0, duct_width_reference_um=35.0,
    )
    assert result.valid is False
    assert result.reason == "scale_geometry_inconsistent"


# ---------------------------------------------------------------- 通道截面

def test_rectangular_duct_is_preserved_and_not_overwritten_as_square() -> None:
    """detector 按 80×40 px（深 100 μm、宽 50 μm @1.25 μm/px）建模时，截面必须原样保留。"""
    frame = _channel_frame(tilt=0.0, offset=110.0, separation=40.0, plugs=PLUGS)
    detector = _detector_rect(100.0, 50.0, 1.25)
    result = _measure(frame, _walls(0.0, 110.0, 40.0), separation_px=40.0,
                      detector=detector, depth_um=100.0)
    assert result.valid, result.reason
    assert result.duct.depth_px == pytest.approx(80.0, abs=0.5)
    assert result.duct.width_px == pytest.approx(40.0, abs=0.5)
    assert result.duct.depth_px != pytest.approx(result.duct.width_px, abs=1.0)
    assert result.duct.detector_geometry_consistent is True
    assert result.duct.depth_source == "declared_chip_geometry"
    # 等效直径必须用 80×40 的矩形截面算，而不是 40×40
    assert result.equivalent_diameters_px[0] == pytest.approx(
        equivalent_sphere_diameter_px(result.plug_lengths_px[0], 80.0, 40.0), rel=0.02)
    assert result.equivalent_diameters_px[0] != pytest.approx(
        equivalent_sphere_diameter_px(result.plug_lengths_px[0], 40.0, 40.0), rel=0.02)


def test_detector_duct_geometry_mismatch_is_rejected() -> None:
    """2026-09-21 的现场错误：图像里通道 40 px，detector 按 50/1.725 ≈ 29 px 建模。"""
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    wrong = _detector_rect(29.0, 29.0, 1.0)
    result = _measure(frame, _walls(0.0, 60.0, 40.0), separation_px=40.0, detector=wrong,
                      depth_um=29.0)
    assert result.valid is False
    assert result.reason == "detector_duct_geometry_mismatch"
    assert result.duct.detector_geometry_consistent is False


def test_unknown_duct_depth_refuses_volume_sizes_but_keeps_lengths() -> None:
    """深度不可见，不能从图高推断：深度未知时拒绝任何体积等效尺寸。"""
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    result = _measure(frame, _walls(0.0, 60.0, 40.0), separation_px=40.0, depth_um=None)
    assert result.valid is False
    assert result.reason == "duct_depth_unknown"
    assert result.duct.depth_source == "unknown"
    assert result.duct.usable_for_volume is False
    assert len(result.plug_lengths_px) == 3
    assert len(result.plug_lengths_um) == 3
    assert result.equivalent_diameters_px == ()
    assert result.equivalent_diameters_um == ()


def test_non_positive_or_non_finite_depth_is_treated_as_unknown() -> None:
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    for depth in (0.0, -5.0, float("nan"), float("inf")):
        result = _measure(frame, _walls(0.0, 60.0, 40.0), separation_px=40.0, depth_um=depth)
        assert result.reason == "duct_depth_unknown"
        assert result.equivalent_diameters_px == ()


# ---------------------------------------------------------------- 复用几何

def test_offset_roi_with_same_separation_and_tilt_is_rejected() -> None:
    """Codex 复现：真实通道上沿 y=110，复用几何上沿 y=40、间距同为 40 —— 必须拒。"""
    frame = _channel_frame(tilt=0.0, offset=110.0, separation=40.0, plugs=PLUGS)
    report = verify_reused_walls(frame, _walls(0.0, 40.0, 40.0), frame_id=7)
    assert report["status"] == "mismatch"
    assert report["consistent"] is False
    assert any("中线位置" in reason for reason in report["rejection_reasons"])
    assert report["reused_roi_covers_current_walls"] is False


def test_aligned_roi_on_the_same_frame_is_accepted() -> None:
    frame = _channel_frame(tilt=0.0, offset=110.0, separation=40.0, plugs=PLUGS)
    report = verify_reused_walls(frame, _walls(0.0, 110.0, 40.0), frame_id=7)
    assert report["status"] == "verified"
    assert report["consistent"] is True
    assert report["mid_offset_px"] == pytest.approx(0.0, abs=6.0)
    assert report["reused_roi_covers_current_walls"] is True
    assert report["verified_frame_id"] == 7


def test_reused_walls_are_rejected_when_the_verification_belongs_to_another_frame() -> None:
    """核验必须绑定帧号：核验帧与测量帧不同就必须拒。"""
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    consistency = verify_reused_walls(frame, _walls(0.0, 60.0, 40.0), frame_id=7)
    assert consistency["consistent"] is True
    result = _measure(frame, _walls(0.0, 60.0, 40.0), separation_px=40.0,
                      wall_source="reused", wall_consistency=consistency,
                      evidence=_evidence(frame_id=9))
    assert result.valid is False
    assert result.reason == "wall_verification_frame_mismatch"


def test_reused_walls_without_current_frame_verification_are_rejected() -> None:
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    result = _measure(frame, _walls(0.0, 60.0, 40.0), separation_px=40.0, wall_source="reused")
    assert result.valid is False
    assert result.reason == "wall_geometry_stale"


def test_reused_walls_declared_unverifiable_are_rejected() -> None:
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    result = _measure(frame, _walls(0.0, 60.0, 40.0), separation_px=40.0,
                      wall_source="reused",
                      wall_consistency={"status": "unverifiable", "consistent": False,
                                        "verified_frame_id": 7})
    assert result.valid is False
    assert result.reason == "wall_geometry_unverified"


def test_reused_walls_verified_on_this_frame_are_accepted() -> None:
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    consistency = verify_reused_walls(frame, _walls(0.0, 60.0, 40.0), frame_id=7)
    result = _measure(frame, _walls(0.0, 60.0, 40.0), separation_px=40.0,
                      wall_source="reused", wall_consistency=consistency)
    assert result.valid, result.reason
    assert result.walls is not None and result.walls.source == "reused"


def test_verify_reused_walls_is_unverifiable_on_a_blank_frame() -> None:
    blank = np.full((HEIGHT, WIDTH), 128, np.uint8)
    report = verify_reused_walls(blank, _walls(0.0, 60.0, 40.0), frame_id=7)
    assert report["status"] == "unverifiable"
    assert report["consistent"] is False


def test_background_texture_is_not_accepted_as_the_reused_channel() -> None:
    """只有横向纹理、没有通道的画面不得被判成与复用几何一致。"""
    frame = _textured_frame(bands=[(40.0, 4, 14.0), (80.0, 4, 14.0), (120.0, 4, 14.0)])
    report = verify_reused_walls(frame, _walls(0.0, 150.0, 40.0), frame_id=7)
    assert report["consistent"] is False
    assert report["status"] in {"mismatch", "unverifiable", "invalid"}


# ---------------------------------------------------------------- 裁剪截断

def test_wall_lines_too_short_to_form_a_quad_are_unusable() -> None:
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    degenerate = [
        {"x1": 0.0, "y1": 0.30, "x2": 1.0, "y2": 0.30},
        {"x1": 0.5, "y1": 0.50, "x2": 0.5, "y2": 0.52},
    ]
    result = _measure(frame, degenerate, separation_px=40.0)
    assert result.valid is False
    assert result.reason == "wall_geometry_unusable"


def test_plug_touching_the_axial_edge_is_not_reported_with_a_fabricated_length() -> None:
    """截断的柱塞不能被当成完整柱塞量出一个假长度。"""
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=[(0, 120), (300, 420)])
    result = _measure(frame, _walls(0.0, 60.0, 40.0), separation_px=40.0)
    assert result.valid
    # 该 detector 对贴着边界的柱塞没有两侧侧翼证据，直接拒收而不是截半报出
    assert result.plug_count == 1
    assert result.complete_plug_count == result.plug_count
    assert result.plug_lengths_px[0] == pytest.approx(120.0, rel=0.1)


# ---------------------------------------------------------------- 空画面 / 壁面误检

def test_empty_frame_is_an_explicit_invalid_state() -> None:
    result = _measure(np.empty((0, 0), np.uint8), _walls(0.0, 60.0, 40.0), separation_px=40.0)
    assert result.valid is False
    assert result.reason == "frame_empty"


def test_wall_bands_alone_do_not_produce_plugs() -> None:
    """只有通道与两侧管壁、没有液柱：测量链有效但观测为空，不是“零尺寸”。"""
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=[])
    result = _measure(frame, _walls(0.0, 60.0, 40.0), separation_px=40.0)
    assert result.valid is True
    assert result.reason == "no_complete_plug"
    assert result.plug_count == 0
    assert result.has_complete_plug is False
    assert result.equivalent_diameters_um == ()


# ---------------------------------------------------------------- 帧身份 / 采集时间

@pytest.mark.parametrize("frame_id", (0, -1))
def test_non_positive_frame_id_is_rejected(frame_id: int) -> None:
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    result = _measure(frame, _walls(0.0, 60.0, 40.0), separation_px=40.0,
                      evidence=_evidence(frame_id=frame_id))
    assert result.valid is False
    assert result.reason == "frame_id_invalid"


def test_zero_hardware_frame_id_is_rejected() -> None:
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    result = _measure(frame, _walls(0.0, 60.0, 40.0), separation_px=40.0,
                      evidence=_evidence(frame_id=7, hardware_frame_id=0))
    assert result.valid is False
    assert result.reason == "hardware_frame_id_invalid"


def test_differing_software_and_hardware_frame_ids_are_accepted() -> None:
    """软件序号与硬件帧号来自不同计数器，本就不同；不得要求二者相等。

    本仓库实测：软件序号 0 对应硬件帧号 74，录像末尾相差 317。
    要求相等只会逼调用方把一个字段填成另一个字段的值。
    """
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    result = _measure(frame, _walls(0.0, 60.0, 40.0), separation_px=40.0,
                      evidence=_evidence(frame_id=7, hardware_frame_id=81))
    assert result.valid is True
    assert result.reason == "ok"
    # 两个身份都如实记录，不互相冒名。
    assert result.frame.frame_id == 7
    assert result.frame.hardware_frame_id == 81


def test_wall_localization_frame_mismatch_is_rejected() -> None:
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    result = _measure(frame, _walls(0.0, 60.0, 40.0), separation_px=40.0,
                      evidence=_evidence(frame_id=7, localization_frame_id=6))
    assert result.valid is False
    assert result.reason == "wall_localization_frame_mismatch"


def test_undeclared_time_source_is_rejected() -> None:
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    result = _measure(frame, _walls(0.0, 60.0, 40.0), separation_px=40.0,
                      evidence=_evidence(time_source="something_else"))
    assert result.valid is False
    assert result.reason == "time_source_undeclared"


def test_host_clock_proxy_is_accepted_but_flagged() -> None:
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    result = _measure(frame, _walls(0.0, 60.0, 40.0), separation_px=40.0,
                      evidence=_evidence(time_source="host_clock_proxy"))
    assert result.valid, result.reason
    assert result.frame.time_is_proxy is True
    payload = result.to_dict()
    assert payload["frame"]["time_is_proxy"] is True
    assert payload["frame"]["time_source"] == "host_clock_proxy"


@pytest.mark.parametrize("moment", (0.0, -1.0, float("nan"), float("inf")))
def test_invalid_capture_time_is_rejected(moment: float) -> None:
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    result = _measure(frame, _walls(0.0, 60.0, 40.0), separation_px=40.0,
                      evidence=_evidence(capture_monotonic=moment))
    assert result.valid is False
    assert result.reason == "capture_time_invalid"


# ---------------------------------------------------------------- 尺寸定义

def test_plug_length_and_equivalent_diameter_are_reported_separately() -> None:
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    result = _measure(frame, _walls(0.0, 60.0, 40.0), separation_px=40.0)
    assert result.valid
    assert len(result.plug_lengths_um) == len(result.equivalent_diameters_um) == result.plug_count
    for length_um, diameter_um in zip(result.plug_lengths_um, result.equivalent_diameters_um):
        assert diameter_um < length_um
    payload = result.to_dict()
    assert payload["frame"]["coordinate_space"] == "rectified_full_resolution"
    assert payload["scale"]["source"] == "channel_width_reference"
    assert payload["scale"]["usable"] is True
    assert payload["duct"]["width_source"] == "rectified_image_measurement"
    assert payload["has_complete_plug"] is True
    assert "rectified_preview" not in payload


def test_measurement_requires_frame_evidence_argument() -> None:
    """帧证据是必填参数，缺省即报错——不允许静默使用默认帧号。"""
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    with pytest.raises(TypeError):
        measure_generation_plugs(
            frame, detector=_detector_for(40.0), wall_lines=_walls(0.0, 60.0, 40.0),
            scale=_scale_for(40.0), duct_depth_um=CHANNEL_UM,
        )


def test_duct_depth_argument_is_required() -> None:
    frame = _channel_frame(tilt=0.0, offset=60.0, separation=40.0, plugs=PLUGS)
    with pytest.raises(TypeError):
        measure_generation_plugs(
            frame, detector=_detector_for(40.0), wall_lines=_walls(0.0, 60.0, 40.0),
            scale=_scale_for(40.0), frame_evidence=_evidence(),
        )
