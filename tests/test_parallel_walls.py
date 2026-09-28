"""平行管壁定位的验收测试（对应任务书 §5 的场景表）。

合成图只证明软件行为，不证明物理精度：这里的“管道”是灰度带，不代表真实相衬对比度、
噪声或弯月面形状。
"""
from __future__ import annotations

import threading

import cv2
import numpy as np
import pytest

from backend.vision.parallel_walls import (
    GEOMETRY_VERSION,
    LocalizationThresholds,
    ParallelWallLocalizer,
    assess_coverage,
    pair_candidates,
    prepare_frame,
)

WIDTH = 720
HEIGHT = 540
DUCT_GRAY = 40.0
OUTSIDE_GRAY = 20.0
PLUG_GRAY = 55.0
TRAIN_A = ((20, 120), (170, 270), (320, 420), (470, 570), (620, 690))
TRAIN_B = ((70, 170), (220, 320), (370, 470), (520, 620))


def _frame(*, offset=60.0, separation=40.0, tilt=0.0, plugs=TRAIN_A, plug_gray=PLUG_GRAY,
           duct_gray=DUCT_GRAY, outside_gray=OUTSIDE_GRAY, width=WIDTH, height=HEIGHT,
           interior_line=False, interior_line_span=None, bend=0.0, second_channel=False,
           seed=11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    canvas = np.full((height, width), outside_gray, np.float32) + rng.normal(0.0, 0.8, (height, width))
    rows = int(round(separation))
    centers = offset + tilt * np.arange(width) + bend * (np.arange(width) - width / 2.0) ** 2 / width
    for x in range(width):
        top = int(round(centers[x]))
        canvas[top:top + rows, x] = duct_gray + rng.normal(0.0, 0.8, rows)
    if interior_line:
        for x in range(width):
            y = int(round(centers[x] + separation * 0.5))
            canvas[y - 1:y + 2, x] += 22.0
    for left, right in plugs:
        for x in range(left, right):
            top = int(round(centers[x]))
            canvas[top:top + rows, x] = plug_gray + rng.normal(0.0, 0.8, rows)
    if interior_line:
        # 带内长边画在柱塞之后：它代表管内的固定结构（气泡/纹理），应当穿过柱塞列。
        span = interior_line_span or (0, width)
        for x in range(int(span[0]), int(span[1])):
            y = int(round(centers[x] + separation * 0.5))
            canvas[y - 1:y + 2, x] = min(255.0, plug_gray + 26.0) + rng.normal(0.0, 0.5, 3)
    if second_channel:
        second_top = int(round(offset + 5 * separation))
        for x in range(width):
            canvas[second_top:second_top + rows, x] = duct_gray + rng.normal(0.0, 0.8, rows)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _observe_all(frames, *, thresholds=None, frame_ids=None, times=None, work_scale=1.0):
    limits = thresholds or LocalizationThresholds()
    localizer = ParallelWallLocalizer(thresholds=limits, work_scale=work_scale)
    for index, frame in enumerate(frames):
        localizer.observe(
            frame,
            frame_id=(frame_ids[index] if frame_ids else 100 + index),
            capture_monotonic=(times[index] if times else 100.0 + index),
        )
    return localizer


# ---------------------------------------------------------------- 1 平移 / 旋转 / 分辨率

@pytest.mark.parametrize("offset,tilt", ((60.0, 0.0), (100.0, 0.06), (40.0, -0.05), (150.0, 0.04)))
def test_translation_and_rotation_are_relocalized_without_old_coordinates(offset, tilt) -> None:
    frames = [_frame(offset=offset, tilt=tilt, plugs=TRAIN_A, seed=11),
              _frame(offset=offset, tilt=tilt, plugs=TRAIN_B, seed=12),
              _frame(offset=offset, tilt=tilt, plugs=TRAIN_A, seed=13)]
    localizer = _observe_all(frames)
    result = localizer.localize(now_monotonic=102.0)
    assert result.status == "localized", result.reason
    geometry = result.geometry
    assert geometry["separation_px"] == pytest.approx(40.0, abs=2.5)
    assert geometry["tilt_ratio"] == pytest.approx(tilt, abs=0.01)
    # 带中线的原图 y 应落在管道高度范围内（含倾斜带来的整体位移）
    expected_mid = offset + 20.0 + tilt * (WIDTH - 1) / 2.0
    assert geometry["mid_y_px"] == pytest.approx(expected_mid, abs=6.0)
    assert geometry["coordinate_mapping"]["invertible"] is True
    assert geometry["geometry_version"] == GEOMETRY_VERSION


def test_resolution_change_keeps_the_same_normalized_geometry() -> None:
    import cv2

    frames = [_frame(plugs=TRAIN_A, seed=11), _frame(plugs=TRAIN_B, seed=12),
              _frame(plugs=TRAIN_A, seed=13)]
    base = _observe_all(frames).localize(now_monotonic=102.0)
    doubled_frames = [cv2.resize(frame, (WIDTH * 2, HEIGHT * 2), interpolation=cv2.INTER_NEAREST)
                      for frame in frames]
    doubled = _observe_all(doubled_frames).localize(now_monotonic=102.0)
    assert base.status == doubled.status == "localized"
    assert doubled.geometry["separation_px"] == pytest.approx(
        base.geometry["separation_px"] * 2.0, rel=0.05)
    for base_line, doubled_line in zip(base.wall_lines, doubled.wall_lines):
        assert doubled_line["y1"] == pytest.approx(base_line["y1"], abs=0.01)
        assert doubled_line["x1"] == pytest.approx(base_line["x1"], abs=0.01)


def test_downscaled_work_image_maps_back_to_original_pixels() -> None:
    frames = [_frame(seed=11), _frame(plugs=TRAIN_B, seed=12), _frame(seed=13)]
    result = _observe_all(frames, work_scale=0.5).localize(now_monotonic=102.0)
    assert result.status == "localized", result.reason
    assert result.geometry["coordinate_mapping"]["work_scale"] == 0.5
    assert result.geometry["separation_px"] == pytest.approx(40.0, abs=3.0)


# ---------------------------------------------------------------- 2 壁面固定、液柱移动

def test_moving_liquid_does_not_move_the_locked_wall_edges() -> None:
    frames = [_frame(plugs=TRAIN_A, seed=11), _frame(plugs=TRAIN_B, seed=12),
              _frame(plugs=TRAIN_A, seed=13)]
    localizer = _observe_all(frames)
    result = localizer.localize(now_monotonic=102.0)
    assert result.status == "localized", result.reason
    assert result.geometry["interior_parallel_competitor_ids"] == []
    assert result.geometry["support"]["separation_px"]["max"] - \
        result.geometry["support"]["separation_px"]["min"] <= 2.0
    assert result.geometry["motion"]["available"] is True


def test_interior_parallel_line_forces_ambiguity() -> None:
    """带内一条贯通的长平行边会让“壁面 + 带内线”成为同样可信的线对：必须判歧义。"""
    frames = [_frame(interior_line=True, seed=11), _frame(plugs=TRAIN_B, interior_line=True, seed=12),
              _frame(interior_line=True, seed=13)]
    result = _observe_all(frames).localize(now_monotonic=102.0)
    assert result.status == "ambiguous"
    assert result.wall_lines == []
    assert result.usable is False
    assert (result.geometry["competition"]["ambiguous"]
            or result.geometry["static_interior_competitor_ids"])


def test_static_interior_edge_inside_the_selected_band_is_reported() -> None:
    """壁面线对胜出、带内另有一条静止长边：仍必须判歧义，不能认证为内壁。"""
    frames = [_frame(interior_line=True, interior_line_span=(150, 570), seed=11),
              _frame(plugs=TRAIN_B, interior_line=True, interior_line_span=(150, 570), seed=12),
              _frame(interior_line=True, interior_line_span=(150, 570), seed=13)]
    result = _observe_all(frames).localize(now_monotonic=102.0)
    assert result.status == "ambiguous"
    assert result.wall_lines == []
    assert (result.geometry["static_interior_competitor_ids"]
            or result.geometry["competition"]["ambiguous"])


# ---------------------------------------------------------------- 3 停泵 / 静态 / 重复帧

def test_stopped_pump_does_not_fabricate_motion_evidence() -> None:
    frames = [_frame(plugs=TRAIN_A, seed=11), _frame(plugs=TRAIN_A, seed=12),
              _frame(plugs=TRAIN_A, seed=13)]
    result = _observe_all(frames).localize(now_monotonic=102.0)
    assert result.status == "pending_motion"
    assert result.geometry["motion"]["available"] is False
    assert result.wall_lines == []
    assert result.usable is False


def test_repeated_identical_frames_do_not_increase_independent_support() -> None:
    frame = _frame(seed=11)
    result = _observe_all([frame, frame.copy(), frame.copy()]).localize(now_monotonic=102.0)
    assert result.status == "insufficient_support"
    assert result.geometry["support"]["independent_frames"] == 1
    assert result.geometry["support"]["repeated_frame_count"] >= 1
    assert result.wall_lines == []


def test_blank_and_flat_frames_are_rejected_with_a_reason() -> None:
    blank = np.full((HEIGHT, WIDTH), 128, np.uint8)
    result = _observe_all([blank, blank.copy(), blank.copy()]).localize(now_monotonic=102.0)
    assert result.status == "rejected"
    assert result.reason in {"no_credible_pair", "insufficient_support"}
    assert result.wall_lines == []


# ---------------------------------------------------------------- 4 多条管道 / 多组纹理

def test_two_equally_credible_channels_are_reported_as_ambiguous() -> None:
    frames = [_frame(second_channel=True, seed=11),
              _frame(plugs=TRAIN_B, second_channel=True, seed=12),
              _frame(second_channel=True, seed=13)]
    result = _observe_all(frames).localize(now_monotonic=102.0)
    assert result.status == "ambiguous", result.reason
    assert result.geometry["competition"]["ambiguous"] is True
    assert result.geometry["competition"]["rivals"]
    assert result.wall_lines == []


def test_separation_is_measured_along_the_normal_not_by_vertical_difference() -> None:
    """倾斜 0.2 时，垂直 y 差会比法向间距大约 2%，两者必须可区分。"""
    limits = LocalizationThresholds()
    frame = _frame(tilt=0.2, plugs=TRAIN_A, seed=11)
    prepared = prepare_frame(frame, thresholds=limits)
    pairs = [pair for pair in pair_candidates(prepared, (HEIGHT, WIDTH), thresholds=limits)
             if not pair.rejection]
    assert pairs
    pair = max(pairs, key=lambda item: item.score)
    vertical = abs(pair.first.y_at(200.0) - pair.second.y_at(200.0))
    assert pair.separation_px == pytest.approx(vertical * np.cos(np.arctan(0.2)), rel=0.02)
    assert pair.separation_px < vertical


# ---------------------------------------------------------------- 5 沿程覆盖

def test_band_leaving_the_frame_at_the_ends_fails_along_track_coverage() -> None:
    """两壁在中心对齐良好，但沿测量段走到右端时下壁出画：中心点检查会放行。"""
    limits = LocalizationThresholds()
    frame = _frame(offset=485.0, tilt=0.02, plugs=((80, 300), (400, 640)), seed=11)
    prepared = prepare_frame(frame, thresholds=limits)
    pairs = [pair for pair in pair_candidates(prepared, (HEIGHT, WIDTH), thresholds=limits)
             if not pair.rejection]
    assert pairs, "两条壁在大部分区间可见，应被检出"
    pair = max(pairs, key=lambda item: item.score)
    coverage = assess_coverage(prepared.gray, pair, thresholds=limits)
    centre = coverage["positions"][len(coverage["positions"]) // 2]
    assert centre["band_inside_frame"] is True
    assert coverage["positions"][-1]["band_inside_frame"] is False
    assert coverage["ok"] is False
    assert coverage["reason"] == "along_track_coverage_incomplete"


def test_localization_rejects_when_coverage_is_incomplete() -> None:
    frames = [_frame(offset=485.0, tilt=0.02, plugs=((80, 300),), seed=11),
              _frame(offset=485.0, tilt=0.02, plugs=((380, 620),), seed=12),
              _frame(offset=485.0, tilt=0.02, plugs=((120, 320),), seed=13)]
    result = _observe_all(frames).localize(now_monotonic=102.0)
    assert result.usable is False


# ---------------------------------------------------------------- 6 单壁 / 交汇口 / 弯道 / 低对比

def test_single_wall_gives_no_pair() -> None:
    canvas = np.full((HEIGHT, WIDTH), OUTSIDE_GRAY, np.float32)
    canvas[60:100, :] = DUCT_GRAY
    canvas[300:302, :] = 200.0                      # 只留一条强长边
    frame = np.clip(canvas, 0, 255).astype(np.uint8)
    limits = LocalizationThresholds()
    prepared = prepare_frame(frame, thresholds=limits)
    pairs = [pair for pair in pair_candidates(prepared, (HEIGHT, WIDTH), thresholds=limits)
             if not pair.rejection]
    assert len({pair.ids for pair in pairs}) <= 1


def test_bent_channel_is_not_forced_into_a_parallel_pair() -> None:
    frames = [_frame(bend=0.6, plugs=TRAIN_A, seed=11),
              _frame(bend=0.6, plugs=TRAIN_B, seed=12),
              _frame(bend=0.6, plugs=TRAIN_A, seed=13)]
    result = _observe_all(frames).localize(now_monotonic=102.0)
    assert result.usable is False


def test_low_contrast_duct_is_not_accepted() -> None:
    """1 个灰阶量级的“管道”不得产出可用的有效直管尺寸。"""
    frames = [_frame(duct_gray=21.0, plug_gray=22.0, seed=11),
              _frame(duct_gray=21.0, plug_gray=22.0, seed=12),
              _frame(duct_gray=21.0, plug_gray=22.0, seed=13)]
    result = _observe_all(frames).localize(now_monotonic=102.0)
    assert result.usable is False
    assert result.wall_lines == []
    assert result.status in {"rejected", "pending_motion", "insufficient_support", "ambiguous"}


def test_crossing_line_at_a_junction_is_not_taken_as_a_wall() -> None:
    canvas = _frame(plugs=TRAIN_A, seed=11).astype(np.float32)
    # 一条大角度交叉线（交汇口/支路），坡度远超候选上限
    for x in range(WIDTH):
        y = int(round(300 + 0.8 * x))
        if 0 <= y < HEIGHT:
            canvas[y - 1:y + 2, x] = 220.0
    frame = np.clip(canvas, 0, 255).astype(np.uint8)
    limits = LocalizationThresholds()
    prepared = prepare_frame(frame, thresholds=limits)
    assert all(abs(candidate.slope) <= limits.max_slope_for_any_candidate
               for candidate in prepared.candidates)


# ---------------------------------------------------------------- 8 证据失效

def test_evidence_older_than_the_limit_is_expired() -> None:
    frames = [_frame(seed=11), _frame(plugs=TRAIN_B, seed=12), _frame(seed=13)]
    localizer = _observe_all(frames)
    fresh = localizer.localize(now_monotonic=102.0)
    assert fresh.status == "localized"
    stale = localizer.localize(now_monotonic=102.0 + localizer.thresholds.evidence_max_age_s + 1.0)
    assert stale.status == "rejected"
    assert stale.reason == "evidence_expired"
    assert stale.wall_lines == []


def test_sudden_translation_after_localization_makes_the_track_unstable() -> None:
    frames = [_frame(offset=60.0, seed=11), _frame(offset=60.0, plugs=TRAIN_B, seed=12),
              _frame(offset=60.0, seed=13)]
    localizer = _observe_all(frames)
    assert localizer.localize(now_monotonic=102.0).status == "localized"
    # 定位后画面突然平移很远：新证据与旧簇分开，且线对不再稳定
    for index, frame in enumerate([_frame(offset=300.0, seed=21), _frame(offset=300.0, plugs=TRAIN_B, seed=22)]):
        localizer.observe(frame, frame_id=200 + index, capture_monotonic=104.0 + index)
    result = localizer.localize(now_monotonic=105.0)
    assert result.geometry["support"]["cluster_count"] >= 2
    assert result.status in {"ambiguous", "insufficient_support", "pending_motion", "rejected"}
    assert result.geometry["support"]["separation_px"] is not None


def test_buffer_is_bounded_and_never_grows_without_limit() -> None:
    limits = LocalizationThresholds(max_buffer_frames=4)
    localizer = ParallelWallLocalizer(thresholds=limits)
    for index in range(40):
        localizer.observe(_frame(plugs=TRAIN_A if index % 2 else TRAIN_B, seed=index),
                          frame_id=index + 1, capture_monotonic=float(index))
    assert localizer.buffered_frames <= limits.max_buffer_frames


def test_reset_clears_all_state() -> None:
    localizer = _observe_all([_frame(seed=11), _frame(plugs=TRAIN_B, seed=12)])
    assert localizer.buffered_frames == 2
    localizer.reset()
    assert localizer.buffered_frames == 0
    assert localizer.localize(now_monotonic=100.0).reason == "no_frames"


def test_localization_does_not_use_an_expired_geometry_version() -> None:
    result = _observe_all([_frame(seed=11), _frame(plugs=TRAIN_B, seed=12)]).localize(
        now_monotonic=101.0)
    assert result.geometry["geometry_version"] == GEOMETRY_VERSION


# ---------------------------------------------------------------- 9 并发

def test_concurrent_observe_and_localize_do_not_deadlock_or_mix_state() -> None:
    localizer = ParallelWallLocalizer(thresholds=LocalizationThresholds())
    errors: list[BaseException] = []

    def producer(start: int) -> None:
        try:
            for index in range(6):
                localizer.observe(_frame(plugs=TRAIN_A if index % 2 else TRAIN_B, seed=start + index),
                                  frame_id=start + index, capture_monotonic=float(start + index))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def consumer() -> None:
        try:
            for _ in range(6):
                localizer.localize(now_monotonic=5.0)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=producer, args=(10,)), threading.Thread(target=consumer)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    assert not errors, errors
    assert all(not thread.is_alive() for thread in threads)
    assert localizer.buffered_frames <= localizer.thresholds.max_buffer_frames


# ---------------------------------------------------------------- 当前帧证据（第三轮反例）

def test_blank_newest_frame_is_rejected_even_though_history_was_localized() -> None:
    """Codex 第三轮反例：历史三帧可定位，最新帧全黑时必须拒绝，不得改写帧号冒充新证据。"""
    frames = [_frame(plugs=TRAIN_A, seed=11), _frame(plugs=TRAIN_B, seed=12),
              _frame(plugs=TRAIN_A, seed=13)]
    localizer = _observe_all(frames)
    assert localizer.localize(now_monotonic=102.0).status == "localized"
    localizer.observe(np.zeros_like(frames[0]), frame_id=103, capture_monotonic=103.0)
    result = localizer.localize(now_monotonic=103.0)
    assert result.status == "rejected"
    assert result.reason == "no_current_frame_evidence"
    assert result.wall_lines == []
    assert result.usable is False
    assert result.geometry["frame_id"] == 103
    assert result.geometry["current_frame"]["has_pair"] is False
    # 历史支持帧仍然如实列出，但它们不再被当成“最新帧的定位结果”
    assert result.geometry["support"]["frame_ids"] == [100, 101, 102]


def test_fully_occluded_newest_frame_is_rejected() -> None:
    frames = [_frame(plugs=TRAIN_A, seed=11), _frame(plugs=TRAIN_B, seed=12),
              _frame(plugs=TRAIN_A, seed=13)]
    localizer = _observe_all(frames)
    assert localizer.localize(now_monotonic=102.0).status == "localized"
    occluded = np.full_like(frames[0], 255)
    localizer.observe(occluded, frame_id=104, capture_monotonic=104.0)
    result = localizer.localize(now_monotonic=104.0)
    assert result.status == "rejected"
    assert result.reason == "no_current_frame_evidence"
    assert result.wall_lines == []


def test_sudden_switch_to_another_channel_is_not_silently_accepted() -> None:
    """定位后画面突然换到另一条通道：目标轨道必须以当前帧为准，另一条只能当竞争者。"""
    frames = [_frame(offset=60.0, plugs=TRAIN_A, seed=11),
              _frame(offset=60.0, plugs=TRAIN_B, seed=12),
              _frame(offset=60.0, plugs=TRAIN_A, seed=13)]
    localizer = _observe_all(frames)
    assert localizer.localize(now_monotonic=102.0).status == "localized"
    for index in range(3):
        localizer.observe(_frame(offset=300.0, plugs=TRAIN_B if index % 2 else TRAIN_A,
                                 seed=21 + index),
                          frame_id=110 + index, capture_monotonic=110.0 + index)
    result = localizer.localize(now_monotonic=112.0)
    geometry = result.geometry
    # 当前帧所在的轨道被选中，旧轨道作为竞争者出现；两者差距过大时判歧义
    assert geometry["current_frame"]["has_pair"] is True
    assert geometry["support"]["cluster_count"] >= 2
    assert result.status in {"ambiguous", "insufficient_support", "pending_motion", "coverage_incomplete"}


def test_global_brightness_ramp_is_not_flow() -> None:
    """Codex 第三轮反例：整体加 0/10/20 灰阶不得被判成流动。"""
    base = _frame(plugs=TRAIN_A, seed=11)
    flicker = [np.clip(base.astype(float) + value, 0, 255).astype(np.uint8) for value in (0, 10, 20)]
    result = _observe_all(flicker).localize(now_monotonic=102.0)
    assert result.status == "pending_motion"
    assert result.geometry["motion"]["available"] is False
    current = result.geometry["motion"]["current_frame"]
    assert current["interior_change"] == pytest.approx(0.0, abs=0.5)
    assert current["photometric_offset"] == pytest.approx(10.0, abs=1.0)
    assert current["checks"]["axial_shift_measured"] is False
    assert result.wall_lines == []


def test_gain_change_is_not_flow() -> None:
    """整体增益/曝光变化同样不得被判成流动：校正后管内不再有可分辨变化。"""
    base = _frame(plugs=TRAIN_A, seed=11).astype(float)
    scaled = [np.clip(base * factor, 0, 255).astype(np.uint8) for factor in (1.0, 1.15, 1.3)]
    result = _observe_all(scaled).localize(now_monotonic=102.0)
    assert result.status == "pending_motion"
    assert result.geometry["motion"]["available"] is False
    current = result.geometry["motion"]["current_frame"]
    # 校正把整体变化解掉：带外几乎不变，带内也不再把亮度变化记成内容变化
    assert current["background_change"] < 0.5
    assert current["photometric_residual"] < 1.0
    assert current["checks"]["axial_shift_measured"] is False
    assert result.wall_lines == []


def test_local_light_spot_without_displacement_is_not_flow() -> None:
    """局部光斑出现但不产生沿轴位移时不得认证流动。"""
    frames = []
    for index in range(4):
        frame = _frame(plugs=TRAIN_A, seed=10 + index).astype(float)
        if index == 1:
            cv2.circle(frame, (360, 80), 30, 200.0, -1)
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    result = _observe_all(frames).localize(now_monotonic=103.0)
    assert result.status != "localized"
    assert result.wall_lines == []
    assert result.geometry["motion"]["available"] is False
    # 光斑出现的那一帧之后：有内容变化，但没有可判定的沿轴位移
    after_spot = result.geometry["motion"]["per_frame"][2]
    assert after_spot["checks"]["axial_shift_measured"] is False


def test_static_noise_is_not_flow() -> None:
    frames = [_frame(plugs=TRAIN_A, seed=11 + index) for index in range(4)]
    result = _observe_all(frames).localize(now_monotonic=103.0)
    assert result.status != "localized"
    assert result.geometry["motion"]["available"] in {False, None}


def test_uniform_duct_without_plugs_has_no_axial_structure() -> None:
    """无柱塞的均匀管道没有沿轴结构，不能凭“像素有变化”认证流动。"""
    frames = [_frame(plugs=(), seed=11 + index) for index in range(4)]
    result = _observe_all(frames).localize(now_monotonic=103.0)
    assert result.status != "localized"
    assert result.geometry["motion"]["available"] is False




# ---------------------------------------------------------------- 输出契约

def test_output_contract_contains_every_required_field() -> None:
    frames = [_frame(seed=11), _frame(plugs=TRAIN_B, seed=12), _frame(seed=13)]
    geometry = _observe_all(frames).localize(now_monotonic=102.0).to_dict()
    for key in ("status", "reason", "measurement_segment_px", "wall_lines", "candidates",
                "competition", "support", "motion", "coverage", "coordinate_mapping",
                "geometry_version", "thresholds", "threshold_provenance",
                "generation_zone_exclusion", "pair_rejections"):
        assert key in geometry, key
    assert geometry["support"]["frame_ids"] == [100, 101, 102]
    assert geometry["support"]["time_range_monotonic"] == [100.0, 102.0]
    assert geometry["threshold_provenance"], "阈值必须记录来源"


def test_thresholds_are_documented_with_provenance() -> None:
    from dataclasses import fields as dataclass_fields

    from backend.vision.parallel_walls import THRESHOLD_PROVENANCE

    for item in dataclass_fields(LocalizationThresholds):
        assert item.name in THRESHOLD_PROVENANCE, item.name
        assert THRESHOLD_PROVENANCE[item.name].strip(), item.name


def test_unknown_separation_never_falls_back_to_a_default_channel_width() -> None:
    """没有旧坐标时不允许用 50 µm 反推间距：间距只来自画面。"""
    frames = [_frame(separation=26.0, seed=11), _frame(separation=26.0, plugs=TRAIN_B, seed=12),
              _frame(separation=26.0, seed=13)]
    result = _observe_all(frames).localize(now_monotonic=102.0)
    assert result.status == "localized", result.reason
    assert result.geometry["separation_px"] == pytest.approx(26.0, abs=2.0)
