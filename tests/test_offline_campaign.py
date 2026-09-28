"""离线采集入口的无硬件端到端测试：原始帧 → 采集入口 → CSV → 离线稳态分析。

设备（相机、泵）由桩替代；**定位器与检测器都是真实的**，帧取自刚写出的合成 PNG，
因此走的是原始帧包路径，不经过 `frame_jpeg` 预览。合成正路径证明数据通路与门控，
不证明物理精度；真实动态片段不存在，这一点在交付文档里如实注明。
"""
from __future__ import annotations

import importlib
import math
import sys
import threading

import cv2
import numpy as np
import pytest

import backend.vision.service as vision_service_module
from backend.orchestrator.offline_campaign import (
    CSV_FIELDS,
    CampaignConfig,
    FramePacket,
    PhaseSpec,
    SavedFrameSequence,
    analyse_steady,
    packet_stream,
    read_series,
    run_campaign,
)
from backend.orchestrator.vision_adapter import PipelineVisionService

WIDTH = 720
HEIGHT = 540
CHANNEL_UM = 50.0
DUCT_PX = 40.0
SCALE = CHANNEL_UM / DUCT_PX
OFFSET = 60.0
TRAIN_A = ((40, 180), (250, 390), (460, 600))
TRAIN_B = ((110, 250), (320, 460), (530, 670))


class _StubCameraService:
    def __init__(self, logger=None) -> None:
        self._logger = logger

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *args, **kwargs: {}

    def stop(self) -> None:
        pass


def _frame(*, plugs=TRAIN_A, offset=OFFSET, seed=11) -> np.ndarray:
    """柱塞带弯月面亮边（柱塞外缘亮、中间暗）：真实检测器的间隔分割器需要这个横向剖面。"""
    rng = np.random.default_rng(seed)
    rows = int(round(DUCT_PX))
    canvas = np.full((HEIGHT, WIDTH), 20.0, np.float32) + rng.normal(0.0, 0.6, (HEIGHT, WIDTH))
    for x in range(WIDTH):
        top = int(round(offset))
        canvas[top:top + rows, x] = 40.0 + rng.normal(0.0, 0.6, rows)
    for left, right in plugs:
        for x in range(left, right):
            top = int(round(offset))
            profile = np.full(rows, 48.0, np.float32)
            outer = max(1, int(round(rows * 0.22)))
            profile[:outer] += 9.0
            profile[rows - outer:] += 9.0
            profile[outer:rows - outer] -= 9.0
            canvas[top:top + rows, x] = profile + rng.normal(0.0, 0.6, rows)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _walls(offset=OFFSET):
    def norm(y: float) -> float:
        return float(y) / float(HEIGHT)

    return [
        {"x1": 0.0, "y1": norm(offset), "x2": 1.0, "y2": norm(offset)},
        {"x1": 0.0, "y1": norm(offset + DUCT_PX), "x2": 1.0, "y2": norm(offset + DUCT_PX)},
    ]


MODE_KEYS = ("localization_enabled", "strict_detection_localization")


def _roi_config(**overrides):
    """ROI 几何配置。定位模式键不在这里——它们走 set_localization_mode()。"""
    values = {
        "enabled": True, "user_defined": True,
        "x_start_ratio": 0.0, "x_end_ratio": 1.0, "y_start_ratio": 0.1, "y_end_ratio": 0.6,
        "channel_calibration_enabled": True, "channel_width_um": CHANNEL_UM,
        "wall_lines": _walls(), "flow_direction": "negative",
        "generation_measurement_enabled": True, "scale_validated": True,
        "chip_depth_um": CHANNEL_UM,
    }
    values.update(overrides)
    for key in MODE_KEYS:
        values.pop(key, None)
    return values


def _apply_roi(instance, **overrides) -> None:
    """几何与定位模式分开提交；本地默认为「开启定位、非严格」，与原 _roi_config 一致。"""
    values = dict(overrides)
    mode = {key: values.pop(key) for key in MODE_KEYS if key in values}
    instance.set_recognition_roi(_roi_config(**values))
    instance.set_localization_mode(
        strict=bool(mode.get("strict_detection_localization", False)),
        enabled=bool(mode.get("localization_enabled", True)),
        reason="test_apply_roi")


def _service(monkeypatch, **overrides) -> PipelineVisionService:
    monkeypatch.setattr(vision_service_module, "VisionCameraService", _StubCameraService)
    instance = PipelineVisionService()
    instance._log = lambda _message: None
    instance._video_source_type = "industrial_camera"
    instance._video_source = "synthetic"
    instance._camera_unique_id = "synthetic"
    _apply_roi(instance, **overrides)
    instance._channel_calibration_status = "calibrated"
    instance._channel_calibration_confidence = 1.0
    instance._channel_width_um = CHANNEL_UM
    instance._channel_width_px = DUCT_PX
    instance._pixel_to_micron = SCALE
    instance._configured_pixel_to_micron = 1.725
    instance._frame_metadata = {}
    instance._pinned_batch_metadata = {}
    instance._ensure_pipeline().detector.configure_expected_diameter(0.0, SCALE)
    return instance


def _write_frames(directory, *, count: int) -> list:
    paths = []
    for index in range(count):
        plugs = TRAIN_B if index % 2 else TRAIN_A
        path = directory / f"raw_{index:04d}.png"
        cv2.imwrite(str(path), _frame(plugs=plugs, seed=11 + index))
        paths.append(path)
    return paths


def _source(paths, *, time_source="camera_frame_timestamp", interval_s=1.0):
    return SavedFrameSequence(paths=paths, frame_ids=[], capture_monotonic=[],
                              interval_s=interval_s, time_source=time_source)


# ---------------------------------------------------------------- 导入安全

def test_importing_the_offline_entry_does_not_touch_hardware(monkeypatch) -> None:
    """导入入口不得枚举相机、连接串口或操作泵。"""
    calls: list[str] = []

    class _Forbidden:
        def __init__(self, *args, **kwargs):
            calls.append("camera_service")
            raise AssertionError("导入离线入口时构造了相机服务")

    monkeypatch.setattr(vision_service_module, "VisionCameraService", _Forbidden)
    module_name = "backend.orchestrator.offline_campaign"
    sys.modules.pop(module_name, None)
    importlib.import_module(module_name)
    assert calls == []


# ---------------------------------------------------------------- 端到端正路径

def test_campaign_runs_from_raw_frames_to_csv_to_steady_analysis(monkeypatch, tmp_path) -> None:
    """真实定位器 + 真实检测器 + 真实适配器共同工作的离线正路径（只有设备是桩）。"""
    vision = _service(monkeypatch)
    paths = _write_frames(tmp_path, count=30)
    config = CampaignConfig(
        output_dir=tmp_path / "session",
        phases=(PhaseSpec("q1_70_20", 70.0, 20.0, 30),),
        duct_depth_um=CHANNEL_UM,
    )
    summary = run_campaign(vision=vision, frame_source=_source(paths), config=config)
    assert summary["rows"] == 30
    assert summary["pump_commands_sent"] == 0
    assert summary["accepted_rows"] >= 20, summary["rejection_reasons"]
    assert summary["duct_depth_um_used_for_measurement"] == CHANNEL_UM
    assert summary["duct_depth_source_used"] == "declared_chip_geometry"

    rows = read_series(config.output_dir / "diameter_series.csv")
    assert list(rows[0]) == list(CSV_FIELDS)
    accepted = [row for row in rows if row["accepted"] == "True"]
    assert accepted
    for row in accepted:
        assert row["localization_status"] == "localized"
        assert row["measurement_valid"] == "True"
        assert row["scale_validated"] == "True"
        assert row["duct_depth_source"] == "declared_chip_geometry"
        assert row["equivalent_diameters_um"]
        assert row["time_source"] == "camera_frame_timestamp"
        assert row["time_is_proxy"] == "False"
        assert row["command_started_monotonic"] == ""
        assert row["readback_completed_monotonic"] == ""
    # 真实检测器给出的等效直径应当落在合理量级（合成图 ~90 µm），不是 0 也不是桩值
    first = float(accepted[0]["equivalent_diameters_um"].split("|")[0])
    assert 60.0 < first < 140.0

    steady = analyse_steady(config.output_dir / "diameter_series.csv",
                            window_s=8.0, cadence_s=2.0, slope_limit=0.1)
    assert steady["accepted_rows"] == len(accepted)
    phase = steady["phases"]["q1_70_20"]
    assert phase["current_status"] == "currently_stable"
    assert phase["confirmed_window_median_um"] > 0
    assert phase["time_origin"] == "observation"
    assert phase["command_to_steady_s"] is None, "离线无指令时刻，不得给指令到稳态时长"


def test_rejected_frames_are_excluded_from_the_size_series(monkeypatch, tmp_path) -> None:
    vision = _service(monkeypatch, localization_enabled=False, wall_lines=_walls(offset=150.0))
    paths = _write_frames(tmp_path, count=12)
    config = CampaignConfig(output_dir=tmp_path / "session2",
                            phases=(PhaseSpec("q1_70_20", 70.0, 20.0, 12),),
                            duct_depth_um=CHANNEL_UM)
    summary = run_campaign(vision=vision, frame_source=_source(paths), config=config)
    assert summary["accepted_rows"] == 0
    assert summary["rejected_rows"] == 12
    assert summary["rejection_reasons"]
    rows = read_series(config.output_dir / "diameter_series.csv")
    assert all(row["accepted"] == "False" for row in rows)
    assert all(row["equivalent_diameters_um"] == "" for row in rows)
    steady = analyse_steady(config.output_dir / "diameter_series.csv",
                            window_s=4.0, cadence_s=2.0)
    assert steady["accepted_rows"] == 0
    # 全拒绝的档必须仍然出现，并给出状态与拒绝理由（不能从结果里消失）
    assert "q1_70_20" in steady["phases"]
    phase = steady["phases"]["q1_70_20"]
    assert phase["status"] == "no_accepted_samples"
    assert phase["accepted_frames"] == 0
    assert phase["frames"] == 12
    assert phase["rejection_reasons"]


def test_blank_newest_frames_reach_the_csv_as_rejected(monkeypatch, tmp_path) -> None:
    """当前帧无管道的拒绝理由必须落到 CSV 的 accepted=false，而不是只留在诊断里。"""
    vision = _service(monkeypatch)
    paths = _write_frames(tmp_path, count=6)
    blank = tmp_path / "raw_0006.png"
    cv2.imwrite(str(blank), np.zeros((HEIGHT, WIDTH), np.uint8))
    paths.append(blank)
    config = CampaignConfig(output_dir=tmp_path / "session12",
                            phases=(PhaseSpec("q1_70_20", 70.0, 20.0, 7),),
                            duct_depth_um=CHANNEL_UM)
    summary = run_campaign(vision=vision, frame_source=_source(paths), config=config)
    rows = read_series(config.output_dir / "diameter_series.csv")
    assert rows[-1]["accepted"] == "False"
    assert rows[-1]["localization_status"] == "rejected"
    assert rows[-1]["localization_reason"] == "no_current_frame_evidence"
    assert rows[-1]["equivalent_diameters_um"] == ""
    assert "localization:no_current_frame_evidence" in summary["rejection_reasons"]


def test_full_occlusion_after_a_good_run_is_rejected_in_the_csv(monkeypatch, tmp_path) -> None:
    vision = _service(monkeypatch)
    paths = _write_frames(tmp_path, count=8)
    for index in range(8, 11):
        occluded = tmp_path / f"raw_{index:04d}.png"
        cv2.imwrite(str(occluded), np.full((HEIGHT, WIDTH), 255, np.uint8))
        paths.append(occluded)
    config = CampaignConfig(output_dir=tmp_path / "session13",
                            phases=(PhaseSpec("q1_70_20", 70.0, 20.0, 11),),
                            duct_depth_um=CHANNEL_UM)
    run_campaign(vision=vision, frame_source=_source(paths), config=config)
    rows = read_series(config.output_dir / "diameter_series.csv")
    assert all(row["accepted"] == "False" for row in rows[8:])
    assert all(row["equivalent_diameters_um"] == "" for row in rows[8:])


def test_undeclared_depth_is_recorded_without_volume_sizes(monkeypatch, tmp_path) -> None:
    vision = _service(monkeypatch, chip_depth_um=None)
    paths = _write_frames(tmp_path, count=12)
    config = CampaignConfig(output_dir=tmp_path / "session3",
                            phases=(PhaseSpec("q1_70_20", 70.0, 20.0, 12),))
    summary = run_campaign(vision=vision, frame_source=_source(paths), config=config)
    rows = read_series(config.output_dir / "diameter_series.csv")
    assert all(row["duct_depth_source"] == "unknown" for row in rows)
    assert all(row["equivalent_diameters_um"] == "" for row in rows)
    localized = [row for row in rows if row["localization_status"] == "localized"]
    assert localized, "应当有定位成功的帧"
    assert all(row["measurement_reason"] == "duct_depth_unknown" for row in localized)
    assert all(row["measurement_valid"] == "False" for row in localized)
    assert summary["accepted_rows"] == 0
    assert "duct_depth_unknown" in summary["rejection_reasons"]


def test_host_clock_proxy_frames_are_recorded_as_proxy(monkeypatch, tmp_path) -> None:
    vision = _service(monkeypatch)
    paths = _write_frames(tmp_path, count=6)
    config = CampaignConfig(output_dir=tmp_path / "session4",
                            phases=(PhaseSpec("q1_70_20", 70.0, 20.0, 6),),
                            duct_depth_um=CHANNEL_UM)
    run_campaign(vision=vision, frame_source=_source(paths, time_source="host_clock_proxy"),
                 config=config)
    rows = read_series(config.output_dir / "diameter_series.csv")
    assert all(row["time_source"] == "host_clock_proxy" for row in rows)
    assert all(row["time_is_proxy"] == "True" for row in rows)


# ---------------------------------------------------------------- 深度声明来源

def test_config_depth_is_passed_into_the_measurement(monkeypatch, tmp_path) -> None:
    """配置里声明的深度必须真的进入测量链，而不是只写进 summary。"""
    vision = _service(monkeypatch, chip_depth_um=None, generation_measurement_enabled=True)
    paths = _write_frames(tmp_path, count=12)
    config = CampaignConfig(output_dir=tmp_path / "session5",
                            phases=(PhaseSpec("q1_70_20", 70.0, 20.0, 12),),
                            duct_depth_um=100.0)
    summary = run_campaign(vision=vision, frame_source=_source(paths), config=config)
    assert summary["duct_depth_um_used_for_measurement"] == 100.0
    rows = read_series(config.output_dir / "diameter_series.csv")
    localized = [row for row in rows if row["localization_status"] == "localized"]
    assert localized
    assert all(float(row["duct_depth_um"]) == 100.0 for row in localized)
    assert all(row["duct_depth_source"] == "declared_chip_geometry" for row in localized)


def test_conflicting_depth_declarations_are_refused(monkeypatch, tmp_path) -> None:
    """配置声明与服务声明冲突时必须拒绝运行，不能让元数据与实际计算不符。"""
    vision = _service(monkeypatch)          # 服务声明 50
    paths = _write_frames(tmp_path, count=6)
    config = CampaignConfig(output_dir=tmp_path / "session6",
                            phases=(PhaseSpec("q1_70_20", 70.0, 20.0, 6),),
                            duct_depth_um=100.0)
    with pytest.raises(ValueError) as error:
        run_campaign(vision=vision, frame_source=_source(paths), config=config)
    assert "冲突" in str(error.value)


def test_matching_depth_declarations_are_accepted(monkeypatch, tmp_path) -> None:
    vision = _service(monkeypatch)          # 服务声明 50
    paths = _write_frames(tmp_path, count=6)
    config = CampaignConfig(output_dir=tmp_path / "session7",
                            phases=(PhaseSpec("q1_70_20", 70.0, 20.0, 6),),
                            duct_depth_um=CHANNEL_UM)
    summary = run_campaign(vision=vision, frame_source=_source(paths), config=config)
    assert summary["duct_depth_um_used_for_measurement"] == CHANNEL_UM
    assert summary["duct_depth_declared_by_campaign"] == CHANNEL_UM
    assert summary["duct_depth_declared_by_vision"] == CHANNEL_UM


# ---------------------------------------------------------------- 稳态分析的稳健性

def _write_series(path, rows):
    import csv as csv_module

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv_module.DictWriter(
            handle, fieldnames=["capture_monotonic", "phase", "accepted",
                                "equivalent_diameters_um", "measurement_reason",
                                "command_started_monotonic", "readback_completed_monotonic"])
        writer.writeheader()
        writer.writerows(rows)


def _series_rows(name, count=301, *, value_fn=None, reject=None, command=None, readback=None):
    rows = []
    for t in range(count):
        accepted = not (reject and reject(t))
        value = 80.0 if value_fn is None else value_fn(t)
        rows.append(dict(
            capture_monotonic=1000 + t, phase=name, accepted=accepted,
            equivalent_diameters_um=str(value) if accepted else "",
            measurement_reason="" if accepted else "rejected",
            command_started_monotonic="" if command is None else str(command),
            readback_completed_monotonic="" if readback is None else str(readback)))
    return rows


def test_gap_then_recovery_does_not_confirm_inside_the_gap(tmp_path) -> None:
    """第四轮审核 A：130–140 全拒、141 起恢复时，140 秒窗口的最后有效点在 129 秒，
    距窗口终点 11 秒 > 5 秒新鲜度，不得判通过。"""
    path = tmp_path / "gap.csv"
    _write_series(path, _series_rows("gap_then_recovery", reject=lambda t: 130 <= t <= 140))
    phase = analyse_steady(path)["phases"]["gap_then_recovery"]
    window_140 = next(c for c in phase["window_detail"] if c and c["end_s"] == 140.0)
    assert window_140["last_sample_in_window_s"] == pytest.approx(129.0)
    assert window_140["age_of_last_sample_s"] == pytest.approx(11.0)
    assert window_140["fresh"] is False
    assert window_140["pass"] is False
    assert window_140["max_gap_s"] > 5.0
    # 确认时间因此被推到缺口之后
    assert phase["first_confirmation_s"] > 140.0
    # 首次“真实通过”的检查在缺口之前：改名后它不再指首次连续三次通过，
    # 两者混用会让人以为 120 秒的检查没通过。
    assert phase["first_passing_check_s"] == 120.0
    assert phase["first_confirmation_s"] > phase["first_passing_check_s"]


def test_future_data_does_not_change_a_past_check(tmp_path) -> None:
    """第四轮审核 A：同一检查时刻，完整记录与截断到该时刻的记录必须一致。"""
    full_rows = _series_rows("prefix", reject=lambda t: 130 <= t <= 140)
    full_path = tmp_path / "full.csv"
    _write_series(full_path, full_rows)
    full = analyse_steady(full_path)["phases"]["prefix"]
    full_checks = {c["end_s"]: c for c in full["window_detail"] if c}

    for cutoff in (120, 150, 200):
        prefix_path = tmp_path / f"prefix_{cutoff}.csv"
        _write_series(prefix_path, [row for row in full_rows
                                    if row["capture_monotonic"] <= 1000 + cutoff])
        prefix = analyse_steady(prefix_path)["phases"]["prefix"]
        prefix_checks = {c["end_s"]: c for c in prefix["window_detail"] if c}
        for end_s, check in prefix_checks.items():
            assert end_s in full_checks, (cutoff, end_s)
            assert check == full_checks[end_s], (cutoff, end_s)


def _dense_rows(name, *, count, step_s, origin, value=80.0, reject=None):
    """一秒内多帧、零点非整数的记录：只保留实际采样时间，不取整成秒。"""
    rows = []
    for index in range(count):
        moment = origin + index * step_s
        accepted = not (reject and reject(moment))
        rows.append(dict(
            capture_monotonic=moment, phase=name, accepted=accepted,
            equivalent_diameters_um=str(value) if accepted else "",
            measurement_reason="" if accepted else "rejected",
            command_started_monotonic="", readback_completed_monotonic=""))
    return rows


def test_sub_second_frames_with_a_fractional_origin_are_analysed_by_real_time(tmp_path) -> None:
    """第五轮必修1 回归：5 帧/秒、零点非整数时仍按实际采样时间聚合成一秒。"""
    origin = 1000.37
    rows = _dense_rows("dense", count=701, step_s=0.2, origin=origin)
    path = tmp_path / "dense.csv"
    _write_series(path, rows)
    phase = analyse_steady(path)["phases"]["dense"]

    assert phase["samples"] == 701
    assert phase["current_status"] == "currently_stable"
    # 首次“真实通过”与“首次连续三次通过”是两件事，不能互相顶替。
    assert phase["first_passing_check_s"] == 120.0
    assert phase["first_confirmation_s"] == 140.0

    window_120 = next(c for c in phase["window_detail"] if c and c["end_s"] == 120.0)
    assert window_120["window_bounds_s"] == [0.0, 120.0]
    # 窗口内样本只按实际采样时间取，一秒桶由窗口内的样本现算，桶标号是 floor(采样时间)。
    moments = [row["capture_monotonic"] - origin for row in rows]
    inside = [moment for moment in moments if 0.0 < moment <= 120.0]
    assert window_120["samples"] == len(inside)
    assert window_120["second_buckets"] == len({math.floor(moment) for moment in inside})
    # 左界那一秒只是部分秒，不进覆盖率分子：覆盖率不超过 1
    assert window_120["covered_seconds"] <= window_120["coverage_denominator_s"]
    assert window_120["coverage"] <= 1.0
    assert window_120["pass"] is True


def test_coverage_never_exceeds_one_on_a_half_second_grid(tmp_path) -> None:
    """第五轮小修：2 Hz 网格左界那一秒是部分秒，不计入分子，覆盖率不得超过 1。"""
    rows = _dense_rows("half", count=280, step_s=0.5, origin=1000.0)
    path = tmp_path / "half.csv"
    _write_series(path, rows)
    phase = analyse_steady(path)["phases"]["half"]
    window_120 = next(c for c in phase["window_detail"] if c and c["end_s"] == 120.0)
    assert window_120["second_buckets"] == 121          # 0–120 秒都有样本
    assert window_120["covered_seconds"] == 120         # 标号严格大于左界的整秒
    assert window_120["coverage"] == 1.0
    assert window_120["coverage_ok"] is True


def test_prefix_consistency_holds_on_a_sub_second_grid(tmp_path) -> None:
    """第五轮必修1 回归：非整数网格上，同一检查时刻在完整记录与前缀记录中必须一致。"""
    origin = 1000.37
    rows = _dense_rows("dense_prefix", count=701, step_s=0.2, origin=origin,
                       reject=lambda moment: 130.0 <= moment <= 140.0)
    full_path = tmp_path / "dense_full.csv"
    _write_series(full_path, rows)
    full = analyse_steady(full_path)["phases"]["dense_prefix"]
    full_checks = {c["end_s"]: c for c in full["window_detail"] if c}
    assert 120.0 in full_checks

    for cutoff in (125.0, 130.0, 137.4):
        prefix_path = tmp_path / f"dense_prefix_{cutoff}.csv"
        _write_series(prefix_path, [row for row in rows
                                    if row["capture_monotonic"] <= origin + cutoff + 1e-9])
        prefix = analyse_steady(prefix_path)["phases"]["dense_prefix"]
        prefix_checks = {c["end_s"]: c for c in prefix["window_detail"] if c}
        assert 120.0 in prefix_checks, cutoff
        compared = 0
        for end_s, check in prefix_checks.items():
            # 非整数截断会多出一个以实际末帧结尾的窗口，只在共同检查时刻上比对。
            if end_s not in full_checks:
                continue
            assert check == full_checks[end_s], (cutoff, end_s)
            compared += 1
        assert compared >= 1, cutoff


def test_stable_then_drift_is_not_reported_as_currently_stable(tmp_path) -> None:
    """第四轮审核 B：确认后失稳必须单独输出状态，不能只给一个 ok。"""
    path = tmp_path / "drift.csv"
    _write_series(path, _series_rows("steady_then_drift",
                                     value_fn=lambda t: 80 + max(0, t - 160) * 0.5))
    phase = analyse_steady(path)["phases"]["steady_then_drift"]
    assert phase["current_status"] == "lost_after_confirmation"
    assert phase["status"] == "lost_after_confirmation"
    assert phase["first_confirmation_s"] is not None      # 历史仍在
    assert phase["final_confirmation_s"] is None          # 当前不再稳定
    assert phase["later_violation"] is True
    assert phase["reason"]


def test_re_stabilising_gives_a_new_confirmation_time(tmp_path) -> None:
    """第四轮审核 B：稳→失稳→再稳时给出新的持续确认时间，不沿用首次确认。

    末段平台必须长于一个窗口，否则窗口永远跨着上升段，本来就不该判稳定。
    """
    def value(t: int) -> float:
        if t <= 160:
            return 80.0
        if t <= 220:
            return 80 + (t - 160) * 0.5
        return 110.0

    path = tmp_path / "restable.csv"
    _write_series(path, _series_rows("restable", count=361, value_fn=value,
                                     command=1000, readback=1005))
    phase = analyse_steady(path)["phases"]["restable"]
    assert phase["current_status"] == "currently_stable"
    assert phase["first_confirmation_s"] is not None
    assert phase["final_confirmation_s"] is not None
    assert phase["final_confirmation_s"] > phase["first_confirmation_s"]
    assert phase["final_confirmation_source"] == "reconfirmed_after_loss"


def test_command_and_readback_times_are_reported_separately(tmp_path) -> None:
    """第四轮审核 C：只有回读时不得输出“指令到稳态”。"""
    cases = {
        "both": (1000, 1005),
        "command_only": (1000, None),
        "readback_only": (None, 1005),
        "neither": (None, None),
    }
    expected = {
        "both": ("command_start", 140.0, 135.0),
        "command_only": ("command_start", 140.0, None),
        "readback_only": ("readback", None, 140.0),
        "neither": ("observation", None, None),
    }
    for name, (command, readback) in cases.items():
        path = tmp_path / f"{name}.csv"
        _write_series(path, _series_rows(name, command=command, readback=readback))
        phase = analyse_steady(path)["phases"][name]
        origin_kind, command_to, readback_to = expected[name]
        assert phase["time_origin"] == origin_kind
        assert phase["command_to_confirmation_s"] == command_to
        assert phase["readback_to_confirmation_s"] == readback_to
        # 观测窗时长总是给出，且以该档首个观测帧（绝对 1000）为起点
        assert phase["observation_to_confirmation_s"] == pytest.approx(
            phase["first_confirmation"]["absolute_monotonic"] - 1000.0)
        assert phase["observation_to_confirmation_s"] is not None
        assert "不是泵指令响应时间" in phase["observation_to_confirmation_note"]


def test_conflicting_command_and_readback_times_are_refused(tmp_path) -> None:
    """指令晚于回读：时间链自相矛盾，不得用来推导响应时间。"""
    path = tmp_path / "conflict.csv"
    _write_series(path, _series_rows("conflict", command=1010, readback=1005))
    phase = analyse_steady(path)["phases"]["conflict"]
    assert phase["time_conflict"] is not None
    assert phase["command_to_confirmation_s"] is None
    assert phase["readback_to_confirmation_s"] is None
    assert phase["observation_to_confirmation_s"] is not None


def test_multiple_distinct_command_times_in_one_phase_are_refused(tmp_path) -> None:
    rows = _series_rows("multi_cmd", command=1000)
    rows[10]["command_started_monotonic"] = "2000"
    path = tmp_path / "multi_cmd.csv"
    _write_series(path, rows)
    phase = analyse_steady(path)["phases"]["multi_cmd"]
    assert phase["time_conflict"] is not None
    assert "command_started_monotonic" in phase["time_conflict"]


def test_engineering_defaults_are_declared_as_engineering(tmp_path) -> None:
    path = tmp_path / "defaults.csv"
    _write_series(path, _series_rows("defaults"))
    result = analyse_steady(path)
    assert result["window_s"] == 120.0 and result["cadence_s"] == 10.0
    assert result["slope_limit_um_s"] == 0.1
    assert result["min_second_coverage"] == 0.6
    assert "不是物理标定结果" in result["criterion_is_engineering_default"]
    assert "时间常数" in result["criterion_is_engineering_default"]


def test_distribution_width_is_reported_separately(tmp_path) -> None:
    def value(t: int) -> float:
        return 80.0 + (10.0 if t % 2 else 0.0)

    path = tmp_path / "width.csv"
    _write_series(path, _series_rows("width", value_fn=value))
    phase = analyse_steady(path)["phases"]["width"]
    assert phase["confirmed_window_median_um"] == pytest.approx(85.0, abs=1.0)
    assert phase["confirmed_window_p10_p90_span_um"] > 0.0


def test_sparse_data_is_not_reported_as_steady(tmp_path) -> None:
    """Codex 第三轮反例：四个相隔 40 秒的点不得被判成稳态。"""
    path = tmp_path / "sparse.csv"
    _write_series(path, [dict(capture_monotonic=t, phase="sparse", accepted=True,
                              equivalent_diameters_um="80", measurement_reason="ok",
                              command_started_monotonic="", readback_completed_monotonic="")
                         for t in (1000, 1040, 1080, 1120)])
    result = analyse_steady(path)
    phase = result["phases"]["sparse"]
    assert phase["status"] != "ok"
    assert phase["first_confirmation_s"] is None
    assert "覆盖率" in phase["reason"] or "空档" in phase["reason"]


def test_dense_accepted_seconds_are_reported_with_real_times(tmp_path) -> None:
    path = tmp_path / "dense.csv"
    _write_series(path, [dict(capture_monotonic=1000 + t, phase="dense", accepted=True,
                              equivalent_diameters_um="80", measurement_reason="ok",
                              command_started_monotonic="", readback_completed_monotonic="")
                         for t in range(0, 200)])
    result = analyse_steady(path, window_s=60.0, cadence_s=10.0, slope_limit=0.1)
    phase = result["phases"]["dense"]
    assert phase["current_status"] == "currently_stable"
    assert phase["first_confirmation_s"] is not None
    assert phase["time_origin"] == "observation"
    assert phase["command_to_steady_s"] is None


def test_command_times_enable_command_to_steady(tmp_path) -> None:
    path = tmp_path / "commanded.csv"
    _write_series(path, [dict(capture_monotonic=1000 + t, phase="cmd", accepted=True,
                              equivalent_diameters_um="80", measurement_reason="ok",
                              command_started_monotonic="1005",
                              readback_completed_monotonic="1010")
                         for t in range(0, 200)])
    result = analyse_steady(path, window_s=60.0, cadence_s=10.0)
    phase = result["phases"]["cmd"]
    assert phase["time_origin"] == "command_start"
    assert phase["command_to_steady_s"] is not None
    assert phase["command_to_steady_s"] > 0


def test_tail_rejection_marks_the_result_stale(tmp_path) -> None:
    """最近一段持续被拒时，不得仍以旧样本末时刻报告当前稳定。"""
    path = tmp_path / "tail.csv"
    rows = [dict(capture_monotonic=1000 + t, phase="tail", accepted=True,
                 equivalent_diameters_um="80", measurement_reason="ok",
                 command_started_monotonic="", readback_completed_monotonic="")
            for t in range(0, 150)]
    rows += [dict(capture_monotonic=1000 + t, phase="tail", accepted=False,
                  equivalent_diameters_um="", measurement_reason="localization_pending_motion",
                  command_started_monotonic="", readback_completed_monotonic="")
             for t in range(150, 260)]
    _write_series(path, rows)
    result = analyse_steady(path, window_s=60.0, cadence_s=10.0)
    phase = result["phases"]["tail"]
    assert phase["tail_stale"] is True
    assert phase["current_status"] in {"stale_after_confirmation", "insufficient_coverage"}
    assert phase["last_accepted_s"] < phase["phase_last_s"]


def test_out_of_order_timestamps_are_refused(tmp_path) -> None:
    path = tmp_path / "unordered.csv"
    _write_series(path, [dict(capture_monotonic=t, phase="messy", accepted=True,
                              equivalent_diameters_um="80", measurement_reason="ok",
                              command_started_monotonic="", readback_completed_monotonic="")
                         for t in (1000, 1040, 1020, 1060)])
    result = analyse_steady(path, window_s=20.0, cadence_s=5.0)
    assert result["time_anomalies"]["out_of_order_rows"] >= 1
    assert result["phases"]["messy"]["status"] == "time_anomaly"


def test_duplicate_timestamps_are_refused(tmp_path) -> None:
    path = tmp_path / "duplicate.csv"
    _write_series(path, [dict(capture_monotonic=t, phase="dup", accepted=True,
                              equivalent_diameters_um="80", measurement_reason="ok",
                              command_started_monotonic="", readback_completed_monotonic="")
                         for t in (1000, 1000, 1040, 1080)])
    result = analyse_steady(path, window_s=20.0, cadence_s=5.0)
    assert result["time_anomalies"]["duplicate_timestamps"] >= 1
    assert result["phases"]["dup"]["status"] == "time_anomaly"


# ---------------------------------------------------------------- 帧包契约与有界运行

@pytest.mark.parametrize("kwargs,message", [
    ({"frame_id": 0}, "frame_id"),
    ({"hardware_frame_id": 0}, "hardware_frame_id"),
    ({"hardware_frame_id": 5}, "不一致"),
    ({"capture_monotonic": 0.0}, "capture_monotonic"),
    ({"time_source": "wall_clock"}, "时间来源"),
])
def test_frame_packet_rejects_bad_identity(kwargs, message) -> None:
    base = dict(image=np.zeros((8, 8), np.uint8), frame_id=3, hardware_frame_id=3,
                capture_monotonic=10.0, time_source="camera_frame_timestamp")
    base.update(kwargs)
    with pytest.raises(ValueError) as error:
        FramePacket(**base).validate()
    assert message in str(error.value)


def test_frame_packet_rejects_an_empty_image() -> None:
    with pytest.raises(ValueError):
        FramePacket(image=np.zeros((0, 0), np.uint8), frame_id=1, hardware_frame_id=1,
                    capture_monotonic=1.0).validate()


def test_declared_frame_shape_mismatch_is_refused() -> None:
    """调用方声明了期望尺寸时必须核对：不符说明这帧可能不是原始分辨率。"""
    packet = FramePacket(image=np.zeros((41, 640), np.uint8), frame_id=1, hardware_frame_id=1,
                         capture_monotonic=1.0, expected_shape=(720, 540))
    with pytest.raises(ValueError) as error:
        packet.validate()
    assert "尺寸与声明不符" in str(error.value)
    FramePacket(image=np.zeros((540, 720), np.uint8), frame_id=1, hardware_frame_id=1,
                capture_monotonic=1.0, expected_shape=(720, 540)).validate()


def test_packet_stream_from_objects_is_usable(tmp_path, monkeypatch) -> None:
    vision = _service(monkeypatch)
    packets = [FramePacket(image=_frame(seed=index), frame_id=index + 1,
                           hardware_frame_id=index + 1, capture_monotonic=100.0 + index,
                           time_source="camera_frame_timestamp")
               for index in range(6)]
    config = CampaignConfig(output_dir=tmp_path / "session8",
                            phases=(PhaseSpec("q1_70_20", 70.0, 20.0, 6),),
                            duct_depth_um=CHANNEL_UM)
    summary = run_campaign(vision=vision, frame_source=packet_stream(packets), config=config)
    assert summary["rows"] == 6


def test_campaign_does_not_require_threads_or_wait_for_hardware(monkeypatch, tmp_path) -> None:
    """入口本身不起线程、不等待设备：整个运行在调用线程内完成。"""
    before = threading.active_count()
    vision = _service(monkeypatch)
    paths = _write_frames(tmp_path, count=6)
    config = CampaignConfig(output_dir=tmp_path / "session9",
                            phases=(PhaseSpec("q1_70_20", 70.0, 20.0, 6),),
                            duct_depth_um=CHANNEL_UM)
    run_campaign(vision=vision, frame_source=_source(paths), config=config)
    assert threading.active_count() == before


def test_row_limit_stops_the_run_and_keeps_written_rows(monkeypatch, tmp_path) -> None:
    vision = _service(monkeypatch)
    paths = _write_frames(tmp_path, count=20)
    config = CampaignConfig(output_dir=tmp_path / "session10",
                            phases=(PhaseSpec("q1_70_20", 70.0, 20.0, 20),),
                            duct_depth_um=CHANNEL_UM, max_rows=5)
    summary = run_campaign(vision=vision, frame_source=_source(paths), config=config)
    assert summary["rows"] == 5
    assert summary["aborted"]["status"] == "row_limit_reached"
    assert len(read_series(config.output_dir / "diameter_series.csv")) == 5


def test_exception_keeps_already_written_rows(monkeypatch, tmp_path) -> None:
    """异常必须保留已采数据：CSV 里应当留下异常之前写出的行。"""
    vision = _service(monkeypatch)
    paths = _write_frames(tmp_path, count=20)
    calls = {"count": 0}
    original = vision.measure_generation_zone

    def flaky(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 4:
            raise RuntimeError("模拟采集异常")
        return original(*args, **kwargs)

    monkeypatch.setattr(vision, "measure_generation_zone", flaky)
    config = CampaignConfig(output_dir=tmp_path / "session11",
                            phases=(PhaseSpec("q1_70_20", 70.0, 20.0, 20),),
                            duct_depth_um=CHANNEL_UM)
    with pytest.raises(RuntimeError):
        run_campaign(vision=vision, frame_source=_source(paths), config=config)
    rows = read_series(config.output_dir / "diameter_series.csv")
    assert len(rows) == 3
