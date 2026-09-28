"""采集事实与分析结果分离的回归测试。

对应复查任务书 §3。核心是两条：

1. 逐帧记录的是**采集事实**，取不到就记 null 并附原因，**绝不把 dataclass 的默认 0
   当成实测值**——``FrameData`` 在默认（legacy）路径下根本不填硬件帧号、硬件时间戳、
   曝光与丢包，这些字段默认都是 ``0``／``None``，与真实读数无法区分。
2. 时间轴不得再用「收到的帧数 ÷ 设定帧率」冒充实机时间；只能用实测时间戳，或在离线
   旧录像上按使用者显式给出的采样率推算并**标记为假定**。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import plant_flow_transient_capture as tool  # noqa: E402
from backend.vision.cameras.models import FrameData  # noqa: E402


def legacy_frame(frame_id: int = 1, wall_clock: float = 1000.0) -> FrameData:
    """默认（legacy）路径的帧：只有 7 个字段被适配器填充，其余保持 dataclass 默认值。"""
    return FrameData(None, frame_id, wall_clock, valid=True, source_backend="hikrobot")


def direct_frame(*, hardware_frame_id: int = 1, monotonic: float = 1.0,
                 ticks: int = 500, exposure: float = 80.0) -> FrameData:
    """direct 路径的帧：硬件帧号、时间戳 ticks、单调时钟与曝光都被填充。"""
    return FrameData(
        None, hardware_frame_id, 1000.0, valid=True, source_backend="hikrobot",
        host_monotonic_timestamp=monotonic, hardware_frame_id=hardware_frame_id,
        hardware_timestamp_ticks=ticks, sdk_host_timestamp_ticks=ticks + 7,
        lost_packet_count=0, exposure_time_us=exposure,
    )


# ---------------------------------------------------------------- 逐帧事实

def test_unpopulated_fields_are_null_with_reason_not_default_zero() -> None:
    """legacy 路径下的 0 必须记成 null + 原因，并保留 raw 值。"""
    facts = tool.frame_facts(legacy_frame(), software_index=0, sampled_monotonic=1000.0)

    for field in ("hardware_frame_id", "hardware_timestamp_ticks", "sdk_host_timestamp_ticks",
                  "host_monotonic_timestamp", "exposure_time_us"):
        assert facts[field]["value"] is None, field
        assert facts[field]["reason"], f"{field} 必须说明为何不可用"
    # 原始默认值仍在，信息没被丢掉
    assert facts["hardware_frame_id"]["raw"] == 0
    assert facts["host_monotonic_timestamp"]["raw"] == 0.0
    # 适配器确实提供了的少数字段照实记录
    assert facts["adapter_frame_id"] == 1
    assert facts["adapter_timestamp"] == pytest.approx(1000.0)


def test_hardware_timestamp_units_are_never_invented() -> None:
    """硬件 ticks 的单位与时基仓库没有记录，因此只能登记 ticks，不得假定单位。"""
    facts = tool.frame_facts(direct_frame(ticks=123456), software_index=0, sampled_monotonic=5.0)
    assert facts["hardware_timestamp_ticks"]["value"] == 123456
    assert facts["hardware_timestamp_units"]["value"] is None
    assert "未记录" in facts["hardware_timestamp_units"]["reason"]


def test_populated_fields_are_recorded_for_the_direct_path() -> None:
    facts = tool.frame_facts(direct_frame(hardware_frame_id=42, monotonic=3.5, ticks=999,
                                          exposure=80.0),
                             software_index=7, sampled_monotonic=9.9)
    assert facts["software_frame_index"] == 7
    assert facts["sampled_host_monotonic"] == pytest.approx(9.9)
    assert facts["hardware_frame_id"]["value"] == 42
    assert facts["host_monotonic_timestamp"]["value"] == pytest.approx(3.5)
    assert facts["exposure_time_us"]["value"] == pytest.approx(80.0)
    assert facts["sdk_host_timestamp_ticks"]["value"] == 1006
    assert facts["sampled_host_monotonic"] != facts["host_monotonic_timestamp"]["value"]


def test_unavailable_image_and_crop_carry_reasons() -> None:
    facts = tool.frame_facts(legacy_frame(), software_index=0, sampled_monotonic=1.0)
    assert facts["image_ref"]["value"] is None and facts["image_ref"]["reason"]
    assert facts["crop"]["value"] is None and facts["crop"]["reason"]

    facts = tool.frame_facts(legacy_frame(), software_index=0, sampled_monotonic=1.0,
                             crop={"x": 10, "width": 200}, block_index=3)
    assert facts["image_ref"]["value"] == {"kind": "block_index", "index": 3}
    assert facts["crop"]["value"] == {"x": 10, "width": 200}


# ---------------------------------------------------------------- 记录器

def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_records_correspond_one_to_one_with_frame_indices(tmp_path: Path) -> None:
    recorder = tool.FrameFactsRecorder(path=tmp_path / "frames.ndjson", max_backlog=64)
    for index in range(5):
        recorder.submit(direct_frame(hardware_frame_id=index + 1, monotonic=1.0 + index))
    recorder.close()

    records = read_records(tmp_path / "frames.ndjson")
    assert [row["software_frame_index"] for row in records] == [0, 1, 2, 3, 4]
    assert [row["hardware_frame_id"]["value"] for row in records] == [1, 2, 3, 4, 5]
    assert recorder.accepted == 5


def test_duplicate_missing_and_non_monotonic_frames_are_recorded(tmp_path: Path) -> None:
    recorder = tool.FrameFactsRecorder(path=tmp_path / "frames.ndjson", max_backlog=64)
    # 1,1 重复；1 -> 5 缺 2/3/4；时间戳 1.0 -> 0.5 回退
    for hardware_id, moment in ((1, 1.0), (1, 1.1), (5, 0.5)):
        recorder.submit(direct_frame(hardware_frame_id=hardware_id, monotonic=moment))
    recorder.close()

    assert recorder.duplicate_frames == 1
    assert recorder.missing_frames == 3
    assert recorder.non_monotonic_timestamps >= 1
    kinds = {row["kind"] for row in recorder.anomalies}
    assert {"duplicate_hardware_frame_id", "missing_hardware_frames"} <= kinds
    assert "non_monotonic_timestamp" in kinds


def test_assumed_timeline_refuses_to_claim_the_sampling_premise(tmp_path: Path) -> None:
    recorder = tool.FrameFactsRecorder(path=tmp_path / "frames.ndjson", max_backlog=8,
                                       timeline_source=tool.TIMELINE_ASSUMED_USER_RATE,
                                       timeline_assumed=True)
    recorder.submit(legacy_frame())
    recorder.close()

    ok, reason = recorder.sampling_premise_ok()
    assert ok is False
    assert "假定" in reason
    assert recorder.summary()["timeline_assumed"] is True


def test_clean_stream_satisfies_the_sampling_premise(tmp_path: Path) -> None:
    recorder = tool.FrameFactsRecorder(path=tmp_path / "frames.ndjson", max_backlog=64)
    for index in range(6):
        recorder.submit(direct_frame(hardware_frame_id=index + 1, monotonic=1.0 + index))
    recorder.close()

    ok, reason = recorder.sampling_premise_ok()
    assert ok is True and reason == ""
    assert recorder.summary()["sampling_premise_ok"] is True


def test_backlog_cap_fails_instead_of_dropping_records(tmp_path: Path) -> None:
    """写盘被延后时必须失败，不能悄悄漏写。"""
    recorder = tool.FrameFactsRecorder(path=tmp_path / "frames.ndjson", max_backlog=3,
                                       flush_every=1000)
    for index in range(3):
        recorder.submit(direct_frame(hardware_frame_id=index + 1, monotonic=float(index)))
    with pytest.raises(tool.StorageBacklogError):
        recorder.submit(direct_frame(hardware_frame_id=4, monotonic=3.0))


# ---------------------------------------------------------------- 时间轴

def test_tracker_marks_the_count_based_timeline_as_assumed() -> None:
    """不给逐帧时间信息时只能按帧数推算，样本必须同时带 t_assumed 与 dt_assumed。"""
    import numpy as np

    tracker = tool.TransientTracker(dt=0.01, window=4, step=1, lock_frames=4)
    stack = np.random.default_rng(0).integers(0, 255, (12, 40, 128)).astype(np.uint8)
    for frame in stack:
        tracker.push(frame)

    assert tracker.samples, "应当产出样本"
    assert all(row["t_assumed"] is True for row in tracker.samples)
    assert all(row["dt_assumed"] is True for row in tracker.samples)
    assert all(row["t_source"] == tool.TIMELINE_ASSUMED_USER_RATE for row in tracker.samples)
    assert all(row["timing_ok"] is True for row in tracker.samples)
    assert tracker.samples[0]["t_s"] < tracker.samples[-1]["t_s"]


def test_replay_marks_every_sample_with_the_assumed_axis(tmp_path: Path) -> None:
    """离线回放的时间轴是假定的，逐样本标记必须一致（物理量与稳定性结论见分析测试）。"""
    import numpy as np

    from test_plant_flow_transient_capture import ramp_stack

    source = tmp_path / "stack.npy"
    np.save(source, ramp_stack(frames=200, ramp_frames=60))
    report = tool.run_replay(source, rate=100.0, directory=tmp_path / "out", window=20,
                             step=5, slope_tolerance=0.05, dwell=5.0)

    assert report["time_axis"]["assumed"] is True
    payload = json.loads((tmp_path / "out" / "series.json").read_text(encoding="utf-8"))
    assert payload["analysis"]["time_axis"]["assumed"] is True
    assert all(row["t_assumed"] for row in payload["samples"])
    assert all(row["t_source"] == tool.TIMELINE_ASSUMED_USER_RATE for row in payload["samples"])


# ---------------------------------------------------------------- 与采集会话联通

def test_session_records_commands_and_frame_facts(tmp_path: Path) -> None:
    """命令记录与逐帧事实必须分别落盘，且命令顺序与实际操作一致。"""
    from test_transient_capture_lifecycle import (
        FakeClock, FakeFrame, RecordingLock, build, make_plan,
    )

    plan = make_plan(termination={"max_session_s": 1.0},
                     capture_plan={"max_no_valid_frame_s": 5.0, "max_backlog_frames": 64})
    sink = tool.FrameFactsRecorder(path=tmp_path / "out" / "frames.ndjson", max_backlog=64)
    session, _pump, _camera, lock, _sink, _order = build(plan, tmp_path, clock_step=0.1, sink=sink)
    session.camera.frames = [FakeFrame(frame_id=i) for i in range(40)]
    report = session.run()

    assert report["verdict"] == "CAPTURE_COMPLETE"
    command_ids = [row["command_id"] for row in report["commands"]]
    assert command_ids == ["connect-and-probe", "wsp-ch1", "wsp-ch2", "start-infusion",
                           "stop-attempt-1"]
    for row in report["commands"]:
        assert row["sent_monotonic"] is not None
        assert row["readback_monotonic"] is not None
        assert row["readback_monotonic"] >= row["sent_monotonic"]
        assert row["ok"] is True

    assert (tmp_path / "out" / "commands.ndjson").exists()
    written = read_records(tmp_path / "out" / "commands.ndjson")
    assert [row["command_id"] for row in written] == command_ids

    facts = report["frame_facts"]
    assert facts["records"] == report["frames_accepted"] > 0
    assert (tmp_path / "out" / "frames.ndjson").exists()
    assert len(read_records(tmp_path / "out" / "frames.ndjson")) == report["frames_accepted"]
    assert lock.released == 1


def test_session_summary_is_written_even_when_empty(tmp_path: Path) -> None:
    from test_transient_capture_lifecycle import build, make_plan

    plan = make_plan()
    session, _pump, _camera, _lock, _sink, _order = build(plan, tmp_path, lock_busy=True)
    report = session.run()

    assert report["verdict"] == "FAILED"
    stored = json.loads((tmp_path / "out" / "session_summary.json").read_text(encoding="utf-8"))
    assert stored["verdict"] == "FAILED"
    # 设备被占用时从未发出任何设备命令（只有清理时的断开），如实为 False。
    # 2026-09-24 前这里是硬编码 True，把「没碰过设备」报成了「碰过」。
    assert stored["touched_hardware"] is False


# ---------------------------------------------------------------- 采集链的标尺边界

def test_capture_chain_has_no_default_pixel_scale() -> None:
    """采集链不得存在任何默认标尺：物理量只能经由通过校验的标定记录产生。"""
    tracker = tool.TransientTracker(dt=0.01)
    assert tracker.pixel_calibration is None
    assert not hasattr(tracker, "pixel_to_micron"), "采集链不应再有 pixel_to_micron 默认值"

    # 未标定时即便混叠已条件解除也给不出物理量
    physical = tracker._physical({"status": "CONDITIONAL", "selected_px_per_frame": 10.0}, 0.01)
    assert physical["mm_per_second"] is None
    assert "标定" in physical["reason"]


def test_capture_chain_does_not_import_the_library_physical_entry_points() -> None:
    """不得把 flow_locking 里带默认标尺的物理输出入口引进采集链。

    ``measure_flow`` 与 ``ChannelLockFlow`` 的 ``pixel_to_micron`` 默认是 1.725，
    一旦引入就会产出未标定的物理量（该库 CLI 仍如此，已登记为后续整改项）。
    """
    for name in ("measure_flow", "ChannelLockFlow"):
        assert not hasattr(tool, name), f"采集链不应引入 {name}"


def test_only_a_validated_calibration_record_enables_physical_output() -> None:
    scale = tool.unvalidated_scale(1.725)
    assert scale["validated"] is False
    assert scale["source"] == "command_line_unsourced"
