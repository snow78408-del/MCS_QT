"""时间轴资格与「实测间隔换算速度」的回归测试。

对应复查意见里「时间轴修复还不完整」这一条。修复前的问题是：``push(timestamp=...)``
只改了样本的时间**标签** ``t_s``，而 ``measure()`` 仍把 ``self.dt``（设定帧率）交给
``estimate_velocity``，``_physical()`` 也用 ``selected / self.dt`` 换算——于是标签是实测的、
速度却是按设定帧率算出来的。本文件钉住修复后的不变量：

* 窗口必须**通过资格检查**（时间戳有限、严格递增、采样间隔均匀、设备帧号连续）才能参与速度换算；
* 合格窗口一律使用**实测间隔**，不再用设定帧率；
* 不合格窗口不产出 ``px_per_second``，也不参与任何稳定结论；
* 设备 ticks 的单位未知，不得充当时间轴；墙钟与主机接收时间分开记录，互不顶替。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import plant_flow_transient_capture as tool  # noqa: E402

FrameTiming = tool.FrameTiming


def constant_speed_stack(frames: int = 40, speed: float = 20.0) -> np.ndarray:
    """恒定位移的液滴序列：每帧位移相同，便于验证速度只随时间间隔变化。"""
    from test_plant_flow_transient_capture import ramp_stack

    return ramp_stack(frames=frames, ramp_frames=1, start_speed=speed, end_speed=speed)


# ---------------------------------------------------------------- 资格检查

def test_window_without_any_timestamp_falls_back_to_the_assumed_rate() -> None:
    ok, reason, dt, source = tool.qualify_window([FrameTiming() for _ in range(5)], nominal_dt=0.01)
    assert ok is True and reason == ""
    assert dt == pytest.approx(0.01)
    assert source == tool.TIMELINE_ASSUMED_USER_RATE


def test_device_ticks_alone_are_not_a_timeline() -> None:
    """设备 ticks 的单位与时基未记录，不能当作采集时钟，只能退回假定时间轴。"""
    timings = [FrameTiming(device_ticks=1000 + index) for index in range(5)]
    ok, _reason, dt, source = tool.qualify_window(timings, nominal_dt=0.02)
    assert ok is True
    assert source == tool.TIMELINE_ASSUMED_USER_RATE
    assert dt == pytest.approx(0.02)


def test_measured_host_timestamps_qualify_and_yield_the_measured_interval() -> None:
    timings = [FrameTiming(host_received=100.0 + index * 0.02) for index in range(5)]
    ok, reason, dt, source = tool.qualify_window(timings, nominal_dt=0.01)
    assert ok is True and reason == ""
    assert dt == pytest.approx(0.02)
    assert source == tool.CLOCK_HOST_RECEIVED


def test_host_received_is_preferred_over_wall_clock() -> None:
    timings = [FrameTiming(host_received=50.0 + index * 0.05,
                           wall_clock=1000.0 + index * 5.0) for index in range(5)]
    _ok, _reason, dt, source = tool.qualify_window(timings, nominal_dt=0.01)
    assert source == tool.CLOCK_HOST_RECEIVED
    assert dt == pytest.approx(0.05)


def test_wall_clock_is_used_only_when_host_received_is_absent() -> None:
    timings = [FrameTiming(wall_clock=1000.0 + index * 0.03) for index in range(5)]
    ok, reason, dt, source = tool.qualify_window(timings, nominal_dt=0.01)
    assert ok is True and reason == ""
    assert source == tool.CLOCK_WALL_CLOCK
    assert dt == pytest.approx(0.03)


def test_partial_timestamps_are_rejected() -> None:
    timings = [FrameTiming(host_received=100.0 + index * 0.02) for index in range(4)]
    timings.append(FrameTiming())              # 最后一帧没有时间戳
    ok, reason, dt, _source = tool.qualify_window(timings, nominal_dt=0.01)
    assert ok is False and dt is None
    assert "不完整" in reason


@pytest.mark.parametrize(
    "values,expect",
    [
        ([1.0, 2.0, float("nan"), 4.0, 5.0], "非有限"),
        ([1.0, 2.0, float("inf"), 4.0, 5.0], "非有限"),
        ([1.0, 2.0, 1.5, 4.0, 5.0], "严格递增"),      # 乱序
        ([1.0, 2.0, 2.0, 4.0, 5.0], "严格递增"),      # 重复时间戳
        ([1.0, 2.0, 6.0, 7.0, 8.0], "不均匀"),        # 中间缺帧：间隔被拉长
    ],
)
def test_bad_timelines_are_rejected(values: list, expect: str) -> None:
    timings = [FrameTiming(host_received=value) for value in values]
    ok, reason, dt, source = tool.qualify_window(timings, nominal_dt=0.01)
    assert ok is False
    assert dt is None
    assert expect in reason, reason
    assert source == tool.CLOCK_HOST_RECEIVED


def test_hardware_frame_id_gap_is_rejected() -> None:
    timings = [FrameTiming(host_received=100.0 + index * 0.02,
                           hardware_frame_id=10 + index) for index in range(4)]
    timings.append(FrameTiming(host_received=100.0 + 4 * 0.02, hardware_frame_id=99))
    ok, reason, _dt, _source = tool.qualify_window(timings, nominal_dt=0.01)
    assert ok is False
    assert "帧号不连续" in reason


def test_empty_window_is_not_qualified() -> None:
    ok, reason, dt, _source = tool.qualify_window([], nominal_dt=0.01)
    assert ok is False and dt is None and reason


# ---------------------------------------------------------------- 实测间隔换算速度

def test_velocity_uses_the_measured_interval_not_the_nominal_rate() -> None:
    """核心回归：名义 dt=0.01，实际间隔 0.02 ⇒ 速度必须按 0.02 换算。"""
    stack = constant_speed_stack()
    tracker = tool.TransientTracker(dt=0.01, window=8, step=1, lock_frames=8)
    for index, frame in enumerate(stack):
        tracker.push(frame, timing=FrameTiming(host_received=100.0 + index * 0.02))

    assert tracker.samples
    sample = tracker.samples[-1]
    assert sample["timing_ok"] is True
    assert sample["dt_s"] == pytest.approx(0.02)
    assert sample["dt_source"] == tool.CLOCK_HOST_RECEIVED
    assert sample["dt_assumed"] is False
    assert sample["px_per_second"] == pytest.approx(sample["px_per_frame"] / 0.02)
    # 若仍按名义 dt=0.01 换算，速度会大一倍——这正是修复前的结果
    assert sample["px_per_second"] != pytest.approx(sample["px_per_frame"] / 0.01, rel=0.01)


def test_doubling_the_real_frame_interval_halves_the_speed() -> None:
    """同样的逐帧位移，真实帧间隔加倍时速度必须减半。"""
    stack = constant_speed_stack()
    speeds = {}
    for label, spacing in (("fast", 0.01), ("slow", 0.02)):
        tracker = tool.TransientTracker(dt=0.01, window=8, step=1, lock_frames=8)
        for index, frame in enumerate(stack):
            tracker.push(frame, timing=FrameTiming(host_received=100.0 + index * spacing))
        speeds[label] = tracker.samples[-1]

    assert speeds["fast"]["px_per_frame"] == pytest.approx(speeds["slow"]["px_per_frame"], rel=0.05)
    assert speeds["fast"]["px_per_second"] == pytest.approx(
        2.0 * speeds["slow"]["px_per_second"], rel=0.05)


def test_disqualified_window_yields_no_pixel_rate_or_physical_speed() -> None:
    stack = constant_speed_stack()
    tracker = tool.TransientTracker(dt=0.01, window=8, step=1, lock_frames=8)
    for index, frame in enumerate(stack):
        # 第 20 帧处插入一个双倍间隔 => 窗口间隔不均匀，直接拒绝
        moment = 100.0 + index * 0.01 + (0.01 if index >= 20 else 0.0)
        tracker.push(frame, timing=FrameTiming(host_received=moment))

    assert tracker.samples
    disqualified = [row for row in tracker.samples if not row["timing_ok"]]
    assert disqualified, "缺帧窗口必须被判为不合格"
    for row in disqualified:
        assert row["px_per_second"] is None, "不合格窗口不得产出像素速率"
        assert row["mm_per_second"] is None, "不合格窗口不得产出物理速度"
        assert row["timing_reason"]
    # 合格窗口仍然照常产出（诊断不丢弃）
    assert any(row["timing_ok"] and row["px_per_frame"] is not None for row in tracker.samples)


def test_analysis_rejects_everything_when_any_window_is_disqualified() -> None:
    """只要存在不合格窗口，本轮就不给任何稳定平台。"""
    times = list(range(0, 200, 5))
    rows = []
    for index, moment in enumerate(times):
        rows.append({
            "t_s": float(moment), "t_source": tool.CLOCK_HOST_RECEIVED, "t_assumed": False,
            "dt_s": 5.0, "dt_source": tool.CLOCK_HOST_RECEIVED, "dt_assumed": False,
            "timing_ok": index != 7, "timing_reason": "" if index != 7 else "采样间隔不均匀",
            "px_per_frame": 10.0, "px_per_second": 2.0, "ok": True, "confidence": "multi_gap",
            "residual_px": 0.1, "alias_margin": 0.1, "agreeing_gaps": [1],
            "aliasing": {"status": "ALIAS_UNRESOLVED", "candidates": [10.0, 160.0],
                         "selected_px_per_frame": None, "control_authorized": False},
            "selected_px_per_frame": None, "mm_per_second": None, "mm_per_second_reason": "x",
        })

    report = tool.analyse(rows, dwell_s=60.0)

    assert report["timing"]["timeline_usable"] is False
    assert report["timing"]["disqualified_windows"] == 1
    assert report["stability"]["timeline_usable"] is False
    assert report["stability"]["pixel_domain_plateau_s"] is None
    assert report["verdict"] == "PREMISE_REJECTED"


def test_tracker_timing_summary_reports_sources_and_usability() -> None:
    stack = constant_speed_stack()
    tracker = tool.TransientTracker(dt=0.01, window=8, step=1, lock_frames=8)
    for index, frame in enumerate(stack):
        tracker.push(frame, timing=FrameTiming(host_received=100.0 + index * 0.02))

    summary = tracker.timing_summary()
    assert summary["disqualified"] == 0
    assert summary["qualified"] == len(tracker.samples) > 0
    assert summary["timeline_usable"] is True
    assert summary["dt_sources"] == {tool.CLOCK_HOST_RECEIVED: len(tracker.samples)}
