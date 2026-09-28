"""瞬时采集的分析路径回归测试。

§4 改变了输出契约，因此这些断言也随契约更新（这不是弱化测试）：

* 速度输出被拆成「候选像素位移 / 混叠状态 / 独立约束 / 条件选定速度 / 像素标定 / 稳定性」；
* 没有独立速度约束时状态只能是 ``ALIAS_UNRESOLVED``——不再把 ``estimate_velocity().ok``
  当作「真实速度已经确定」；
* 物理量（mm/s）只在混叠条件解除**且**像素标定通过校验时出现；
* 稳定窗口必须覆盖完整驻留时长，中间不得夹着无效样本。
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import plant_flow_transient_capture as tool  # noqa: E402

CALIBRATION_JSON = {
    "schema_version": 1,
    "calibration_id": "cal-test-20260920",
    "created_at": "2026-09-20T00:00:00Z",
    "magnification": "10x",
    "view_id": "generation-zone",
    "pixel_to_micron": 1.725,
    "uncertainty_um_per_px": 0.01,
    "calibration_image_sha256": "a" * 64,
}


def write_calibration(tmp_dir: Path) -> Path:
    path = tmp_dir / "calibration.json"
    path.write_text(json.dumps(CALIBRATION_JSON), encoding="utf-8")
    return path


@pytest.fixture()
def tmp_dir():
    directory = Path(tempfile.gettempdir()) / f"mcs-transient-{uuid.uuid4().hex[:12]}"
    directory.mkdir()
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def ramp_stack(frames: int = 690, height: int = 100, width: int = 512, band=(40, 80),
               pitch: float = 150.0, droplet_length: float = 105.0,
               start_speed: float = 10.0, end_speed: float = 30.0, ramp_frames: int = 90,
               seed: int = 11) -> np.ndarray:
    """A droplet train that ramps linearly and then holds, like a real start-up.

    The timescale is compressed 100x (dt = 10 ms) so the whole run fits in a test,
    but the slope stays at the real magnitude: 20 px/frame over 0.9 s is
    3.8 mm/s^2, the same order as the 3.9 mm/s^2 measured on the pump.
    """
    random = np.random.default_rng(seed)
    columns = np.arange(width, dtype=np.float32)
    background = np.full((height, width), 140.0, np.float32)
    background += random.normal(0.0, 2.0, (height, width)).astype(np.float32)
    background[band[0] - 4:band[0] - 2, :] += 60.0
    background[band[1] + 3:band[1] + 5, :] += 60.0
    rows = np.zeros((height, 1), np.float32)
    rows[band[0]:band[1] + 1] = 1.0
    kernel = np.exp(-0.5 * (np.arange(-9, 10) / 2.5) ** 2)
    kernel /= kernel.sum()
    positions = np.concatenate([
        np.linspace(start_speed, end_speed, ramp_frames),
        np.full(frames - ramp_frames, end_speed),
    ])
    stack = np.empty((frames, height, width), np.uint8)
    phase_offset = 0.0
    for index in range(frames):
        phase = np.mod(columns - phase_offset, pitch)
        train = np.convolve((phase < droplet_length).astype(np.float32), kernel, mode="same")
        stack[index] = np.clip(background + 55.0 * rows * train[None, :], 0, 255).astype(np.uint8)
        phase_offset += positions[index]
    return stack


def make_samples(times, values, *, ok=True, t_assumed=True, aliasing_status="ALIAS_UNRESOLVED",
                 selected=None, mm_per_second=None) -> list:
    """构造样本；默认是「混叠未解除、时间轴假定」的常见情形。"""
    rows = []
    for index, (moment, value) in enumerate(zip(times, values)):
        valid = ok if isinstance(ok, bool) else ok[index]
        rows.append({
            "t_s": float(moment),
            "t_source": tool.TIMELINE_ASSUMED_USER_RATE if t_assumed
            else tool.TIMELINE_HOST_MONOTONIC,
            "t_assumed": bool(t_assumed),
            "px_per_frame": float(value),
            "px_per_second": float(value) / 0.01,
            "ok": bool(valid),
            "confidence": "multi_gap",
            "residual_px": 0.1,
            "alias_margin": 0.1,
            "agreeing_gaps": [1, 2],
            "aliasing": {"status": aliasing_status, "candidates": [value, value + 150.0],
                         "candidates_exhaustive": aliasing_status == "CONDITIONAL",
                         "selected_px_per_frame": selected,
                         "alias_period_px_per_frame": 150.0,
                         "control_authorized": False},
            "selected_px_per_frame": selected,
            "mm_per_second": mm_per_second,
            "mm_per_second_reason": None if mm_per_second is not None else "未条件解除混叠",
        })
    return rows


# ---------------------------------------------------------------- 离线回放

def test_replay_reports_pixel_domain_only_without_an_independent_constraint(tmp_dir: Path) -> None:
    """没有独立速度约束：平台仍能在像素域找到，但不出任何物理量。"""
    source = tmp_dir / "stack.npy"
    np.save(source, ramp_stack())

    report = tool.run_replay(source, rate=100.0, directory=tmp_dir / "out", window=20, step=5,
                             slope_tolerance=0.05, dwell=5.0)

    assert report["verdict"] == "ALIAS_UNRESOLVED"
    assert report["aliasing"]["status"] == "ALIAS_UNRESOLVED"
    assert report["independent_constraint"]["provided"] is False
    assert "从同一录像推断的流向不作为独立约束" == report["independent_constraint"]["note"]
    assert report["selected_velocity"]["available"] is False
    assert report["pixel_calibration"]["validated"] is False
    assert report["time_axis"]["assumed"] is True
    assert report["stability"]["pixel_domain_plateau_s"] == pytest.approx(0.90, abs=0.60)
    assert report["usable"] == report["samples"] > 20

    payload = json.loads((tmp_dir / "out" / "series.json").read_text(encoding="utf-8"))
    assert all(row["mm_per_second"] is None for row in payload["samples"])
    assert all(row["mm_per_second_reason"] for row in payload["samples"])
    assert all(row["px_per_frame"] is not None for row in payload["samples"]), "像素域诊断必须保留"
    rows = list(csv.DictReader((tmp_dir / "out" / "series.csv").open(encoding="utf-8")))
    assert len(rows) == report["samples"]


def test_replay_with_constraint_and_validated_calibration_gives_a_conditional_result(
    tmp_dir: Path,
) -> None:
    """独立约束 + 有效标定 + 假定时间轴 => 条件稳定结论，且不授权控制。"""
    source = tmp_dir / "stack.npy"
    np.save(source, ramp_stack())
    calibration = tool.load_pixel_calibration(write_calibration(tmp_dir))

    report = tool.run_replay(source, rate=100.0, directory=tmp_dir / "out", window=20, step=5,
                             slope_tolerance=0.05, dwell=5.0,
                             pixel_calibration=calibration,
                             max_displacement=60.0, bound_source="标定尺规实测上限")

    assert report["independent_constraint"]["provided"] is True
    assert report["independent_constraint"]["source"] == "标定尺规实测上限"
    assert report["pixel_calibration"]["validated"] is True
    assert report["pixel_calibration"]["source"].startswith("calibration_record:")
    assert report["selected_velocity"]["available"] is True
    assert report["selected_velocity"]["control_authorized"] is False
    assert report["aliasing"]["status"] == "CONDITIONAL"
    assert report["verdict"] == "CONDITIONAL_STEADY"

    payload = json.loads((tmp_dir / "out" / "series.json").read_text(encoding="utf-8"))
    physical = [row["mm_per_second"] for row in payload["samples"]
                if row["mm_per_second"] is not None]
    assert physical, "条件解除且有有效标定时应当给出物理速度"
    expected_level = 30.0 * 1.725 / 0.01 / 1000.0
    assert max(physical) == pytest.approx(expected_level, rel=0.3)


def test_command_line_scale_does_not_authorise_physical_output(tmp_dir: Path) -> None:
    """显式 --scale 只是未验证标尺：能出像素域结果，不能出物理速度。"""
    source = tmp_dir / "stack.npy"
    np.save(source, ramp_stack(frames=300, ramp_frames=60))

    report = tool.run_replay(source, rate=100.0, directory=tmp_dir / "out", window=20, step=5,
                             slope_tolerance=0.05, dwell=5.0,
                             pixel_calibration=tool.unvalidated_scale(1.725),
                             max_displacement=60.0, bound_source="标定尺规实测上限")

    assert report["pixel_calibration"]["validated"] is False
    assert report["pixel_calibration"]["source"] == "command_line_unsourced"
    assert report["aliasing"]["status"] == "CONDITIONAL"
    assert report["verdict"] == "PIXEL_DOMAIN_ONLY"
    payload = json.loads((tmp_dir / "out" / "series.json").read_text(encoding="utf-8"))
    assert all(row["mm_per_second"] is None for row in payload["samples"])


def test_inferred_direction_is_not_used_to_resolve_aliasing(tmp_dir: Path) -> None:
    """从同一录像推断的流向不能用来筛掉候选分支。"""
    source = tmp_dir / "stack.npy"
    np.save(source, ramp_stack(frames=300, ramp_frames=60))

    report = tool.run_replay(source, rate=100.0, directory=tmp_dir / "out", window=20, step=5,
                             slope_tolerance=0.05, dwell=5.0,
                             pixel_calibration=tool.unvalidated_scale(1.725))

    payload = json.loads((tmp_dir / "out" / "series.json").read_text(encoding="utf-8"))
    learned = [row["direction"] for row in payload["samples"] if row["direction"] is not None]
    assert learned, "该序列应当推断出流向"
    # 尽管推断出了流向，混叠仍未解除，候选分支也没有被剪掉
    assert report["aliasing"]["status"] == "ALIAS_UNRESOLVED"
    assert len(report["aliasing"]["candidates_px_per_frame"]) > 1
    assert report["selected_velocity"]["available"] is False


def test_replay_reports_an_assumed_time_axis(tmp_dir: Path) -> None:
    source = tmp_dir / "stack.npy"
    np.save(source, ramp_stack(frames=200, ramp_frames=60))
    report = tool.run_replay(source, rate=100.0, directory=tmp_dir / "out", window=20,
                             step=5, slope_tolerance=0.05, dwell=5.0)

    assert report["time_axis"]["source"] == tool.TIMELINE_ASSUMED_USER_RATE
    assert report["time_axis"]["assumed"] is True
    assert "不授权控制" in report["time_axis"]["detail"]
    payload = json.loads((tmp_dir / "out" / "series.json").read_text(encoding="utf-8"))
    assert payload["analysis"]["time_axis"]["assumed"] is True
    assert all(row["t_assumed"] for row in payload["samples"])


# ---------------------------------------------------------------- 稳定窗口

def test_a_ramp_shorter_than_the_dwell_is_not_reported_as_steady() -> None:
    """平台必须在远长于瞬态的窗口上成立。"""
    samples = make_samples((0.0, 10.0, 20.0, 30.0, 40.0), (0.0, 5.0, 12.0, 20.0, 30.0),
                           aliasing_status="CONDITIONAL", selected=10.0,
                           mm_per_second=0.5)

    report = tool.analyse(samples, slope_tolerance_mm_s2=0.05, dwell_s=60.0,
                          pixel_calibration={"um_per_px": 1.725, "validated": True},
                          time_axis_assumed=False)

    assert report["stability"]["pixel_domain_plateau_s"] is None
    assert report["verdict"] == "STILL_RISING"


def test_steady_state_time_needs_a_full_dwell_window() -> None:
    samples = make_samples(range(0, 200, 5), [10.0] * 40)

    assert tool.steady_state_time(samples, dwell_s=60.0) == pytest.approx(0.0)
    assert tool.steady_state_time(samples[:6], dwell_s=60.0) is None


def test_a_window_short_of_the_full_dwell_is_not_accepted() -> None:
    """0.8×dwell 的折扣已废除：窗口必须真正覆盖要求时长。"""
    # 采样间隔 10 s、驻留要求 60 s；只有 50 s 的跨度不足以成立
    samples = make_samples(range(0, 60, 10), [10.0] * 6)
    assert tool.steady_state_time(samples, dwell_s=60.0) is None

    # 补足到 60 s 跨度后成立
    samples = make_samples(range(0, 70, 10), [10.0] * 7)
    assert tool.steady_state_time(samples, dwell_s=60.0) == pytest.approx(0.0)


def test_invalid_sample_inside_the_window_breaks_continuity() -> None:
    """中间夹着无效样本时，不得把两侧有效样本拼成一段连续平台。

    正确行为不是「整轮判负」，而是**跳过**含失效样本的窗口：平台只能从失效样本之后
    成立，不能被平滑跨过去。
    """
    times = list(range(0, 200, 5))
    ok = [True] * len(times)
    ok[5] = False                     # t = 25 s 处失效

    samples = make_samples(times, [10.0] * len(times), ok=ok)
    plateau = tool.steady_state_time(samples, dwell_s=60.0)

    assert plateau is not None
    assert plateau >= 30.0, "平台不得跨过 t=25 的失效样本"

    # 对照：若先把失效样本滤掉再判，平台会从 0.0 就被认定成立——那正是被禁止的做法。
    filtered = make_samples([t for index, t in enumerate(times) if ok[index]],
                            [10.0] * (len(times) - 1))
    assert tool.steady_state_time(filtered, dwell_s=60.0) == pytest.approx(0.0)


def test_no_usable_samples_is_reported_as_no_signal() -> None:
    assert tool.analyse([])["verdict"] == "NO_SIGNAL"
    rejected = make_samples((0.0,), (0.0,), ok=False)
    assert tool.analyse(rejected)["verdict"] == "NO_SIGNAL"


def test_sampling_premise_rejection_dominates_the_verdict() -> None:
    samples = make_samples(range(0, 200, 5), [10.0] * 40)
    report = tool.analyse(samples, dwell_s=60.0,
                          sampling_premise=(False, "存在帧号缺口"))
    assert report["verdict"] == "PREMISE_REJECTED"
    assert report["sampling_premise"]["ok"] is False


def test_tracker_holds_the_direction_it_learned(tmp_dir: Path) -> None:
    stack = ramp_stack()
    tracker = tool.TransientTracker(dt=0.01, window=20, step=5)
    for frame in stack:
        tracker.push(frame.astype(np.float32))

    report = tool.analyse(tracker.samples, dwell_s=5.0)

    assert tracker.direction is not None
    assert all(row["direction"] == tracker.direction for row in tracker.samples)
    assert report["displacement"]["max_abs_alias_margin"] < 0.45
    # 流向是从同一录像推断的，因此混叠仍未解除
    assert report["verdict"] == "ALIAS_UNRESOLVED"
