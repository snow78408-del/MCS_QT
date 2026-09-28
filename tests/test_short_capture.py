"""短时验证采集的计划、预算、拒绝路径与运行期守卫测试：全程不接触设备。

计划侧只校验数值与现场字段；运行期测试把注入帧来源喂给入口，核对原始帧确实先落盘、
可逐帧还原，并且时长/间隔/帧数/容量/写盘异常都能收尾。设备接入由现场侧装配，
这些测试不证明真机场景，只证明离线接线与停止判据本身成立。
"""
from __future__ import annotations

import csv
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import backend.vision.service as vision_service_module
from backend.orchestrator.offline_campaign import FramePacket, analyse_steady
from backend.orchestrator.short_capture import (
    CaptureGuards,
    DepthEvidencePlan,
    DeviceReadiness,
    RawFrameWriter,
    ScaleEvidencePlan,
    ShortCapturePlan,
    StorageBudget,
    plan_report,
    plan_to_dict,
    run_short_capture,
)

WIDTH = 720
HEIGHT = 540
SMALL = (32, 24)          # 守卫测试用小图：容量/帧数上限按实际像素计，不靠预算估算


def _ready_plan(**overrides) -> ShortCapturePlan:
    values = dict(
        duration_s=90.0,
        scale=ScaleEvidencePlan(source="scale_bar", reference_um=100.0, reference_px=94.0,
                                image_path="scale_bar.png", uncertainty_um_per_px=0.01,
                                captured_with_same_optics=True),
        depth=DepthEvidencePlan(depth_um=50.0, source="declared_chip_geometry",
                               evidence_note="芯片规格书"),
        readiness=DeviceReadiness(camera_unique_id="HIKROBOT:DIRECT:0", pump_port="COM12",
                                  pump_connection_verified=True, oil_remaining="现场填写",
                                  water_remaining="现场填写",
                                  allowed_q1_range=(50.0, 90.0), allowed_q2_range=(20.0, 40.0),
                                  available_travel_note="现场填写", stop_boundary_note="现场填写"),
        budget=StorageBudget(free_space_note="现场确认剩余空间充足"),
    )
    values.update(overrides)
    return ShortCapturePlan(**values)


# ---------------------------------------------------------------- 导入安全

def test_importing_the_short_capture_module_does_not_touch_hardware(monkeypatch) -> None:
    calls: list[str] = []

    class _Forbidden:
        def __init__(self, *args, **kwargs):
            calls.append("camera_service")
            raise AssertionError("导入短时采集模块时构造了相机服务")

    monkeypatch.setattr(vision_service_module, "VisionCameraService", _Forbidden)
    sys.modules.pop("backend.orchestrator.short_capture", None)
    importlib.import_module("backend.orchestrator.short_capture")
    assert calls == []


# ---------------------------------------------------------------- 默认拒绝执行

def test_default_plan_is_not_executable_and_refuses_to_run() -> None:
    plan = ShortCapturePlan()
    report = plan_report(plan)
    assert plan.execution_enabled is False
    assert report["execution_enabled"] is False
    # 两个状态必须分开：默认计划连离线回放准备都不合格，更谈不上现场就绪。
    assert report["offline_replay_ready"] is False
    assert report["site_ready_to_execute"] is False
    result = run_short_capture(plan, vision=object(), frame_source=object(),
                               output_dir=Path("unused"))
    assert result["status"] == "refused"
    assert "execution_enabled=False" in result["reason"]


def test_ready_plan_still_refuses_without_a_frame_source() -> None:
    plan = _ready_plan(execution_enabled=True)
    report = plan_report(plan)
    assert report["blockers"] == []
    assert report["offline_replay_ready"] is True
    assert report["site_ready_to_execute"] is True
    result = run_short_capture(plan, output_dir=Path("unused"))
    assert result["status"] == "refused"
    assert "设备接入由现场侧装配" in result["reason"]


def test_unconfirmed_site_fields_block_execution_even_when_enabled() -> None:
    """现场字段没填全时，execution_enabled=True 也不能变成执行许可。"""
    plan = _ready_plan(execution_enabled=True, budget=StorageBudget())
    report = plan_report(plan)
    assert "free_space_note" in report["site_fields_pending"]
    assert report["offline_replay_ready"] is True
    assert report["site_ready_to_execute"] is False
    result = run_short_capture(plan, vision=_StubVision(), frame_source=_frames(2),
                               output_dir=Path("unused"))
    assert result["status"] == "refused"
    assert "现场字段" in result["reason"]


def test_short_capture_module_never_opens_devices_even_when_enabled(monkeypatch,
                                                                   tmp_path) -> None:
    """即使计划通过校验，本模块也不自己连设备：只消费注入的帧来源。"""
    opened: list[str] = []

    class _Forbidden:
        def __init__(self, *args, **kwargs):
            opened.append("camera_service")
            raise AssertionError("短时采集入口不应构造相机服务")

    monkeypatch.setattr(vision_service_module, "VisionCameraService", _Forbidden)
    plan = _ready_plan(execution_enabled=True)
    result = run_short_capture(plan, vision=_StubVision(), frame_source=_frames(4),
                               output_dir=tmp_path / "short")
    assert result["status"] == "executed"
    assert opened == [], "不得构造相机服务"
    assert result["summary"]["pump_commands_sent"] == 0
    assert result["summary"]["frames_persisted"] == 4
    assert result["summary"]["scale_binding"]["mode"] == "bound_to_vision"


# ---------------------------------------------------------------- 计划校验项

def test_scale_evidence_must_be_independent() -> None:
    pending = _ready_plan(scale=ScaleEvidencePlan())
    report = plan_report(pending)
    assert report["scale_is_independent"] is False
    assert any("独立证据" in item for item in report["blockers"])

    assumption_only = _ready_plan(scale=ScaleEvidencePlan(
        source="channel_width_assumption", reference_um=50.0, reference_px=47.0,
        captured_with_same_optics=True))
    assert plan_report(assumption_only)["scale_is_independent"] is False
    assert any("只有通道宽度假设" in item or "独立证据" in item
               for item in plan_report(assumption_only)["blockers"])


def test_a_declared_independent_scale_needs_its_reference_image() -> None:
    """声明独立标尺来源却没有登记标尺原图，不算独立证据。"""
    without_image = ScaleEvidencePlan(source="scale_bar", reference_um=100.0,
                                      reference_px=94.0, captured_with_same_optics=True)
    assert without_image.is_independent is False
    report = plan_report(_ready_plan(scale=without_image))
    assert any("标尺原图" in item for item in report["blockers"])

    measured = ScaleEvidencePlan(source="scale_bar", reference_um=100.0, reference_px=94.0,
                                 image_path="scale_bar.png", captured_with_same_optics=True)
    assert measured.is_independent is True
    assert measured.um_per_px == pytest.approx(100.0 / 94.0)


def test_channel_width_assumption_is_not_labelled_as_independent_calibration() -> None:
    assumption = ScaleEvidencePlan(source="channel_width_assumption", reference_um=50.0,
                                   reference_px=47.0, image_path="assumed.png",
                                   captured_with_same_optics=True)
    measured = ScaleEvidencePlan(source="scale_bar", reference_um=100.0, reference_px=94.0,
                                 image_path="scale_bar.png", captured_with_same_optics=True)
    assert assumption.is_independent is False
    assert measured.is_independent is True


@pytest.mark.parametrize("bad", [-50.0, 0.0, float("nan"), float("inf")])
def test_an_illegal_scale_can_never_reach_site_ready(bad: float) -> None:
    """第五轮必修3 复现：负值/NaN/inf 的物理长度即使来源写成 scale_bar 也不能放行。"""
    plan = _ready_plan(execution_enabled=True, scale=ScaleEvidencePlan(
        source="scale_bar", reference_um=bad, reference_px=40.0,
        image_path="scale_bar.png", captured_with_same_optics=True))
    report = plan_report(plan)
    assert report["scale_um_per_px"] is None
    assert report["scale_is_independent"] is False
    assert report["site_ready_to_execute"] is False
    assert any("reference_um" in item for item in report["parameter_blockers"])


@pytest.mark.parametrize("bad", [-1.0, 0.0, float("nan"), float("inf")])
def test_illegal_depth_and_frame_rate_are_parameter_blockers(bad: float) -> None:
    depth_report = plan_report(_ready_plan(depth=DepthEvidencePlan(
        depth_um=bad, source="declared_chip_geometry")))
    assert depth_report["depth_usable_for_volume"] is False
    assert any("depth_um" in item for item in depth_report["parameter_blockers"])

    budget_report = plan_report(_ready_plan(budget=StorageBudget(
        capture_fps=bad, free_space_note="现场确认剩余空间充足")))
    assert budget_report["offline_replay_ready"] is False
    assert any("capture_fps" in item for item in budget_report["parameter_blockers"])


def test_unknown_depth_is_reported_as_axial_only() -> None:
    plan = _ready_plan(depth=DepthEvidencePlan())
    report = plan_report(plan)
    assert report["depth_usable_for_volume"] is False
    assert any("只输出轴向长度" in item for item in report["warnings"])


def test_duration_outside_the_validation_range_is_a_blocker() -> None:
    for duration in (30.0, 300.0):
        report = plan_report(_ready_plan(duration_s=duration))
        assert any("不在" in item for item in report["blockers"])


def test_prime_and_purge_must_be_marked_separately() -> None:
    report = plan_report(_ready_plan(prime_and_purge_marked_separately=False))
    assert any("排气/充液" in item for item in report["blockers"])


def test_device_readiness_gaps_are_listed_for_the_site() -> None:
    """未填的现场字段逐个列出并计入 blockers，不因布尔开关变成执行许可。"""
    report = plan_report(_ready_plan(readiness=DeviceReadiness()))
    pending = report["site_fields_pending"]
    for name in ("camera_unique_id", "pump_port", "oil_remaining", "water_remaining",
                 "available_travel_note", "stop_boundary_note"):
        assert name in pending, name
    assert "pump_connection_verified" in pending
    for name in pending:
        assert f"现场字段未填写：{name}" in report["blockers"]
    assert report["site_ready_to_execute"] is False


def test_manual_comparison_points_must_be_at_least_ten() -> None:
    report = plan_report(_ready_plan(manual_comparison_points=5))
    assert any("人工对照点" in item for item in report["blockers"])


def test_storage_budget_is_estimated_and_bounded() -> None:
    plan = _ready_plan()
    budget = plan_report(plan)["storage_budget"]
    assert budget["persisted_fps"] == 100.0
    assert budget["frames"] == 9000
    assert budget["bytes_per_frame"] == WIDTH * HEIGHT
    assert budget["within_capacity"] is True
    assert budget["estimated_gib"] > 1.0

    huge = _ready_plan(duration_s=120.0,
                       budget=StorageBudget(capacity_limit_bytes=1024 ** 3))
    assert plan_report(huge)["storage_budget"]["within_capacity"] is False
    assert any("容量上限" in item for item in plan_report(huge)["blockers"])


def test_frame_decimation_is_warned_about() -> None:
    plan = _ready_plan(budget=StorageBudget(persist_every_n=4,
                                            free_space_note="现场确认剩余空间充足"))
    report = plan_report(plan)
    assert report["storage_budget"]["persisted_fps"] == pytest.approx(25.0)
    assert any("抽帧" in item for item in report["warnings"])


def test_plan_serialises_with_execution_flag_and_report() -> None:
    payload = plan_to_dict(ShortCapturePlan())
    assert payload["execution_enabled"] is False
    assert payload["report"]["execution_enabled"] is False
    assert payload["report"]["not_a_physical_result"]
    assert payload["report"]["acceptance_criteria"]
    assert payload["report"]["uncancellable_source_acknowledged"] is False


def test_stop_chain_separates_enforced_from_declared_and_device_side() -> None:
    """停止边界按「谁负责」三分：声明过的意图不得混进运行中生效的判据。"""
    report = plan_report(_ready_plan())
    chain = report["stop_chain"]
    enforced = " ".join(chain["enforced_by_recorder"])
    declared = " ".join(chain["declared_but_not_enforced"])
    device_side = " ".join(chain["device_side_responsibility"])

    assert "帧时间戳达到计划" in enforced
    assert "no_frame_timeout_s" in enforced
    assert "max_bytes" in enforced
    assert "processing_budget_s" in enforced
    # 记录器不监测液量、也不按定位连续拒绝停止：只能留在「声明但未实现」里
    assert "定位连续拒绝" in declared
    assert "液量" in declared
    assert "液量" not in enforced
    assert "定位连续拒绝" not in enforced
    assert "设备" in device_side
    assert "既不测量也不据此停止" in report["volume_monitoring"]
    assert "不控制泵" in device_side or "不下发任何泵指令" in device_side


def test_a_plan_without_a_wall_clock_budget_is_warned_about() -> None:
    assert any("未设墙钟期限" in item for item in plan_report(_ready_plan())["warnings"])
    guarded = _ready_plan(guards=CaptureGuards(processing_budget_s=600.0))
    assert not any("未设墙钟期限" in item for item in plan_report(guarded)["warnings"])


# ---------------------------------------------------------------- 运行期守卫

class _StubVision:
    """只记录调用；定位链与测量链的行为由 offline_campaign 的测试覆盖。"""

    def __init__(self, *, um_per_px: float | None = 100.0 / 94.0,
                 measure_error: Exception | None = None) -> None:
        self.calls = 0
        self._um_per_px = um_per_px
        self._measure_error = measure_error

    def declared_chip_depth(self) -> dict:
        return {"depth_um": 50.0, "source": "declared_chip_geometry", "validated": False}

    def generation_measurement_scale(self) -> dict:
        return {"um_per_px": self._um_per_px, "source": "channel_width_reference",
                "validated": True}

    def localize_parallel_walls(self, frame, *, frame_id, capture_monotonic, region=None) -> dict:
        return {}

    def measure_generation_zone(self, frame, *, frame_id, hardware_frame_id,
                               capture_monotonic, time_source, duct_depth_um=None,
                               duct_depth_source=None) -> dict:
        self.calls += 1
        if self._measure_error is not None:
            raise self._measure_error
        return {"valid": False, "reason": "localization_pending_motion"}


class _PacketSource:
    """实现了可取消读取契约的来源：``read`` 立刻给帧，取完抛 ``StopIteration``。"""

    def __init__(self, packets: list[FramePacket]) -> None:
        self._packets = list(packets)
        self._index = 0

    def packets(self):
        return iter(self._packets)

    def read(self, timeout_s: float) -> FramePacket | None:
        if self._index >= len(self._packets):
            raise StopIteration
        packet = self._packets[self._index]
        self._index += 1
        return packet


class _BlockingSource:
    """只有阻塞迭代、没有有限等待读取契约的来源。"""

    def __init__(self, times, *, shape=SMALL) -> None:
        self._source = _packets(times, shape=shape)

    def packets(self):
        return self._source.packets()


class _StalledSource:
    """停滞来源：在有限等待内给不出帧，并真的等满请求的时间。"""

    def __init__(self, *, stall_s: float = 30.0) -> None:
        self._stall_s = stall_s

    def read(self, timeout_s: float) -> None:
        time.sleep(min(float(timeout_s), self._stall_s))
        return None


class _MicronVision(_StubVision):
    """返回 valid=true 且带 µm 的 vision：用来验证诊断模式的输出闸门。"""

    def measure_generation_zone(self, frame, **kwargs) -> dict:
        self.calls += 1
        return {"valid": True, "reason": "ok", "has_complete_plug": True,
                "plug_count": 1, "complete_plug_count": 1,
                "plug_lengths_px": [40.0], "plug_lengths_um": [69.0],
                "equivalent_diameters_px": [40.0], "equivalent_diameters_um": [69.0],
                "duct": {"width_px": 40.0, "width_um": 69.0, "depth_um": 50.0,
                         "depth_source": "declared_chip_geometry"},
                "scale": {"um_per_px": 1.725, "source": "channel_width_reference",
                          "validated": True}}


def _packets(times, *, shape=SMALL) -> _PacketSource:
    """按给定采集时刻造帧包；每帧填不同常量，便于核对原始像素是否逐个还原。"""
    packets = []
    for index, moment in enumerate(times):
        image = np.full(shape, 30 + index % 200, np.uint8)
        packets.append(FramePacket(image=image, frame_id=index + 1, hardware_frame_id=index + 1,
                                   capture_monotonic=float(moment),
                                   time_source="camera_frame_timestamp"))
    return _PacketSource(packets)


def _frames(count: int) -> _PacketSource:
    return _PacketSource([
        FramePacket(image=np.zeros((HEIGHT, WIDTH), np.uint8), frame_id=index + 1,
                    hardware_frame_id=index + 1, capture_monotonic=100.0 + index,
                    time_source="camera_frame_timestamp")
        for index in range(count)])


def _restore_frames(manifest_path: Path, binary_path: Path) -> list[np.ndarray]:
    """按 manifest 的 offset/nbytes 从 frames.bin 还原原始像素。"""
    with Path(manifest_path).open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    blob = Path(binary_path).read_bytes()
    restored = []
    for row in manifest:
        offset = int(row["offset_bytes"])
        nbytes = int(row["nbytes"])
        shape = (int(row["height"]), int(row["width"]))
        buffer = np.frombuffer(blob[offset:offset + nbytes],
                               dtype=np.dtype(row["dtype"])).reshape(shape)
        restored.append(buffer)
    return restored


def _run(plan, *, vision, source, tmp_path, name="run", writer_factory=RawFrameWriter):
    result = run_short_capture(plan, vision=vision, frame_source=source,
                               output_dir=tmp_path / name, writer_factory=writer_factory)
    assert result["status"] == "executed", result
    return result["summary"]


def test_raw_frames_are_persisted_losslessly_and_restorable(tmp_path) -> None:
    """每帧原始像素无损落盘，manifest 的 offset/nbytes 能逐帧还原。"""
    times = [100.0 + index * 0.5 for index in range(6)]
    summary = _run(_ready_plan(execution_enabled=True), vision=_StubVision(),
                   source=_packets(times), tmp_path=tmp_path)
    assert summary["frames_persisted"] == 6
    raw_dir = tmp_path / "run" / "raw_frames"
    restored = _restore_frames(raw_dir / "frames_manifest.csv", raw_dir / "frames.bin")
    assert len(restored) == 6
    assert sum(frame.size for frame in restored) * restored[0].itemsize == \
        summary["raw_bytes_persisted"]
    for index, frame in enumerate(restored):
        assert np.array_equal(frame, np.full(SMALL, 30 + index, np.uint8))


def test_measurement_failures_do_not_drop_captured_frames(tmp_path) -> None:
    """检测吞吐不足或逐帧报错都不得静默丢弃采集帧：原始帧先落盘。"""
    vision = _StubVision(measure_error=RuntimeError("检测器抛错"))
    summary = _run(_ready_plan(execution_enabled=True), vision=vision,
                   source=_packets([100.0 + index for index in range(5)]), tmp_path=tmp_path)
    assert summary["frames_persisted"] == 5
    assert summary["frames_measured"] == 5
    assert summary["processing_error_count"] == 5
    assert summary["stop_reason"] == ""
    rows = list(csv.DictReader((tmp_path / "run" / "diameter_series.csv")
                               .open(encoding="utf-8-sig", newline="")))
    assert len(rows) == 5, "测量失败也要逐帧留下记录"
    assert all(row["accepted"] == "False" for row in rows)


def test_a_slow_source_is_stopped_at_the_planned_duration(tmp_path) -> None:
    """低于计划 FPS 时按实际采集时间（帧时间戳）收尾，不会一直跑下去。"""
    times = [100.0 + index * 0.1 for index in range(2000)]
    summary = _run(_ready_plan(execution_enabled=True), vision=_StubVision(),
                   source=_packets(times), tmp_path=tmp_path)
    assert summary["status"] == "stopped_duration"
    assert summary["frames_persisted"] == 901          # 0.0–90.0 s，第 902 帧已超计划
    assert summary["capture_span_s"] == pytest.approx(90.0)


def test_a_long_gap_stops_the_capture(tmp_path) -> None:
    """帧间隔超过无帧超时即停止，迟到的帧不再计入采集。"""
    summary = _run(_ready_plan(execution_enabled=True), vision=_StubVision(),
                   source=_packets([100.0, 101.0, 102.0, 202.0]), tmp_path=tmp_path)
    assert summary["status"] == "stopped_gap_timeout"
    assert summary["frames_persisted"] == 3
    assert "100.000" in summary["stop_reason"]


def test_frame_and_capacity_limits_stop_the_capture(tmp_path) -> None:
    plan = _ready_plan(execution_enabled=True, guards=CaptureGuards(max_frames=3))
    summary = _run(plan, vision=_StubVision(),
                   source=_packets([100.0 + index for index in range(10)]),
                   tmp_path=tmp_path, name="frames")
    assert summary["status"] == "stopped_frame_limit"
    assert summary["frames_persisted"] == 3

    per_frame = SMALL[0] * SMALL[1]
    plan = _ready_plan(execution_enabled=True,
                       guards=CaptureGuards(max_bytes=per_frame * 2))
    summary = _run(plan, vision=_StubVision(),
                   source=_packets([100.0 + index for index in range(10)]),
                   tmp_path=tmp_path, name="capacity")
    assert summary["status"] == "stopped_capacity"
    assert summary["frames_persisted"] == 2
    assert summary["raw_bytes_persisted"] == per_frame * 2


def test_a_storage_write_failure_keeps_the_frames_already_written(tmp_path) -> None:
    """写盘异常要收尾并保留已写帧，不能把已经采到的原始帧丢掉。"""

    class _FailingWriter:
        def __init__(self, directory, *, fail_after: int) -> None:
            self._inner = RawFrameWriter(directory)
            self._fail_after = fail_after

        @property
        def frames_written(self) -> int:
            return self._inner.frames_written

        @property
        def bytes_written(self) -> int:
            return self._inner.bytes_written

        @property
        def manifest_path(self):
            return self._inner.manifest_path

        @property
        def binary_path(self):
            return self._inner.binary_path

        def write(self, packet):
            if self._inner.frames_written >= self._fail_after:
                raise OSError("磁盘写入失败")
            return self._inner.write(packet)

        def close(self) -> None:
            self._inner.close()

    summary = _run(_ready_plan(execution_enabled=True), vision=_StubVision(),
                   source=_packets([100.0 + index for index in range(6)]),
                   tmp_path=tmp_path, name="failing",
                   writer_factory=lambda directory: _FailingWriter(directory, fail_after=2))
    assert summary["status"] == "stopped_storage_error"
    assert summary["frames_persisted"] == 2
    assert "磁盘写入失败" in summary["stop_reason"]

    raw_dir = tmp_path / "failing" / "raw_frames"
    assert len(_restore_frames(raw_dir / "frames_manifest.csv",
                               raw_dir / "frames.bin")) == 2, "异常前写出的帧必须仍可还原"


def test_a_plan_scale_that_does_not_match_vision_is_refused(tmp_path) -> None:
    """计划标尺必须与实际测量标尺一致，否则不输出 µm。"""
    result = run_short_capture(_ready_plan(execution_enabled=True),
                               vision=_StubVision(um_per_px=2.0),
                               frame_source=_packets([100.0, 101.0]),
                               output_dir=tmp_path / "mismatch")
    assert result["status"] == "refused"
    assert "不一致" in result["reason"]


def test_missing_scale_evidence_refuses_instead_of_claiming_microns(tmp_path) -> None:
    result = run_short_capture(_ready_plan(execution_enabled=True),
                               vision=_StubVision(um_per_px=None),
                               frame_source=_packets([100.0, 101.0]),
                               output_dir=tmp_path / "no_scale")
    assert result["status"] == "refused"
    assert "标尺证据" in result["reason"]


def test_pixel_diagnostic_mode_runs_without_physical_scale_evidence(tmp_path) -> None:
    """像素诊断模式允许没有物理标尺证据，但明确不认证物理尺寸。"""
    plan = _ready_plan(execution_enabled=True, pixel_diagnostic_only=True)
    summary = _run(plan, vision=_StubVision(um_per_px=None),
                   source=_packets([100.0 + index for index in range(3)]),
                   tmp_path=tmp_path, name="diagnostic")
    assert summary["scale_binding"]["used_for_microns"] is False
    assert summary["scale_binding"]["mode"] == "pixel_diagnostic_only"
    assert summary["frames_persisted"] == 3
    assert summary["depth_um"] == 50.0
    assert summary["measurement_mode"] == "pixel_diagnostic_only"
    assert summary["physical_sizes_withheld"] == ["equivalent_diameters_um",
                                                  "plug_lengths_um", "duct_width_um"]


def test_pixel_diagnostic_mode_withholds_the_microns_a_vision_would_emit(tmp_path) -> None:
    """诊断模式的闸门：vision 返回 valid=true 且带 µm 时，记录里也不能出现物理尺寸。"""
    plan = _ready_plan(execution_enabled=True, pixel_diagnostic_only=True)
    summary = _run(plan, vision=_MicronVision(um_per_px=None),
                   source=_packets([100.0 + index for index in range(3)]),
                   tmp_path=tmp_path, name="gate")
    assert summary["frames_persisted"] == 3
    path = tmp_path / "gate" / "diameter_series.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for column in ("equivalent_diameters_um", "plug_lengths_um", "duct_width_um"):
        assert [row[column] for row in rows] == ["", "", ""], column
    assert rows[0]["scale_source"] == "pixel_diagnostic_only"
    assert rows[0]["scale_um_per_px"] == ""
    assert rows[0]["equivalent_diameters_px"]          # 像素量保留，检测链仍可审
    assert rows[0]["duct_depth_um"] == "50.0"          # 深度是声明的结构尺寸，不来自标尺
    # 稳态分析拿不到任何物理尺寸：即使 accepted，也不能产出尺寸序列
    steady = analyse_steady(path)["phases"][plan.label]
    assert steady["accepted_frames"] == 3
    assert steady.get("samples", 0) == 0


def test_the_same_vision_without_diagnostic_mode_does_write_microns(tmp_path) -> None:
    """对照：同一 vision 在标尺绑定模式下确实会写入 µm，说明闸门才是拦截点。"""
    _run(_ready_plan(execution_enabled=True), vision=_MicronVision(),
         source=_packets([100.0, 101.0]), tmp_path=tmp_path, name="bound")
    path = tmp_path / "bound" / "diameter_series.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["equivalent_diameters_um"] == "69.000"
    assert rows[0]["scale_source"] == "channel_width_reference"


# ---------------------------------------------------------------- 阻塞与墙钟

def test_a_blocking_source_is_refused_unless_the_plan_acknowledges_it(tmp_path) -> None:
    """没有可取消读取契约时拒绝执行：卡死无法收尾，不能假装有停止链。"""
    result = run_short_capture(_ready_plan(execution_enabled=True), vision=_StubVision(),
                               frame_source=_BlockingSource([100.0, 101.0]),
                               output_dir=tmp_path / "blocking")
    assert result["status"] == "refused"
    assert "无法自行收尾" in result["reason"]
    assert not (tmp_path / "blocking").exists(), "拒绝时不得建输出目录或写任何帧"


def test_an_acknowledged_blocking_source_runs_with_the_risk_recorded(tmp_path) -> None:
    plan = _ready_plan(execution_enabled=True, uncancellable_source_acknowledged=True)
    report = plan_report(plan)
    assert report["source_contract"]["acknowledged_uncancellable"] is True
    assert any("不可取消" in item for item in report["warnings"])
    assert any("卡死" in item for item in report["stop_chain"]["declared_but_not_enforced"])

    summary = _run(plan, vision=_StubVision(),
                   source=_BlockingSource([100.0 + index for index in range(3)]),
                   tmp_path=tmp_path, name="acknowledged")
    assert summary["frames_persisted"] == 3
    assert summary["source_contract"] == {"cancellable_read": False,
                                         "uncancellable_acknowledged": True,
                                         "self_stop_possible_on_stalled_source": False}


def test_a_source_that_never_delivers_a_frame_stops_on_the_read_timeout(tmp_path) -> None:
    plan = _ready_plan(execution_enabled=True, guards=CaptureGuards(no_frame_timeout_s=0.05))
    summary = _run(plan, vision=_StubVision(), source=_StalledSource(),
                   tmp_path=tmp_path, name="stalled_read")
    assert summary["status"] == "stopped_no_frame_timeout"
    assert summary["frames_persisted"] == 0
    assert "0.050" in summary["stop_reason"]


def test_a_stalled_source_is_stopped_by_the_wall_clock_guard(tmp_path) -> None:
    """墙钟期限是独立守卫：停滞来源靠它收尾，且它不充当采集时间。"""
    plan = _ready_plan(execution_enabled=True,
                       guards=CaptureGuards(no_frame_timeout_s=30.0, processing_budget_s=0.05))
    summary = _run(plan, vision=_StubVision(), source=_StalledSource(),
                   tmp_path=tmp_path, name="stalled_clock")
    assert summary["status"] == "stopped_wall_clock_guard"
    assert summary["frames_persisted"] == 0
    assert summary["guard_clock"]["wall_clock_stop_used"] is True
    assert summary["guard_clock"]["not_a_measurement_time"]
    assert summary["guard_clock"]["processing_budget_s"] == 0.05
    assert summary["capture_span_s"] is None, "没有帧就没有采集时间，不能被墙钟顶替"


def test_hardware_frame_id_gaps_are_recorded(tmp_path) -> None:
    """上游丢帧必须如实记录：硬件帧号不连续就是缺口，不能推断无缝采集。"""
    packets = [
        FramePacket(image=np.zeros(SMALL, np.uint8), frame_id=10, hardware_frame_id=10,
                    capture_monotonic=100.0, time_source="camera_frame_timestamp"),
        FramePacket(image=np.zeros(SMALL, np.uint8), frame_id=13, hardware_frame_id=13,
                    capture_monotonic=101.0, time_source="camera_frame_timestamp"),
        FramePacket(image=np.zeros(SMALL, np.uint8), frame_id=14, hardware_frame_id=14,
                    capture_monotonic=102.0, time_source="camera_frame_timestamp"),
    ]
    summary = _run(_ready_plan(execution_enabled=True), vision=_StubVision(),
                   source=_PacketSource(packets), tmp_path=tmp_path, name="gaps")
    assert summary["frames_received"] == 3
    assert summary["frames_persisted"] == 3
    gaps = summary["hardware_frame_id_gaps"]
    assert gaps["count"] == 1
    assert gaps["first"][0]["missing"] == 2
    assert gaps["first"][0]["after_hardware_frame_id"] == 10
    assert "丢帧" in summary["backpressure"] or "取帧" in summary["backpressure"]
