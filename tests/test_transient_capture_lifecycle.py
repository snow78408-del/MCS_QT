"""实机采集会话的设备生命周期回归测试（全部用 mock，不连接任何设备）。

对应复查任务书 §2 与 §6 的场景 ②③④⑤⑥⑦⑪。最重要的一条是
``test_pump_may_be_running_is_set_before_the_start_command``：旧实现把 ``started`` 标记
放在启动回读**之后**才置位，于是「指令已发出但回读失败」时标记仍为假，清理路径会跳过停泵，
把一台可能正在转的泵留在原地。这里钉死修复后的不变量。
"""
from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import plant_flow_transient_capture as tool  # noqa: E402
from backend.device_lock import DeviceLockError  # noqa: E402
from test_transient_capture_entrypoints import valid_plan  # noqa: E402


# ---------------------------------------------------------------- 假件

class FakeResult:
    def __init__(self, ok: bool = True, reason: str | None = None, error: str | None = None):
        self.ok = ok
        self.reason = reason
        self.error = error


class FakeState:
    def __init__(self, established: bool = True, failed: str = ""):
        self.comm_established = established
        self.failed = failed


class FakePump:
    def __init__(self, order: list) -> None:
        self._order = order
        self.calls: list = []
        self.state_established = True
        self.write_ok = {1: True, 2: True}
        self.start_ok = True
        self.start_raises: BaseException | None = None
        self.stop_plan: list = [True]
        self.stop_raises: list = []
        self.current_q_state_raises: BaseException | None = None

    def _log(self, name: str) -> None:
        self.calls.append(name)
        self._order.append(name)

    def connect_and_probe(self):
        self._log("connect_and_probe")
        return FakeState(self.state_established, "" if self.state_established else "no reply")

    def channel_params_for_flow(self, channel, flow):
        self._log(f"channel_params_for_flow:{channel}")
        return {"channel": channel, "flow": flow}

    def flow_from_channel_params(self, params):
        return float(params["flow"])

    def write_wsp_and_verify(self, channel, params):
        self._log(f"write_wsp_and_verify:{channel}")
        ok = self.write_ok.get(channel, True)
        return FakeResult(ok, reason=None if ok else "readback mismatch")

    def start_infusion_and_verify(self, channels):
        self._log("start_infusion_and_verify")
        if self.start_raises is not None:
            raise self.start_raises
        return FakeResult(self.start_ok, reason=None if self.start_ok else "no start ack")

    def stop_system_and_verify(self):
        self._log("stop_system_and_verify")
        if self.stop_raises:
            raise self.stop_raises.pop(0)
        ok = self.stop_plan.pop(0) if len(self.stop_plan) > 1 else self.stop_plan[0]
        return FakeResult(ok, reason=None if ok else "stop readback unconfirmed")

    def get_current_q_state(self):
        self._log("get_current_q_state")
        if self.current_q_state_raises is not None:
            raise self.current_q_state_raises
        return (17.5, 5.0)

    def disconnect(self):
        self._log("disconnect")


@dataclass
class FakeDevice:
    unique_id: str = "HIKROBOT:DIRECT:0"
    source_backend: str = "hikrobot"
    available: bool = True


class FakeFrame:
    def __init__(self, valid: bool = True, frame_id: int = 0) -> None:
        self.valid = valid
        self.frame_id = frame_id


class FakeCamera:
    def __init__(self, order: list) -> None:
        self._order = order
        self.calls: list = []
        self.features = {"exposure": 80.0, "frame_rate": 320.0}
        self.readback_override: dict = {}
        self.discover_returns = [FakeDevice()]
        self.frames: list = []
        self.read_raises: BaseException | None = None

    def _log(self, name: str) -> None:
        self.calls.append(name)
        self._order.append(name)

    def discover_devices(self):
        self._log("discover_devices")
        return list(self.discover_returns)

    def open(self, device):
        self._log("open")

    def close(self):
        self._log("close")

    def start_stream(self):
        self._log("start_stream")

    def stop_stream(self):
        self._log("stop_stream")

    def set_feature(self, name, value):
        self._log(f"set_feature:{name}")
        self.features[name] = value

    def get_feature(self, name):
        self._log(f"get_feature:{name}")
        if name in self.readback_override:
            return self.readback_override[name]
        return self.features.get(name)

    def read_frame(self, timeout_ms):
        self._log("read_frame")
        if self.read_raises is not None:
            raise self.read_raises
        if self.frames:
            return self.frames.pop(0)
        return FakeFrame(valid=False)


class RecordingLock:
    """记录获取/释放顺序；release 只在确实持锁时计数，与真实 DeviceLock 语义一致。"""

    def __init__(self, order: list, busy: bool = False, key: str = "test-device") -> None:
        self._order = order
        self.busy = busy
        self.key = key
        self.acquired = 0
        self.released = 0
        self._held = False

    def acquire(self):
        if self.busy:
            raise DeviceLockError("设备锁已被其他进程占用")
        self.acquired += 1
        self._held = True
        self._order.append("lock.acquire")

    def release(self):
        if not self._held:
            return
        self._held = False
        self.released += 1
        self._order.append("lock.release")


class FakeClock:
    """每次读取都前进固定步长，让有界循环可预期地结束。"""

    def __init__(self, step: float = 0.1) -> None:
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


class RecordingSink(tool.BoundedFrameSink):
    def __init__(self, max_backlog: int, order: list) -> None:
        super().__init__(max_backlog=max_backlog)
        self._order = order

    def submit(self, frame) -> None:
        super().submit(frame)
        self._order.append("sink.submit")


class FailingSink(tool.BoundedFrameSink):
    """模拟写盘失败／跟不上。"""

    def submit(self, frame) -> None:
        raise OSError("磁盘写满")


# ---------------------------------------------------------------- 组装

def pump_commands(pump: FakePump) -> list:
    """会改变泵状态或让泵转起来的命令。

    ``disconnect`` 不在其列：清理时无条件断开是刻意的（半连接状态下必须释放串口），
    对尚未连接的泵是空操作，不会改变设备状态。
    """
    return [call for call in pump.calls
            if call.startswith(("write_wsp_and_verify", "start_infusion_and_verify",
                                "channel_params_for_flow"))]


def make_plan(**overrides) -> dict:
    plan = copy.deepcopy(valid_plan())
    plan["session_id"] = "session-under-test"
    for section, values in overrides.items():
        plan[section].update(values)
    return plan


def build(plan: dict, tmp_path: Path, *, lock_busy: bool = False, clock_step: float = 0.1,
          sink=None, device_free: bool = True):
    """构造一个测试会话。

    ``device_free=True``（默认）经 ``for_device_free_tests()`` 显式声明「本轮不做追溯」——
    这是无设备单元测试的专用接口，真机入口没有对应开关。
    ``device_free=False`` 走真实构造路径（不带追溯规格），用于验证现场入口的拒绝行为。
    """
    order: list = []
    pump = FakePump(order)
    camera = FakeCamera(order)
    lock = RecordingLock(order, busy=lock_busy)
    actual_sink = sink if sink is not None else RecordingSink(
        max_backlog=int(plan["capture_plan"]["max_backlog_frames"]), order=order)
    factory = (tool.LiveCaptureSession.for_device_free_tests if device_free
               else tool.LiveCaptureSession)
    session = factory(
        plan=plan, pump=pump, camera=camera, locks=[lock], sink=actual_sink,
        output_dir=tmp_path / "out", log=lambda _m: None, clock=FakeClock(clock_step),
    )
    return session, pump, camera, lock, actual_sink, order


# ---------------------------------------------------------------- ② 设备被占用

def test_busy_device_lock_prevents_any_pump_command(tmp_path: Path) -> None:
    session, pump, _camera, lock, _sink, _order = build(make_plan(), tmp_path, lock_busy=True)
    report = session.run()

    assert report["verdict"] == "FAILED"
    assert pump.calls == ["disconnect"], f"设备被占用时不得连接或驱动泵，实际：{pump.calls}"
    assert pump_commands(pump) == []
    assert lock.released == 0, "没拿到锁就不应释放"


# ---------------------------------------------------------------- ③ 启动回读失败

def test_pump_may_be_running_is_set_before_the_start_command(tmp_path: Path) -> None:
    """核心回归：标记必须在启动指令**之前**置位。

    旧实现把 started 放在回读成功之后，于是「已发出但回读失败」时清理会跳过停泵。
    """
    session, pump, _camera, _lock, _sink, order = build(make_plan(), tmp_path)
    pump.start_ok = False
    report = session.run()

    assert pump.calls.count("start_infusion_and_verify") == 1
    assert "stop_system_and_verify" in pump.calls, "启动回读失败后必须尝试停泵"
    assert report["verdict"] == "FAILED"
    assert report["stop"]["verified"] is True
    assert report["stop"]["verdict"] == "STOPPED"
    assert report["completed"] is False
    # 停泵发生在锁仍被持有的窗口内
    assert order.index("lock.acquire") < order.index("stop_system_and_verify")
    assert order.index("stop_system_and_verify") < order.index("lock.release")


def test_start_command_raising_still_stops_the_pump(tmp_path: Path) -> None:
    session, pump, _camera, _lock, _sink, _order = build(make_plan(), tmp_path)
    pump.start_raises = TimeoutError("no start ack")
    report = session.run()

    assert "stop_system_and_verify" in pump.calls
    assert report["failure_kind"] == "TimeoutError"
    assert report["stop"]["verified"] is True


# ---------------------------------------------------------------- ④ 第二路写入失败

def test_second_channel_write_failure_cleans_up_without_claiming_success(tmp_path: Path) -> None:
    session, pump, camera, lock, _sink, _order = build(make_plan(), tmp_path)
    pump.write_ok = {1: True, 2: False}
    report = session.run()

    assert report["verdict"] == "FAILED"
    assert report["failure_kind"] == "ChannelWriteError"
    assert report["completed"] is False
    # 泵从未启动，因此不需要停泵；但相机、泵连接与锁都必须收干净。
    assert "stop_system_and_verify" not in pump.calls
    assert "disconnect" in pump.calls
    assert "stop_stream" in camera.calls and "close" in camera.calls
    assert lock.released == 1


# ---------------------------------------------------------------- ⑤ 相机 / 无帧 / 写盘

def test_camera_feature_readback_mismatch_aborts_before_touching_the_pump(tmp_path: Path) -> None:
    session, pump, camera, lock, _sink, _order = build(make_plan(), tmp_path)
    camera.readback_override = {"exposure": 999.0}
    report = session.run()

    assert report["verdict"] == "FAILED"
    assert report["failure_kind"] == "CameraSetupError"
    # 泵的连接按任务书顺序发生在相机配置之前，但绝不允许写参数或启动。
    assert pump_commands(pump) == [], "相机设置未确认时不得写泵参数或启动"
    assert "close" in camera.calls
    assert lock.released == 1


def test_camera_feature_without_readback_is_refused(tmp_path: Path) -> None:
    session, _pump, camera, _lock, _sink, _order = build(make_plan(), tmp_path)
    camera.readback_override = {"frame_rate": None}
    report = session.run()

    assert report["failure_kind"] == "CameraSetupError"
    assert "frame_rate" in report["original_error"]


@pytest.mark.parametrize("readback", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_camera_readback_is_rejected(tmp_path: Path, readback: float) -> None:
    """NaN 必须先于容差比较被拒绝。

    ``abs(nan - target) > tolerance`` 恒为 False，只比较差值会让 NaN 静默「通过」设置检查。
    """
    session, _pump, camera, lock, _sink, _order = build(make_plan(), tmp_path)
    camera.readback_override = {"exposure": readback}
    report = session.run()

    assert report["verdict"] == "FAILED"
    assert report["failure_kind"] == "CameraSetupError"
    assert "不是有限数" in report["original_error"]
    assert lock.released == 1


@pytest.mark.parametrize("target", [float("nan"), float("inf")])
def test_non_finite_camera_target_is_never_sent(tmp_path: Path, target: float) -> None:
    """设定值本身非有限时不得下发，也不得写泵参数。"""
    plan = make_plan(capture_plan={"requested_exposure_us": target})
    session, pump, _camera, lock, _sink, _order = build(plan, tmp_path)
    report = session.run()

    assert report["failure_kind"] == "CameraSetupError"
    assert "设定值不是有限数" in report["original_error"]
    assert pump_commands(pump) == []
    assert lock.released == 1


def test_no_valid_frame_times_out_bounded_and_stops_the_pump(tmp_path: Path) -> None:
    plan = make_plan(capture_plan={"max_no_valid_frame_s": 0.5},
                     termination={"max_session_s": 60.0})
    session, pump, _camera, _lock, _sink, _order = build(plan, tmp_path)
    report = session.run()

    assert report["failure_kind"] == "NoValidFrameError"
    assert report["frames_accepted"] == 0
    assert "stop_system_and_verify" in pump.calls
    assert report["stop"]["verified"] is True


def test_storage_backlog_failure_stops_the_run(tmp_path: Path) -> None:
    """写盘跟不上必须标记失败并终止受影响的试验，而不是写完警告继续。"""
    plan = make_plan(termination={"max_session_s": 60.0})
    session, pump, _camera, _lock, _sink, _order = build(plan, tmp_path, sink=FailingSink(max_backlog=8))
    session.camera.frames = [FakeFrame() for _ in range(5)]
    report = session.run()

    assert report["verdict"] == "FAILED"
    assert "磁盘写满" in report["original_error"]
    assert "stop_system_and_verify" in pump.calls


def test_baseline_write_failure_aborts_before_starting_the_pump(tmp_path: Path) -> None:
    blocker = tmp_path / "out"
    blocker.write_text("not a directory", encoding="utf-8")
    session, pump, camera, lock, _sink, _order = build(make_plan(), tmp_path)
    session.output_dir = blocker  # 基线落盘失败；由于同一路径，最终摘要也写不下
    report = session.run()

    # 落盘失败必须把结论降级为失败，而不是保留「已完成」
    assert report["verdict"] == "OUTPUT_WRITE_FAILED"
    assert report["completed"] is False
    assert report["failure_record_saved"] is False
    assert "失败记录也未能保存" in report["failure_record_error"]
    assert "stop_system_and_verify" not in pump.calls, "基线都写不下时不该已经启动泵"
    assert lock.released == 1
    assert "close" in camera.calls


def test_output_write_failure_downgrades_a_completed_session(tmp_path: Path,
                                                            monkeypatch) -> None:
    """正常跑完但摘要落盘失败：结论必须降级，不能仍报 CAPTURE_COMPLETE。"""
    plan = make_plan(termination={"max_session_s": 1.0},
                     capture_plan={"max_no_valid_frame_s": 5.0})
    session, _pump, _camera, _lock, _sink, _order = build(plan, tmp_path)
    session.camera.frames = [FakeFrame() for _ in range(40)]
    original = session._write_json

    def failing_on_summary(name: str, payload: dict) -> None:
        if name == "session_summary.json":
            raise OSError("磁盘写满")
        return original(name, payload)

    monkeypatch.setattr(session, "_write_json", failing_on_summary)
    report = session.run()

    assert report["verdict"] == "OUTPUT_WRITE_FAILED"
    assert report["completed"] is False
    assert "磁盘写满" in report["output_write_error"]
    assert report["failure_record_saved"] is False
    # 停泵仍然执行且已验证——降级只影响结论与记录，不影响安全路径
    assert report["stop"]["verified"] is True


def test_downgraded_failure_record_is_what_gets_saved(tmp_path: Path,
                                                      monkeypatch) -> None:
    """第一次落盘失败后重试一次：保存的必须是**降级后**的失败记录。"""
    plan = make_plan(termination={"max_session_s": 1.0},
                     capture_plan={"max_no_valid_frame_s": 5.0})
    session, _pump, _camera, _lock, _sink, _order = build(plan, tmp_path)
    session.camera.frames = [FakeFrame() for _ in range(40)]
    original = session._write_json
    attempts = {"count": 0}

    def flaky(name: str, payload: dict) -> None:
        if name == "session_summary.json":
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise OSError("第一次写盘失败")
        return original(name, payload)

    monkeypatch.setattr(session, "_write_json", flaky)
    report = session.run()

    assert report["failure_record_saved"] is True
    stored = json.loads((tmp_path / "out" / "session_summary.json").read_text(encoding="utf-8"))
    assert stored["verdict"] == "OUTPUT_WRITE_FAILED"
    assert stored["completed"] is False
    assert "第一次写盘失败" in stored["output_write_error"]


# ---------------------------------------------------------------- ⑥ 取消 / Ctrl-C

@pytest.mark.parametrize("exc", [KeyboardInterrupt(), RuntimeError("operator cancelled")],
                         ids=["keyboard_interrupt", "cancelled"])
def test_interrupt_enters_cleanup_and_preserves_the_original_reason(
    tmp_path: Path, exc: BaseException
) -> None:
    plan = make_plan(termination={"max_session_s": 60.0})
    session, pump, camera, lock, _sink, _order = build(plan, tmp_path)
    session.camera.frames = [FakeFrame()]
    session.camera.read_raises = exc
    report = session.run()

    assert report["verdict"] == "FAILED"
    assert report["failure_kind"] == type(exc).__name__
    assert report["stop"]["verified"] is True, "中断也必须走停泵验证"
    assert "disconnect" in pump.calls
    assert "close" in camera.calls
    assert lock.released == 1


# ---------------------------------------------------------------- ⑦ 停机失败

def test_stop_failure_gives_stop_unverified_after_bounded_retries(tmp_path: Path) -> None:
    plan = make_plan(termination={"max_session_s": 60.0, "max_stop_retries": 2},
                     capture_plan={"max_no_valid_frame_s": 0.5})
    session, pump, _camera, _lock, _sink, _order = build(plan, tmp_path)
    pump.stop_plan = [False, False, False]
    report = session.run()

    assert pump.calls.count("stop_system_and_verify") == 3, "重试次数 = max_stop_retries + 1"
    assert report["stop"]["verdict"] == "STOP_UNVERIFIED"
    assert report["stop"]["verified"] is False
    assert report["verdict"] == "STOP_UNVERIFIED"
    assert report["completed"] is False
    assert report["requires_onsite_confirmation"] is True
    assert "禁止继续实验或自动重启" in report["onsite_instruction"]
    # 不得自动重启：全程只发出一次启动指令，且停机之后没有任何泵命令
    assert pump.calls.count("start_infusion_and_verify") == 1
    assert pump.calls[-1] in {"stop_system_and_verify", "disconnect"}


def test_stop_recovers_on_a_retry(tmp_path: Path) -> None:
    plan = make_plan(termination={"max_session_s": 60.0, "max_stop_retries": 3},
                     capture_plan={"max_no_valid_frame_s": 0.5})
    session, pump, _camera, _lock, _sink, _order = build(plan, tmp_path)
    pump.stop_plan = [False, False, True]
    report = session.run()

    assert pump.calls.count("stop_system_and_verify") == 3
    assert report["stop"]["verified"] is True
    assert report["stop"]["attempts"] == 3


def test_cleanup_exception_does_not_overwrite_the_original_error(tmp_path: Path) -> None:
    """原始异常与清理异常分别保存，不能相互覆盖。"""
    plan = make_plan(termination={"max_session_s": 60.0},
                     capture_plan={"max_no_valid_frame_s": 0.5})
    session, pump, _camera, _lock, _sink, _order = build(plan, tmp_path)
    session.camera.read_raises = ValueError("frame pipeline broke")
    pump.stop_plan = [False, False, False, False]
    report = session.run()

    assert report["failure_kind"] == "ValueError"
    assert "frame pipeline broke" in report["original_error"]
    assert report["stop"]["verdict"] == "STOP_UNVERIFIED"
    assert report["completed"] is False


# ---------------------------------------------------------------- ⑪ 正常完整运行

def test_successful_run_follows_the_required_order_and_reports_completion(tmp_path: Path) -> None:
    plan = make_plan(termination={"max_session_s": 1.0},
                     capture_plan={"max_no_valid_frame_s": 5.0})
    session, pump, camera, lock, sink, _order = build(plan, tmp_path)
    session.camera.frames = [FakeFrame(frame_id=i) for i in range(40)]
    report = session.run()

    assert report["verdict"] == "CAPTURE_COMPLETE"
    assert report["completed"] is True
    assert report["stop"]["verified"] is True
    assert report["requires_onsite_confirmation"] is False
    assert report["cleanup_errors"] == []
    assert report["original_error"] is None
    assert sink.accepted == report["frames_accepted"] > 0

    assert report["events"] == [
        # 无设备测试接口显式跳过追溯，事件序列里如实记一笔（真机入口没有这条路径）。
        "provenance_skipped_device_free_test_interface",
        "device_lock_acquired",
        "pump_connected",
        "camera_open",
        "camera_streaming",
        "camera_exposure_verified",
        "camera_frame_rate_verified",
        "baseline_saved",
        "ch1_written_verified",
        "ch2_written_verified",
        "pump_start_verified",
        "bounded_run_finished",
        "pump_stopped_verified_attempt_1",
        "camera_closed",
        "pump_disconnected",
        "device_lock_released",
    ]
    # 指令前基线必须真的落盘，且早于任何泵写入
    baseline = json.loads((tmp_path / "out" / "pre_command_baseline.json").read_text(encoding="utf-8"))
    assert baseline["channels"] == {"available": True, "q1": 17.5, "q2": 5.0}
    assert pump.calls.index("get_current_q_state") < pump.calls.index("write_wsp_and_verify:1")


def test_baseline_is_recorded_as_unavailable_with_reason_when_unreadable(tmp_path: Path) -> None:
    plan = make_plan(termination={"max_session_s": 1.0},
                     capture_plan={"max_no_valid_frame_s": 5.0})
    session, pump, _camera, _lock, _sink, _order = build(plan, tmp_path)
    pump.current_q_state_raises = RuntimeError("pump reported no state")
    session.camera.frames = [FakeFrame() for _ in range(40)]
    report = session.run()

    assert report["verdict"] == "FAILED"
    assert report["failure_kind"] == "ChannelWriteError"
    assert report["baseline"]["channels"]["available"] is False
    assert "pump reported no state" in report["baseline"]["channels"]["reason"]


def test_lifecycle_errors_are_all_capture_lifecycle_errors() -> None:
    """便于调用方用一个基类捕获整族失败。"""
    for name in ("ChannelWriteError", "PumpStartUnverifiedError", "CameraSetupError",
                 "NoValidFrameError", "StorageBacklogError"):
        assert issubclass(getattr(tool, name), tool.CaptureLifecycleError)
