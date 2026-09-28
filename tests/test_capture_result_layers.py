"""入口退出码与三层结果：运行完成 / 测量验收 / 停止状态必须分别表达。

修复前两个入口都写 ``return 0 if result.get("stop", {}).get("verified") else 3``：

* ``PREMISE_REJECTED`` + 已停泵 → 退出码 ``0``，与「成功」不可区分；
* ``stop`` 为 ``None``（启动前就失败的尝试）→ ``AttributeError``，把「没有停泵证据」
  变成崩溃，掩盖真实失败原因。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import plant_flow_transient_capture as tool  # noqa: E402
from test_transient_capture_lifecycle import build, make_plan  # noqa: E402


def _layers(*, run_completed: bool, measurement_accepted: bool, stop_state: str,
            task_goal: str) -> dict:
    verified = stop_state == "STOPPED"
    if task_goal == tool.TASK_GOAL_DIAGNOSTIC_CAPTURE:
        met = run_completed and verified
    else:
        met = run_completed and verified and measurement_accepted
    return {"run_completed": run_completed, "measurement_accepted": measurement_accepted,
            "sampling_premise_ok": measurement_accepted,
            "stop": {"state": stop_state, "verified": verified},
            "task_goal": task_goal, "task_goal_met": met}


CAMPAIGN = tool.TASK_GOAL_CAMPAIGN
DIAGNOSTIC = tool.TASK_GOAL_DIAGNOSTIC_CAPTURE


# ------------------------------------------------------------------ 五种结局

def test_successful_campaign_is_exit_zero() -> None:
    report = {"verdict": "CAPTURE_COMPLETE", "completed": True,
              "result_layers": _layers(run_completed=True, measurement_accepted=True,
                                       stop_state="STOPPED", task_goal=CAMPAIGN)}
    assert tool.exit_code(report) == tool.EXIT_OK


def test_premise_rejected_with_verified_stop_is_not_exit_zero_for_a_campaign() -> None:
    """测量未验收的试验不得因「停泵成功」而返回 0。"""
    report = {"verdict": "PREMISE_REJECTED", "completed": False,
              "result_layers": _layers(run_completed=True, measurement_accepted=False,
                                       stop_state="STOPPED", task_goal=CAMPAIGN)}
    assert tool.exit_code(report) == tool.EXIT_TASK_GOAL_UNMET


def test_diagnostic_capture_completes_without_measurement_acceptance() -> None:
    """诊断入口声明的是「采集」，测量未验收仍然算目标达成——但必须分别报告。"""
    layers = _layers(run_completed=True, measurement_accepted=False,
                     stop_state="STOPPED", task_goal=DIAGNOSTIC)
    assert tool.exit_code({"result_layers": layers}) == tool.EXIT_OK
    assert layers["measurement_accepted"] is False
    assert layers["sampling_premise_ok"] is False


def test_stop_unverified_takes_priority_over_everything_else() -> None:
    for goal in (CAMPAIGN, DIAGNOSTIC):
        for completed in (True, False):
            report = {"result_layers": _layers(run_completed=completed,
                                                measurement_accepted=completed,
                                                stop_state="STOP_UNVERIFIED",
                                                task_goal=goal)}
            assert tool.exit_code(report) == tool.EXIT_STOP_UNVERIFIED


def test_failed_with_stop_none_does_not_raise_and_is_not_exit_zero() -> None:
    """修复前这里是 AttributeError。"""
    report = {"verdict": "FAILED", "completed": False, "stop": None,
              "result_layers": _layers(run_completed=False, measurement_accepted=False,
                                       stop_state="NOT_ATTEMPTED", task_goal=CAMPAIGN)}
    assert tool.exit_code(report) == tool.EXIT_TASK_GOAL_UNMET
    assert tool.exit_code({"verdict": "FAILED", "stop": None}) != tool.EXIT_OK


def test_preflight_block_before_touching_the_device_is_reported_separately() -> None:
    report = {"verdict": "LIVE_BLOCKED", "touched_hardware": False, "unmet": ["x"]}
    assert tool.exit_code(report) != tool.EXIT_OK
    assert tool.EXIT_LIVE_BLOCKED != tool.EXIT_OK
    assert tool.EXIT_LIVE_BLOCKED != tool.EXIT_TASK_GOAL_UNMET


def test_the_five_outcomes_map_to_the_documented_codes() -> None:
    success = {"result_layers": _layers(run_completed=True, measurement_accepted=True,
                                        stop_state="STOPPED", task_goal=CAMPAIGN)}
    rejected = {"result_layers": _layers(run_completed=True, measurement_accepted=False,
                                         stop_state="STOPPED", task_goal=CAMPAIGN)}
    stop_unverified = {"result_layers": _layers(run_completed=True, measurement_accepted=True,
                                                stop_state="STOP_UNVERIFIED", task_goal=CAMPAIGN)}
    failed_none = {"result_layers": _layers(run_completed=False, measurement_accepted=False,
                                            stop_state="NOT_ATTEMPTED", task_goal=CAMPAIGN)}

    assert tool.exit_code(success) == tool.EXIT_OK
    assert tool.exit_code(rejected) == tool.EXIT_TASK_GOAL_UNMET
    assert tool.exit_code(failed_none) == tool.EXIT_TASK_GOAL_UNMET
    assert tool.exit_code(stop_unverified) == tool.EXIT_STOP_UNVERIFIED
    # 非零码彼此不重合，也都不是成功码。
    assert len({tool.EXIT_OK, tool.EXIT_USAGE, tool.EXIT_LIVE_BLOCKED,
                tool.EXIT_STOP_UNVERIFIED, tool.EXIT_TASK_GOAL_UNMET}) == 5
    assert tool.EXIT_OK not in {tool.EXIT_USAGE, tool.EXIT_LIVE_BLOCKED,
                                tool.EXIT_STOP_UNVERIFIED, tool.EXIT_TASK_GOAL_UNMET}
    # 「测量未验收」与「没有停泵证据」退出码相同，靠分层字段区分。
    assert rejected["result_layers"]["stop"]["state"] == "STOPPED"
    assert failed_none["result_layers"]["stop"]["state"] == "NOT_ATTEMPTED"


# ------------------------------------------------- 真实 report() 路径：测量验收证据

class _Sink:
    """可编程的 sink：只回报它被要求回报的东西。"""

    def __init__(self, summary: dict | None = None) -> None:
        self._summary = summary

    def summary(self) -> dict | None:
        return self._summary


def _session_with_sink(tmp_path: Path, sink) -> object:
    """直接构造会话并调用真实 report()，不走 run()（不需要硬件）。"""
    session = tool.LiveCaptureSession.for_device_free_tests(
        plan={"session_id": "layers", "capture_plan": {}, "termination": {}},
        pump=None, camera=None, locks=[], sink=sink,
        output_dir=tmp_path / "out", log=lambda _m: None)
    session.started = True
    session.run_finished = True
    session.stop_outcome = tool.StopOutcome("STOPPED", 1, True)
    return session


def test_report_without_any_measurement_evidence_rejects_the_campaign_goal(tmp_path: Path) -> None:
    """sink 只声明采样前提、没有任何测量验收字段 → 不得默认通过。"""
    report = _session_with_sink(tmp_path, _Sink({"sampling_premise_ok": True})).report()
    layers = report["result_layers"]
    assert layers["run_completed"] is True
    assert layers["measurement_accepted"] is False, "缺测量证据不得默认验收通过"
    assert layers["measurement_acceptance_reason"]
    assert layers["task_goal"] == CAMPAIGN
    assert layers["task_goal_met"] is False
    assert tool.exit_code(report) == tool.EXIT_TASK_GOAL_UNMET
    assert tool.exit_code(report) != tool.EXIT_OK


def test_report_with_no_sink_summary_at_all_rejects_the_campaign_goal(tmp_path: Path) -> None:
    report = _session_with_sink(tmp_path, _Sink(None)).report()
    layers = report["result_layers"]
    assert layers["measurement_accepted"] is False
    assert layers["measurement_acceptance_reason"] == "sink_provides_no_measurement_evidence"
    assert layers["task_goal_met"] is False


def test_report_with_explicit_acceptance_meets_the_campaign_goal(tmp_path: Path) -> None:
    report = _session_with_sink(tmp_path, _Sink(
        {"sampling_premise_ok": True, "measurement_accepted": True})).report()
    layers = report["result_layers"]
    assert layers["measurement_accepted"] is True
    assert layers["task_goal_met"] is True
    assert tool.exit_code(report) == tool.EXIT_OK


def test_report_with_explicit_rejection_keeps_the_reason(tmp_path: Path) -> None:
    report = _session_with_sink(tmp_path, _Sink(
        {"sampling_premise_ok": True, "measurement_accepted": False,
         "measurement_acceptance_reason": "scale_not_validated"})).report()
    layers = report["result_layers"]
    assert layers["measurement_accepted"] is False
    assert layers["measurement_acceptance_reason"] == "scale_not_validated"
    assert layers["task_goal_met"] is False


def test_diagnostic_capture_goal_can_pass_without_measurement_acceptance(tmp_path: Path) -> None:
    """诊断入口声明的目标就是采集：测量未验收时它仍达成，但必须分别报告。"""
    session = tool.LiveCaptureSession.for_device_free_tests(
        plan={"session_id": "diag", "capture_plan": {}, "termination": {}},
        pump=None, camera=None, locks=[], sink=_Sink({"sampling_premise_ok": False}),
        output_dir=tmp_path / "out", log=lambda _m: None,
        task_goal=DIAGNOSTIC)
    session.started = True
    session.run_finished = True
    session.stop_outcome = tool.StopOutcome("STOPPED", 1, True)
    report = session.report()
    layers = report["result_layers"]
    assert layers["task_goal_met"] is True
    assert layers["measurement_accepted"] is False
    assert tool.exit_code(report) == tool.EXIT_OK


# ---------------------------------------------------------- 真实会话产出分层字段

def test_real_session_reports_the_three_layers_separately(tmp_path: Path) -> None:
    """设备被占用：启动前即失败 → stop 为 None，分层字段仍须完整且不抛异常。"""
    session, _pump, _camera, _lock, _sink, _order = build(make_plan(), tmp_path,
                                                          lock_busy=True)
    report = session.run()
    layers = report["result_layers"]
    assert report["stop"] is None
    assert layers["stop"] == {"state": "NOT_ATTEMPTED", "verified": False}
    assert layers["run_completed"] is False
    assert layers["measurement_accepted"] is False
    assert layers["task_goal"] == CAMPAIGN
    assert layers["task_goal_met"] is False
    assert tool.exit_code(report) == tool.EXIT_TASK_GOAL_UNMET


def test_session_rejects_an_unknown_task_goal(tmp_path: Path) -> None:
    session, *_ = build(make_plan(), tmp_path)
    assert session.task_goal == CAMPAIGN
    with pytest.raises(ValueError):
        tool.LiveCaptureSession(plan=session.plan, pump=session.pump, camera=session.camera,
                               locks=[], sink=session.sink, output_dir=tmp_path / "x",
                               log=lambda _m: None, task_goal="not-a-goal")
