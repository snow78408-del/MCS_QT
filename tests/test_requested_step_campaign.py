"""Device-free checks for the bounded 2026-09-26 step sequence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.plant_flow_transient_capture import CaptureLifecycleError, ChannelWriteError
from tools.run_r1_front4_live import R1FrontFourSession, validate_r1_plan


PLAN = Path(__file__).resolve().parents[1] / "output/step-50-100-20260926/session_plan.json"
Q2_PLAN = Path(__file__).resolve().parents[1] / "output/step-q2-100-10-30-40-20260926/session_plan.json"
Q2_FIRST_PLAN = Path(__file__).resolve().parents[1] / "output/step-q2-100-10-20260926/session_plan.json"
Q2_REPEAT_PLAN = Path(__file__).resolve().parents[1] / "output/step-q2-repeat-100-10-30-20260926/session_plan.json"
Q1_FOUR_PLAN = Path(__file__).resolve().parents[1] / "output/step-q1-60-70-80-90-20260926/session_plan.json"


class PumpStub:
    def __init__(self, *, running: bool = False) -> None:
        self.running = running
        self.q1 = 20.0
        self.q2 = 20.0
        self.writes: list[int] = []

    def read_rse(self):
        return SimpleNamespace(ok=True, parsed_reply=SimpleNamespace(
            system_running=self.running, channel_running=[self.running, self.running, False, False]))

    def read_rsp(self, channel: int):
        return SimpleNamespace(ok=True, parsed_reply=SimpleNamespace(channel=channel, syringe_code=33))

    def _channel_params_preserving_profile(self, current, q: float):
        assert current.syringe_code == 33
        return SimpleNamespace(channel=1, flow=q)

    def channel_params_for_flow(self, channel: int, q: float):
        return SimpleNamespace(channel=channel, flow=q)

    @staticmethod
    def flow_from_channel_params(params):
        return params.flow

    flow_from_channel_params_strict = flow_from_channel_params

    def write_wsp_and_verify(self, channel: int, params):
        self.writes.append(channel)
        if channel == 1:
            self.q1 = params.flow
        else:
            self.q2 = params.flow
        return SimpleNamespace(ok=True)

    def get_current_q_state(self):
        return self.q1, self.q2


def make_session(pump: PumpStub, plan: dict) -> R1FrontFourSession:
    session = R1FrontFourSession(
        plan=plan, pump=pump, camera=object(), locks=[], sink=object(),
        output_dir=PLAN.parent, log=lambda _: None, clock=lambda: 1.0,
    )
    session.baseline = {"channels": {"available": True, "q1": 20.0, "q2": 20.0}}
    return session


def test_requested_two_stage_plan_is_frozen_and_budgeted():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert validate_r1_plan(plan) == []
    broken = copy.deepcopy(plan)
    broken["flow_plan"]["steps"][1]["q1_ul_min"] = 120.0
    assert any("does not match frozen" in issue for issue in validate_r1_plan(broken))


def test_new_q2_three_stage_plan_is_frozen_and_budgeted():
    plan = json.loads(Q2_PLAN.read_text(encoding="utf-8"))
    assert validate_r1_plan(plan) == []
    broken = copy.deepcopy(plan)
    broken["flow_plan"]["steps"][2]["q2_ul_min"] = 30.0
    assert any("does not match frozen" in issue for issue in validate_r1_plan(broken))
    short_budget = copy.deepcopy(plan)
    short_budget["flow_plan"]["max_cumulative_delivery_each_ul"] = 1499.0
    assert any("cumulative delivery limit" in issue
               for issue in validate_r1_plan(short_budget))


def test_q2_first_stage_plan_is_bounded_for_refilled_one_ml_syringe():
    plan = json.loads(Q2_FIRST_PLAN.read_text(encoding="utf-8"))
    assert validate_r1_plan(plan) == []
    assert plan["flow_plan"]["max_cumulative_delivery_each_ul"] == 550.0
    assert plan["flow_plan"]["steps"] == [
        {"label": "Q2_10", "q1_ul_min": 100.0, "q2_ul_min": 10.0,
         "duration_s": 300.0}]


def test_q2_repeat_plan_is_frozen_and_covers_two_stage_budget():
    plan = json.loads(Q2_REPEAT_PLAN.read_text(encoding="utf-8"))
    assert validate_r1_plan(plan) == []
    assert [step["label"] for step in plan["flow_plan"]["steps"]] == ["Q2_10", "Q2_30"]
    assert plan["flow_plan"]["max_cumulative_delivery_each_ul"] >= 1000.0
    shortened = copy.deepcopy(plan)
    shortened["flow_plan"]["max_cumulative_delivery_each_ul"] = 999.0
    assert any("cumulative delivery limit" in issue
               for issue in validate_r1_plan(shortened))


def test_q1_four_stage_plan_is_frozen_and_covers_delivery_budget():
    plan = json.loads(Q1_FOUR_PLAN.read_text(encoding="utf-8"))
    assert validate_r1_plan(plan) == []
    assert [step["q1_ul_min"] for step in plan["flow_plan"]["steps"]] == [60, 70, 80, 90]
    short = copy.deepcopy(plan)
    short["flow_plan"]["max_cumulative_delivery_each_ul"] = 1499.0
    assert any("cumulative delivery limit" in issue for issue in validate_r1_plan(short))


def test_stopped_20_20_is_preconditioned_q1_first_before_normal_write():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    pump = PumpStub()
    session = make_session(pump, plan)
    session._write_pump_parameters()
    assert pump.writes == [1, 1, 2]
    assert session.pump_may_be_running
    assert pump.get_current_q_state() == (50.0, 20.0)
    assert any(row["command_id"] == "stopped-safe-initialization-ch1" and row["ok"]
               for row in session.commands)


def test_precondition_refuses_any_write_if_pump_is_running():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    pump = PumpStub(running=True)
    session = make_session(pump, plan)
    with pytest.raises(ChannelWriteError, match="停止"):
        session._write_pump_parameters()
    assert pump.writes == []


def test_precondition_requires_explicit_plan_flag():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan["flow_plan"]["allow_stopped_safe_initialization"] = False
    pump = PumpStub()
    session = make_session(pump, plan)
    with pytest.raises(ChannelWriteError, match="未允许"):
        session._write_pump_parameters()
    assert pump.writes == []


def test_q2_campaign_requires_verified_stop_and_restart_before_next_stage():
    plan = json.loads(Q2_PLAN.read_text(encoding="utf-8"))
    pump = PumpStub()
    pump.update_flow_while_running = lambda *_: SimpleNamespace(
        ok=True, still_running=True, stop_verified_before_write=False,
        restart_verified=False, command_started_monotonic=1.0,
        readback_completed_monotonic=2.0)
    session = make_session(pump, plan)
    session._wait_segment = lambda _duration: None

    with pytest.raises(CaptureLifecycleError, match="segment 2"):
        session._run_segments()
    assert len(session.segment_records) == 1
    assert not session.commands[-1]["ok"]


def test_q2_repeat_requires_verified_stop_and_restart_before_second_stage():
    plan = json.loads(Q2_REPEAT_PLAN.read_text(encoding="utf-8"))
    pump = PumpStub()
    pump.update_flow_while_running = lambda *_: SimpleNamespace(
        ok=True, still_running=True, stop_verified_before_write=False,
        restart_verified=False, command_started_monotonic=1.0,
        readback_completed_monotonic=2.0)
    session = make_session(pump, plan)
    session._wait_segment = lambda _duration: None
    with pytest.raises(CaptureLifecycleError, match="segment 2"):
        session._run_segments()
