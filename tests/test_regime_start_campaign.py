"""Device-free regression checks for refill-bounded survey pairs."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.plant_flow_transient_capture import CaptureLifecycleError
from tools.run_r1_front4_live import R1FrontFourSession, validate_r1_plan


@pytest.fixture(params=[(50, 70), (90, 70), (60, 50)])
def plan(request):
    root = Path(__file__).resolve().parents[1]
    flows = request.param
    result = json.loads((root / "output/step-50-100-20260926/session_plan.json").read_text(encoding="utf-8"))
    result["flow_plan"].update(
        campaign_profile=f"requested_q1_{flows[0]}_{flows[1]}_20261008",
        baseline_q1_ul_min=float(flows[0]),
        max_cumulative_delivery_each_ul=sum(flows) * 5.0 + 100.0,
        steps=[{"label": f"Q1_{q}", "q1_ul_min": float(q),
                "q2_ul_min": 20.0, "duration_s": 300.0} for q in flows],
    )
    return result


def test_first_two_points_are_frozen_and_budgeted(plan):
    assert validate_r1_plan(plan) == []
    short = copy.deepcopy(plan)
    short["flow_plan"]["max_cumulative_delivery_each_ul"] = sum(
        step["q1_ul_min"] * 5.0 for step in plan["flow_plan"]["steps"]
    ) - 1.0
    assert any("cumulative delivery limit" in issue for issue in validate_r1_plan(short))
    plan["flow_plan"]["steps"][1]["q1_ul_min"] = 90.0
    assert any("does not match frozen" in issue for issue in validate_r1_plan(plan))


def test_final_single_point_is_budgeted_and_does_not_start_extra_point(plan, tmp_path):
    plan["flow_plan"].update(
        campaign_profile="requested_q1_50_only_20261008",
        baseline_q1_ul_min=50.0,
        max_cumulative_delivery_each_ul=300.0,
        steps=[{"label": "Q1_50", "q1_ul_min": 50.0,
                "q2_ul_min": 20.0, "duration_s": 300.0}],
    )
    assert validate_r1_plan(plan) == []
    short = copy.deepcopy(plan)
    short["flow_plan"]["max_cumulative_delivery_each_ul"] = 249.0
    assert any("cumulative delivery limit" in issue for issue in validate_r1_plan(short))

    def unexpected_update(*_):
        pytest.fail("single-point session must not perform a second flow update")

    session = R1FrontFourSession(
        plan=plan, pump=SimpleNamespace(update_flow_while_running=unexpected_update),
        camera=object(), locks=[], sink=object(), output_dir=tmp_path,
        log=lambda _: None, clock=lambda: 1.0,
    )
    waits = []
    session._wait_segment = waits.append
    session._run_segments()
    assert session.run_finished
    assert waits == [300.0]
    assert len(session.segment_records) == 1
    plan["flow_plan"]["steps"].append(copy.deepcopy(plan["flow_plan"]["steps"][0]))
    assert any("unsupported or malformed" in issue for issue in validate_r1_plan(plan))


@pytest.mark.parametrize("verified", [True, False])
def test_second_point_requires_verified_stop_restart_and_ends_batch(plan, verified, tmp_path):
    updates = []

    def update(q1, q2):
        updates.append((q1, q2))
        return SimpleNamespace(ok=True, still_running=True,
                               stop_verified_before_write=verified, restart_verified=verified)

    session = R1FrontFourSession(
        plan=plan, pump=SimpleNamespace(update_flow_while_running=update),
        camera=object(), locks=[], sink=object(), output_dir=tmp_path,
        log=lambda _: None, clock=lambda: 1.0,
    )
    waits = []
    session._wait_segment = waits.append
    if verified:
        session._run_segments()
        assert session.run_finished
        assert waits == [300.0, 300.0]
        assert len(session.segment_records) == 2
    else:
        with pytest.raises(CaptureLifecycleError, match="segment 2"):
            session._run_segments()
        assert not session.run_finished
        assert waits == [300.0]
    assert updates == [(plan["flow_plan"]["steps"][1]["q1_ul_min"], 20.0)]


@pytest.mark.parametrize("decision", ["missing", "declined", "wrong_session", "no_evidence", "approved"])
def test_optional_visual_review_gate_prevents_unreviewed_flow_increase(plan, decision, tmp_path):
    review_path = tmp_path / "transition_review.json"
    image_path = tmp_path / "reviewed_frame.png"
    image_path.write_bytes(b"device-free evidence placeholder")
    plan["flow_plan"]["transition_review_file"] = str(review_path)
    if decision != "missing":
        review_path.write_text(json.dumps({
            "session_id": "wrong" if decision == "wrong_session" else plan["session_id"],
            "next_segment_index": 2,
            "proceed": decision != "declined",
            "evidence_image": str(tmp_path / "absent.png") if decision == "no_evidence" else str(image_path),
        }), encoding="utf-8")
    updates = []

    def update(q1, q2):
        updates.append((q1, q2))
        return SimpleNamespace(ok=True, still_running=True,
                               stop_verified_before_write=True, restart_verified=True)

    session = R1FrontFourSession(
        plan=plan, pump=SimpleNamespace(update_flow_while_running=update),
        camera=object(), locks=[], sink=object(), output_dir=tmp_path,
        log=lambda _: None, clock=lambda: 1.0,
    )
    session._wait_segment = lambda _: None
    if decision == "approved":
        session._run_segments()
        assert updates == [(plan["flow_plan"]["steps"][1]["q1_ul_min"], 20.0)]
        assert session.run_finished
    else:
        with pytest.raises(CaptureLifecycleError, match="visual transition review"):
            session._run_segments()
        assert updates == []
        assert len(session.segment_records) == 1
