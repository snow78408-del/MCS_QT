from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.regime_survey import apply_reviews, survey_records, transition_brackets


def segments(flows=(50, 70, 90)):
    return [{"index": i, "label": f"Q1_{q}", "q1_ul_min": q, "q2_ul_min": 20,
             "observation_started_monotonic": i * 300.,
             "observation_finished_monotonic": (i + 1) * 300.}
            for i, q in enumerate(flows, 1)]


def reviews(labels=("jetting", "dripping", "jetting")):
    return [{"segment_index": i, "regime": regime, "sweep_id": "up-1",
             "scan_channel": "q1", "stable_window_monotonic": [i * 300 + 180, (i + 1) * 300],
             "evidence": [f"review/segment-{i}.json"], "breakup_position_px": [10, 30]}
            for i, regime in enumerate(labels, 1)]


def test_capture_does_not_invent_regime_or_actual_flow():
    records = survey_records(segments())
    assert [row["q1_over_q2_command"] for row in records] == [2.5, 3.5, 4.5]
    assert all(row["regime"] == "unknown" and row["review_status"] == "pending" for row in records)
    assert transition_brackets(records) == []


def test_two_transitions_are_brackets_not_exact_ratios():
    records = apply_reviews(survey_records(segments()), reviews())
    boundaries = transition_brackets(records)
    assert [b["transition"] for b in boundaries] == ["jetting->dripping", "dripping->jetting"]
    assert [b["q1_over_q2_bracket"] for b in boundaries] == [[2.5, 3.5], [3.5, 4.5]]
    assert all(b["fixed_channel_flow_ul_min"] == 20 for b in boundaries)


@pytest.mark.parametrize("middle", ["unknown", "mixed", "unbroken_thread"])
def test_uncertain_points_do_not_bridge_a_transition(middle):
    records = apply_reviews(survey_records(segments()), reviews(("jetting", middle, "dripping")))
    assert transition_brackets(records) == []


def test_reverse_sweep_is_reported_separately():
    records = apply_reviews(survey_records(segments((90, 70, 50))), reviews())
    assert all(b["direction"] == "decreasing" for b in transition_brackets(records))
    assert transition_brackets(apply_reviews(survey_records(segments((50, 90, 70))), reviews())) == []


def test_changing_both_flows_is_not_a_single_channel_boundary():
    points = segments()
    points[1]["q2_ul_min"] = 25
    assert transition_brackets(apply_reviews(survey_records(points), reviews())) == []


@pytest.mark.parametrize("change", [
    {"evidence": []}, {"stable_window_monotonic": [0, 900]},
    {"stable_window_monotonic": [float("nan"), 500]},
    {"regime": "no_detections"}, {"breakup_position_px": [-1, 5]},
])
def test_review_requires_explicit_bounded_evidence(change):
    review = reviews()[0] | change
    with pytest.raises(ValueError):
        apply_reviews(survey_records(segments()), [review])


def test_failed_segment_cannot_supply_verified_regime():
    points = segments()
    points[0]["observation_finished_monotonic"] = None
    with pytest.raises(ValueError):
        apply_reviews(survey_records(points), reviews())


def test_review_cannot_replace_recorded_flow():
    review = reviews()[0] | {"q1_command_ul_min": 999}
    assert apply_reviews(survey_records(segments()), [review])[0]["q1_command_ul_min"] == 50


def test_offline_cli_preserves_source_and_never_overwrites(tmp_path):
    source = tmp_path / "session.json"
    output = tmp_path / "survey.json"
    source.write_text(json.dumps({"segments": segments()}), encoding="utf-8")
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "tools/regime_survey.py"),
               str(source), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["validated_for_mpc"] is False
    before = output.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert output.read_bytes() == before


def test_session_attaches_pending_regime_rows_after_cleanup():
    from tools.run_r1_front4_live import R1FrontFourSession

    session = object.__new__(R1FrontFourSession)
    session.segment_records = segments()
    session.plan = {"regime_exploration": {"phase_mapping": {"q1": "oil_total"}}}
    calls = []
    for name in ("_prepare_provenance", "_acquire_locks", "_connect_and_verify_initial_state",
                 "_configure_camera_and_read_back", "_begin_capture_and_save_baseline",
                 "_start_reader", "_write_pump_parameters", "_start_infusion", "_run_segments",
                 "_cleanup_campaign", "_close_sink"):
        setattr(session, name, lambda name=name: calls.append(name))
    session._record = lambda *_: None
    session.report = lambda: {}
    session._finalize_outputs = lambda payload: calls.append("saved")
    report = session.run()
    assert calls[-3:] == ["_cleanup_campaign", "_close_sink", "saved"]
    assert len(report["regime_survey"]["records"]) == 3
    assert report["regime_survey"]["transition_brackets"] == []
    assert report["regime_survey"]["phase_mapping"] == {"q1": "oil_total"}


def test_tomorrow_recipe_is_budgeted_and_not_a_live_plan():
    path = Path(__file__).resolve().parents[1] / "docs/validation_and_regime_survey_20261008.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    assert plan["status"] == "planning_only_not_live_session_plan"
    assert plan["phase_mapping"]["equal_side_split_verified"] is False
    for block in plan["blocks"].values():
        rows = block["points"]
        assert all(row["q1_ul_min"] > 2 * row["q2_ul_min"] > 0 for row in rows)
        assert block["oil_budget_ml"] == pytest.approx(sum(
            row["q1_ul_min"] * row["observation_duration_s"] / 60 / 1000 for row in rows))
        assert block["water_budget_ml"] == pytest.approx(sum(
            row["q2_ul_min"] * row["observation_duration_s"] / 60 / 1000 for row in rows))
