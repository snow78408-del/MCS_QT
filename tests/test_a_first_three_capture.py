from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tools.run_a_first_three import RawSurveyRecorder, AFirstThreeSession
from tools.plant_flow_transient_capture import CaptureLifecycleError


def frame(stamp, identifier=1):
    return SimpleNamespace(image=np.arange(80, dtype=np.uint8).reshape(8, 10),
                           host_monotonic_timestamp=stamp, hardware_frame_id=identifier)


@pytest.mark.parametrize("phase", ["A1", "A4", "A5", "A6", "A7"])
def test_fixed_grid_and_bursts_preserve_pixels_and_real_frame_facts(tmp_path, monkeypatch, phase):
    def forbidden(*args, **kwargs):
        pytest.fail("VideoWriter must never be used")
    monkeypatch.setattr(cv2, "VideoWriter", forbidden)
    sink = RawSurveyRecorder(directory=tmp_path, interval_s=1, walls=[])
    sink.prepare(frame(1))
    sink.set_phase(phase, 10.)
    for i, stamp in enumerate((10., 10.01, 11.3, 260., 260.01, 262., 262.01), 1):
        sink.submit(frame(stamp, i))
    sink.close()
    rows = [json.loads(line) for line in (tmp_path / "frame_manifest.ndjson").read_text().splitlines()]
    assert [row["host_monotonic_s"] for row in rows] == [10., 260., 260.01, 262.]
    assert [row["burst"] for row in rows] == [None, 1, 1, None]
    assert rows[2]["grid_slot"] is None
    assert 1 not in [row["grid_slot"] for row in rows]
    for row in rows:
        assert np.array_equal(cv2.imread(str(tmp_path / row["image"]), cv2.IMREAD_UNCHANGED), frame(1).image)
    assert sink.summary()["raw_video_saved"] is False
    assert sink.summary()["measurement_accepted"] is False


def test_write_failure_is_propagated_and_never_counted_as_saved(tmp_path, monkeypatch):
    sink = RawSurveyRecorder(directory=tmp_path, interval_s=1, walls=[])
    sink.prepare(frame(1))
    monkeypatch.setattr(cv2, "imwrite", lambda *args: False)
    sink.set_phase("storage_pilot", 1.)
    sink.submit(frame(1.))
    with pytest.raises(CaptureLifecycleError, match="writer failed"):
        sink.close()
    assert sink.saved == 0
    assert not sink.summary()["sampling_premise_ok"]


def test_camera_timestamp_failure_refuses_fabricated_samples(tmp_path):
    sink = RawSurveyRecorder(directory=tmp_path, interval_s=1, walls=[])
    sink.prepare(frame(1))
    sink.set_phase("A1", 1.)
    sink.submit(frame(1., 2))
    with pytest.raises(ValueError, match="timestamp"):
        sink.submit(frame(1., 3))
    sink.close()


def test_stop_and_delivery_limit_interrupt_wait_without_writing_hardware(tmp_path):
    session = AFirstThreeSession(plan={}, pump=object(), camera=object(), locks=[],
                                sink=SimpleNamespace(check=lambda: None), output_dir=tmp_path,
                                log=lambda _: None, clock=lambda: 100.)
    session._delivery_ul = [950., 0.]
    with pytest.raises(CaptureLifecycleError, match="delivery limit"):
        session._wait_segment(300.)
    session._delivery_ul = [0., 0.]
    (tmp_path / "STOP").touch()
    with pytest.raises(CaptureLifecycleError, match="STOP"):
        session._wait_segment(300.)


def test_delivery_budget_includes_ack_and_transaction_intervals(tmp_path):
    now = [0.]
    session = AFirstThreeSession(plan={}, pump=object(), camera=object(), locks=[],
                                sink=object(), output_dir=tmp_path, log=lambda _: None,
                                clock=lambda: now[0])
    session._delivery_last = 0.
    now[0] = 60.
    session._update_delivery()
    assert session._delivery_ul == [70.5, 20.5]
    session._upper_rates = (60.5, 20.5)
    now[0] = 120.
    session._update_delivery()
    assert session._delivery_ul == [131., 41.]


@pytest.mark.parametrize("high", [False, True])
def test_refilled_pair_runs_only_frozen_points_and_reference(tmp_path, monkeypatch, high):
    from tools import run_a_first_three as a
    profile = a.HIGH_PAIR_PROFILE if high else a.REFILL_PAIR_PROFILE
    sequence = a.BLOCK_SEQUENCES[profile]
    plan = {"flow_plan": {"campaign_profile": profile, "steps": [
        {"label": label, "q1_ul_min": q1, "q2_ul_min": q2, "duration_s": 300.}
        for label, q1, q2, _ in sequence]}}
    monkeypatch.setitem(a.live.CAMPAIGN_PROFILES, profile, sequence)
    monkeypatch.setattr(a.live.R1FrontFourSession, "_start_infusion", lambda _: None)
    updates = []
    def update(q1, q2):
        updates.append((q1, q2))
        return SimpleNamespace(ok=True, still_running=True,
                               stop_verified_before_write=True, restart_verified=True)
    phases = []
    sink = SimpleNamespace(summary=lambda: {}, set_phase=lambda *args: phases.append(args))
    session = a.AFirstThreeSession(plan=plan, pump=SimpleNamespace(update_flow_while_running=update),
                                  camera=object(), locks=[], sink=sink, output_dir=tmp_path,
                                  log=lambda _: None, clock=lambda: 10.)
    durations = []
    session._wait_segment = durations.append
    session._start_infusion()
    session._run_segments()
    assert durations == [30. if high else 60., 300., 300.]
    assert [x[0] for x in phases] == ["refill_reference_70_20"] + [x[0] for x in sequence]
    assert updates == [(x[1], x[2]) for x in sequence]
    assert session.run_finished


def test_increasing_transaction_charged_at_next_upper_rate(tmp_path):
    plan = {"flow_plan": {"steps": [{"q1_ul_min": 60., "q2_ul_min": 20.},
                                    {"q1_ul_min": 70., "q2_ul_min": 20.}]}}
    now = [100.]
    session = AFirstThreeSession(plan=plan, pump=object(), camera=object(), locks=[],
                                sink=SimpleNamespace(summary=lambda: {}), output_dir=tmp_path,
                                log=lambda _: None, clock=lambda: now[0])
    session._delivery_last = 100.
    session._upper_rates = (60.5, 20.5)
    session.segment_records = [{"label": "A4"}]
    session._record("segment_1_A4_finished")
    now[0] = 160.
    session._update_delivery()
    assert session._delivery_ul == [70.5, 20.5]


def test_high_side_budget_keeps_reference_and_transition_inside950ul():
    from tools.run_a_first_three import HIGH_PAIR_PROFILE, REFILL_PAIR_PROFILE, refill_block_budget_ul
    assert refill_block_budget_ul(HIGH_PAIR_PROFILE) == pytest.approx(935.5)
    assert refill_block_budget_ul(REFILL_PAIR_PROFILE) == pytest.approx(760.75)


def test_resumed_budget_cannot_reuse_full_syringe_allowance(tmp_path):
    session = AFirstThreeSession(
        plan={"flow_plan": {"baseline_q1_ul_min": 80., "baseline_q2_ul_min": 20.,
                            "max_cumulative_delivery_each_ul": 500.}},
        pump=object(), camera=object(), locks=[], sink=SimpleNamespace(check=lambda: None),
        output_dir=tmp_path, log=lambda _: None, clock=lambda: 100.)
    assert session._upper_rates == (80.5, 20.5)
    session._delivery_ul = [500., 0.]
    with pytest.raises(CaptureLifecycleError, match="500uL delivery limit"):
        session._wait_segment(300.)
