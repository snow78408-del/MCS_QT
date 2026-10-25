from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.run_q2_sb_pilot import SBPilotSession, PilotRecorder, PROFILE, STAGE, validate
from tools.plant_flow_transient_capture import CaptureLifecycleError


def pilot(tmp_path):
    now = [0.]
    s = SBPilotSession(plan=dict(flow_plan=dict(steps=[], campaign_profile=PROFILE,
                       baseline_q1_ul_min=70., baseline_q2_ul_min=25., max_cumulative_delivery_each_ul=900.)),
                       pump=object(), camera=object(), locks=[], sink=SimpleNamespace(summary=lambda: {}, set_phase=lambda *a: None),
                       output_dir=tmp_path, log=lambda _: None, clock=lambda: now[0])
    s._geometry_registered = True
    s._delivery_last = 0.
    s._check_geometry = lambda: None
    def wait(seconds):
        now[0] += seconds
        s._update_delivery()
    s._wait_segment = wait
    return s


def test_pilot_observes_only_60_seconds_and_never_runs_formal_second_stage(tmp_path):
    s = pilot(tmp_path)
    s._run_segments()
    assert s.run_finished and s._clock() == 60.
    assert len(s.segment_records) == 1
    row = s.segment_records[0]
    assert (row['q1_ul_min'], row['q2_ul_min']) == (70., 25.)
    assert row['observation_finished_monotonic'] - row['observation_started_monotonic'] == 60.
    assert row['pilot_not_formal_sb'] and not row['same_setting_transaction']
    assert s._delivery_ul == pytest.approx([70.5, 25.5])


def test_pilot_refuses_geometry_or_budget_failure_before_observation(tmp_path):
    s = pilot(tmp_path)
    s._geometry_registered = False
    with pytest.raises(CaptureLifecycleError, match='geometry'):
        s._run_segments()
    s._geometry_registered = True
    s._delivery_ul = [830., 0.]
    with pytest.raises(CaptureLifecycleError, match='reserve'):
        s._run_segments()
    assert not s.segment_records and not s.run_finished


def test_pilot_sampling_keeps_pilot_burst_and_regular_grid():
    recorder = object.__new__(PilotRecorder)
    assert recorder.full_rate('SB_pilot_60s', 29.9)
    assert not recorder.full_rate('SB_pilot_60s', 30.)
    assert recorder.full_rate('storage_pilot', 5.)


def test_pilot_validation_cannot_turn_into_custom_flow_or_longer_run(tmp_path, monkeypatch):
    monkeypatch.setattr('tools.run_q2_sb_pilot.live.validate_session_plan', lambda _: [])
    reference = tmp_path / 'reference.png'
    reference.write_bytes(b'existence check only')
    plan = dict(flow_plan=dict(campaign_profile=PROFILE, baseline_q1_ul_min=70., baseline_q2_ul_min=25., steps=[],
                              q1_allowed_range_ul_min=[69.5, 70.5], q2_allowed_range_ul_min=[24.5, 25.5],
                              max_cumulative_delivery_each_ul=900., usable_confirmed_ul=dict(oil=1000., water=1000.)),
                geometry_reference_image=str(reference), reference_walls=[{}, {}], identification_stages=[dict(STAGE)],
                capture_mode='pilot_60s_not_formal_sb')
    assert not validate(plan)
    plan['identification_stages'][0]['duration_s'] = 61.
    assert validate(plan)
    plan['identification_stages'] = [dict(STAGE)]
    plan['flow_plan']['baseline_q2_ul_min'] = 20.
    assert validate(plan)
