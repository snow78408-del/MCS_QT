from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_q2_lab_capture import session
from tools.run_q2_lab_capture import (
    BLOCKS, LabRecorder, REVERSE_Q2_PROFILE, SHAM_HIGH_PROFILE, SHAM_LOW_PROFILE,
)
from tools.plant_flow_transient_capture import CaptureLifecycleError


@pytest.mark.parametrize('profile', [SHAM_LOW_PROFILE, SHAM_HIGH_PROFILE, REVERSE_Q2_PROFILE])
def test_next_day_profiles_execute_transaction_and_preserve_budget(tmp_path, profile):
    s, calls = session(tmp_path)
    s.plan['flow_plan']['campaign_profile'] = profile
    stages = BLOCKS[profile]
    before = tuple(stages[0][k] for k in ('q1_ul_min', 'q2_ul_min'))
    after = tuple(stages[1][k] for k in ('q1_ul_min', 'q2_ul_min'))
    s._upper_rates = tuple(q + .5 for q in before)
    s.pump.get_current_q_state = lambda: after
    s._run_segments()
    assert calls == [after]  # Same values must not become a no-op.
    assert s.segment_records[1]['same_setting_transaction'] == (before == after)
    assert [r['observation_finished_monotonic'] - r['observation_started_monotonic']
            for r in s.segment_records] == [300., 300.]
    for axis in range(2):
        assert s._delivery_ul[axis] == pytest.approx(
            (before[axis]+.5)*5 + (max(before[axis],after[axis])+.5)*4/60 + (after[axis]+.5)*5)
    recorder = object.__new__(LabRecorder)
    assert all(recorder.full_rate(stage['label'], 280) for stage in stages)


@pytest.mark.parametrize('profile', [SHAM_LOW_PROFILE, SHAM_HIGH_PROFILE])
def test_sham_requires_real_stop_and_restart_confirmation(tmp_path, profile):
    s, _ = session(tmp_path)
    s.plan['flow_plan']['campaign_profile'] = profile
    s.pump.update_flow_while_running = lambda *args: SimpleNamespace(
        ok=True, still_running=True, stop_verified_before_write=False, restart_verified=True)
    with pytest.raises(CaptureLifecycleError, match='transaction failed'):
        s._run_segments()
    assert len(s.segment_records) == 1


def test_missing_transaction_times_remain_missing(tmp_path):
    s, _ = session(tmp_path)
    s._run_segments()
    assert s.commands[-1]['sent_monotonic'] is None
    assert s.commands[-1]['readback_monotonic'] is None
    assert s.segment_records[1]['command_started_monotonic'] is None
    assert s.segment_records[1]['command_readback_monotonic'] is None


def test_raw_only_sham_keeps_transaction_without_claiming_geometry(tmp_path):
    s,calls=session(tmp_path)
    s.plan['capture_plan']={'raw_only':True,'detector_executed':False}
    s.plan['flow_plan']['campaign_profile']=SHAM_LOW_PROFILE
    s.pump.get_current_q_state=lambda:(70.,20.)
    s._geometry_registered=False
    s._run_segments()
    assert calls==[(70.,20.)] and s.run_finished
    assert not s._geometry_registered


def test_raw_only_still_rejects_failed_stop_transaction(tmp_path):
    s,_=session(tmp_path)
    s.plan['capture_plan']={'raw_only':True,'detector_executed':False}
    s._geometry_registered=False
    s.pump.update_flow_while_running=lambda *args:SimpleNamespace(
        ok=True,still_running=True,stop_verified_before_write=False,restart_verified=True)
    with pytest.raises(CaptureLifecycleError,match='transaction failed'):
        s._run_segments()


def test_raw_only_still_rejects_insufficient_reserve(tmp_path):
    s,_=session(tmp_path)
    s.plan['capture_plan']={'raw_only':True,'detector_executed':False}
    s._geometry_registered=False;s._delivery_ul=[600.,0.]
    with pytest.raises(CaptureLifecycleError,match='reserve'):
        s._run_segments()
