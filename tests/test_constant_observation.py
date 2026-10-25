from __future__ import annotations
import json
from types import SimpleNamespace
import numpy as np
import pytest
from tools.run_constant_observation import ConstantRecorder, ConstantSession, sampling, validate, PROFILE
from tools.plant_flow_transient_capture import CaptureLifecycleError

def frame(t,i):return SimpleNamespace(image=np.arange(80,dtype=np.uint8).reshape(8,10),host_monotonic_timestamp=t,hardware_frame_id=i)

def test_sampling_and_raw_identity(tmp_path):
    sink=ConstantRecorder(directory=tmp_path,interval_s=1,walls=[])
    sink.prepare(frame(1,1));sink.set_phase('H',10)
    for i,t in enumerate([10,10.01,39.99,40.01,40.02,40.05,190,190.01],1):sink.submit(frame(t,i))
    sink.close();rows=[json.loads(l) for l in (tmp_path/'frame_manifest.ndjson').read_text().splitlines()]
    assert [r['host_monotonic_s'] for r in rows]==[10,10.01,39.99,40.01,40.05,190,190.01]
    assert sum(r['host_monotonic_s']==190 for r in rows)==1
    assert sampling(30)[1] is False and sampling(180)[1] is True

def test_writer_failure_and_identity_rejection(tmp_path,monkeypatch):
    import cv2
    sink=ConstantRecorder(directory=tmp_path,interval_s=1,walls=[]);sink.prepare(frame(1,1));sink.set_phase('H',10)
    monkeypatch.setattr(cv2,'imwrite',lambda *args:False);sink.submit(frame(10,1))
    with pytest.raises(ValueError):sink.submit(frame(10.1,1))
    with pytest.raises(CaptureLifecycleError):sink.close()

def test_h_does_not_change_flow_or_append_points(tmp_path):
    sink=SimpleNamespace(summary=lambda:{},set_phase=lambda *args:None)
    session=ConstantSession(plan={'observation_s':720,'flow_plan':{'steps':[]}},pump=object(),camera=object(),locks=[],sink=sink,output_dir=tmp_path,log=lambda _:None,clock=lambda:100)
    waits=[];session._wait_segment=waits.append
    session._run_segments()
    assert waits==[720] and session.run_finished and len(session.segment_records)==1

@pytest.mark.parametrize('duration',[float('nan'),721,-1,True])
def test_invalid_duration_is_blocked(duration):
    p={'observation_s':duration,'flow_plan':{'campaign_profile':PROFILE,'baseline_q1_ul_min':70.,'baseline_q2_ul_min':20.}}
    assert any('observation_s' in e for e in validate(p))

def test_usable_volume_and_extra_steps_blocked():
    p={'observation_s':720,'flow_plan':{'campaign_profile':PROFILE,'baseline_q1_ul_min':70.,'baseline_q2_ul_min':20.,'max_cumulative_delivery_each_ul':900,'usable_confirmed_ul':{'oil':500,'water':1000},'steps':[{}]}}
    errors=validate(p)
    assert any('confirmed' in e for e in errors) and any('additional' in e for e in errors)
