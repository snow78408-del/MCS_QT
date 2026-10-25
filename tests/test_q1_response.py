from __future__ import annotations
import json
from types import SimpleNamespace
import numpy as np
import cv2
import pytest
from tools.run_q1_response import admission,stable_check,response_schedule,ResponseRecorder,ResponseSession,PROFILE,registration_patch
from tools.plant_flow_transient_capture import CaptureLifecycleError

def stable_samples(end):return [(float(t),float(72+.8*np.sin(2*np.pi*t/10))) for t in range(int(end))]

def test_range_stationarity_and_drift_are_distinct():
    assert stable_check(stable_samples(180),180)['passed']
    drifting=[(float(t),float(72+.15*t+.8*np.sin(2*np.pi*t/10))) for t in range(180)]
    assert not stable_check(drifting,180)['passed']
    missing=[(t,None if 100<=t<120 else y) for t,y in stable_samples(180)]
    assert not stable_check(missing,180)['available']

def test_admission_reserves_target_and_transaction():
    assert admission([282.,82.],900.)
    assert not admission([352.5,102.5],900.)
    assert not admission([200.,800.],900.)

def test_phase_sampling_includes_transition_and_target():
    r=object.__new__(ResponseRecorder)
    assert r.full_rate('transition',100.)
    assert r.full_rate('P1_target',59.9) and not r.full_rate('P1_target',60.)
    assert r.full_rate('P1_baseline',180.)

def session(tmp_path,qualifies):
    (tmp_path/'registered_geometry.json').write_text(json.dumps({'walls':[]}))
    now=[0.];updates=[]
    def update(q1,q2):
        updates.append((q1,q2));now[0]+=4
        return SimpleNamespace(ok=True,still_running=True,stop_verified_before_write=True,restart_verified=True)
    pump=SimpleNamespace(update_flow_while_running=update,get_current_q_state=lambda:(60.,20.))
    sink=SimpleNamespace(summary=lambda:{},set_phase=lambda *a:None,
                         guard_frame=lambda:(cv2.imread(str(tmp_path/'latest.png'),0),now[0]) if (tmp_path/'latest.png').exists() else None)
    p={'flow_plan':{'campaign_profile':PROFILE,'steps':[],'max_cumulative_delivery_each_ul':900.}}
    s=ResponseSession(plan=p,pump=pump,camera=object(),locks=[],sink=sink,output_dir=tmp_path,log=lambda _:None,clock=lambda:now[0])
    s._geometry_registered=True;s._delivery_last=0.
    def wait(seconds):now[0]+=seconds;s._update_delivery()
    s._wait_segment=wait
    s._measure_baseline=lambda w:stable_samples(now[0]) if qualifies else [(float(t),None) for t in range(int(now[0]))]
    return s,updates

def test_unqualified_baseline_never_changes_flow(tmp_path):
    s,updates=session(tmp_path,False)
    with pytest.raises(CaptureLifecycleError,match='Baseline not qualified'):s._run_segments()
    assert not updates and not s.run_finished and len(s.segment_records)==1

def test_single_target_preserves_budget_and_command_evidence(tmp_path):
    s,updates=session(tmp_path,True);s._run_segments()
    assert updates==[(60.,20.)]
    assert s.run_finished and [r['label'] for r in s.segment_records]==['P1_baseline','P1_target']
    assert s._delivery_ul[0]==pytest.approx(70.5*184/60+60.5*9)
    assert s.commands[-1]['ok']

def test_failed_transaction_does_not_begin_target(tmp_path):
    s,updates=session(tmp_path,True)
    s.pump.update_flow_while_running=lambda *a:SimpleNamespace(ok=True,still_running=True,stop_verified_before_write=False,restart_verified=True)
    with pytest.raises(CaptureLifecycleError,match='transaction failed'):s._run_segments()
    assert len(s.segment_records)==1 and not s.run_finished

def test_actual_baseline_budget_blocks_switch(tmp_path):
    s,updates=session(tmp_path,True)
    s._delivery_ul=[400.,0.]
    with pytest.raises(CaptureLifecycleError,match='Insufficient'):s._run_segments()
    assert not updates


def test_geometry_guard_rejects_translation_before_more_capture(tmp_path):
    s,_=session(tmp_path,True)
    rng=np.random.default_rng(17)
    original=rng.integers(0,256,(100,140),dtype=np.uint8)
    s._guard_template=original[25:55,40:80].copy();s._guard_anchor=(40,25)
    cv2.imwrite(str(tmp_path/'latest.png'),original)
    s._check_geometry()
    assert json.loads((tmp_path/'geometry_guard.json').read_text())['passed']
    shifted=cv2.warpAffine(original,np.float32([[1,0,9],[0,1,14]]),(140,100))
    cv2.imwrite(str(tmp_path/'latest.png'),shifted)
    with pytest.raises(CaptureLifecycleError,match='Geometry moved'):s._check_geometry()
    evidence=json.loads((tmp_path/'geometry_guard.json').read_text())
    assert evidence['dx']==9 and evidence['dy']==14 and not evidence['passed']


def test_geometry_guard_rejects_unresolved_static_feature(tmp_path):
    s,_=session(tmp_path,True);rng=np.random.default_rng(13)
    s._guard_template=rng.integers(0,256,(30,40),dtype=np.uint8);s._guard_anchor=(40,25)
    cv2.imwrite(str(tmp_path/'latest.png'),rng.integers(0,256,(100,140),dtype=np.uint8))
    with pytest.raises(CaptureLifecycleError,match='Geometry moved'):s._check_geometry()


def test_current_field_static_patch_registers_without_weakening_drift_guard(tmp_path):
    s,_=session(tmp_path,True)
    reference=np.random.default_rng(3).integers(0,256,(180,240),dtype=np.uint8)
    reference_path=tmp_path/'reference.png'
    cv2.imwrite(str(reference_path),reference)
    s.plan.update(geometry_reference_image=str(reference_path),
                  reference_walls=[dict(x1=.1,y1=y,x2=.9,y2=y) for y in (.7,.85)],
                  geometry_guard=dict(reference_patch_px=[20,10,60,50]))
    shifted=cv2.warpAffine(reference,np.float32([[1,0,3],[0,1,2]]),(240,180))
    cv2.imwrite(str(tmp_path/'pre_start.png'),shifted)
    walls=s._register_geometry()
    assert walls[0]['x1']==pytest.approx(.1+3/240)
    assert walls[0]['y1']==pytest.approx(.7+2/180)
    evidence=json.loads((tmp_path/'registered_geometry.json').read_text())
    assert evidence['reference_patch_px']==[20,10,60,50]
    assert evidence['dx']==3 and evidence['dy']==2 and evidence['correlation']>=.85
    s._delivery_last=None
    cv2.imwrite(str(tmp_path/'latest.png'),shifted)
    s._check_geometry()
    moved=cv2.warpAffine(shifted,np.float32([[1,0,4],[0,1,0]]),(240,180))
    cv2.imwrite(str(tmp_path/'latest.png'),moved)
    with pytest.raises(CaptureLifecycleError,match='Geometry moved'):
        s._check_geometry()


@pytest.mark.parametrize('roi',[[0,0,10,50],[-1,0,50,50],[200,0,60,50],
                                [0,0,60.,50],[True,0,60,50],[30,120,60,40]])
def test_custom_static_patch_rejects_invalid_or_in_channel_region(roi):
    plan=dict(geometry_guard=dict(reference_patch_px=roi),
              reference_walls=[dict(x1=.1,y1=y,x2=.9,y2=y) for y in (.7,.85)])
    with pytest.raises(CaptureLifecycleError):
        registration_patch(plan,np.zeros((180,240),np.uint8))


def test_default_static_patch_remains_compatible():
    assert registration_patch({},np.zeros((540,720),np.uint8))==(330,150,105,90)


def test_textureless_custom_patch_never_registers(tmp_path):
    s,_=session(tmp_path,True)
    reference=np.full((180,240),128,np.uint8)
    path=tmp_path/'reference.png';cv2.imwrite(str(path),reference)
    cv2.imwrite(str(tmp_path/'pre_start.png'),reference)
    s.plan.update(geometry_reference_image=str(path),
                  reference_walls=[dict(x1=.1,y1=y,x2=.9,y2=y) for y in (.7,.85)],
                  geometry_guard=dict(reference_patch_px=[20,10,60,50]))
    with pytest.raises(CaptureLifecycleError,match='no texture'):
        s._register_geometry()
    assert not s._geometry_registered


def test_guarded_wait_retains_duration_heartbeat_and_early_abort(tmp_path,monkeypatch):
    s,_=session(tmp_path,True);clock=[0.];s._clock=lambda:clock[0]
    s._guard_template=np.zeros((2,2),dtype=np.uint8);checks=[];events=[]
    s._check_geometry=lambda:checks.append(clock[0]);s._record=events.append
    monkeypatch.setattr('tools.run_q1_response.AFirstThreeSession._wait_segment',lambda _,seconds:clock.__setitem__(0,clock[0]+seconds))
    ResponseSession._wait_segment(s,9.)
    assert clock[0]==9 and max(b-a for a,b in zip(checks,checks[1:]))<=2
    assert events==['capture_heartbeat']
    def moved():raise CaptureLifecycleError('Geometry moved')
    s._check_geometry=moved
    with pytest.raises(CaptureLifecycleError,match='Geometry moved'):ResponseSession._wait_segment(s,10.)
    assert clock[0]==9


def test_baseline_worker_failure_propagates_without_device_actions(tmp_path,monkeypatch):
    s,_=session(tmp_path,True)
    monkeypatch.setattr('tools.run_q1_response.subprocess.run',lambda *a,**k:SimpleNamespace(returncode=1,stderr='analysis failed'))
    with pytest.raises(CaptureLifecycleError,match='Baseline analysis failed'):ResponseSession._measure_baseline(s,[])


def test_guard_uses_prestart_only_before_infusion(tmp_path):
    s,_=session(tmp_path,True);rng=np.random.default_rng(41)
    original=rng.integers(0,256,(100,140),dtype=np.uint8)
    s._guard_template=original[25:55,40:80].copy();s._guard_anchor=(40,25)
    cv2.imwrite(str(tmp_path/'pre_start.png'),original)
    s._delivery_last=None;s._check_geometry()
    s._delivery_last=0.
    with pytest.raises(CaptureLifecycleError,match='frame unavailable'):s._check_geometry()


def test_recorder_geometry_snapshot_is_bounded_and_detached(tmp_path):
    recorder=ResponseRecorder(directory=tmp_path,interval_s=1.,walls=[])
    frame=SimpleNamespace(image=np.zeros((20,30),np.uint8),host_monotonic_timestamp=10.,hardware_frame_id=1)
    recorder.prepare(frame)
    try:
        recorder.set_phase('P1_target',10.)
        recorder.submit(frame);snapshot=recorder.guard_frame()
        frame.image[:]=255
        assert np.all(snapshot[0]==0)
        frame.hardware_frame_id=2;frame.host_monotonic_timestamp=10.6
        recorder.submit(frame)
        assert np.all(recorder.guard_frame()[0]==255)
        assert recorder.guard_frame()[1]==10.6
        assert np.all(snapshot[0]==0)
    finally:recorder.close()


def test_geometry_guard_rejects_stale_memory_frame(tmp_path):
    s,_=session(tmp_path,True);s._guard_template=np.zeros((2,2),np.uint8)
    s.sink.guard_frame=lambda:(np.zeros((10,10),np.uint8),-3.)
    with pytest.raises(CaptureLifecycleError,match='frame stale'):s._check_geometry()


def test_extended_baseline_allows_late_qualification_with_five_minute_target(tmp_path):
    s,updates=session(tmp_path,True)
    s.plan['response_plan']={'baseline_check_s':[120,180,240,300,360,420],'target_duration_s':300}
    s._measure_baseline=lambda _:stable_samples(s._clock()) if s._clock()>=360 else [(float(t),None) for t in range(int(s._clock()))]
    s._run_segments()
    assert updates==[(60.,20.)] and s.run_finished
    assert s.segment_records[0]['observation_finished_monotonic']==420
    assert s.segment_records[1]['planned_duration_s']==300
    assert s._delivery_ul[0]==pytest.approx(70.5*424/60+60.5*5)
    assert admission([70.5*7,20.5*7],900.,300.)


def test_extended_baseline_unqualified_does_not_send_target(tmp_path):
    s,updates=session(tmp_path,False)
    s.plan['response_plan']={'baseline_check_s':[120,180,240,300,360,420],'target_duration_s':300}
    with pytest.raises(CaptureLifecycleError,match='by420s'):s._run_segments()
    assert s._clock()==420 and not updates


@pytest.mark.parametrize('response',[
    {'baseline_check_s':[120,180,240,300,360,420],'target_duration_s':540},
    {'baseline_check_s':[120,180,240],'target_duration_s':999},
    {'baseline_check_s':[120,180,240,300],'target_duration_s':300},
])
def test_unsupported_schedule_rejected(response):
    with pytest.raises(ValueError):response_schedule({'response_plan':response})


def diagnostic_session(tmp_path):
    s,updates=session(tmp_path,False)
    s.plan['response_plan']={'baseline_policy':'fixed_diagnostic','baseline_duration_s':120,
        'baseline_check_s':[120],'target_duration_s':300}
    def no_online_fit(_):raise AssertionError('Diagnostic capture must not wait for a numeric steady gate')
    s._measure_baseline=no_online_fit
    return s,updates


def test_fixed_diagnostic_collects_one_step_without_claiming_steady_state(tmp_path):
    s,updates=diagnostic_session(tmp_path);s._run_segments()
    assert updates==[(60.,20.)] and s.run_finished
    assert s.segment_records[0]['observation_finished_monotonic']==120
    assert s.segment_records[1]['planned_duration_s']==300
    record=json.loads((tmp_path/'baseline_checks.json').read_text())
    assert record['qualified_numeric'] is False and record['policy']=='fixed_diagnostic'
    assert s._delivery_ul[0]==pytest.approx(70.5*124/60+60.5*5)


def test_fixed_diagnostic_still_rejects_insufficient_target_budget(tmp_path):
    s,updates=diagnostic_session(tmp_path);s._delivery_ul=[600.,0.]
    with pytest.raises(CaptureLifecycleError,match='Insufficient'):s._run_segments()
    assert not updates


def test_fixed_diagnostic_geometry_failure_blocks_flow_change(tmp_path):
    s,updates=diagnostic_session(tmp_path)
    s._check_geometry=lambda:(_ for _ in ()).throw(CaptureLifecycleError('Geometry moved'))
    with pytest.raises(CaptureLifecycleError,match='Geometry moved'):s._run_segments()
    assert not updates


def test_fixed_diagnostic_transaction_failure_blocks_target_record(tmp_path):
    s,updates=diagnostic_session(tmp_path)
    s.pump.update_flow_while_running=lambda *a:SimpleNamespace(ok=False,still_running=False)
    with pytest.raises(CaptureLifecycleError,match='transaction failed'):s._run_segments()
    assert len(s.segment_records)==1 and not s.run_finished


def test_fixed_diagnostic_five_minutes_each_preserves_single_step_and_budget(tmp_path):
    s,updates=diagnostic_session(tmp_path)
    s.plan['response_plan'].update(baseline_duration_s=300,baseline_check_s=[300])
    s._run_segments()
    assert updates==[(60.,20.)] and s.run_finished
    assert s.segment_records[0]['observation_finished_monotonic']==300
    assert s.segment_records[1]['observation_finished_monotonic']-s.segment_records[1]['observation_started_monotonic']==300
    assert s._delivery_ul[0]==pytest.approx(70.5*304/60+60.5*5)
    assert admission([70.5*5,20.5*5],900.,300.)


def test_fixed_diagnostic_duration_and_checkpoint_must_match():
    with pytest.raises(ValueError):response_schedule({'response_plan':{'baseline_policy':'fixed_diagnostic',
        'baseline_duration_s':300,'baseline_check_s':[120],'target_duration_s':300}})


@pytest.mark.parametrize('policy',['force','ignored','steady'])
def test_unknown_policy_cannot_bypass_qualified_gate(policy):
    with pytest.raises(ValueError):response_schedule({'response_plan':{'baseline_policy':policy}})
