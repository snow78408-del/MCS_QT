from __future__ import annotations
from types import SimpleNamespace
import pytest
from tools.run_q2_lab_capture import Q2LabSession,LabRecorder,target_reserve,STAGES,PROFILE,LOW_Q1_PROFILE,HOLDOUT_PROFILE,REVERSE_Q1_PROFILE,LOW_Q2_PROFILE,BLOCKS,validate
from tools.plant_flow_transient_capture import CaptureLifecycleError


def session(tmp_path):
    now=[0.];updates=[]
    def update(q1,q2):
        updates.append((q1,q2));now[0]+=4
        return SimpleNamespace(ok=True,still_running=True,stop_verified_before_write=True,restart_verified=True)
    pump=SimpleNamespace(update_flow_while_running=update,get_current_q_state=lambda:(70.,25.))
    sink=SimpleNamespace(summary=lambda:{},set_phase=lambda *a:None)
    plan={'flow_plan':{'campaign_profile':PROFILE,'baseline_q1_ul_min':70.,'baseline_q2_ul_min':20.,'steps':[],'max_cumulative_delivery_each_ul':900.}}
    s=Q2LabSession(plan=plan,pump=pump,camera=object(),locks=[],sink=sink,output_dir=tmp_path,log=lambda _:None,clock=lambda:now[0])
    s._geometry_registered=True;s._delivery_last=0.
    s._check_geometry=lambda:None
    def wait(seconds):now[0]+=seconds;s._update_delivery()
    s._wait_segment=wait
    return s,updates


def test_water_step_single_transaction_duration_and_upper_budget(tmp_path):
    s,updates=session(tmp_path);s._run_segments()
    assert updates==[(70.,25.)] and s.run_finished
    assert [r['q2_ul_min'] for r in s.segment_records]==[20.,25.]
    assert [r['observation_finished_monotonic']-r['observation_started_monotonic'] for r in s.segment_records]==[300.,300.]
    assert s._delivery_ul[1]==pytest.approx(20.5*5+25.5*304/60)
    assert s._delivery_ul[0]==pytest.approx(70.5*604/60)
    assert s.commands[-1]['ok']


def test_q2_reserve_covers_rising_water_and_transaction():
    assert target_reserve([352.5,102.5],900.)
    assert not target_reserve([600.,0.],900.)
    assert not target_reserve([0.,800.],900.)


def test_no_target_when_budget_or_geometry_invalid(tmp_path):
    s,updates=session(tmp_path);s._delivery_ul=[600.,0.]
    with pytest.raises(CaptureLifecycleError,match='reserve'):s._run_segments()
    assert not updates
    s,updates=session(tmp_path)
    s._check_geometry=lambda:(_ for _ in ()).throw(CaptureLifecycleError('Geometry moved'))
    with pytest.raises(CaptureLifecycleError,match='Geometry moved'):s._run_segments()
    assert not updates


@pytest.mark.parametrize('state',[(60.,25.),(70.,20.),(70.,26.)])
def test_wrong_readback_does_not_record_target(tmp_path,state):
    s,_=session(tmp_path);s.pump.get_current_q_state=lambda:state
    with pytest.raises(CaptureLifecycleError,match='readback'):s._run_segments()
    assert len(s.segment_records)==1 and not s.run_finished


def test_failed_stop_or_restart_does_not_begin_target(tmp_path):
    s,_=session(tmp_path)
    s.pump.update_flow_while_running=lambda *a:SimpleNamespace(ok=True,still_running=True,stop_verified_before_write=False,restart_verified=True)
    with pytest.raises(CaptureLifecycleError,match='transaction failed'):s._run_segments()
    assert len(s.segment_records)==1


def test_capture_windows_include_both_tails_and_transaction():
    r=object.__new__(LabRecorder)
    for stage in [s for block in BLOCKS.values() for s in block]:
        assert r.full_rate(stage['label'],185.) and r.full_rate(stage['label'],280.)
        assert not r.full_rate(stage['label'],100.)
    assert r.full_rate('transition',2.)


@pytest.mark.parametrize('profile,target',[(LOW_Q1_PROFILE,(50.,20.)),(HOLDOUT_PROFILE,(65.,22.)),(REVERSE_Q1_PROFILE,(60.,20.)),(LOW_Q2_PROFILE,(70.,18.))])
def test_frozen_followup_blocks_keep_exact_times_and_correct_rate_budget(tmp_path,profile,target):
    s,updates=session(tmp_path);s.plan['flow_plan']['campaign_profile']=profile
    current=tuple(BLOCKS[profile][0][key] for key in ['q1_ul_min','q2_ul_min'])
    s._upper_rates=tuple(q+.5 for q in current);s.pump.get_current_q_state=lambda:target
    s._run_segments()
    assert updates==[target] and s.run_finished
    assert [r['label'] for r in s.segment_records]==[r['label'] for r in BLOCKS[profile]]
    assert [r['observation_finished_monotonic']-r['observation_started_monotonic'] for r in s.segment_records]==[300.,300.]
    for axis in range(2):
        assert s._delivery_ul[axis]==pytest.approx((current[axis]+.5)*5+(max(current[axis],target[axis])+.5)*4/60+(target[axis]+.5)*5)


@pytest.mark.parametrize('profile,current,bounds,cap',[
    (REVERSE_Q1_PROFILE,(50.,20.),([49.5,60.5],[19.5,20.5]),590.),
    (LOW_Q2_PROFILE,(70.,20.),([69.5,70.5],[17.5,20.5]),745.)])
def test_followup_plan_validation_and_capacity(tmp_path,monkeypatch,profile,current,bounds,cap):
    from copy import deepcopy
    monkeypatch.setattr('tools.run_q2_lab_capture.live.validate_session_plan',lambda _:[])
    reference=tmp_path/'reference.png';reference.write_bytes(b'reference file existence only')
    plan={'flow_plan':{'campaign_profile':profile,'baseline_q1_ul_min':current[0],'baseline_q2_ul_min':current[1],
        'steps':[],'max_cumulative_delivery_each_ul':900.,'usable_confirmed_ul':{'oil':1000.,'water':1000.},
        'q1_allowed_range_ul_min':bounds[0],'q2_allowed_range_ul_min':bounds[1]},
        'identification_stages':deepcopy(BLOCKS[profile]),'geometry_reference_image':str(reference),'reference_walls':[{},{}]}
    assert validate(plan)==[]
    plan['flow_plan']['max_cumulative_delivery_each_ul']=cap
    assert 'Cap cannot cover whole block' in validate(plan)


def test_custom_validation_rejects_unfrozen_stages_or_refill_bounds(tmp_path,monkeypatch):
    from copy import deepcopy
    monkeypatch.setattr('tools.run_q2_lab_capture.live.validate_session_plan',lambda _:[])
    reference=tmp_path/'reference.png';reference.write_bytes(b'not accessed by offline shape validation')
    plan={'flow_plan':{'campaign_profile':PROFILE,'baseline_q1_ul_min':70.,'baseline_q2_ul_min':20.,
        'steps':[],'max_cumulative_delivery_each_ul':900.,'usable_confirmed_ul':{'oil':1000.,'water':1000.},
        'q1_allowed_range_ul_min':[69.5,70.5],'q2_allowed_range_ul_min':[19.5,25.5]},
        'identification_stages':deepcopy(STAGES),'geometry_reference_image':str(reference),'reference_walls':[{},{}]}
    assert validate(plan)==[]
    for modify in [lambda p:p['flow_plan'].update(campaign_profile='arbitrary'),
                   lambda p:p['flow_plan'].update(max_cumulative_delivery_each_ul=1100),
                   lambda p:p['flow_plan']['usable_confirmed_ul'].update(oil=700),
                   lambda p:p['identification_stages'][1].update(q2_ul_min=35),
                   lambda p:p['flow_plan'].update(q2_allowed_range_ul_min=[0,100])]:
        corrupted=deepcopy(plan);modify(corrupted)
        assert validate(corrupted)
