from copy import deepcopy

import numpy as np
import pytest

from tools.convolution_endpoints import propose, supported_originals
from tools.convolution_geometry import FrozenWallMonitor, QualityGate
from tools.convolution_dynamics import convolve_events, step_basis, predictions


def wall_scene(shift=0, tilt=0):
    y,x=np.mgrid[:160,:360]
    image=np.full((160,360),60.)
    for centre in (55,105):
        image+=90*np.exp(-((y-centre-shift-tilt*(x-180))/2)**2)
    walls=[dict(x1=.1,x2=.9,y1=c/160,y2=c/160) for c in (55,105)]
    return image.astype(np.uint8),walls


def test_frozen_monitor_detects_drift_tilt_missing_without_rebasing():
    original,walls=wall_scene(); monitor=FrozenWallMonitor(original,walls)
    assert monitor.check(original)['state']=='stable_candidate'
    shifted,_=wall_scene(shift=8)
    assert monitor.check(shifted)['state']=='view_drift'
    assert monitor.check(shifted)['state']=='view_drift'
    tilted,_=wall_scene(tilt=.04)
    assert monitor.check(tilted)['state']=='wall_tilt_changed'
    flat=np.full_like(original,60)
    assert monitor.check(flat)['state']=='wall_unresolved'
    assert monitor.check(original)['state']=='stable_candidate'


def test_no_wall_reference_does_not_become_valid():
    image,walls=wall_scene()
    monitor=FrozenWallMonitor(np.full_like(image,60),walls)
    assert not monitor.check(image)['eligible']


def test_replay_gate_stale_outage_and_recovery():
    gate=QualityGate()
    assert gate.check(1,1,'stable_candidate')=='candidate_only'
    assert gate.check(1,1,'stable_candidate')=='hold_stale'
    assert gate.check(2,2,'missing_raw')=='hold_invalid'
    assert gate.check(3,3,'wall_unresolved')=='hold_and_alert_persistent_invalid'
    assert gate.check(4,4,'stable_candidate')=='candidate_only'


def test_derivative_proposals_are_not_outer_edge_replacements():
    signal=np.full(360,-24.);signal[100:180]=24
    image=np.full((32,360),128.);image[2:8]-=signal/2;image[24:30]+=signal/2
    candidates=propose(image.astype(np.uint8),2)
    assert len(candidates)==1
    assert abs(candidates[0][0]-100)<=2 and abs(candidates[0][1]-180)<=2
    assert supported_originals([(90,190)],candidates)==[(90,190)]
    assert supported_originals([],candidates)==[]
    assert propose(np.full((32,360),128,np.uint8),2)==[]


def test_temporal_convolution_is_causal_and_superposes_return_step():
    times=np.array([-1,0,5,100,500.]);coef=np.arange(1.,7.)
    up=dict(time_s=0,log_q2_change=.2); down=dict(time_s=100,log_q2_change=-.2)
    first=convolve_events(times,[up],coef); second=convolve_events(times,[down],coef)
    np.testing.assert_allclose(convolve_events(times,[up,down],coef),first+second)
    assert first[0]==first[1]==0 and second[2]==0
    assert (first+second)[-1]==pytest.approx(0)
    assert np.all(step_basis(np.array([-10.,0.]))==0)
    with pytest.raises(ValueError):
        convolve_events(times,[down,up],coef)


def test_model_selection_does_not_read_held_post_values():
    def pair(group,gain):
        return dict(group=group,uid=group,date='day',q0=np.array([70.,20.]),q1=np.array([70.,25.]),
            pre=np.full(30,80.),post=80*np.exp(gain*np.log(1.25)*(1-np.exp(-np.arange(5,300,10)/60))),base=80.)
    train=[pair('a',-.3),pair('b',-.4)];held=pair('c',-.35)
    before,model=predictions(held,train,{},[],False)
    changed=deepcopy(held);changed['post'][:]=99999
    after,model_after=predictions(changed,train,{},[],False)
    for name in before:
        np.testing.assert_array_equal(before[name],after[name])
    assert model==model_after
    with pytest.raises(ValueError,match='leakage'):
        predictions(held,train+[held],{},[],False)
