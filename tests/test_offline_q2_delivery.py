import numpy as np
import pytest
from tools.offline_q2_delivery import ReplayGate, geometry_candidates, measurement_candidate, uncertain_model
from test_response_transfer_audit import pair


def test_missing_wall_never_runs_candidate_measurement():
    r=measurement_candidate(np.zeros((540,720),np.uint8),[],geometry_valid=False)
    assert r['status']=='geometry_unconfirmed' and not r['candidates']


def test_width_and_tilt_are_separate_from_translation():
    walls=[dict(x1=.1,x2=.9,y1=.3,y2=.3),dict(x1=.1,x2=.9,y1=.4,y2=.4)]
    vv=geometry_candidates(walls,100)
    assert len(vv)==7 and walls[0]['y1']==.3
    assert vv[3][0]['y1']-walls[0]['y1']==pytest.approx(-(vv[3][1]['y1']-walls[1]['y1']))
    assert vv[5][0]['y1']-walls[0]['y1']==pytest.approx(-(vv[5][0]['y2']-walls[0]['y2']))


def test_stale_geometry_loss_and_latched_failure():
    g=ReplayGate();assert g.check(10,100,'candidate_valid')=='eligible'
    assert g.check(10,100,'candidate_valid')=='hold_stale'
    assert g.check(11,101,'wall_missing')=='hold_wall_missing'
    assert g.check(12,102,'candidate_valid',False)=='stop_requested_transaction_failure'
    assert g.check(13,103,'candidate_valid')=='stopped'


def test_resampling_records_scope_and_never_authorizes():
    pp=[pair('a',.2),pair('b',.15),pair('c',.25)]
    r=uncertain_model(pp,repeats=5)
    assert r['block_count']==3 and r['q2_directions']=={'up':3}
    assert r['control_authorized'] is False
    assert r['bootstrap_percentiles_5_50_95']['q2_log_gain'][1]==pytest.approx(-.3,abs=.001)


def test_consensus_rejects_endpoint_instability_without_erasing_original(monkeypatch):
    monkeypatch.setattr('tools.offline_q2_delivery.rectify_channel_frame',lambda *a: np.zeros((30,300),np.uint8))
    answers=iter([[(40,100)],[(44,100)]]+[[(40,100)]]*5)
    monkeypatch.setattr('tools.offline_q2_delivery.calibrated_phase_intervals',lambda *a,**kw:next(answers))
    walls=[dict(x1=.1,x2=.9,y1=.3,y2=.3),dict(x1=.1,x2=.9,y1=.4,y2=.4)]
    r=measurement_candidate(np.zeros((540,720),np.uint8),walls,geometry_valid=True)
    assert r['candidates']==[]
    assert r['rejected'][0]['reason']=='endpoint_geometry_spread_gt_2_strip_px'
    assert r['variants'][0]['intervals']==[(40,100)]


def test_consensus_is_candidate_only_even_if_all_variants_agree(monkeypatch):
    monkeypatch.setattr('tools.offline_q2_delivery.rectify_channel_frame',lambda *a:np.zeros((30,300),np.uint8))
    monkeypatch.setattr('tools.offline_q2_delivery.calibrated_phase_intervals',lambda *a,**kw:[(40,100)])
    walls=[dict(x1=.1,x2=.9,y1=.3,y2=.3),dict(x1=.1,x2=.9,y1=.4,y2=.4)]
    r=measurement_candidate(np.zeros((540,720),np.uint8),walls,geometry_valid=True)
    assert len(r['candidates'])==1 and r['accepted_for_control'] is False
