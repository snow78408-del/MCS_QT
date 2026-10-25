import numpy as np
import pytest
from tools.replacement_pump_model_audit import q2_plan, simulate, OfflineChipSettings


@pytest.mark.parametrize('gain',[-.3,.3])
@pytest.mark.parametrize('target',[-.1,.1])
def test_q1_is_locked_through_entire_plan_and_q2_bounded(gain,target):
    r=q2_plan(22.5,0,target,gain,30,5)
    assert r['status']=='optimized'
    p=np.asarray(r['plan']);assert np.all(p[:,0]==70)
    assert np.all(p[:,1]>=20-1e-7) and np.all(p[:,1]<=25+1e-7)
    assert np.max(abs(np.diff(np.r_[22.5,p[:,1]])))<=.5+1e-7


def test_invalid_measurement_holds_and_bad_numeric_rejected():
    assert q2_plan(22.5,0,.01,.3,30,5,valid=False)['plan']==[[70.,22.5]]*10
    with pytest.raises(ValueError):q2_plan(22.5,float('nan'),0,.3,30,5)


def test_outage_holds_and_transaction_failure_ends_simulation():
    s=simulate(.3,30,5,'measurement_outage')
    assert all(r['status']=='hold' for r in s['steps'][6:10])
    assert len({r['q2'] for r in s['steps'][5:10]})==1
    s=simulate(.3,30,5,'transaction_failure')
    assert len(s['steps'])==7 and s['steps'][-1]['status']=='verified_stop_simulated'


def test_other_chip_preserves_explicit_dimensions_and_flow_envelope():
    cfg=OfflineChipSettings(chip_id='synthetic-other',q1_fixed=60,q2_min=10,q2_max=18,
        q2_reference=14,max_q2_step=.2,channel_width_um=80,channel_depth_um=40,scale_um_per_px=.8)
    r=q2_plan(14,0,.1,.3,30,5,settings=cfg)
    p=np.array(r['plan']);assert np.all(p[:,0]==60)
    assert np.all(p[:,1]>=10-1e-7) and np.all(p[:,1]<=18+1e-7)
    assert max(abs(np.diff(np.r_[14,p[:,1]])))<=.2+1e-7
    assert cfg.channel_depth_um==40


def test_unknown_dimensions_remain_unknown():
    cfg=OfflineChipSettings()
    assert cfg.channel_width_um is None and cfg.scale_um_per_px is None
    with pytest.raises(ValueError):OfflineChipSettings(channel_depth_um=-1)
    with pytest.raises(ValueError):OfflineChipSettings(q1_fixed=40)
