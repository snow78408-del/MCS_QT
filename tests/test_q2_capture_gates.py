from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_q2_lab_capture import session
from tools.plant_flow_transient_capture import CaptureLifecycleError
from tools.q2_capture_gates import validate_gates
from tools.q2_prediction_gate import (BASELINE_METHOD, SOURCE_FILES, ROOT, atomic_json,
    object_hash, predict, load_bundle_for_plan, sha256, run as run_worker, baseline_from_snapshot)


def gated_session(tmp_path, prediction=False):
    s,calls=session(tmp_path)
    s.plan.update(session_id='validation_1',study_id='study_test',
        apparatus=dict(chip_id='chip_test',camera_unique_id='test_camera'),
        capture_plan=dict(requested_exposure_us=80.,requested_gain=10.),
        reference_walls=[dict(x1=.1,x2=.9,y1=.3,y2=.3),dict(x1=.1,x2=.9,y1=.4,y2=.4)],
        stage_gates=dict(schema_version=1,role='validation' if prediction else 'training',
            readiness=dict(mode='operator',remind_after_s=1.,maximum_wait_s=3.),
            prediction=dict(required=prediction,timeout_s=2.)))
    reference=tmp_path/'reference.png';reference.write_bytes(b'fake reference for pure gate tests')
    s.plan['geometry_reference_image']=str(reference)
    bundle=dict(schema_version=1,baseline_method=BASELINE_METHOD,study_id='study_test',chip_id='chip_test',
        q1_fixed=70,q2_range=[20,25],training_block_ids=['train_1'],primary_model='first_order',
        measurement_source_hashes={name:sha256(ROOT/name) for name in SOURCE_FILES},
        baseline_worker_sha256=sha256(ROOT/'tools/q2_prediction_gate.py'),
        reference_image_sha256=sha256(reference),reference_walls_sha256=object_hash(s.plan['reference_walls']),
        camera_settings=dict(camera_unique_id='test_camera',exposure_us=80.,gain=10.),
        models=dict(first_order=dict(kind='fopdt',gain=-.3,tau_s=60.,delay_s=5.),
                    fir=dict(kind='fir',knots_s=[0,30,120,360],coefficients=[-.1,-.1,-.1])))
    path=tmp_path/'bundle.json';atomic_json(path,bundle)
    s.plan['stage_gates']['prediction'].update(bundle_path=str(path),bundle_sha256=sha256(path))
    (tmp_path/'registered_geometry.json').write_text(json.dumps(dict(walls=s.plan['reference_walls'])))
    (tmp_path/'frame_manifest.ndjson').write_text('{}\n')
    return s,calls,bundle


def confirm_after(s, delay=.5, invalid=False):
    original=s._wait_segment
    def wait(seconds):
        original(seconds)
        path=s.output_dir/'readiness_request.json'
        if path.exists() and not (s.output_dir/'readiness_confirmation.json').exists() and s._clock()>=delay:
            request=json.loads(path.read_text())
            atomic_json(s.output_dir/'readiness_confirmation.json',dict(
                session_id=s.plan['session_id'],nonce='old' if invalid else request['nonce'],
                confirmed_monotonic=s._clock(),operator='onsite operator',
                both_walls_visible=True,complete_droplet_endpoints_visible=True))
        if (s.output_dir/'STOP').exists():
            raise CaptureLifecycleError('Operator STOP file received')
    s._wait_segment=wait


def worker_result(s, bins=None):
    request_path=s.output_dir/'prediction_request.json';request=json.loads(request_path.read_text())
    result=predict(s._frozen_bundle,bins if bins is not None else [80.]*6,request['q0'],request['q1'])
    result.update(session_id=s.plan['session_id'],bundle_sha256=s._bundle_hash,request_sha256=sha256(request_path))
    atomic_json(s.output_dir/'prediction_candidate.json',result)
    return SimpleNamespace(poll=lambda:0,returncode=0)


def test_operator_start_is_delayed_but_full_stages_and_budget_preserved(tmp_path):
    s,calls,_=gated_session(tmp_path);confirm_after(s,1.5)
    s._run_segments()
    assert calls==[(70.,25.)]
    assert s.segment_records[0]['observation_started_monotonic']==1.5
    assert s.segment_records[1]['observation_started_monotonic']==305.5
    assert s._delivery_ul[0]==pytest.approx(70.5*605.5/60)
    assert 'operator_readiness_reminder_budget_rechecked' in s.events
    assert json.loads((tmp_path/'readiness_request.json').read_text())['status']=='closed'


@pytest.mark.parametrize('fault', ['timeout','stale','reserve','stop','pause'])
def test_readiness_failure_never_starts_formal_stage_or_switch(tmp_path,fault):
    s,calls,_=gated_session(tmp_path)
    if fault=='stale':confirm_after(s,invalid=True)
    if fault=='reserve':s._delivery_ul[0]=200
    if fault in ('stop','pause'):
        confirm_after(s);(tmp_path/('STOP' if fault=='stop' else 'PAUSE')).touch()
    with pytest.raises(CaptureLifecycleError):s._run_segments()
    assert not calls and not s.segment_records


def test_prediction_is_sealed_before_transaction_and_second_stage_does_not_wait_ready(tmp_path):
    s,calls,_=gated_session(tmp_path,True);confirm_after(s)
    s._launch_prediction_worker=lambda:worker_result(s)
    update=s.pump.update_flow_while_running
    def checked_update(*flows):
        seal=json.loads((tmp_path/'prediction_seal.json').read_text())
        assert seal['prediction_sha256']==sha256(tmp_path/'prospective_prediction.json')
        assert not (tmp_path/'prediction_observation_binding.json').exists()
        return update(*flows)
    s.pump.update_flow_while_running=checked_update
    s._run_segments()
    assert calls==[(70.,25.)] and s.run_finished
    assert (tmp_path/'prediction_observation_binding.json').exists()


@pytest.mark.parametrize('fault',['exit','timeout','stop','pause','interrupt','disk','tamper'])
def test_prediction_failure_never_commands_target_and_reaps_worker(tmp_path,fault,monkeypatch):
    s,calls,_=gated_session(tmp_path,True);confirm_after(s)
    state=dict(running=True,terminated=False)
    def terminate():state.update(running=False,terminated=True)
    worker=SimpleNamespace(poll=lambda:None if state['running'] else 0,returncode=1,terminate=terminate,wait=lambda **kw:None)
    def launch():
        if fault=='exit':state['running']=False;return worker
        if fault in ('stop','pause'):(tmp_path/('STOP' if fault=='stop' else 'PAUSE')).touch()
        if fault=='interrupt':
            def interrupted(seconds):raise KeyboardInterrupt()
            s._wait_segment=interrupted
        if fault in ('disk','tamper'):
            result=worker_result(s)
            if fault=='tamper':
                data=json.loads((tmp_path/'prediction_candidate.json').read_text());data['curves']['hold'][0]=123.
                atomic_json(tmp_path/'prediction_candidate.json',data)
            else:
                original=atomic_json
                def fail(path,payload):
                    if path.name=='prospective_prediction.json':raise OSError('disk failure')
                    original(path,payload)
                monkeypatch.setattr('tools.q2_capture_gates.atomic_json',fail)
            return result
        return worker
    s._launch_prediction_worker=launch
    with pytest.raises((CaptureLifecycleError,KeyboardInterrupt,OSError)):s._run_segments()
    assert not calls and len(s.segment_records)==1
    if fault in ('timeout','stop','pause','interrupt'):assert state['terminated']


def test_preflight_rejects_changed_model_measurement_chip_and_validation_bypass(tmp_path):
    s,_,bundle=gated_session(tmp_path,True)
    assert validate_gates(s.plan)==[]
    for change in (lambda p:p['apparatus'].update(chip_id='other'),
                   lambda p:p['stage_gates']['prediction'].update(required=False),
                   lambda p:p['capture_plan'].update(requested_exposure_us=100)):
        bad=deepcopy(s.plan);change(bad);assert validate_gates(bad)
    bundle['models']['first_order']['gain']=1
    atomic_json(tmp_path/'bundle.json',bundle)
    assert validate_gates(s.plan)


def test_prediction_uses_frozen_parameters_and_baseline_not_any_post_outcome(tmp_path):
    s,_,bundle=gated_session(tmp_path,True)
    with pytest.raises(ValueError,match='Insufficient baseline'):predict(bundle,[80.,None,None,80.,80.,80.],[70,20],[70,25])
    result=predict(bundle,[80.]*6,[70,20],[70,25])
    assert result['curves']['hold']==[80.]*30
    assert result['curves']['first_order'][-1]<80
    reverse=predict(bundle,[80.]*6,[70,25],[70,20])
    assert reverse['curves']['first_order'][-1]>80


def test_worker_selects_first_raw_in_each_cell_and_never_reads_post_stage(tmp_path,monkeypatch):
    import cv2
    image=tmp_path/'image.png';cv2.imwrite(str(image),np.zeros((32,300),np.uint8))
    records=[dict(host_monotonic_s=340+i*.2+.01,hardware_frame_id=i+1,image='image.png') for i in range(300)]
    records.append(dict(host_monotonic_s=400.01,hardware_frame_id=301,image='missing_post_image.png'))
    blob=('\n'.join(json.dumps(r) for r in records)+'\n').encode()
    (tmp_path/'prediction_manifest_snapshot.ndjson').write_bytes(blob)
    monkeypatch.setattr('tools.q2_prediction_gate.make_pipeline',lambda:object())
    monkeypatch.setattr('tools.q2_prediction_gate.measure_image',lambda *args:[80.])
    request=dict(baseline_stage=dict(observation_started_monotonic=100.),registered_walls=[],
                 manifest_sha256=hashlib.sha256(blob).hexdigest())
    bins,images=baseline_from_snapshot(tmp_path,request)
    assert bins==[80.]*6 and len(images)==300


@pytest.mark.parametrize('fault',['readiness_timeout','prediction_failure','interrupt'])
def test_gate_failures_use_existing_finally_verified_stop_path(tmp_path,fault):
    s,calls,_=gated_session(tmp_path,prediction=fault!='readiness_timeout')
    if fault!='readiness_timeout':confirm_after(s)
    stopped=[];connected=[]
    s._prepare_provenance=s._prepare_stage_gates
    s._acquire_locks=lambda:None
    s._connect_and_verify_initial_state=lambda:connected.append(True)
    s._configure_camera_and_read_back=lambda:None
    s._begin_capture_and_save_baseline=lambda:None
    s._start_reader=lambda:None
    s._write_pump_parameters=lambda:None
    def start():s.pump_may_be_running=True
    s._start_infusion=start
    def stop():
        stopped.append(True);s.pump_may_be_running=False
        return SimpleNamespace(verified=True)
    s._stop_pump_with_retries=stop
    s._teardown_camera=lambda:None;s._disconnect_pump=lambda:None;s._close_sink=lambda:None
    s.report=lambda:dict(failed=bool(s.original_error));s._finalize_outputs=lambda payload:None
    def launch():
        if fault=='interrupt':raise KeyboardInterrupt()
        return SimpleNamespace(poll=lambda:1,returncode=1)
    s._launch_prediction_worker=launch
    result=s.run()
    assert result['failed'] and connected and stopped==[True] and not calls
    assert not s.pump_may_be_running


def test_bad_frozen_model_fails_before_connecting_pump(tmp_path):
    s,calls,_=gated_session(tmp_path,True); connected=[]
    s._connect_and_verify_initial_state=lambda:connected.append(True)
    s._cleanup_campaign=lambda:None;s._close_sink=lambda:None
    s.report=lambda:dict(failed=bool(s.original_error));s._finalize_outputs=lambda _:None
    (tmp_path/'bundle.json').write_text('{}')
    assert s.run()['failed'] and not connected and not calls


def test_present_null_gates_are_not_silently_treated_as_legacy():
    assert validate_gates({'stage_gates':None})


@pytest.mark.parametrize('gates',[None,{'readiness':None,'prediction':None}])
def test_malformed_gate_preflight_reports_error_without_hardware(tmp_path,monkeypatch,capsys,gates):
    from tools import run_q2_lab_capture as capture
    plan=tmp_path/'plan.json'
    atomic_json(plan,dict(stage_gates=gates,capture_plan=dict(
        output_directory=str(tmp_path/'capture'),required_free_space_bytes=0)))
    monkeypatch.setattr(capture,'validate',validate_gates)
    monkeypatch.setattr(capture.sys,'argv',['capture','--plan',str(plan)])
    assert capture.main()==2
    result=json.loads(capsys.readouterr().out)
    assert result['preflight_gaps'] and result['hardware_requested'] is False


def test_acknowledgement_helper_requires_active_current_request_and_explicit_flags(tmp_path,monkeypatch):
    from tools.q2_gate_control import acknowledge
    atomic_json(tmp_path/'readiness_request.json',dict(session_id='a',nonce='fresh',status='waiting',
        started_monotonic=100.,deadline_monotonic=110.))
    monkeypatch.setattr('tools.q2_gate_control.time.monotonic',lambda:105.)
    with pytest.raises(ValueError):
        acknowledge(tmp_path,'operator',both_walls_visible=False,complete_droplet_endpoints_visible=True)
    acknowledge(tmp_path,'operator',both_walls_visible=True,complete_droplet_endpoints_visible=True)
    assert json.loads((tmp_path/'readiness_confirmation.json').read_text())['nonce']=='fresh'
    monkeypatch.setattr('tools.q2_gate_control.time.monotonic',lambda:111.)
    with pytest.raises(ValueError):
        acknowledge(tmp_path,'operator',both_walls_visible=True,complete_droplet_endpoints_visible=True)


def test_freezer_packages_matching_parameters_and_refuses_overwrite_or_wrong_baseline(tmp_path):
    from tools.freeze_q2_prediction_bundle import freeze
    s,_,bundle=gated_session(tmp_path,True)
    plan=tmp_path/'plan.json';candidate=tmp_path/'candidate.json';output=tmp_path/'frozen.json'
    atomic_json(plan,s.plan)
    atomic_json(candidate,{k:bundle[k] for k in ('baseline_method','models','primary_model','training_block_ids')})
    digest=freeze(plan,candidate,output)
    s.plan['stage_gates']['prediction'].update(bundle_path=str(output),bundle_sha256=digest)
    assert load_bundle_for_plan(s.plan)[1]['models']==bundle['models']
    with pytest.raises(ValueError,match='overwrite'):freeze(plan,candidate,output)
    bad=json.loads(candidate.read_text());bad['baseline_method']='old_measurement'
    atomic_json(candidate,bad)
    with pytest.raises(ValueError,match='same'):freeze(plan,candidate,tmp_path/'another.json')
