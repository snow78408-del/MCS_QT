"""Real-image measurement candidate, block uncertainty and recorded shadow replay.

No hardware adapter imports. All outputs are non-authorizing development evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.vision.capsule_profile import calibrated_phase_intervals
from backend.vision.rectified_roi import rectify_channel_frame
from tools.evaluate_response_transfer import read, write, build_pairs, fit_model
from tools.replacement_pump_model_audit import q2_plan, OfflineChipSettings


def geometry_candidates(walls: list[dict], height: int) -> list[list[dict]]:
    """Predeclared sensor-pixel width/tilt perturbations; not a truth interval."""
    variants=[deepcopy(walls)]
    for kind in ('translation','width','tilt'):
        for sign in (-1,1):
            candidate=deepcopy(walls)
            for i,w in enumerate(candidate):
                for j,key in enumerate(('y1','y2')):
                    offset=sign if kind=='translation' else sign*(-1 if i==0 else 1) if kind=='width' else sign*(-1 if j==0 else 1)
                    w[key]+=offset/height
            variants.append(candidate)
    return variants


def measurement_candidate(image: np.ndarray, walls: list[dict], *, geometry_valid: bool) -> dict:
    if not geometry_valid:
        return dict(status='geometry_unconfirmed',candidates=[],accepted_for_control=False)
    if image.ndim!=2 or image.dtype!=np.uint8:raise ValueError('Mono8 required')
    variants=[]
    for ww in geometry_candidates(walls,image.shape[0]):
        if any(not 0<=w[k]<=1 for w in ww for k in ('x1','x2','y1','y2')):
            return dict(status='geometry_outside_image',candidates=[],accepted_for_control=False)
        strip=rectify_channel_frame(image,ww)
        if strip is None:return dict(status='geometry_unusable',candidates=[],accepted_for_control=False)
        diag={}
        intervals=calibrated_phase_intervals(strip,float(strip.shape[0]),diagnostics=diag)
        variants.append(dict(intervals=intervals,diagnostics=diag))
    retained=[]; rejected=[]
    for a,b in variants[0]['intervals']:
        endpoints=[(a,b)];reason=None
        for variant in variants[1:]:
            overlaps=[(c,d) for c,d in variant['intervals'] if min(b,d)>max(a,c)]
            if len(overlaps)!=1:
                reason='missing_or_ambiguous_geometry_match';break
            c,d=overlaps[0]
            # Mutual uniqueness avoids merging two main candidates into one.
            if sum(min(d,y)>max(c,x) for x,y in variants[0]['intervals'])!=1:
                reason='merged_geometry_match';break
            endpoints.append((c,d))
        spread=float(np.max(np.ptp(np.asarray(endpoints),axis=0)))
        if reason is None and spread>2:reason='endpoint_geometry_spread_gt_2_strip_px'
        row=dict(left=a,right=b,length=b-a,geometry_endpoint_spread_strip_px=spread)
        if reason:rejected.append(dict(**row,reason=reason))
        else:retained.append(row)
    return dict(status='consistent_candidates' if retained else 'no_consistent_candidate',
                candidates=retained,rejected=rejected,variants=variants,accepted_for_control=False,
                note='Algorithm consistency filter; neither physical truth nor independent repeatability')


class ReplayGate:
    """Offline quality/freshness state; never sends hardware commands."""
    def __init__(self) -> None:
        self.last_time=None; self.last_frame=None; self.stopped=False

    def check(self, stamp: float, frame: int, quality: str, transaction_ok: bool=True) -> str:
        if self.stopped:return 'stopped'
        if not transaction_ok:self.stopped=True;return 'stop_requested_transaction_failure'
        if not np.isfinite(stamp) or stamp<=0 or isinstance(frame,bool) or frame<=0:return 'hold_invalid_identity'
        if self.last_time is not None and (stamp<=self.last_time or frame<=self.last_frame):return 'hold_stale'
        self.last_time=stamp;self.last_frame=frame
        return 'eligible' if quality=='candidate_valid' else 'hold_'+quality


def uncertain_model(pairs: list[dict], repeats: int=80) -> dict:
    if not pairs:raise ValueError('No isolated Q2 steps')
    groups=sorted({p['group'] for p in pairs});rng=np.random.default_rng(20261024)
    estimates=[]
    for _ in range(repeats):
        picked=rng.choice(groups,len(groups),replace=True);sample=[]
        for i,g in enumerate(picked):
            for p in pairs:
                if p['group']==g:
                    pp=dict(p);pp['group']=f'resample_{i}';sample.append(pp)
        m=fit_model(sample);estimates.append([m['coefficients'][1],m['tau'],m['delay']])
    m=fit_model(pairs);near=[x for x in m['grid'] if x['train_loss']<=m['train_loss']*1.05+1e-15]
    return dict(model=m,block_count=len(groups),bootstrap_repetitions=repeats,
                bootstrap_percentiles_5_50_95={name:np.percentile(np.array(estimates)[:,i],[5,50,95]).tolist()
                    for i,name in enumerate(('q2_log_gain','tau_s','delay_s'))},
                near_best_training_grid=near,
                caveat='Descriptive block-resampling stability only; few blocks and unaccepted geometry preclude calibrated confidence intervals',
                observed_q1_range=[float(min(p['q0'][0] for p in pairs)),float(max(p['q1'][0] for p in pairs))],
                q2_directions=Counter('up' if p['q1'][1]>p['q0'][1] else 'down' for p in pairs),
                control_authorized=False)


def deliver(root: Path, review: Path, output: Path) -> None:
    if output.exists():raise ValueError('Refusing overwrite')
    output.mkdir(parents=True);cv2.setNumThreads(1)
    manifest=read(root/'source_manifest.json');stage_map={(s['block'],s['uid'].split('_')[-1]):s for s in manifest['stages'] if s['date']=='20261023'}
    records={};measurement_summary={};candidate_measurements={}
    import json
    for block in ('SA','SB','F1','R1','R2','F2'):
        path=review/block/'grid_main.ndjson';rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
        records[block]=rows;counts=Counter();initial=0;retained=0;candidate_measurements[block]={}
        with (output/f'{block}_measurement_candidates.ndjson').open('x',encoding='utf-8') as f:
            for row in rows:
                # Every fifth second, selected before inspecting candidate counts.
                if row['grid_slot']%25 or not row['raw_available']:continue
                st=stage_map[(block,str(row['stage']))]
                folder=Path(st['frames'][0]['image']).parent.parent
                image_path=folder/row['raw_record']['image'];blob=image_path.read_bytes()
                if hashlib.sha256(blob).hexdigest()!=row['image_sha256']:raise ValueError('Raw image changed')
                im=cv2.imdecode(np.frombuffer(blob,np.uint8),cv2.IMREAD_GRAYSCALE)
                result=measurement_candidate(im,st['walls'],geometry_valid=block not in ('R1','R2'))
                candidate_measurements[block][(row['stage'],row['grid_slot'])]=(
                    float(np.median([c['length'] for c in result['candidates']])) if result['candidates'] else None)
                counts[result['status']]+=1;initial+=len(result.get('variants',[{'intervals':[]}])[0]['intervals']);retained+=len(result['candidates'])
                f.write(json.dumps(dict(block=block,stage=row['stage'],slot=row['grid_slot'],image=str(image_path),sha256=row['image_sha256'],result=result))+'\n')
        measurement_summary[block]=dict(status_counts=dict(counts),original_candidates=initial,retained_candidates=retained)
        print(block,'measurement candidate completed',flush=True)
    data=read(root/'offline_completion_20261024/cross_date_1hz_curated.json')
    models={}
    for variant in ('main','minus1','plus1'):
        pairs,_=build_pairs(data['stages'],variant)
        isolated=[p for p in pairs if abs(p['q0'][0]-p['q1'][0])<1e-8 and abs(p['q0'][1]-p['q1'][1])>1e-8]
        models[variant]=uncertain_model(isolated)
    write(output/'q2_parameter_stability.json',models)
    # Historical data only to avoid fitting replayed current-session outcomes.
    pairs,_=build_pairs(data['stages'],'main')
    train=[p for p in pairs if p['date']<'20261023' and abs(p['q0'][0]-p['q1'][0])<1e-8 and abs(p['q0'][1]-p['q1'][1])>1e-8]
    # Remeasure historical training with the SAME candidate algorithm. Do not
    # reuse an old-measurement gain silently after changing the measurement.
    candidate_train=[];training_evidence=[]
    by_uid={s['uid']:s for s in manifest['stages']}
    for pair in train:
        observed=[];tail_count=0
        for number,uid in enumerate(pair['uid'].split('->')):
            st=by_uid[uid];sample=[]
            for second in range(0,300,5):
                frame=next((f for f in st['frames'] if second<=f['t']<second+1),None)
                values=[]
                if frame is not None:
                    path=Path(frame['image'])
                    im=np.load(path,allow_pickle=False) if path.suffix=='.npy' else cv2.imread(str(path),0)
                    result=measurement_candidate(im,st['walls'],geometry_valid=frame.get('geometry_ok',True))
                    values=[c['length'] for c in result['candidates']]
                    training_evidence.append(dict(uid=uid,time_s=second,image=str(path),result=result))
                sample.append(float(np.median(values)) if values else np.nan)
            if number==0:tail_count=int(np.isfinite(sample[-12:]).sum())
            observed.append(np.array([float(np.nanmedian(sample[i:i+2])) if np.isfinite(sample[i:i+2]).any() else np.nan for i in range(0,60,2)]))
        if tail_count>=8 and np.isfinite(observed[1]).sum()>=24 and np.isfinite(observed[1][18:]).sum()>=10:
            candidate_train.append(dict(pair,pre=observed[0],post=observed[1],base=float(np.nanmedian(observed[0][-6:]))))
    write(output/'shadow_training_remeasurement.json',training_evidence)
    model=fit_model(candidate_train) if candidate_train else None
    write(output/'shadow_model.json',dict(model=model,measurement_method='same seven-geometry consensus, fixed 5s sample; exploratory 10s bins',
        training_accepted_for_control=False,reason=None if model else 'historical candidate measurement coverage insufficient'))
    inv=read(review.parent/'completed_six_block_inventory.json')
    summary=[]
    for block,rows in records.items():
        pre=[v for (stage,slot),v in candidate_measurements[block].items() if stage==1 and slot>=1200 and v is not None]
        # Sparse candidate replay is distinct from primary 5 Hz scoring. Require
        # 8/12 fixed 5-second tail samples; never substitute the old baseline.
        base=float(np.median(pre)) if len(pre)>=8 else None
        cfg=OfflineChipSettings(q2_reference=stage_map[(block,'1')]['q2'])
        tx=[json.loads(x) for x in (Path(inv['blocks'][block]['capture'])/'commands.ndjson').read_text(encoding='utf-8').splitlines()]
        events=[e for e in tx if 'q2_ul_min' in e.get('request',{})]
        if len(events)!=1:raise ValueError('Expected one recorded change transaction')
        for scenario in ('recorded','view_drift','wall_missing','measurement_outage','stale_frame','transaction_failure'):
            gate=ReplayGate();stats=Counter();elapsed=[];previous=None
            with (output/f'{block}_{scenario}_shadow.ndjson').open('x',encoding='utf-8') as f:
                for row in rows:
                    if row['stage']!=2:continue
                    slot=row['grid_slot'];raw=row.get('raw_record');quality='candidate_valid'
                    measured=candidate_measurements[block].get((2,slot))
                    if block in ('R1','R2'):quality='geometry_unconfirmed'
                    elif base is None:quality='insufficient_candidate_baseline'
                    elif slot%25:quality='not_remeasured_at_this_slot'
                    elif measured is None:quality='no_consistent_candidate'
                    stamp=raw['host_monotonic_s'] if raw else -1;frame=raw['hardware_frame_id'] if raw else -1
                    if 300<=slot<400:
                        if scenario in ('view_drift','wall_missing','measurement_outage'):quality=scenario
                        if scenario=='stale_frame' and previous:stamp,frame=previous
                    transaction_ok=bool(events[0].get('ok')) and not(scenario=='transaction_failure' and slot>=300)
                    state=gate.check(stamp,frame,quality,transaction_ok);stats[state]+=1
                    if state=='eligible':previous=(stamp,frame)
                    proposal=None;duration=None
                    # Recommendations every 10 s; recorded flow is never replaced by a recommendation.
                    if slot%50==0 and state=='eligible' and model is not None:
                        start=time.perf_counter();proposal=q2_plan(stage_map[(block,'2')]['q2'],np.log(measured/base),0.,
                            model['coefficients'][1],model['tau'],model['delay'],settings=cfg)
                        duration=time.perf_counter()-start;elapsed.append(duration)
                    f.write(json.dumps(dict(slot=slot,source_monotonic_s=raw['host_monotonic_s'] if raw else None,
                        replay_monotonic_s=stamp,hardware_frame_id=frame,quality_state=state,
                        recorded_q1=70.,recorded_q2=stage_map[(block,'2')]['q2'],recommendation=proposal,
                        solve_seconds=duration,hardware_command_sent=False))+'\n')
            summary.append(dict(block=block,scenario=scenario,quality_counts=dict(stats),recommendations=len(elapsed),
                                candidate_baseline_px=base,baseline_candidate_samples=len(pre),
                                solve_seconds_p50_p95=np.percentile(elapsed,[50,95]).tolist() if elapsed else None))
    write(output/'delivery.json',dict(measurement=measurement_summary,shadow=summary,
        model_training_ids=model['training_ids'] if model else [],measurement_accepted=False,control_authorized=False,
        retrospective=True,hardware_requested=False,
        limitations=['Measurement candidate uses seven predefined width/tilt/translation geometries; not independent human truth.',
            'Known missing walls reject measurement rather than inventing coordinates.',
            'Shadow uses recorded real frame identities/times and command log; injected faults are synthetic.',
            'All measurements remain exploratory; candidate_valid means software eligibility only.',
            'Candidate measurements recomputed every 5 seconds, tail baseline needs 8/12; separate from primary 5Hz scoring.',
            'Recommendations are never applied to recorded future observations; no counterfactual closed-loop score.',
            'Prototype proposes every 10 seconds for diagnostics; not an authorized control scheduling frequency.']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();deliver(a.root,a.review,a.output)
