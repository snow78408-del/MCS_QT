"""Causal Q2 finite-impulse-response comparison using isolated recorded transitions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_convolution import digest
from tools.evaluate_response_transfer import (TIMES, TAUS, DELAYS, build_pairs, forecast, read, score, serial, write)

KNOTS=np.array([0.,10.,30.,60.,120.,240.,360.])
RIDGES=(.0001,.001,.01,.1)


def step_basis(time: np.ndarray) -> np.ndarray:
    """Integral of piecewise constant impulse bins, zero before an event."""
    time=np.asarray(time,float)
    if not np.isfinite(time).all():
        raise ValueError('Finite times required')
    return np.clip((time[...,None]-KNOTS[:-1])/np.diff(KNOTS),0,1)


def convolve_events(times: np.ndarray, events: list[dict], coefficients: np.ndarray) -> np.ndarray:
    coef=np.asarray(coefficients,float)
    if coef.shape!=(len(KNOTS)-1,) or not np.isfinite(coef).all():
        raise ValueError('Finite impulse coefficients required')
    effect=np.zeros_like(np.asarray(times,float))
    previous=-np.inf
    for event in events:
        stamp=float(event['time_s']); delta=float(event['log_q2_change'])
        if not np.isfinite([stamp,delta]).all() or stamp<previous:
            raise ValueError('Finite ordered input events required')
        effect+=delta*(step_basis(np.asarray(times)-stamp)@coef); previous=stamp
    return effect


def context_offset(pair: dict, timing: dict) -> float:
    return timing.get(pair['group'],{}).get('readback_relative_to_observation_s',0.)


def sham_curve(pair: dict, shams: list[dict], enabled: bool) -> np.ndarray:
    if not enabled:
        return np.zeros(30)
    matches=[s for s in shams if s['date']==pair['date'] and np.allclose(s['q0'],pair['q0'])]
    if not matches:
        return np.zeros(30)
    curves=np.asarray([np.log(s['post']/s['base']) for s in matches])
    # All bins must be present: no extrapolated or silently zero-filled sham correction.
    if not np.isfinite(curves).all():
        raise ValueError('Incomplete sham curve; cannot subtract unobserved response')
    return np.mean(curves,axis=0)


def fit_fir(pairs: list[dict], timing: dict, shams: list[dict], use_sham: bool, ridge: float) -> np.ndarray:
    xx=[]; yy=[]; ww=[]
    counts={p['group']:sum(q['group']==p['group'] for q in pairs) for p in pairs}
    for p in pairs:
        y=np.log(p['post']/p['base'])-sham_curve(p,shams,use_sham); mask=np.isfinite(y)
        x=step_basis(TIMES-context_offset(p,timing))*np.log(p['q1'][1]/p['q0'][1])
        xx.append(x[mask]); yy.append(y[mask]); ww.extend([1/(counts[p['group']]*mask.sum())]*mask.sum())
    x=np.vstack(xx); y=np.concatenate(yy); w=np.asarray(ww)
    return np.linalg.solve(x.T@(w[:,None]*x)+ridge*np.eye(x.shape[1]),x.T@(w*y))


def choose_ridge(pairs: list[dict], timing: dict, shams: list[dict], use_sham: bool) -> tuple[float,list]:
    groups=sorted({p['group'] for p in pairs}); losses=[]
    if len(groups)<2:
        return .01,[]
    for ridge in RIDGES:
        group_losses=[]
        for group in groups:
            training=[p for p in pairs if p['group']!=group]
            coef=fit_fir(training,timing,shams,use_sham,ridge)
            errors=[]
            for p in pairs:
                if p['group']!=group:
                    continue
                pred=(step_basis(TIMES-context_offset(p,timing))@coef)*np.log(p['q1'][1]/p['q0'][1])+sham_curve(p,shams,use_sham)
                target=np.log(p['post']/p['base']); mask=np.isfinite(target)
                errors.append(float(np.mean((pred[mask]-target[mask])**2)))
            group_losses.append(float(np.mean(errors)))
        losses.append(dict(ridge=ridge,inner_block_loss=float(np.mean(group_losses))))
    return min(losses,key=lambda r:r['inner_block_loss'])['ridge'],losses


def fit_first_order(pairs: list[dict], timing: dict, shams: list[dict], use_sham: bool) -> dict:
    best=None
    for tau in TAUS:
        for delay in DELAYS:
            xs=[];ys=[];weights=[]
            for p in pairs:
                y=np.log(p['post']/p['base'])-sham_curve(p,shams,use_sham); valid=np.isfinite(y)
                x=(1-np.exp(-np.maximum(TIMES-context_offset(p,timing)-delay,0)/tau))*np.log(p['q1'][1]/p['q0'][1])
                xs.extend(x[valid]);ys.extend(y[valid]);weights.extend([1/valid.sum()]*valid.sum())
            x=np.asarray(xs);y=np.asarray(ys);w=np.asarray(weights)
            gain=float(np.sum(w*x*y)/(np.sum(w*x*x)+1e-6));loss=float(np.sum(w*(gain*x-y)**2)/w.sum())
            candidate=dict(tau=tau,delay=delay,gain=gain,training_loss=loss)
            if best is None or loss<best['training_loss']:
                best=candidate
    return best


def predictions(pair: dict, train: list[dict], timing: dict, shams: list[dict], use_sham: bool) -> tuple[dict,dict]:
    if any(p['group']==pair['group'] for p in train):
        raise ValueError('Whole-block leakage')
    ridge,inner=choose_ridge(train,timing,shams,use_sham)
    coef=fit_fir(train,timing,shams,use_sham,ridge); fopdt=fit_first_order(train,timing,shams,use_sham)
    offset=context_offset(pair,timing); dq=float(np.log(pair['q1'][1]/pair['q0'][1]))
    event=dict(time_s=offset,log_q2_change=dq)
    sham=sham_curve(pair,shams,use_sham)
    pred=forecast(pair['pre'],pair['q0'],pair['q1'])
    pred['fir']=pair['base']*np.exp(sham+convolve_events(TIMES,[event],coef))
    pred['first_order']=pair['base']*np.exp(sham+fopdt['gain']*dq*(1-np.exp(-np.maximum(TIMES-offset-fopdt['delay'],0)/fopdt['tau'])))
    meta=dict(coefficients=coef.tolist(),ridge=ridge,inner_selection=inner,first_order=fopdt,
        training_ids=[p['uid'] for p in train],training_groups=sorted({p['group'] for p in train}),
        sham_ids=[s['uid'] for s in shams] if use_sham else [],timing=timing.get(pair['group']),
        event=event,stop_start_adjustment=use_sham,
        matched_sham_available=bool([s for s in shams if s['date']==pair['date'] and np.allclose(s['q0'],pair['q0'])]))
    return pred,meta


def run(dataset: Path, primary: Path, inventory: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError('Refusing overwrite')
    output.mkdir(parents=True)
    timing={}
    for block,entry in read(inventory)['blocks'].items():
        summary_path=Path(entry['capture'])/'session_summary.json'; summary=read(summary_path)
        segment=summary['segments'][1]; origin=segment['observation_started_monotonic']
        timing['20261023/'+block]=dict(readback_relative_to_observation_s=segment['command_readback_monotonic']-origin,
            request_relative_to_observation_s=segment['command_started_monotonic']-origin,
            source_sha256=digest(summary_path),timing_basis='command readback proxy, not measured chip-flow onset')
    data=read(dataset); primary_data=read(primary); records=[]; excluded={}
    for variant in ('main','minus1','plus1'):
        pairs,excluded[variant]=build_pairs(data['stages'],variant)
        local=[p for p in pairs if np.allclose(p['q0'][0],p['q1'][0]) and abs(p['q0'][1]-p['q1'][1])>1e-8]
        shams=[p for p in pairs if np.allclose(p['q0'],p['q1']) and np.isfinite(p['post']).all()]
        for held in local:
            for split in ('leave_block','leave_date'):
                train=[p for p in local if p['group']!=held['group'] and (split!='leave_date' or p['date']!=held['date'])]
                if not train:
                    continue
                for use_sham in (False,True):
                    # A held-out date supplies no training/calibration outcomes, including its shams.
                    available_shams=[s for s in shams if split!='leave_date' or s['date']!=held['date']]
                    pred,model=predictions(held,train,timing,available_shams,use_sham)
                    records.append(dict(uid=held['uid'],variant=variant,split=split,rate='1Hz development',
                        model=model,score=score(held,pred),actual=serial(held['post']),base=held['base'],predictions={k:serial(v) for k,v in pred.items()}))
        primary_pairs,_=build_pairs(primary_data['stages'],variant)
        historical=[p for p in local if p['date']<'20261023']
        for held in primary_pairs:
            if not historical or held['block'] not in ('F1','F2'):
                continue
            pred,model=predictions(held,historical,timing,[],False)
            records.append(dict(uid=held['uid'],variant=variant,split='historical_only',rate='5Hz primary bins retrospective',
                model=model,score=score(held,pred),actual=serial(held['post']),base=held['base'],predictions={k:serial(v) for k,v in pred.items()}))
    for record in records:
        record['prediction_sha256']=hashlib.sha256(json.dumps(record['predictions'],sort_keys=True).encode()).hexdigest()
        frozen={k:record['model'][k] for k in ('coefficients','ridge','first_order','training_ids','training_groups','sham_ids','stop_start_adjustment')}
        frozen.update(knots_s=KNOTS.tolist(),output='log relative strip-pixel length',control_authorized=False)
        record['frozen_model']=frozen
        record['model_sha256']=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
    result=dict(records=records,excluded=excluded,knots_seconds=KNOTS.tolist(),timing=timing,
        input_sha256={str(p):digest(p) for p in (dataset,primary,inventory,Path(__file__))},
        measurement_accepted=False,control_authorized=False,hardware_requested=False,
        limitations=['Only isolated Q2 changes used; no inference from joint Q1/Q2 transitions.',
            'Actual command readback relative to observation used when logged; older stages use observation zero, explicitly unknown onset.',
            'Stop/start curve is an optional same-date, same-initial-state empirical diagnostic, never an unbiased causal subtraction.',
            'Only upward isolated transitions pass existing eligibility; reverse response remains unsupported.',
            'General event superposition tested synthetically; experimental records contain one isolated Q2 change per evaluated block.',
            'Unknown pre-recording input history is not reconstructed; model anchors to pre-stage tail baseline.',
            'Whole-block/date exclusions and inner ridge selection prevent fitting held outcomes; all evaluations remain retrospective.',
            'Historical 1Hz model fitting and current 5Hz primary-bin scoring are reported separately.'])
    write(output/'summary.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','primary','inventory','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); r=run(a.dataset,a.primary,a.inventory,a.output);print('Scored records:',len(r['records']))
