"""All registered replacement-pump dates: retrospective folds and offline Q2 MPC.

Pixel/log-relative model only. No hardware imports, calibration export or claim
that simulated counterfactual actions were observed in recorded experiments.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_response_transfer import read, write, build_pairs, fit_model, record


@dataclass(frozen=True)
class OfflineChipSettings:
    """Explicit per-chip design settings; dimensions are not inferred from pixels."""
    chip_id: str = 'offline_candidate_not_field_confirmed'
    q1_fixed: float = 70.
    q2_min: float = 20.
    q2_max: float = 25.
    q2_reference: float = 22.5
    max_q2_step: float = .5
    channel_width_um: float | None = None
    channel_depth_um: float | None = None
    scale_um_per_px: float | None = None

    def __post_init__(self) -> None:
        if not self.chip_id.strip():raise ValueError('Chip identity required')
        if not np.isfinite([self.q1_fixed,self.q2_min,self.q2_max,self.q2_reference,self.max_q2_step]).all():
            raise ValueError('Finite chip settings required')
        if not 0 < self.q2_min < self.q2_max or not self.q2_min <= self.q2_reference <= self.q2_max or self.max_q2_step<=0:
            raise ValueError('Invalid chip flow envelope')
        if self.q1_fixed<=2*self.q2_max:raise ValueError('Chip envelope violates Q1 > 2 Q2')
        for x in (self.channel_width_um,self.channel_depth_um,self.scale_um_per_px):
            if x is not None and (not np.isfinite(x) or x<=0):raise ValueError('Invalid physical dimension')


def q2_plan(current_q2: float, current_log_length: float, target_log_length: float,
            gain: float, tau: float, delay: float, period: float = 30.,
            valid: bool = True, settings: OfflineChipSettings | None = None) -> dict:
    """Separate offline prototype; Q1 is locked in every planned move."""
    cfg=settings or OfflineChipSettings()
    scalars = [current_q2, current_log_length, target_log_length, gain, tau, delay, period]
    if not np.isfinite(scalars).all() or not cfg.q2_min <= current_q2 <= cfg.q2_max or tau <= 0 or delay < 0 or period <= 0:
        raise ValueError('Invalid offline state/model')
    if not valid or abs(gain) < 1e-8:
        return dict(status='hold', plan=[[cfg.q1_fixed, current_q2]]*10, reason='invalid measurement or no authority')
    n=10; cumulative=np.tril(np.ones((n,n))); times=np.arange(1,n+1)*period
    response=np.zeros((n,n))
    for i,t in enumerate(times):
        for j in range(i+1):
            response[i,j]=(1-np.exp(-max(t-j*period-delay,0)/tau))*gain/current_q2
    nominal=gain*np.log(current_q2/cfg.q2_reference)
    free=nominal+(current_log_length-nominal)*np.exp(-times/tau)
    matrix=response/.02;error=(free-target_log_length)/.02
    h=matrix.T@matrix+.1*np.eye(n);linear=matrix.T@error
    result=minimize(lambda x:float(x@h@x+2*linear@x),np.zeros(n),jac=lambda x:2*(h@x+linear),
        method='SLSQP',bounds=Bounds(-cfg.max_q2_step,cfg.max_q2_step),constraints=LinearConstraint(cumulative,cfg.q2_min-current_q2,cfg.q2_max-current_q2),
        options={'maxiter':100,'ftol':1e-9})
    q=current_q2+cumulative@result.x
    if not result.success or not np.isfinite(q).all() or np.any(q<cfg.q2_min-1e-7) or np.any(q>cfg.q2_max+1e-7) or np.any(abs(result.x)>cfg.max_q2_step+1e-7):
        return dict(status='hold',plan=[[cfg.q1_fixed,current_q2]]*n,reason='solver failure or infeasible plan')
    return dict(status='optimized',plan=[[cfg.q1_fixed,float(v)] for v in q],prediction_log=(free+response@result.x).tolist())


def simulate(gain: float, tau: float, delay: float, scenario: str) -> dict:
    # Slow response-window prototype, explicit pure input delay in simulated plant.
    period=max(30.,delay+3*tau); q=22.5;y=0.; history=[(0.,q)];steps=[]
    actual_gain=gain*(.5 if scenario=='half_gain' else -1 if scenario=='sign_reversal' else 1)
    actual_tau=tau*(2 if scenario=='slow_plant' else 1)
    for k in range(24):
        target=.015 if k<12 else -.015
        valid=not (scenario=='measurement_outage' and 6<=k<10)
        sensed=y+(.03 if scenario=='unflagged_bias' and k>=6 else 0)
        proposal=q2_plan(q,sensed,target,gain,tau,delay,period,valid)
        new=proposal['plan'][0][1]
        if scenario=='transaction_failure' and k==6:
            steps.append(dict(step=k,status='verified_stop_simulated',q1=70.,q2=q,y=y,target=target));break
        q=new;history.append((k*period,q))
        # Integration step <=1 s; no actual plant or hardware interaction.
        dt=period/np.ceil(period)
        for j in range(int(np.ceil(period))):
            t=k*period+(j+1)*dt-delay
            delayed=next((v for when,v in reversed(history) if when<=t),22.5)
            equilibrium=actual_gain*np.log(delayed/22.5)
            y=equilibrium+(y-equilibrium)*np.exp(-dt/actual_tau)
        steps.append(dict(step=k,status=proposal['status'],q1=70.,q2=q,y=y,target=target,valid=valid))
    return dict(scenario=scenario,period_s=period,steps=steps,
                rmse_log=float(np.sqrt(np.mean([(s['y']-s['target'])**2 for s in steps]))),
                hold_rmse_log=.015,source='synthetic plant; not recorded experimental response')


def audit(dataset: Path, output: Path) -> None:
    if output.exists():raise ValueError('Refusing to overwrite')
    data=read(dataset);output.mkdir(parents=True)
    # Keep common geometry coverage for all fits and tests.
    for s in data['stages']:
        for i in range(30):
            if any(s['bins'][v][i]['value'] is None for v in ('main','minus1','plus1')):
                for v in ('main','minus1','plus1'):s['bins'][v][i]['value']=None
    variants={}
    for v in ('main','minus1','plus1'):
        pairs,excluded=build_pairs(data['stages'],v)
        changed=[p for p in pairs if np.any(abs(p['q1']-p['q0'])>1e-8)]
        folds=[]
        for p in changed:
            train=[x for x in changed if x['group']!=p['group']]
            model=fit_model(train)
            folds.append(record(p,model,'leave_whole_refill_block_out'))
        dayfolds=[]
        for day in sorted({p['date'] for p in changed}):
            train=[p for p in changed if p['date']!=day]
            model=fit_model(train)
            dayfolds.extend(record(p,model,'leave_whole_date_out') for p in changed if p['date']==day)
        local_models={};local_folds=[]
        for day in sorted({p['date'] for p in changed}):
            members=[p for p in changed if p['date']==day]
            local_models[day]=fit_model(members)
            for p in members:
                train=[x for x in members if x['group']!=p['group']]
                if train:local_folds.append(record(p,fit_model(train),'same_date_leave_block_out'))
        model=fit_model(changed)
        variants[v]=dict(model=model,block_folds=folds,date_folds=dayfolds,excluded=excluded,
            date_local_models=local_models,date_local_block_folds=local_folds,
            inventory=[dict(uid=p['uid'],date=p['date'],block=p['block'],q_before=p['q0'].tolist(),q_after=p['q1'].tolist(),
                q2_direction='up' if p['q1'][1]>p['q0'][1] else 'down' if p['q1'][1]<p['q0'][1] else 'constant',
                isolated_q2=bool(abs(p['q1'][0]-p['q0'][0])<1e-8)) for p in pairs])
    m=variants['main']['model'];g=m['coefficients'][1]
    sims=[simulate(g,m['tau'],m['delay'],s) for s in ('nominal','half_gain','slow_plant','sign_reversal','measurement_outage','unflagged_bias','transaction_failure')]
    result=dict(variants=variants,simulations=sims,dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        measurement_accepted=False,control_authorized=False,prospective_validation=False,
        caveats=['All registered post-replacement dates included; exclusions are segment evidence, not pump/date exclusions.',
                 'Leave-date-out may train on later dates; retrospective transfer diagnostic, not chronological forecast.',
                 'Historical joint Q1/Q2 steps do not constitute isolated Q2 reverse experiments.',
                 'Candidates remain measured with unaccepted geometry; no physical gain truth claimed.',
                 'Offline Q2 prototype is not the production diameter controller; no unit substitution or authorization bypass.',
                 'No sham subtraction for historical blocks without matched controls.'])
    write(output/'results.json',result)
    lines=['# 换泵后全范围建模与离线Q2 MPC','', '所有已登记日期纳入；既有气泡、端点争议、覆盖缺失和下壁不可见按段保留。原测量尚未验收，以下为开发结果。', '',
           '|留出方式|转换数|同时优于保持/趋势20%的转换数|','|---|---:|---:|']
    for name,key in [('全范围整补液块','block_folds'),('整日期','date_folds'),('同日期整补液块','date_local_block_folds')]:
        rr=variants['main'][key];scored=[r for r in rr if r['score']['status']=='scored']
        wins=sum(r['score']['metrics']['flow']['rmse_px']<=.8*min(r['score']['metrics'][b]['rmse_px'] for b in ('hold','trend')) for r in scored)
        lines.append(f'|{name}|{len(scored)}|{wins}|')
    lines+=['','|转换|Q2方向|Q1固定|整块留出保持RMSE(px)|模型RMSE(px)|','|---|---|---|---:|---:|']
    for r in variants['main']['block_folds']:
        p=next(p for p in variants['main']['inventory'] if p['uid']==r['uid']);s=r['score']
        if s['status']=='scored':lines.append(f"|{r['uid']}|{p['q2_direction']}|{p['isolated_q2']}|{s['metrics']['hold']['rmse_px']:.3f}|{s['metrics']['flow']['rmse_px']:.3f}|")
    lines+=['','|日期局部模型|激励秩|Q1候选系数|Q2候选系数|τ(s)|延迟(s)|','|---|---:|---:|---:|---:|---:|']
    for day,local in variants['main']['date_local_models'].items():
        lines.append(f"|{day}|{local['excitation_rank']}|{local['coefficients'][0]:.4f}|{local['coefficients'][1]:.4f}|{local['tau']}|{local['delay']}|")
    lines+=['','日期不等于芯片身份；局部模型保留不同参数，仅作开发比较。秩1模型未激励的方向不可解释为已识别增益；正式使用需芯片/几何/配方绑定。',
            '',f"全范围候选Q2对数增益={g:.5f}，τ={m['tau']}秒，延迟={m['delay']}秒；训练拟合仅用于构造离线候选，不等于验证。",'',
            '|仿真场景|log输出RMSE|保持基线|','|---|---:|---:|']
    lines += [f"|{s['scenario']}|{s['rmse_log']:.5f}|{s['hold_rmse_log']:.5f}|" for s in sims]
    lines+=['','仿真每步Q1固定70，Q2限制20–25，单次≤0.5；不是实机运行。事务失败场景以核验停止结束，截断分数不可与全程分数直接排名。模型反号、未标记测量偏差场景用于暴露失败，不作稳健保证。', '',
            '各方向数据清单、全部折训练名单、三种几何、预测和失败场景逐步日志见results.json。没有将已看过的数据追认为事前验证；没有启用MPC或写出可加载标定。']
    (output/'报告.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();audit(a.dataset,a.output)
