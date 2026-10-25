"""Retrospective block/date-isolated forecasts, never a controller calibration.

Transfer uses log length relative to the observed pre-stage baseline. This
removes a constant scale factor, not geometry error or changes in droplet phase.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

TIMES = np.arange(5., 300., 10.)
TAUS = (10., 20., 30., 60., 90., 120., 240.)
DELAYS = (0., 5., 15., 30., 60.)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def write(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def array(values: list) -> np.ndarray:
    return np.asarray([np.nan if v is None else v for v in values], float)


def values(stage: dict, variant: str) -> np.ndarray:
    return array([b['value'] for b in stage['bins'][variant]])


def baseline(pre: np.ndarray) -> float:
    tail = pre[-6:]
    good = np.isfinite(tail) & (tail > 0)
    if good.sum() < 4:
        raise ValueError('At least four qualified pre-stage tail bins required')
    return float(np.median(tail[good]))


def build_pairs(stages: list[dict], variant: str) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = {}
    for stage in stages:
        groups.setdefault(stage['date'] + '/' + stage['block'], []).append(stage)
    pairs, excluded = [], []
    for group, members in groups.items():
        for before, after in zip(members, members[1:]):
            uid = before['uid'] + '->' + after['uid']
            reason = before.get('exclusion') or after.get('exclusion')
            pre, post = values(before, variant), values(after, variant)
            if not reason:
                try:
                    base = baseline(pre)
                except ValueError as error:
                    reason = str(error)
            if not reason and (np.isfinite(post).sum() < 24 or np.isfinite(post[18:]).sum() < 10):
                reason = 'Post coverage below 24/30 or tail below 10/12'
            if reason:
                excluded.append(dict(uid=uid, reason=reason))
                continue
            q0 = np.array([before['q1'], before['q2']], float)
            q1 = np.array([after['q1'], after['q2']], float)
            if np.any(q0 <= 0) or np.any(q1 <= 0):
                raise ValueError('Positive flows required for log ratios')
            pairs.append(dict(uid=uid, group=group, block=before['block'], date=before['date'],
                              pre=pre, post=post, base=base, q0=q0, q1=q1))
    return pairs, excluded


def features(pair: dict, directional: bool) -> np.ndarray:
    delta = np.log(pair['q1'] / pair['q0'])
    if directional:
        if abs(delta[0]) > 1e-8:
            raise ValueError('Directional local model only supports constant Q1')
        return np.array([max(delta[1], 0), min(delta[1], 0)])
    return delta


def kernel(tau: float, delay: float) -> np.ndarray:
    return 1 - np.exp(-np.maximum(TIMES - delay, 0) / tau)


def sham_for(pair: dict, shams: dict[str, list] | None) -> np.ndarray:
    if shams is None:
        return np.zeros(30)
    key = str(int(round(pair['q0'][1])))
    if key not in shams:
        raise ValueError('No matching pre-state sham')
    return array(shams[key])


def fit_at(pairs: list[dict], tau: float, delay: float, directional: bool,
           shams: dict[str, list] | None) -> tuple[np.ndarray, float]:
    xs, ys, ws = [], [], []
    counts = Counter(p['group'] for p in pairs)
    for p in pairs:
        target = np.log(p['post'] / p['base']) - sham_for(p, shams)
        valid = np.isfinite(target)
        if not valid.any():
            raise ValueError('No training observations')
        x = kernel(tau, delay)[:, None] * features(p, directional)[None, :]
        xs.append(x[valid]); ys.append(target[valid])
        ws.append(np.full(valid.sum(), 1 / (counts[p['group']] * valid.sum())))
    x, y, w = np.vstack(xs), np.concatenate(ys), np.concatenate(ws)
    # Fixed weak ridge only stabilises poorly excited directions. No intercept:
    # an unmodelled stop/start response must not be silently called flow gain.
    coef = np.linalg.solve(x.T @ (w[:, None] * x) + 1e-6 * np.eye(2), x.T @ (w * y))
    loss = float(np.sum(w * (x @ coef - y) ** 2) / w.sum())
    return coef, loss


def fit_model(pairs: list[dict], *, directional: bool = False,
              shams: dict[str, list] | None = None) -> dict:
    if not pairs:
        raise ValueError('Training blocks required')
    grid = []
    for tau in TAUS:
        for delay in DELAYS:
            coef, loss = fit_at(pairs, tau, delay, directional, shams)
            grid.append(dict(tau=tau, delay=delay, coefficients=coef.tolist(), train_loss=loss))
    best = min(grid, key=lambda r: r['train_loss'])
    excitation = np.vstack([features(p, directional) for p in pairs])
    return dict(**best, directional=directional, shams=shams, training_ids=[p['uid'] for p in pairs],
                training_groups=sorted({p['group'] for p in pairs}), grid=grid,
                excitation_rank=int(np.linalg.matrix_rank(excitation)),
                excitation_vectors=excitation.tolist(),
                selection='fixed tau/delay grid; training blocks equally weighted; test outcomes unavailable',
                control_authorized=False)


def forecast(pre: np.ndarray, q0: np.ndarray, q1: np.ndarray,
             model: dict | None = None) -> dict[str, np.ndarray]:
    """Only inputs known at the switch are accepted; no post-stage argument."""
    base = baseline(pre)
    good = np.isfinite(pre[-6:]) & (pre[-6:] > 0)
    slope, intercept = np.polyfit((TIMES[-6:] - 300)[good], np.log(pre[-6:][good] / base), 1)
    predictions = dict(hold=np.full(30, base),
                       trend=base * np.exp(intercept + slope * TIMES),
                       damped_trend=base * np.exp(intercept + slope * 60 * (1 - np.exp(-TIMES / 60))))
    if model is not None:
        context = dict(q0=q0, q1=q1)
        effect = float(features(context, model['directional']) @ np.array(model['coefficients']))
        predictions['flow'] = base * np.exp(sham_for(context, model['shams']) +
                                            effect * kernel(model['tau'], model['delay']))
    return predictions


def score(pair: dict, predictions: dict[str, np.ndarray]) -> dict:
    mask = np.isfinite(pair['post'])
    for pred in predictions.values():
        mask &= np.isfinite(pred)
    if mask.sum() < 24 or mask[18:].sum() < 10:
        return dict(status='insufficient_common_bins', bins=int(mask.sum()))
    result = {}
    for name, pred in predictions.items():
        error = pred[mask] - pair['post'][mask]
        tail = mask & (TIMES >= 180)
        result[name] = dict(rmse_px=float(np.sqrt(np.mean(error ** 2))),
                            rmse_relative_percent=float(100 * np.sqrt(np.mean(error ** 2)) / pair['base']),
                            tail_center_error_px=float(np.median(pred[tail]) - np.median(pair['post'][tail])))
    return dict(status='scored', bins=int(mask.sum()), metrics=result)


def serial(values_: np.ndarray) -> list:
    return [float(v) if np.isfinite(v) else None for v in values_]


def record(pair: dict, model: dict, label: str) -> dict:
    if pair['group'] in model['training_groups']:
        raise ValueError('Whole-block leakage')
    pred = forecast(pair['pre'], pair['q0'], pair['q1'], model)
    excitation = np.array(model['excitation_vectors'])
    target = features(pair, model['directional'])
    projected = excitation.T @ np.linalg.lstsq(excitation.T, target, rcond=None)[0]
    result = dict(uid=pair['uid'], group=pair['group'], date=pair['date'], block=pair['block'],
                  model=label, base=pair['base'], q0=pair['q0'].tolist(), q1=pair['q1'].tolist(),
                  training_ids=model['training_ids'], predictions={k: serial(v) for k, v in pred.items()},
                  excitation_rank=model['excitation_rank'],
                  unexcited_direction_norm=float(np.linalg.norm(target - projected)),
                  actual=serial(pair['post']), score=score(pair, pred))
    # These hashes establish reproducibility after collection, not prospective timing.
    result['prediction_sha256'] = hashlib.sha256(json.dumps(result['predictions'], sort_keys=True).encode()).hexdigest()
    return result


def aggregate(records: list[dict]) -> list[dict]:
    result = []
    for date in sorted({r['date'] for r in records}):
        rr = [r for r in records if r['date'] == date and r['score']['status'] == 'scored']
        if not rr:
            continue
        groups = sorted({r['group'] for r in rr})
        means = {method: float(np.mean([np.mean([r['score']['metrics'][method]['rmse_relative_percent']
                    for r in rr if r['group'] == group]) for group in groups]))
                 for method in ('hold', 'trend', 'damped_trend', 'flow')}
        result.append(dict(date=date, blocks=len(groups), transitions=len(rr),
                           mean_block_rmse_relative_percent=means))
    return result


def evaluate(dataset: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError('Refusing to overwrite evaluation')
    data = read(dataset)
    # Geometry comparisons must not improve merely by dropping difficult bins.
    # Use the same eligible bins for all three variants, keeping the original
    # replay's per-variant availability in dataset.json for the missingness audit.
    for stage in data['stages']:
        for i in range(30):
            if any(stage['bins'][v][i]['value'] is None for v in ('main', 'minus1', 'plus1')):
                for v in ('main', 'minus1', 'plus1'):
                    stage['bins'][v][i]['value'] = None
    all_results = {}
    for variant in ('main', 'minus1', 'plus1'):
        pairs, excluded = build_pairs(data['stages'], variant)
        cross_date, models = [], {}
        dates = sorted({p['date'] for p in pairs})
        for date in dates[1:]:
            train = [p for p in pairs if p['date'] < date and np.any(p['q0'] != p['q1'])]
            test = [p for p in pairs if p['date'] == date]
            if not train:
                continue
            model = fit_model(train)
            models[date] = model
            cross_date.extend(record(p, model, 'past_dates_flow') for p in test)
        # Local 20261023 development split is fixed by the existing plan.
        current = {p['block']: p for p in pairs if p['date'] == '20261023'}
        local_records, local_models = [], {}
        if {'SA', 'SB', 'F1', 'R1', 'R2', 'F2'} <= current.keys():
            shams = {str(int(p['q0'][1])): serial(np.log(p['post'] / p['base']))
                     for p in (current['SA'], current['SB'])}
            for label, directional, background in [('local_shared', False, None),
                    ('local_sham_shared', False, shams), ('local_sham_directional', True, shams)]:
                model = fit_model([current['F1'], current['R1']], directional=directional, shams=background)
                if background is not None:
                    model['sham_training_ids'] = [current['SA']['uid'], current['SB']['uid']]
                local_models[label] = model
                local_records.extend(record(current[n], model, label) for n in ('R2', 'F2'))
        all_results[variant] = dict(pair_count=len(pairs), exclusions=excluded,
                    cross_date_models=models, cross_date=cross_date, cross_date_summary=aggregate(cross_date),
                    local_models=local_models, local=local_records)
    output.mkdir(parents=True)
    result = dict(variants=all_results, dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
                  measurement_method=data.get('method', 'See input dataset'),
                  evaluation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  measurement_accepted=False, prospective_validation=False, control_authorized=False,
                  caveats=['Retrospective development evaluation; all experiments were already collected.',
                           'All variants use common eligible bins; original availability is retained in the replay dataset.',
                           'Log ratios cancel constant magnification only, not phase or geometry changes.',
                           'Historical transitions lack matched sham controls.',
                           'Separate A/B sham curves assume transfer to subsequent change transactions.',
                           'One sham per state cannot estimate between-run uncertainty.',
                           'Damped trend uses a fixed 60s decay of pre-stage log slope; not a fitted Holt ETS.',
                           'Sampling and coverage rules are inherited from the input dataset; do not mix audit rates.'])
    write(output / 'results.json', result)
    report(result, data, output)
    return result


def report(result: dict, dataset: dict, output: Path) -> None:
    main = result['variants']['main']
    lines = ['跨领域方法在本项目的离线验证', '',
             '本报告为事后开发评估，测量未验收、没有前瞻留出资格、不能加载为控制标定。', '',
             f"纳入{len(dataset['stages'])}段原图重放，覆盖日期：" +
             '、'.join(sorted({s['date'] for s in dataset['stages']})) + '。',
             f"主几何可评分相邻转换{main['pair_count']}个，排除转换{len(main['exclusions'])}个。单段仍保留测量审计。",
             '输入测量口径：' + result['measurement_method'],
             '主几何与±1px使用共同箱；不同采样口径分别保存、分别评分。', '',
             '方法一：借鉴预测学的阻尼趋势，用切换前末60秒趋势做固定60秒衰减外推。',
             '方法二：借鉴跨域验证，将整个较晚日期隔离，历史流量模型只在更早日期拟合。',
             '方法三：使用本轮A/B同值停启对照，分别建立背景曲线，拟合共享/分方向响应。', '',
             '|测试日期|训练截止|测试块|保持值|事前直线|衰减趋势|历史流量模型|',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in main['cross_date_summary']:
        dates = [p.split('_')[0] for p in main['cross_date_models'][row['date']]['training_ids']]
        m = row['mean_block_rmse_relative_percent']
        lines.append(f"|{row['date']}|{max(dates)}|{row['blocks']}|{m['hold']:.3f}|{m['trend']:.3f}|{m['damped_trend']:.3f}|{m['flow']:.3f}|")
    lines += ['', '数值为按补液块等权的相对前段基线长度RMSE（%），越小越好；不跨日期汇总绝对像素。',
              '训练阶段缺少的流量激励方向由excitation_rank和unexcited_direction_norm标出；低秩岭解不代表该方向已辨识。', '',
              '|本轮模型|测试块|保持值RMSE(px)|事前直线|衰减趋势|模型|比保持改善|比直线改善|',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in main['local']:
        if r['score']['status'] != 'scored':
            continue
        m = r['score']['metrics']; f = m['flow']['rmse_px']
        a = 100 * (1 - f / m['hold']['rmse_px']) if m['hold']['rmse_px'] else None
        b = 100 * (1 - f / m['trend']['rmse_px']) if m['trend']['rmse_px'] else None
        improve_hold = '未定义' if a is None else f'{a:.1f}%'
        improve_trend = '未定义' if b is None else f'{b:.1f}%'
        lines.append(f"|{r['model']}|{r['block']}|{m['hold']['rmse_px']:.3f}|{m['trend']['rmse_px']:.3f}|{m['damped_trend']['rmse_px']:.3f}|{f:.3f}|{improve_hold}|{improve_trend}|")
    local_note = ('模型只在SA/SB/F1/R1范围建立，R2/F2之间不重拟合。两个测试块早已采集、看过，因此不能追认成事前预测。'
                  if main['local'] else '本轮所需六块未全部具备可评分资格；没有拟合或评分局部双向模型。旧局部预测不能沿用为本报告结果。')
    lines += ['', 'local_shared：F1/R1共享增益；local_sham_shared：加SA/SB状态各自停启项；local_sham_directional：正反各一增益。',
              local_note, '',
              '|模型/块|主几何RMSE|−1px RMSE|+1px RMSE|', '|---|---:|---:|---:|']
    for r in main['local']:
        metrics = []
        for variant in ('main', 'minus1', 'plus1'):
            match = next(x for x in result['variants'][variant]['local'] if x['model'] == r['model'] and x['uid'] == r['uid'])
            metrics.append(match['score']['metrics']['flow']['rmse_px'])
        lines.append(f"|{r['model']}/{r['block']}|" + '|'.join(f'{v:.3f}' for v in metrics) + '|')
    lines += ['', '局限与项目决定：', '',
              '- 像素长度仍是候选量测，人工物相与端点一致性未补齐；几何有回顾性估计。',
              '- 每状态仅一条停启对照，无法估计对照间方差。用对照扣除假定背景可迁移，不是因果证明。',
              '- 对数相对长度只能抵消常数倍率，不保证不同日期可共享物理增益。',
              '- 保持值和事前趋势是必须击败的基线；新增模型没有通过时保留失败结果，不以训练拟合优度推进MPC。',
              '- 所有输入、逐格测量、训练名单、预测和哈希已保存；后续可用同一工具评估新实验。', '',
              '排除记录：', '']
    lines.extend(f"- {r['uid']}：{r['reason']}" for r in main['exclusions'])
    lines += ['', '参考方法：', '',
              '- [FPP：基线与时间序列验证](https://otexts.com/fpp3/toolbox.html)',
              '- [FPP：阻尼趋势](https://otexts.com/fpp3/exponential-smoothing.html)',
              '- [scikit-learn：分组交叉验证](https://scikit-learn.org/stable/modules/cross_validation.html)',
              '- [NIST：测量过程时间变化](https://www.itl.nist.gov/div898/handbook/mpc/section1/mpc123.htm)']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.dataset, args.output)
    print(json.dumps(result['variants']['main']['cross_date_summary'], indent=2))


if __name__ == '__main__':
    main()
