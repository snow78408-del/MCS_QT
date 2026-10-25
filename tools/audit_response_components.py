"""Offline ablation of saved local predictions; never refits on test outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_response_transfer import TIMES, array, kernel, read, serial, write


def components(model: dict, base: float, q0: list, q1: list) -> dict[str, np.ndarray]:
    if base <= 0 or not np.isfinite(base):
        raise ValueError('Positive baseline required')
    delta = np.log(np.asarray(q1) / np.asarray(q0))
    if model['directional']:
        if abs(delta[0]) > 1e-8:
            raise ValueError('Directional model requires fixed Q1')
        delta = np.array([max(delta[1], 0), min(delta[1], 0)])
    flow = float(delta @ np.asarray(model['coefficients'])) * kernel(model['tau'], model['delay'])
    sham = np.zeros(30) if model['shams'] is None else array(model['shams'][str(int(round(q0[1])))])
    return dict(hold=np.full(30, base), sham_only=base * np.exp(sham),
                flow_only=base * np.exp(flow), full=base * np.exp(sham + flow))


def residuals(actual: np.ndarray, predictions: dict[str, np.ndarray]) -> dict:
    mask = np.isfinite(actual)
    for values in predictions.values():
        mask &= np.isfinite(values)
    result = {}
    for name, pred in predictions.items():
        windows = {}
        for label, start, end in [('all', 0, 300), ('startup', 0, 60), ('middle', 60, 180), ('tail', 180, 300)]:
            valid = mask & (TIMES >= start) & (TIMES < end)
            e = pred[valid] - actual[valid]
            bias = float(np.mean(e)) if len(e) else None
            windows[label] = dict(n=int(len(e)), rmse=float(np.sqrt(np.mean(e * e))) if len(e) else None,
                                 bias=bias, centered_rmse=float(np.sqrt(np.mean((e-bias)**2))) if len(e) else None)
        result[name] = windows
    return result


def audit(source: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError('Refusing to overwrite')
    saved = read(source)
    records = []
    for variant, data in saved['variants'].items():
        for row in data['local']:
            model = data['local_models'][row['model']]
            if row['group'] in model['training_groups']:
                raise ValueError('Training/test overlap')
            pred = components(model, row['base'], row['q0'], row['q1'])
            if not np.allclose(pred['full'], array(row['predictions']['flow']), equal_nan=True):
                raise ValueError('Saved prediction does not reproduce')
            pred['pre_trend'] = array(row['predictions']['trend'])
            actual = array(row['actual'])
            # Select plausible alternatives using TRAINING loss only. This is
            # a descriptive tolerance set, not a confidence interval.
            near = [g for g in model['grid'] if g['train_loss'] <= model['train_loss'] * 1.05]
            alternatives = []
            for grid in near:
                alternative = dict(model, **grid)
                candidate = components(alternative, row['base'], row['q0'], row['q1'])['full']
                error = residuals(actual, {'candidate': candidate, 'hold': pred['hold']})
                alternatives.append(dict(tau=grid['tau'], delay=grid['delay'],
                                         train_loss=grid['train_loss'], rmse=error['candidate']['all']['rmse'],
                                         beats_hold=error['candidate']['all']['rmse'] < error['hold']['all']['rmse']))
            records.append(dict(variant=variant, block=row['block'], model=row['model'],
                                predictions={k: serial(v) for k, v in pred.items()}, actual=serial(actual),
                                errors=residuals(actual, pred),
                                within_5_percent_training_loss=alternatives,
                                note='flow_only removes sham without refitting; centered error is diagnostic, not an available corrected prediction'))
    output.mkdir(parents=True)
    result = dict(records=records, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  measurement_accepted=False, control_authorized=False, prospective_validation=False)
    write(output/'components.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.source, args.output)
    for r in result['records']:
        if r['variant'] == 'main':
            print(r['model'], r['block'], {k: round(v['all']['rmse'], 3) for k, v in r['errors'].items()})


if __name__ == '__main__':
    main()
