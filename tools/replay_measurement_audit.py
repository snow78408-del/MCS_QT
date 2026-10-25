"""Offline common-rate replay of saved images; no acquisition imports."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.analyze_q2_v2 import make_pipeline, measure_image, read_json, sha256, verify_contract, write_json


def first_per_second(frames: list[dict]) -> list[dict | None]:
    cells: list[dict | None] = [None] * 300
    previous = -math.inf
    for row in frames:
        t = row['t']
        if not math.isfinite(t) or t < previous:
            raise ValueError('Nonfinite or unordered stage time')
        previous = t
        if 0 <= t < 300 and cells[int(t)] is None:
            cells[int(t)] = row
    return cells


def audit_bins(rows: list[dict]) -> list[dict]:
    if len(rows) != 300:
        raise ValueError('Expected 300 cells')
    bins = []
    for i in range(30):
        group = rows[i * 10:(i + 1) * 10]
        values = [r['length'] for r in group if r['length'] is not None]
        raw = sum(r['raw_available'] for r in group)
        bins.append(dict(t=i * 10 + 5, raw=raw, candidates=len(values),
                         value=float(np.median(values)) if raw >= 8 and len(values) >= 3 else None))
    return bins


def replay(manifest: Path, contract: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError('Refusing to overwrite output')
    source_hashes = verify_contract(read_json(contract))
    description = read_json(manifest)
    stages = description['stages']
    if len({s['uid'] for s in stages}) != len(stages):
        raise ValueError('Duplicate stages')
    output.mkdir(parents=True)
    provenance = dict(manifest_sha256=sha256(manifest), contract_sha256=sha256(contract),
                      source_hashes=source_hashes, replay_source_sha256=sha256(Path(__file__)))
    write_json(output / 'provenance.json', provenance)
    measured = []
    cv2.setNumThreads(1)
    for stage in stages:
        metadata = {k: v for k, v in stage.items() if k != 'frames'}
        rows = {v: [] for v in ('main', 'minus1', 'plus1')}
        pipes = {v: make_pipeline() for v in rows}
        for second, record in enumerate(first_per_second(stage['frames'])):
            im = None
            fact = dict(second=second, length=None, raw_available=False, count=0,
                        physical_state='unreviewed', source=record)
            if record is not None:
                path = Path(record['image'])
                if not path.is_file():
                    raise FileNotFoundError(path)
                im = np.load(path, allow_pickle=False) if path.suffix == '.npy' else cv2.imread(str(path), 0)
                if im is None or im.ndim != 2 or im.dtype != np.uint8:
                    raise ValueError(f'Invalid Mono8 original: {path}')
                fact.update(raw_available=True, image_sha256=sha256(path))
            for variant, shift in [('main', 0), ('minus1', -1), ('plus1', 1)]:
                row = dict(fact)
                if im is not None and record.get('geometry_ok', True):
                    walls = [dict(w, y1=w['y1'] + shift / im.shape[0],
                                  y2=w['y2'] + shift / im.shape[0]) for w in stage['walls']]
                    if any(not 0 <= w[k] <= 1 for w in walls for k in ('x1', 'y1', 'x2', 'y2')):
                        raise ValueError('Wall perturbation outside image')
                    values = measure_image(pipes[variant], im, walls, record['t'])
                    row.update(length=float(np.median(values)) if values else None,
                               count=len(values), complete_candidate_lengths=values)
                rows[variant].append(row)
        pack = dict(**metadata, rows=rows, bins={v: audit_bins(r) for v, r in rows.items()})
        write_json(output / (stage['uid'] + '.json'), pack)
        measured.append({k: v for k, v in pack.items() if k != 'rows'})
        print(stage['uid'], [sum(b['value'] is not None for b in pack['bins'][v]) for v in rows], flush=True)
    result = dict(stages=measured, provenance=provenance, measurement_accepted=False,
                  method='Common 1 Hz first original; 10s bins >=8 raw cells and >=3 complete-candidate cells; sensitivity audit, not replacement for registered 5 Hz primary',
                  control_authorized=False, hardware_requested=False)
    write_json(output / 'dataset.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    replay(args.manifest, args.contract, args.output)


if __name__ == '__main__':
    main()
