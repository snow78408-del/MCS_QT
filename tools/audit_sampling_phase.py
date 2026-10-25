"""Measure systematic downsampling phase sensitivity without changing raw data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def phase_audit(rows: list[dict]) -> list[dict]:
    keys = {(r['stage'], r['grid_slot']) for r in rows}
    if len(rows) != 3000 or keys != {(s, k) for s in (1, 2) for k in range(1500)}:
        raise ValueError('Expected both complete 5Hz time grids, including missing rows')
    result = []
    for offset in range(5):
        stages = []
        for stage in (1, 2):
            grid = {r['grid_slot']: r for r in rows if r['stage'] == stage}
            bins = []
            for box in range(30):
                group = [grid[k] for k in range(box * 50 + offset, (box + 1) * 50, 5)]
                lengths = [r['frame_median_strip_px'] for r in group
                           if r['frame_median_strip_px'] is not None]
                bins.append(float(np.median(lengths)) if
                            sum(r['raw_available'] for r in group) >= 8 and len(lengths) >= 3 else None)
            stages.append(bins)
        y = np.array([np.nan if v is None else v for v in stages[0][-6:]])
        valid = np.isfinite(y)
        slope = float(np.polyfit(np.arange(-55., 0., 10.)[valid], y[valid], 1)[0]) if valid.sum() >= 4 else None
        result.append(dict(offset_s=offset / 5, bins=stages, pre_last60_slope_px_per_s=slope))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite output')
    blocks, sources = {}, {}
    for path in sorted(args.review_dir.glob('*/grid_main.ndjson')):
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        blocks[path.parent.name] = phase_audit(rows)
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not blocks:
        raise ValueError('No stored 5Hz grids found')
    payload = dict(blocks=blocks, source_sha256=sources,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   method='Five systematic 1Hz subgrids from identical 5Hz measurements; no interpolation; no independent-repeat claim',
                   measurement_accepted=False, control_authorized=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
