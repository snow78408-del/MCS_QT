"""Exploratory multi-frame wall sensitivity; no detector-outcome optimisation."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.analyze_q2_v2 import read_json, write_json, sha256
from tools.replay_measurement_audit import first_per_second


def fit_wall(image: np.ndarray, seed: dict, radius: int = 8) -> tuple[dict, dict]:
    height, width = image.shape
    x0, x1 = seed['x1'] * width, seed['x2'] * width
    xs = np.arange(int(x0)+10, int(x1)-9, 5)
    expected = np.interp(xs, [x0, x1], [seed['y1']*height, seed['y2']*height])
    ys, boundary = [], 0
    for x, y in zip(xs, expected):
        lo, hi = round(y)-radius, round(y)+radius+1
        if lo < 1 or hi >= height or x < 4 or x+5 >= width:
            raise ValueError('Wall search leaves image')
        profile = image[lo:hi, x-4:x+5].mean(axis=1)
        index = int(np.argmax(profile))
        boundary += index in (0, len(profile)-1)
        ys.append(lo+index)
    ys = np.array(ys)
    keep = np.ones(len(xs), bool)
    for _ in range(4):
        if keep.sum() < len(xs)//2:
            raise ValueError('Insufficient coherent ridge points')
        slope, intercept = np.polyfit(xs[keep], ys[keep], 1)
        error = ys-(slope*xs+intercept)
        keep = np.abs(error) <= 2.5
    wall = dict(x1=seed['x1'], x2=seed['x2'], y1=float(slope*x0+intercept)/height,
                y2=float(slope*x1+intercept)/height)
    return wall, dict(retained=int(keep.sum()), total=len(xs), search_boundary_hits=boundary,
                      ridge_residual_p90=float(np.percentile(abs(error),90)))


def build(source: Path, output: Path) -> None:
    if output.exists():
        raise ValueError('Refusing to overwrite')
    original = read_json(source)
    stages = [deepcopy(s) for s in original['stages'] if s['date'] == '20261023']
    groups = {}
    for stage in stages:
        groups.setdefault(stage['block'], []).append(stage)
    output.mkdir(parents=True)
    evidence = {}
    for block, members in groups.items():
        if len(members) != 2:
            raise ValueError('Expected two stages per block')
        grid = first_per_second(members[0]['frames'])
        selected = [grid[s] for s in range(180,300,2)]
        if any(r is None for r in selected):
            raise ValueError('Missing pre-stage geometry references')
        images = [cv2.imread(r['image'],0) for r in selected]
        if any(im is None for im in images):
            raise ValueError('Unreadable original')
        stack = np.stack(images)
        composites = [np.median(stack,axis=0),np.median(stack[::2],axis=0),np.median(stack[1::2],axis=0)]
        vis=cv2.cvtColor(cv2.convertScaleAbs(composites[0],alpha=4,beta=-280),cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(output/f'{block}_temporal_median.png'),vis)
        try:
            fitted = [[fit_wall(im,w) for w in members[0]['walls']] for im in composites]
        except ValueError as error:
            evidence[block] = dict(status='rejected',reason=str(error),
                                  references=[dict(image=r['image'],t=r['t'],sha256=sha256(Path(r['image']))) for r in selected])
            continue
        walls = [p[0] for p in fitted[0]]
        differences = [[(new[k]-old[k])*540 for k in ['y1','y2']] for new,old in zip(walls,members[0]['walls'])]
        split = [[(a[0][k]-b[0][k])*540 for k in ['y1','y2']] for a,b in zip(fitted[1],fitted[2])]
        evidence[block] = dict(status='candidate',original_walls=members[0]['walls'],walls=walls,
            endpoint_shift_sensor_px=differences,split_half_endpoint_difference_px=split,
            diagnostics=[p[1] for p in fitted[0]],
            references=[dict(image=r['image'],t=r['t'],sha256=sha256(Path(r['image']))) for r in selected])
        vis=cv2.cvtColor(cv2.convertScaleAbs(composites[0],alpha=4,beta=-280),cv2.COLOR_GRAY2BGR)
        for ww,color in [(members[0]['walls'],(0,165,255)),(walls,(0,255,0))]:
            for w in ww:
                cv2.line(vis,(round(w['x1']*720),round(w['y1']*540)),(round(w['x2']*720),round(w['y2']*540)),color,1)
        cv2.imwrite(str(output/f'{block}_overlay.png'),vis)
        for stage in members:
            stage['walls']=walls
            stage['geometry_source']='pre-stage180..298s every2s temporal median bright ridge; exploratory; not human accepted'
    write_json(output/'geometry_evidence.json',dict(blocks=evidence,source_sha256=sha256(source),
                script_sha256=sha256(Path(__file__)),measurement_accepted=False,
                note='Seeded +/-8px search; split halves are correlated diagnostics, not reproducibility or truth. No post-stage frames or length outcomes used.'))
    write_json(output/'manifest.json',dict(stages=[s for s in stages if evidence[s['block']]['status']=='candidate']))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build(args.source,args.output)
