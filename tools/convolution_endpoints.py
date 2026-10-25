"""Multiscale derivative proposals: offline only; weak outer edges remain authoritative."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_convolution import digest, match_intervals, measure
from backend.vision.rectified_roi import rectify_channel_frame


def propose(image: np.ndarray, sigma: float) -> list[tuple[int, int]]:
    if image.ndim != 2 or image.dtype != np.uint8 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Mono8 and positive finite sigma required')
    h, w = image.shape
    if h < 18 or w < max(100, h*3):
        return []
    upper = image[max(1, int(h*.08)):int(h*.25)].mean(axis=0)
    lower = image[int(h*.76):int(h*.94)].mean(axis=0)
    signal = lower - upper
    if np.ptp(signal) < 15:
        return []
    gradient = gaussian_filter1d(signal, sigma, order=1, mode='reflect')
    noise = np.median(abs(gradient - np.median(gradient)))
    prominence = max(.5, 2*noise)
    rises, _ = find_peaks(gradient, prominence=prominence, height=prominence, distance=max(2, h//3))
    falls, _ = find_peaks(-gradient, prominence=prominence, height=prominence, distance=max(2, h//3))
    proposals = []; used = set()
    for left in rises:
        choices = [int(right) for right in falls if h-1 < right-left < 4*(h-1) and right not in used]
        if choices:
            right = choices[0]
            # Do not bridge a separate rising interface to reach a later falling edge.
            if any(left < r < right for r in rises):
                continue
            proposals.append((int(left), right)); used.add(right)
    return proposals


def supported_originals(original: list, proposals: list) -> list:
    """Use proposals only as support; never replace weak outer edges by strong inner gradients."""
    matches = match_intervals(proposals, original)
    return [original[j] for i, j in matches
            if original[j][0] <= proposals[i][0] < proposals[i][1] <= original[j][1]]


def compare(labels: Path, manifest: Path, samples: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError('Refusing overwrite')
    output.mkdir(parents=True); cv2.setNumThreads(1)
    modes = ('original', 'gradient_1', 'gradient_2', 'gradient_4', 'supported_original')
    rows = []
    annotations = json.loads(labels.read_text(encoding='utf8'))
    stages = {s['uid']: s for s in json.loads(manifest.read_text(encoding='utf8'))['stages']}
    pool = [('labelled', r['id'], labels.parent/(r['id']+'.png'), None, r) for r in annotations]
    pool += [('unlabelled', r['uid']+':'+str(r['second']), Path(r['source']['image']),
              stages[r['uid']]['walls'], None)
             for r in json.loads(samples.read_text(encoding='utf8')) if r['status']=='candidate_only']
    for kind, uid, path, walls, annotation in pool:
        im = np.load(path, allow_pickle=False) if path.suffix=='.npy' else cv2.imread(str(path), 0)
        if im is None:
            raise ValueError(f'Unreadable image: {path}')
        strip = rectify_channel_frame(im, walls) if walls else im
        if strip is None:
            raise ValueError('Previously usable geometry no longer rectifies')
        predictions = {'original': measure(strip, 0)}
        for sigma in (1, 2, 4):
            predictions[f'gradient_{sigma}'] = propose(strip, sigma)
        predictions['supported_original'] = supported_originals(predictions['original'], predictions['gradient_2'])
        rows.append(dict(kind=kind, uid=uid, image=str(path), sha256=digest(path), predictions=predictions,
                         truth=[(d['left'], d['right']) for d in annotation['droplets']] if annotation else None))
    metrics = {}
    for mode in modes:
        matched=missed=unmatched=negative_false=0; errors=[]; lengths=[]
        for row in rows:
            if row['truth'] is None:
                continue
            pred=row['predictions'][mode]; truth=row['truth']; pairs=match_intervals(pred, truth)
            matched+=len(pairs); missed+=len(truth)-len(pairs); unmatched+=len(pred)-len(pairs)
            negative_false+=int(not truth and bool(pred))
            for i,j in pairs:
                a,b=pred[i]; c,d=truth[j]
                errors.extend([abs(a-c),abs(b-d)]); lengths.append(abs(b-a-d+c)/(d-c))
        metrics[mode]=dict(matched=matched,missed=missed,unmatched=unmatched,negative_false_frames=negative_false,
            endpoint_p90=float(np.percentile(errors,90)) if errors else None,
            length_relative_median=float(np.median(lengths)) if lengths else None,
            unlabelled_candidates=sum(len(r['predictions'][mode]) for r in rows if r['truth'] is None))
    result=dict(metrics=metrics,labelled_frames=len(annotations),unlabelled_frames=len(pool)-len(annotations),
        source_sha256={str(p):digest(p) for p in (labels,manifest,samples,Path(__file__))},
        control_authorized=False, measurement_accepted=False,
        limitations=['Calibration-set comparison, not independent accuracy.',
            'Gradient proposals locate strong interfaces, not necessarily weak outer endpoints.',
            'Supported originals retain existing full phase checks and coordinates; support can reduce recall.',
            'Unknown walls cannot be reconstructed; no production or hardware integration.'])
    for name,value in [('summary.json',result),('rows.json',rows)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('labels','manifest','samples','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); print(json.dumps(compare(a.labels,a.manifest,a.samples,a.output)['metrics'],indent=2))
