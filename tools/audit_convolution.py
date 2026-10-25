"""Offline horizontal convolution ablation; never changes production measurement."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.vision.capsule_profile import calibrated_phase_intervals
from backend.vision.rectified_roi import rectify_channel_frame
from tools.replay_measurement_audit import first_per_second

SIGMAS = (0.0, 0.7, 1.4, 2.8)


def convolve_horizontal(image: np.ndarray, sigma: float) -> np.ndarray:
    """Symmetric normalized kernel, axial only; preserve transverse geometry."""
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError('Mono8 required')
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError('Nonnegative finite sigma required')
    if sigma == 0:
        return image.copy()
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel.reshape(1, -1), borderType=cv2.BORDER_REFLECT_101)


def measure(image: np.ndarray, sigma: float) -> list[tuple[int, int]]:
    # Candidate-only common rule, without temporal detector history or physical-unit inference.
    return calibrated_phase_intervals(convolve_horizontal(image, sigma), float(image.shape[0] - 1))


def match_intervals(predicted: list, truth: list, threshold: float = .3) -> list[tuple[int, int]]:
    """One-to-one matching, maximizing admissible match count before IoU."""
    if not predicted or not truth:
        return []
    iou = np.zeros((len(predicted), len(truth)))
    for i, (a, b) in enumerate(predicted):
        for j, (c, d) in enumerate(truth):
            iou[i, j] = max(0, min(b, d) - max(a, c)) / max(1, max(b, d) - min(a, c))
    valid = iou > threshold
    weight = valid * (min(iou.shape) + 1 + iou)
    ii, jj = linear_sum_assignment(-weight)
    return [(int(i), int(j)) for i, j in zip(ii, jj) if valid[i, j]]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(labels: Path, manifest: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError('Refusing overwrite')
    output.mkdir(parents=True)
    cv2.setNumThreads(1)
    records = json.loads(labels.read_text(encoding='utf-8'))
    annotated = []
    for record in records:
        path = labels.parent / (record['id'] + '.png')
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f'Unreadable image: {path}')
        truth = [(d['left'], d['right']) for d in record['droplets']]
        annotated.append(dict(id=record['id'], sha256=digest(path), truth=truth,
                              negative=record['negative_frame'],
                              predictions={str(s): measure(image, s) for s in SIGMAS}))
    metrics = {}
    for sigma in SIGMAS:
        count = Counter(); endpoint_errors = []; length_errors = []
        for row in annotated:
            pred = row['predictions'][str(sigma)]; truth = row['truth']
            matches = match_intervals(pred, truth)
            count.update(truth=len(truth), predictions=len(pred), matched=len(matches),
                         unmatched_predictions=len(pred) - len(matches), missed=len(truth) - len(matches),
                         negative_frames=int(row['negative']),
                         negative_frames_with_candidates=int(row['negative'] and bool(pred)))
            for i, j in matches:
                a, b = pred[i]; c, d = truth[j]
                endpoint_errors.extend([abs(a-c), abs(b-d)])
                length_errors.append(abs((b-a) - (d-c)) / (d-c))
        metrics[str(sigma)] = dict(count,
            endpoint_absolute_error_p90_px=float(np.percentile(endpoint_errors, 90)) if endpoint_errors else None,
            relative_length_error_median=float(np.median(length_errors)) if length_errors else None)
    source = json.loads(manifest.read_text(encoding='utf-8'))
    stage_results = []; sampled = []
    for stage in source['stages']:
        counts = {str(s): Counter() for s in SIGMAS}
        for second, frame in enumerate(first_per_second(stage['frames'])):
            if second % 30:
                continue
            known_missing_wall = stage['date'] == '20261023' and stage['block'] in ('R1', 'R2')
            row = dict(uid=stage['uid'], second=second, geometry_exclusion=known_missing_wall,
                       source=frame, predictions={})
            if frame is None or known_missing_wall or not frame.get('geometry_ok', True):
                row['status'] = 'missing_raw' if frame is None else 'geometry_unconfirmed'
            else:
                path = Path(frame['image'])
                image = np.load(path, allow_pickle=False) if path.suffix == '.npy' else cv2.imread(str(path), 0)
                if image is None:
                    raise ValueError(f'Unreadable image: {path}')
                strip = rectify_channel_frame(image, stage['walls'])
                if strip is None:
                    row['status'] = 'geometry_unusable'
                else:
                    row.update(status='candidate_only', sha256=digest(path),
                               predictions={str(s): measure(strip, s) for s in SIGMAS})
            for sigma in SIGMAS:
                tally = counts[str(sigma)]
                tally[row['status']] += 1
                pred = row['predictions'].get(str(sigma), [])
                tally['candidates'] += len(pred)
                tally['frames_with_candidates'] += bool(pred)
                baseline = row['predictions'].get('0.0', [])
                tally['count_changed_vs_baseline'] += len(pred) != len(baseline)
            sampled.append(row)
        stage_results.append(dict(uid=stage['uid'], date=stage['date'], block=stage['block'],
                                  walls=stage['walls'], source_exclusion=stage.get('exclusion'), metrics=counts))
    result = dict(sigmas_strip_px=SIGMAS, calibration=metrics, stages=stage_results,
        measurement_accepted=False, control_authorized=False, hardware_requested=False,
        provenance=dict(manifest_sha256=digest(manifest), labels_sha256=digest(labels),
            audit_source_sha256=digest(Path(__file__)),
            phase_source_sha256=digest(Path(__file__).resolve().parents[1]/'backend/vision/capsule_profile.py')),
        limitations=[
            'Extra axial Gaussian convolution before the existing phase rule, which already smooths its signal.',
            '60 annotated calibration crops are not independent validation. Errors are conditional on matching; misses are separate.',
            'All dates use fixed 30s first-frame sampling, not 5Hz primary scoring; unlabelled candidate counts are not accuracy.',
            'Per-stage geometry retained; known R1/R2 missing-wall observations cannot be restored by convolution.',
            'No CNN training, no automatic production parameter selection, no physical-unit inference.'])
    for name, value in [('summary.json', result), ('calibration_rows.json', annotated), ('sampled_rows.json', sampled)]:
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.labels, args.manifest, args.output)
    print(json.dumps(result['calibration'], ensure_ascii=False, indent=2))
