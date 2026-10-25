"""Validate human review and describe repeat agreement, never authorize control."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

KINDS = {'complete_droplet', 'gap', 'filament', 'bubble', 'truncated', 'uncertain'}


def points(value: object) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError('Exactly two endpoints required')
    if any(not isinstance(p, list) or len(p) != 2 for p in value):
        raise ValueError('Each endpoint must have x/y')
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
           for p in value for x in p):
        raise ValueError('Finite numeric coordinates required')
    arr = np.array(value, float)
    if np.any(arr < 0) or np.any(arr[:, 0] > 719) or np.any(arr[:, 1] > 539):
        raise ValueError('Coordinates outside original 720x540 image')
    arr = arr[np.argsort(arr[:, 0])]
    if arr[1, 0] <= arr[0, 0]:
        raise ValueError('Distinct left/right endpoints required')
    return arr


def validate(answer: dict, public: dict) -> dict:
    for name in ('schema', 'package_id', 'coordinate_system'):
        if answer.get(name) != public[name]:
            raise ValueError(f'Mismatched {name}')
    rows = answer.get('images')
    if not isinstance(rows, list) or [r.get('id') for r in rows] != [r['id'] for r in public['images']]:
        raise ValueError('Missing, duplicated or reordered sample identities')
    pending = []
    for row in rows:
        if not isinstance(row.get('review_complete'), bool) or not isinstance(row.get('objects'), list):
            raise ValueError('Invalid review status or objects')
        for side in ('upper', 'lower'):
            visibility = row.get(side + '_wall_visibility')
            if visibility not in (None, 'visible', 'invisible', 'uncertain'):
                raise ValueError('Invalid wall visibility')
            coords = row.get(side + '_wall_points')
            if coords is not None:
                if visibility != 'visible':
                    raise ValueError('Cannot annotate an invisible/unconfirmed wall')
                points(coords)
            if row['review_complete'] and (visibility is None or (visibility == 'visible' and coords is None)):
                raise ValueError('Completed image requires wall status and points for visible walls')
        for obj in row['objects']:
            if obj.get('kind') not in KINDS:
                raise ValueError('Invalid object class')
            points(obj.get('points'))
        if row['review_complete'] and not row['objects'] and not str(row.get('note', '')).strip():
            raise ValueError('Empty completed image requires a note; empty is not no-droplet truth')
        if not row['review_complete']:
            pending.append(row['id'])
    complete = not pending and bool(str(answer.get('reviewer') or '').strip())
    if answer.get('human_review_complete') is True and not complete:
        raise ValueError('Claimed completion inconsistent with actual answers/reviewer')
    return dict(status='ready_to_score' if complete else 'human_review_pending',
                completed_images=len(rows)-len(pending), total_images=len(rows), pending_ids=pending,
                reviewer_present=bool(str(answer.get('reviewer') or '').strip()),
                measurement_accepted=False, control_authorized=False)


def compare_objects(a: list[dict], b: list[dict]) -> dict:
    aa, bb = [points(o['points']) for o in a], [points(o['points']) for o in b]
    # Conservative matching: uniquely overlapping axial intervals only. Do not
    # force an assignment through missing objects, splits or merged labels.
    overlap = np.zeros((len(a), len(b)), bool)
    for i, x in enumerate(aa):
        for j, y in enumerate(bb):
            overlap[i, j] = min(x[1, 0], y[1, 0]) > max(x[0, 0], y[0, 0])
    matches = [(i, j) for i, j in zip(*np.where(overlap))
               if overlap[i].sum() == 1 and overlap[:, j].sum() == 1]
    errors, agreements = [], []
    for i, j in matches:
        agreements.append(a[i]['kind'] == b[j]['kind'])
        if a[i]['kind'] == b[j]['kind'] == 'complete_droplet':
            errors.extend(np.linalg.norm(aa[i]-bb[j], axis=1).tolist())
    return dict(objects_first=len(a), objects_second=len(b), uniquely_matched=len(matches),
                unmatched_or_ambiguous_first=len(a)-len(matches), unmatched_or_ambiguous_second=len(b)-len(matches),
                matching_class_count=sum(agreements), endpoint_errors_sensor_px=errors,
                count_changed=len(a) != len(b))


def score(answer: dict, public: dict, key: list[dict]) -> dict:
    result = validate(answer, public)
    if result['status'] != 'ready_to_score':
        raise ValueError('Complete independent review before opening the key')
    if [r['id'] for r in key] != [r['id'] for r in public['images']]:
        raise ValueError('Key identities mismatch')
    expected = hashlib.sha256(''.join(r['source']['sha256'] for r in key).encode()).hexdigest()
    if expected != public['package_id']:
        raise ValueError('Key package hash mismatch')
    groups = {}
    for row, entry in zip(answer['images'], key):
        groups.setdefault(entry['source']['sha256'], []).append(row)
    if any(len(g) != 2 for g in groups.values()):
        raise ValueError('Exactly two repetitions per original required')
    pairs = []
    for group in groups.values():
        a, b = group
        row = dict(ids=[a['id'], b['id']], **compare_objects(a['objects'], b['objects']))
        row['wall_visibility_agreement'] = {side: a[side+'_wall_visibility'] == b[side+'_wall_visibility']
                                            for side in ('upper', 'lower')}
        pairs.append(row)
    errors = [e for p in pairs for e in p['endpoint_errors_sensor_px']]
    matched = sum(p['uniquely_matched'] for p in pairs)
    result.update(status='descriptive_repeatability_only', pairs=pairs,
        matched_class_agreement=sum(p['matching_class_count'] for p in pairs)/matched if matched else None,
        uniquely_matched_objects=matched,
        unmatched_or_ambiguous_first=sum(p['unmatched_or_ambiguous_first'] for p in pairs),
        unmatched_or_ambiguous_second=sum(p['unmatched_or_ambiguous_second'] for p in pairs),
        complete_droplet_endpoint_p90_sensor_px=float(np.percentile(errors, 90)) if errors else None,
        endpoint_sample_count=len(errors), engineering_threshold_passed=None,
        limitations=['Same-person repetition, not between-person reproducibility.',
                    'Raw sensor Euclidean endpoint errors cannot be compared to the 2 strip-px gate.',
                    'Unique axial-overlap matching is provisional; unmatched and ambiguous objects remain explicit.',
                    'Class agreement only covers unique matches; not the acceptance rate of all clear examples.',
                    'Repeated image agreement does not establish physical truth or model qualification.'])
    return result


def run(answer_path: Path, package: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError('Refusing to overwrite scoring result')
    answer = json.loads(answer_path.read_text(encoding='utf-8'))
    public = json.loads((package/'blank_answers.json').read_text(encoding='utf-8'))
    result = validate(answer, public)
    # Never read the closed key while human review is incomplete.
    if result['status'] == 'ready_to_score':
        key = json.loads((package/'key_keep_closed.json').read_text(encoding='utf-8'))
        result = score(answer, public, key)
    result['answer_sha256'] = hashlib.sha256(answer_path.read_bytes()).hexdigest()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--answers', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(run(args.answers, args.package, args.output)['status'])
