"""Device-free frozen prediction worker for the Q2 capture transaction gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.analyze_q2_v2 import SOURCE_FILES, ROOT, contained_path, make_pipeline, measure_image, sha256

BASELINE_METHOD = 'tail60_fresh_pipeline_5hz_v1'
TIMES = np.arange(5., 300., 10.)


def json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode('utf-8')


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(json_bytes(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def object_hash(value: object) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_bundle(bundle: dict) -> None:
    if bundle.get('schema_version') != 1 or bundle.get('baseline_method') != BASELINE_METHOD:
        raise ValueError('Unsupported frozen prediction / baseline schema')
    for key in ('study_id', 'chip_id'):
        if not isinstance(bundle.get(key), str) or not bundle[key].strip():
            raise ValueError('Missing frozen '+key)
    if bundle.get('q1_fixed') != 70 or bundle.get('q2_range') != [20, 25]:
        raise ValueError('Prediction gate supports fixed Q1=70, Q2=20..25 only')
    hashes = bundle.get('measurement_source_hashes', {})
    if bundle.get('baseline_worker_sha256') != sha256(Path(__file__)):
        raise ValueError('Frozen baseline worker source mismatch')
    if any(hashes.get(name) != sha256(ROOT/name) for name in SOURCE_FILES):
        raise ValueError('Frozen measurement source mismatch')
    models = bundle.get('models', {})
    if not models or bundle.get('primary_model') not in models or set(models) & {'hold', 'trend'}:
        raise ValueError('Missing primary model or reserved model name')
    if not isinstance(bundle.get('training_block_ids'), list) or not bundle['training_block_ids']:
        raise ValueError('Frozen training identities required')
    for model in models.values():
        if model.get('kind') == 'fopdt':
            if (not all(finite_number(model.get(k)) for k in ('gain', 'tau_s', 'delay_s'))
                    or model['tau_s'] <= 0 or model['delay_s'] < 0):
                raise ValueError('Invalid first-order parameters')
        elif model.get('kind') == 'fir':
            knots = model.get('knots_s', []); coef = model.get('coefficients', [])
            if (not 2 <= len(knots) <= 32 or len(coef) != len(knots)-1
                    or not all(finite_number(v) for v in knots+coef)
                    or knots[0] != 0 or np.any(np.diff(knots) <= 0)):
                raise ValueError('Invalid FIR parameters')
        else:
            raise ValueError('Unsupported prediction model')


def load_bundle_for_plan(plan: dict) -> tuple[bytes, dict]:
    config = plan['stage_gates']['prediction']
    blob = Path(config['bundle_path']).read_bytes()
    if hashlib.sha256(blob).hexdigest() != config['bundle_sha256']:
        raise ValueError('Frozen model file hash mismatch')
    bundle = json.loads(blob); validate_bundle(bundle)
    if (bundle['study_id'] != plan.get('study_id') or bundle['chip_id'] != plan['apparatus'].get('chip_id')
            or plan['session_id'] in bundle['training_block_ids']):
        raise ValueError('Study/chip mismatch or validation block used for training')
    if bundle.get('reference_image_sha256') != sha256(Path(plan['geometry_reference_image'])):
        raise ValueError('Frozen reference image mismatch')
    if bundle.get('reference_walls_sha256') != object_hash(plan['reference_walls']):
        raise ValueError('Frozen reference walls mismatch')
    expected = dict(camera_unique_id=plan['apparatus']['camera_unique_id'],
                    exposure_us=plan['capture_plan']['requested_exposure_us'],
                    gain=plan['capture_plan']['requested_gain'])
    if bundle.get('camera_settings') != expected:
        raise ValueError('Frozen camera settings mismatch')
    # Exercise both supported directions before hardware, without inspecting experimental outcomes.
    predict(bundle, [1.]*6, [70,20], [70,25])
    predict(bundle, [1.]*6, [70,25], [70,20])
    return blob, bundle


def predict(bundle: dict, bins: list[float | None], q0: list, q1: list) -> dict:
    if len(bins) != 6 or any(v is not None and (not finite_number(v) or v <= 0) for v in bins):
        raise ValueError('Six positive/null baseline bins required')
    valid = np.array([v is not None for v in bins])
    if valid.sum() < 5:
        raise ValueError(f'Insufficient baseline: {valid.sum()}/6 qualified bins (need 5)')
    if q0[0] != 70 or q1[0] != 70 or any(q not in (20, 25) for q in (q0[1], q1[1])):
        raise ValueError('Unsupported prediction flows')
    values = np.array([v if v is not None else np.nan for v in bins]); base = float(np.median(values[valid]))
    slope, intercept = np.polyfit(np.arange(-55., 0., 10.)[valid], np.log(values[valid]/base), 1)
    curves = dict(hold=np.full(30, base), trend=base*np.exp(intercept+slope*TIMES))
    delta = math.log(q1[1]/q0[1])
    for name, model in bundle['models'].items():
        if model['kind'] == 'fopdt':
            response = model['gain']*(1-np.exp(-np.maximum(TIMES-model['delay_s'], 0)/model['tau_s']))
        else:
            knots = np.asarray(model['knots_s'])
            basis = np.clip((TIMES[:, None]-knots[:-1])/np.diff(knots), 0, 1)
            response = basis @ np.asarray(model['coefficients'])
        curves[name] = base*np.exp(delta*response)
    if any(not np.isfinite(v).all() or np.any(v <= 0) for v in curves.values()):
        raise ValueError('Nonfinite prediction; no transaction allowed')
    return dict(baseline_strip_px=base, baseline_bins=bins, times_s=TIMES.tolist(),
                curves={k:v.tolist() for k,v in curves.items()}, primary_model=bundle['primary_model'])


def baseline_from_snapshot(directory: Path, request: dict) -> tuple[list, list]:
    start = request['baseline_stage']['observation_started_monotonic']
    end = start+300.; cells = [None]*300; rows = []
    snapshot = (directory/'prediction_manifest_snapshot.ndjson').read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != request['manifest_sha256']:
        raise ValueError('Manifest snapshot changed')
    last_stamp = -math.inf; last_id = -1
    for line in snapshot.splitlines():
        record = json.loads(line)
        stamp = record['host_monotonic_s']; frame_id = record['hardware_frame_id']
        if (not finite_number(stamp) or stamp <= 0 or stamp <= last_stamp or not isinstance(frame_id, int)
                or isinstance(frame_id, bool) or frame_id <= 0 or frame_id <= last_id):
            raise ValueError('Invalid/duplicate raw identity in prediction snapshot')
        last_stamp, last_id = stamp, frame_id
        # Ignore recorder phase labels; use the actual formal first-stage clock.
        if end-60 <= stamp < end:
            slot = min(299, int((stamp-(end-60))*5))
            if cells[slot] is None:
                cells[slot] = record
    pipeline = make_pipeline(); values = []
    for slot, record in enumerate(cells):
        value = None
        if record is not None:
            path = contained_path(directory, record['image'])
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError('Unreadable baseline image')
            lengths = measure_image(pipeline, image, request['registered_walls'], record['host_monotonic_s'])
            value = float(np.median(lengths)) if lengths else None
            rows.append(dict(slot=slot, image=record['image'], sha256=sha256(path),
                             hardware_frame_id=record['hardware_frame_id'], length_strip_px=value))
        values.append(value)
    bins = []
    for begin in range(0, 300, 50):
        valid = [v for v in values[begin:begin+50] if v is not None]
        raw = sum(v is not None for v in cells[begin:begin+50])
        bins.append(float(np.median(valid)) if raw >= 40 and len(valid) >= 10 else None)
    return bins, rows


def run(request_path: Path) -> dict:
    cv2.setNumThreads(1)
    directory = request_path.parent; request_blob = request_path.read_bytes(); request = json.loads(request_blob)
    bundle_blob = (directory/'frozen_prediction_bundle.json').read_bytes()
    if hashlib.sha256(bundle_blob).hexdigest() != request['bundle_sha256']:
        raise ValueError('Frozen bundle changed during capture')
    bundle = json.loads(bundle_blob); validate_bundle(bundle)
    bins, rows = baseline_from_snapshot(directory, request)
    result = predict(bundle, bins, request['q0'], request['q1'])
    result.update(request_sha256=hashlib.sha256(request_blob).hexdigest(), bundle_sha256=request['bundle_sha256'],
                  session_id=request['session_id'], baseline_images=rows,
                  measurement_accepted=False, control_authorized=False,
                  time_origin='second-stage observation start, bound after verified transaction')
    atomic_json(directory/'prediction_candidate.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--request', type=Path, required=True)
    run(parser.parse_args().request)
