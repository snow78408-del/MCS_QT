"""Offline evidence curation and image diagnostics; never authorizes control.

Image diagnostics concern saved pixel data, not physical wall visibility or
droplet identity. User field observations remain a separate evidence source.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_response_transfer import evaluate

BLOCKS = ('SA', 'SB', 'F1', 'R1', 'R2', 'F2')
FIELD_REASON = 'User reported view displacement and invisible lower wall; two-wall measurement unaccepted; exact interval unknown'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def curate(data: dict) -> dict:
    """Retain every stage and any existing exclusions; reject known bad pairs."""
    result = deepcopy(data)
    found = set()
    for stage in result['stages']:
        if stage['date'] == '20261023' and stage['block'] in ('R1', 'R2'):
            found.add(stage['block'])
            previous = stage.get('exclusion')
            stage['exclusion'] = f'{previous}; {FIELD_REASON}' if previous else FIELD_REASON
            stage['field_observation'] = dict(lower_wall_visible=False,
                source='user_statement_20261024', exact_affected_interval=None,
                scope='whole-block conservative exclusion pending interval review')
    if found != {'R1', 'R2'}:
        raise ValueError('Both affected blocks must be present in this campaign')
    result.update(measurement_accepted=False, control_authorized=False,
                  prospective_validation=False,
                  curation='Known R1/R2 two-wall invalidity excluded; all remaining measurements are still exploratory')
    return result


def pixel_metrics(image: np.ndarray) -> dict:
    if image.dtype != np.uint8 or image.ndim != 2 or image.size == 0:
        raise ValueError('Expected nonempty Mono8 image')
    hist = np.bincount(image.ravel(), minlength=256)
    cumulative = np.cumsum(hist)
    quantiles = [int(np.searchsorted(cumulative, max(1, image.size * p))) for p in (.1, .5, .9)]
    return dict(mean_gray=float(image.mean()), p10=quantiles[0], median_gray=quantiles[1],
                p90=quantiles[2], contrast_p90_p10=quantiles[2] - quantiles[0],
                black_fraction=float(hist[0] / image.size), white_fraction=float(hist[255] / image.size),
                laplacian_variance=float(cv2.Laplacian(image, cv2.CV_32F).var()))


def texture_shift(reference: np.ndarray, image: np.ndarray) -> dict:
    if reference.shape != image.shape or reference.ndim != 2:
        raise ValueError('Matching 2D texture regions required')
    if reference.std() < 1 or image.std() < 1:
        return dict(dx_px=None, dy_px=None, response=None, status='insufficient_texture')
    window = cv2.createHanningWindow((image.shape[1], image.shape[0]), cv2.CV_32F)
    delta, response = cv2.phaseCorrelate(reference.astype(np.float32), image.astype(np.float32), window)
    if not np.isfinite([*delta, response]).all():
        return dict(dx_px=None, dy_px=None, response=None, status='nonfinite')
    # This threshold is diagnostic triage, not a validated geometry quality gate.
    status = 'candidate_displacement' if response >= .3 and max(abs(x) for x in delta) > 2 else (
        'low_correlation' if response < .3 else 'small_estimated_displacement')
    return dict(dx_px=float(delta[0]), dy_px=float(delta[1]), response=float(response), status=status)


def validate_grid(rows: list[dict]) -> None:
    expected = {(stage, slot) for stage in (1, 2) for slot in range(1500)}
    if len(rows) != 3000 or {(r['stage'], r['grid_slot']) for r in rows} != expected:
        raise ValueError('Both complete 5 Hz grids, including explicit missing cells, required')


def transaction_evidence(meta: dict) -> dict:
    verification = read(Path(meta['verification']))
    path = Path(meta['capture']) / 'commands.ndjson'
    events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    txs = []
    for event in events:
        if not {'q1_ul_min', 'q2_ul_min'} <= event.get('request', {}).keys():
            continue
        a, b = event.get('sent_monotonic'), event.get('readback_monotonic')
        if a is not None and b is not None and (not np.isfinite([a, b]).all() or b < a):
            raise ValueError('Invalid transaction time ordering')
        txs.append(dict(original_event=event, request_to_readback_s=b-a if a is not None and b is not None else None,
                        stop_confirmation_time=None, parameter_verification_time=None,
                        restart_confirmation_time=None,
                        note='Readback is not used to invent missing transaction sub-event times or downtime'))
    return dict(verification_sha256=meta['verification_sha256'], command_log_sha256=digest(path),
                stage_durations_s=verification.get('stage_durations_s'), transactions=txs)


def audit_images(review: Path, inventory: dict, output: Path) -> dict:
    summaries = {}
    review_samples = []
    cv2.setNumThreads(1)
    for block in BLOCKS:
        grid_path = review / block / 'grid_main.ndjson'
        rows = [json.loads(line) for line in grid_path.read_text(encoding='utf-8').splitlines()]
        validate_grid(rows)
        rows.sort(key=lambda r: (r['stage'], r['grid_slot']))
        capture = Path(inventory['blocks'][block]['capture']).resolve()
        reference = None
        facts = []
        with (output / f'{block}_image_audit.ndjson').open('x', encoding='utf-8') as stream:
            for row in rows:
                fact = dict(stage=row['stage'], slot=row['grid_slot'], time_s=row['grid_slot'] / 5,
                            raw_available=row['raw_available'], physical_state='unreviewed')
                if row['raw_available']:
                    original = row['raw_record']
                    path = (capture / original['image']).resolve()
                    if not path.is_relative_to(capture):
                        raise ValueError('Image leaves capture directory')
                    blob = path.read_bytes()
                    actual_hash = hashlib.sha256(blob).hexdigest()
                    if actual_hash != row['image_sha256']:
                        raise ValueError(f'Original changed: {path}')
                    im = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_UNCHANGED)
                    if im is None or im.shape != (540, 720) or im.dtype != np.uint8:
                        raise ValueError('This audit requires the saved 720x540 Mono8 setup')
                    texture = im[10:110, 80:620]
                    if reference is None:
                        reference = texture.copy()
                    fact.update(image=str(path), image_sha256=actual_hash,
                                host_monotonic_s=original['host_monotonic_s'],
                                hardware_frame_id=original['hardware_frame_id'],
                                metrics=pixel_metrics(im), texture_shift=texture_shift(reference, texture))
                    # Uniform grid for review, not outcome-selected examples.
                    if row['grid_slot'] in (300, 1200):
                        review_samples.append(dict(block=block, stage=row['stage'], time_s=fact['time_s'],
                                                   image=str(path), sha256=actual_hash))
                stream.write(json.dumps(fact, ensure_ascii=False, allow_nan=False) + '\n')
                facts.append(fact)
        stage_summaries = []
        for number in (1, 2):
            valid = [r for r in facts if r['stage'] == number and r['raw_available']]
            def percentiles(key: str) -> list[float] | None:
                return [float(x) for x in np.percentile([r['metrics'][key] for r in valid], [10, 50, 90])] if valid else None
            shifts = [r['texture_shift'] for r in valid]
            credible = [s for s in shifts if s['response'] is not None and s['response'] >= .3]
            stage_summaries.append(dict(stage=number, raw_cells=len(valid), missing_cells=1500-len(valid),
                mean_gray_p10_p50_p90=percentiles('mean_gray'), focus_proxy_p10_p50_p90=percentiles('laplacian_variance'),
                shift_status_counts={status: sum(s['status'] == status for s in shifts) for status in
                    ('insufficient_texture', 'low_correlation', 'candidate_displacement', 'small_estimated_displacement', 'nonfinite')},
                max_credible_abs_dy_px=max((abs(s['dy_px']) for s in credible), default=None)))
        summaries[block] = dict(grid_sha256=digest(grid_path), stages=stage_summaries,
                               reference='first available stage1 grid frame; fixed top texture ROI x80:620 y10:110')
        print(f'{block}: image audit complete ({sum(r["raw_available"] for r in facts)} originals)', flush=True)
    result = dict(blocks=summaries, measurement_accepted=False, control_authorized=False,
        method='Every original selected in the prior 5 Hz primary grid; no substitution; raw SHA256 verified',
        limitations=['Not every saved 100fps/20Hz frame is decoded.',
                    'Texture shift is not a wall detector, truth label or droplet velocity.',
                    'Small shift within a block cannot show that its initial view or walls were correct.',
                    'Fixed diagnostic thresholds response>=0.3 and shift>2 sensor px have not been field validated.',
                    'Brightness and focus proxies do not determine droplet visibility.'])
    write(output / 'image_audit_summary.json', result)
    write(output / 'review_sample_sources.json', review_samples)
    return result


def prepare(root: Path, review: Path, inventory_path: Path, output: Path) -> None:
    if output.exists():
        raise ValueError('Refusing to overwrite evidence bundle')
    inventory = read(inventory_path)
    for block in BLOCKS:
        meta = inventory['blocks'][block]
        if digest(Path(meta['verification'])) != meta['verification_sha256']:
            raise ValueError('Capture verification changed')
    output.mkdir(parents=True)
    transactions = {}
    for block in BLOCKS:
        meta = inventory['blocks'][block]
        transactions[block] = transaction_evidence(meta)
    write(output / 'transaction_evidence.json', transactions)
    for name, source in [('primary_5hz', root / 'primary_5hz_dataset.json'), ('cross_date_1hz', root / 'replay/dataset.json')]:
        data = curate(read(source))
        data['source_dataset_sha256'] = digest(source)
        path = output / f'{name}_curated.json'
        write(path, data)
        evaluate(path, output / f'{name}_evaluation')
    audit_images(review, inventory, output)
    write(output / 'provenance.json', dict(script_sha256=digest(Path(__file__)),
        inventory_sha256=digest(inventory_path), hardware_requested=False,
        control_authorized=False, measurement_accepted=False, human_review_complete=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--review', required=True, type=Path)
    parser.add_argument('--inventory', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.root, args.review, args.inventory, args.output)
