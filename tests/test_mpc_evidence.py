from copy import deepcopy

import cv2
import numpy as np
import pytest

from tools.prepare_mpc_evidence import curate, pixel_metrics, texture_shift, validate_grid, transaction_evidence
from tools.evaluate_response_transfer import build_pairs, evaluate


def dataset():
    return dict(stages=[dict(uid=f'{block}_{i}', date='20261023', block=block, q1=70, q2=20+i,
                            exclusion=None, bins={'main': [dict(value=80.) for _ in range(30)]})
                        for block in ('F1', 'R1', 'R2') for i in range(2)])


def test_field_exclusion_prevents_both_reverse_pairs_without_erasing_data():
    data = dataset()
    previous = deepcopy(data)
    result = curate(data)
    pairs, rejected = build_pairs(result['stages'], 'main')
    assert [p['block'] for p in pairs] == ['F1']
    assert len(rejected) == 2
    assert len(result['stages']) == 6
    assert result['control_authorized'] is False
    assert result['measurement_accepted'] is False
    assert data == previous


def test_curation_keeps_older_exclusions_and_is_scoped_to_date():
    data = dataset()
    data['stages'][2]['exclusion'] = 'bubble'
    old = deepcopy(data['stages'][2]); old.update(uid='older', date='20261016', exclusion='old')
    data['stages'].append(old)
    result = curate(data)
    assert result['stages'][2]['exclusion'].startswith('bubble;')
    assert result['stages'][-1] == old
    assert result['stages'][2]['field_observation']['exact_affected_interval'] is None


def test_wrong_campaign_is_not_silently_accepted():
    with pytest.raises(ValueError, match='Both affected'):
        curate(dict(stages=[]))


def test_texture_shift_recovers_synthetic_translation():
    rng = np.random.default_rng(11)
    im = cv2.GaussianBlur(rng.uniform(10, 240, (100, 540)).astype(np.float32), (5, 5), 0)
    moved = cv2.warpAffine(im, np.float32([[1, 0, 3], [0, 1, -4]]), (540, 100), borderMode=cv2.BORDER_REFLECT)
    result = texture_shift(im, moved)
    assert result['dx_px'] == pytest.approx(3, abs=.3)
    assert result['dy_px'] == pytest.approx(-4, abs=.3)
    assert result['status'] == 'candidate_displacement'


def test_flat_image_is_not_a_zero_shift_confirmation():
    flat = np.full((100, 540), 80, np.uint8)
    result = texture_shift(flat, flat)
    assert result['status'] == 'insufficient_texture'
    assert result['dx_px'] is None


def test_dark_pixels_do_not_become_no_droplet_ground_truth():
    result = pixel_metrics(np.zeros((100, 100), np.uint8))
    assert result['black_fraction'] == 1
    assert result['contrast_p90_p10'] == 0
    assert 'physical_state' not in result
    with pytest.raises(ValueError, match='Mono8'):
        pixel_metrics(np.zeros((100, 100), np.float32))


def test_grid_duplicate_cannot_hide_a_missing_cell():
    rows = [dict(stage=s, grid_slot=k) for s in (1, 2) for k in range(1500)]
    validate_grid(rows)
    rows[-1] = rows[0]
    with pytest.raises(ValueError, match='complete'):
        validate_grid(rows)


def test_excluded_reverse_blocks_disable_local_model_and_report(tmp_path):
    import json
    data = curate(dataset())
    for stage in data['stages']:
        stage['bins']['minus1'] = deepcopy(stage['bins']['main'])
        stage['bins']['plus1'] = deepcopy(stage['bins']['main'])
    path = tmp_path / 'data.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    result = evaluate(path, tmp_path / 'result')
    assert result['variants']['main']['local_models'] == {}
    assert result['variants']['main']['local'] == []
    assert '没有拟合或评分局部双向模型' in (tmp_path / 'result/report.md').read_text(encoding='utf-8')


def test_transaction_uses_original_log_when_summary_omits_events(tmp_path):
    import json
    summary = tmp_path / 'verification.json'
    summary.write_text('{}', encoding='utf-8')
    event = dict(command_id='sham', request=dict(q1_ul_min=70, q2_ul_min=20),
                 sent_monotonic=10., readback_monotonic=13.5)
    (tmp_path / 'commands.ndjson').write_text(json.dumps(event)+'\n', encoding='utf-8')
    result = transaction_evidence(dict(verification=str(summary), capture=str(tmp_path), verification_sha256='test'))
    assert result['transactions'][0]['request_to_readback_s'] == 3.5
    assert result['transactions'][0]['stop_confirmation_time'] is None
    event['readback_monotonic'] = None
    (tmp_path / 'commands.ndjson').write_text(json.dumps(event)+'\n', encoding='utf-8')
    assert transaction_evidence(dict(verification=str(summary), capture=str(tmp_path), verification_sha256='test'))['transactions'][0]['request_to_readback_s'] is None
