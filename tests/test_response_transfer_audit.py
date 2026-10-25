from __future__ import annotations

from copy import deepcopy
import json
import numpy as np
import pytest

from tools.replay_measurement_audit import first_per_second, audit_bins
from tools.audit_sampling_phase import phase_audit
from tools.evaluate_response_transfer import (
    TIMES, baseline, build_pairs, evaluate, fit_model, forecast, kernel, record, score,
)


def pair(uid: str, delta: float, gain: float = -.3) -> dict:
    pre = np.full(30, 80.)
    return dict(uid=uid, group=uid, date='20261023', block=uid, pre=pre, base=80.,
                q0=np.array([70., 20.]), q1=np.array([70., 20. * np.exp(delta)]),
                post=80 * np.exp(gain * delta * kernel(30, 5)))


def test_sampling_preserves_missing_first_cell_without_later_substitution():
    cells = first_per_second([dict(t=.1, image='bad'), dict(t=.2, image='good'),
                              dict(t=2.1, image='next')])
    assert cells[0]['image'] == 'bad'
    assert cells[1] is None
    with pytest.raises(ValueError):
        first_per_second([dict(t=2), dict(t=1)])


def test_missing_candidate_not_zero_or_raw_missing():
    rows = [dict(raw_available=True, length=None) for _ in range(300)]
    for i in range(3):
        rows[i]['length'] = 80.
    assert audit_bins(rows)[0]['value'] == 80.
    assert audit_bins(rows)[1]['value'] is None
    for i in range(7, 10):
        rows[i]['raw_available'] = False
    assert audit_bins(rows)[0]['value'] is None


def test_synthetic_signed_gain_and_delay_recovery():
    model = fit_model([pair('forward', .2), pair('reverse', -.2)])
    assert model['tau'] == 30
    assert model['delay'] == 5
    assert model['coefficients'][1] == pytest.approx(-.3, abs=.0001)
    assert model['excitation_rank'] == 1  # Q1 gain is not identified.


def test_forecast_is_invariant_to_hidden_post_outcomes():
    model = fit_model([pair('train1', .2), pair('train2', -.2)])
    p = pair('test', .2)
    a = record(p, model, 'test')
    p['post'][:] = 400
    b = record(p, model, 'test')
    assert a['predictions'] == b['predictions']
    assert a['prediction_sha256'] == b['prediction_sha256']
    assert a['score'] != b['score']


def test_entire_block_is_excluded_from_own_test():
    p = pair('same_block', .2)
    model = fit_model([p])
    p['uid'] = 'different_transition_in_same_block'
    with pytest.raises(ValueError, match='leakage'):
        record(p, model, 'bad')


def test_damped_trend_limits_extrapolation_from_pre_only():
    pre = 80 * np.exp(.001 * (TIMES - 300))
    pred = forecast(pre, np.array([70, 20]), np.array([70, 25]))
    assert pred['damped_trend'][-1] < pred['trend'][-1]
    assert pred['damped_trend'][-1] < 86
    assert np.all(pred['hold'] == baseline(pre))


def test_sham_model_recovers_flow_after_known_background():
    sham = .02 * np.exp(-TIMES / 90)
    pp = [pair('a', .2), pair('b', -.2)]
    for p in pp:
        p['post'] *= np.exp(sham)
    model = fit_model(pp, shams={'20': sham.tolist()})
    assert model['coefficients'][1] == pytest.approx(-.3, abs=.0001)


def test_score_uses_common_bins_and_requires_tail_coverage():
    p = pair('test', .2)
    predictions = forecast(p['pre'], p['q0'], p['q1'])
    p['post'][0] = np.nan
    predictions['trend'][1] = np.nan
    assert score(p, predictions)['bins'] == 28
    p['post'][18:22] = np.nan
    assert score(p, predictions)['status'] == 'insufficient_common_bins'


def test_event_exclusion_breaks_adjacent_pair_without_bridging():
    stages = []
    for i in range(3):
        stages.append(dict(uid=str(i), block='a', date='20261023', q1=70, q2=20+i,
                           exclusion='bubble' if i == 1 else None,
                           bins={'main': [dict(value=80.) for _ in range(30)]}))
    pairs, excluded = build_pairs(stages, 'main')
    assert pairs == []
    assert len(excluded) == 2


def test_input_data_not_mutated_by_fit():
    p = pair('a', .2)
    copy = deepcopy(p)
    fit_model([p])
    np.testing.assert_array_equal(p['post'], copy['post'])


def test_sampling_phase_exposes_alias_without_inventing_drift():
    rows = [dict(stage=stage, grid_slot=k, raw_available=True,
                 frame_median_strip_px=80 + (1 if k % 5 == 0 else -1) * k / 1500)
            for stage in (1, 2) for k in range(1500)]
    result = phase_audit(rows)
    assert result[0]['pre_last60_slope_px_per_s'] > 0
    assert result[1]['pre_last60_slope_px_per_s'] < 0
    assert len(result) == 5


def test_sampling_phase_requires_missing_cells_to_remain_explicit():
    with pytest.raises(ValueError, match='complete 5Hz'):
        phase_audit([])


def test_end_to_end_preserves_method_and_forbids_controller_qualification(tmp_path):
    stages = []
    for block, before, after in [('SA', 20, 20), ('SB', 25, 25), ('F1', 20, 25),
                                  ('R1', 25, 20), ('R2', 25, 20), ('F2', 20, 25)]:
        for number, q in enumerate((before, after)):
            length = np.full(30, 80.) if number == 0 else 80 * np.exp(-.3 * np.log(after / before) * kernel(30, 5))
            stages.append(dict(uid=f'{block}_{number}', date='20261023', block=block,
                               q1=70, q2=q, exclusion=None,
                               bins={v: [dict(value=float(y)) for y in length]
                                     for v in ('main', 'minus1', 'plus1')}))
    path = tmp_path / 'dataset.json'
    path.write_text(json.dumps(dict(stages=stages, method='synthetic 5Hz primary')), encoding='utf-8')
    result = evaluate(path, tmp_path / 'output')
    assert result['measurement_method'] == 'synthetic 5Hz primary'
    assert result['control_authorized'] is False
    assert result['prospective_validation'] is False
    assert len(result['variants']['main']['local']) == 6
    assert 'synthetic 5Hz primary' in (tmp_path / 'output' / 'report.md').read_text(encoding='utf-8')
    with pytest.raises(ValueError, match='overwrite'):
        evaluate(path, tmp_path / 'output')
