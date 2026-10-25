from __future__ import annotations

import numpy as np
import pytest

from tools.audit_response_components import components, residuals


def test_components_preserve_multiplicative_decomposition():
    model = dict(directional=False, coefficients=[0., -.2], tau=30., delay=5., shams={'20':[.01]*30})
    p = components(model, 80, [70,20], [70,25])
    np.testing.assert_allclose(p['full'], p['flow_only'] * p['sham_only'] / p['hold'])


def test_zero_step_is_sham_and_no_background_is_hold():
    model = dict(directional=False, coefficients=[0., -.2], tau=30., delay=5., shams=None)
    p = components(model, 80, [70,20], [70,20])
    for v in p.values():
        np.testing.assert_allclose(v, 80)


def test_bias_is_not_removed_from_forecast_score():
    result = residuals(np.full(30, 80.), {'biased': np.full(30, 82.)})['biased']
    assert result['all']['rmse'] == 2
    assert result['all']['bias'] == 2
    assert result['all']['centered_rmse'] == 0


def test_common_missing_mask_and_window_counts():
    actual = np.full(30, 80.); actual[0] = np.nan
    pred = np.full(30, 81.); pred[1] = np.nan
    result = residuals(actual, {'a':pred, 'b':np.full(30,80.)})
    assert result['a']['all']['n'] == result['b']['all']['n'] == 28
    assert result['a']['startup']['n'] == 4
    assert result['a']['tail']['n'] == 12


def test_invalid_baseline_rejected():
    with pytest.raises(ValueError):
        components({}, 0, [70,20], [70,25])
