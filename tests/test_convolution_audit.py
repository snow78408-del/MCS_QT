import numpy as np
import pytest

from tools.audit_convolution import convolve_horizontal, match_intervals, measure


def test_convolution_preserves_transverse_rows_and_constant_background():
    image = np.repeat(np.arange(32, dtype=np.uint8)[:, None], 300, axis=1)
    np.testing.assert_array_equal(convolve_horizontal(image, 2.8), image)
    assert measure(image, 2.8) == []


def test_zero_is_exact_identity_without_mutating_source():
    image = np.random.default_rng(0).integers(0, 256, (32, 300), dtype=np.uint8)
    result = convolve_horizontal(image, 0)
    np.testing.assert_array_equal(result, image)
    assert not np.shares_memory(result, image)


@pytest.mark.parametrize('sigma', [-1, float('nan'), float('inf')])
def test_bad_kernel_parameters_rejected(sigma):
    with pytest.raises(ValueError):
        convolve_horizontal(np.zeros((32, 300), np.uint8), sigma)


def test_matching_never_counts_one_prediction_as_two_droplets():
    assert len(match_intervals([(0, 100)], [(0, 45), (55, 100)])) == 1
    assert match_intervals([(0, 10)], [(20, 30)]) == []
    assert match_intervals([], [(20, 30)]) == []
