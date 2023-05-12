import unittest

import numpy as np
import pytensor
import pytest
from numpy.testing import assert_allclose

from pymc_statespace.filters import (
    CholeskyFilter,
    SingleTimeseriesFilter,
    StandardFilter,
    UnivariateFilter,
    SteadyStateFilter
)
from tests.utilities.test_helpers import (
    initialize_filter,
    make_test_inputs,
    nile_test_test_helper,
    get_expected_shape
)

standard_inout = initialize_filter(StandardFilter())
cholesky_inout = initialize_filter(CholeskyFilter())
univariate_inout = initialize_filter(UnivariateFilter())
single_inout = initialize_filter(SingleTimeseriesFilter())
steadystate_inout = initialize_filter(SteadyStateFilter())

f_standard = pytensor.function(*standard_inout)
f_cholesky = pytensor.function(*cholesky_inout)
f_univariate = pytensor.function(*univariate_inout)
f_single_ts = pytensor.function(*single_inout)
f_steady = pytensor.function(*steadystate_inout)

filter_funcs = [f_standard, f_cholesky, f_univariate, f_single_ts, f_steady]

filter_names = ["StandardFilter", "CholeskyFilter", "UnivariateFilter", "SingleTimeSeriesFilter", "SteadyStateFilter"]
output_names = ['filtered_states', 'predicted_states', 'smoothed_states',
                'filtered_covs', 'predicted_covs', 'smoothed_covs',
                'log_likelihood', 'll_obs']


@pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
@pytest.mark.parametrize(('output_idx', 'name'), list(enumerate(output_names)), ids=output_names)
def test_output_shapes_one_state_one_observed(filter_func, output_idx, name):
    p, m, r, n = 1, 1, 1, 10
    inputs = make_test_inputs(p, m, r, n)

    outputs = filter_func(*inputs)
    expected_output = get_expected_shape(name, p, m, r, n)

    assert outputs[output_idx].shape == expected_output


@pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
@pytest.mark.parametrize(('output_idx', 'name'), list(enumerate(output_names)), ids=output_names)
def test_output_shapes_when_all_states_are_stochastic(filter_func, output_idx, name):
    p, m, r, n = 1, 2, 2, 10
    inputs = make_test_inputs(p, m, r, n)

    outputs = filter_func(*inputs)
    expected_output = get_expected_shape(name, p, m, r, n)

    assert outputs[output_idx].shape == expected_output


@pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
@pytest.mark.parametrize(('output_idx', 'name'), list(enumerate(output_names)), ids=output_names)
def test_output_shapes_when_some_states_are_deterministic(filter_func, output_idx, name):
    p, m, r, n = 1, 5, 2, 10
    inputs = make_test_inputs(p, m, r, n)

    outputs = filter_func(*inputs)
    expected_output = get_expected_shape(name, p, m, r, n)

    assert outputs[output_idx].shape == expected_output


@pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
@pytest.mark.parametrize(('output_idx', 'name'), list(enumerate(output_names)), ids=output_names)
def test_output_with_deterministic_observation_equation(filter_func, output_idx, name):
    p, m, r, n = 1, 5, 1, 10
    inputs = make_test_inputs(p, m, r, n)

    outputs = filter_func(*inputs)
    expected_output = get_expected_shape(name, p, m, r, n)

    assert outputs[output_idx].shape == expected_output


@pytest.mark.parametrize(("filter_func", 'filter_name'), zip(filter_funcs, filter_names), ids=filter_names)
@pytest.mark.parametrize(('output_idx', 'name'), list(enumerate(output_names)), ids=output_names)
def test_output_with_multiple_observed(filter_func, filter_name, output_idx, name):
    p, m, r, n = 5, 5, 1, 10
    inputs = make_test_inputs(p, m, r, n)
    expected_output = get_expected_shape(name, p, m, r, n)

    if filter_name == 'SingleTimeSeriesFilter':
        with pytest.raises(AssertionError, match='UnivariateTimeSeries filter requires data be at most 1-dimensional'):
            filter_func(*inputs)

    else:
        outputs = filter_func(*inputs)
        assert outputs[output_idx].shape == expected_output


@pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
@pytest.mark.parametrize(('output_idx', 'name'), list(enumerate(output_names)), ids=output_names)
def test_missing_data(filter_func, output_idx, name):
    p, m, r, n = 1, 5, 1, 10
    inputs = make_test_inputs(p, m, r, n, missing_data=1)
    outputs = filter_func(*inputs)

    assert not np.any(np.isnan(outputs[output_idx]))


@pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
@pytest.mark.parametrize('output_idx', [(0, 2), (3, 5)], ids=['smoothed_states', 'smoothed_covs'])
def test_last_smoother_is_last_filtered(filter_func, output_idx):
    p, m, r, n = 1, 5, 1, 10
    inputs = make_test_inputs(p, m, r, n)
    outputs = filter_func(*inputs)

    filtered = outputs[output_idx[0]]
    smoothed = outputs[output_idx[1]]

    assert_allclose(filtered[-1], smoothed[-1])


# @pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
# def test_state_calculations(filter_func):
#     nile_test_test_helper(filter_func, test_ll=False, test_states=True)
#
#
# @pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
# def test_state_calculations_with_missing(filter_func):
#     nile_test_test_helper(filter_func, test_ll=False, test_states=True, n_missing=5)
#
#
# @pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
# def test_loglike_calculation(filter_func):
#     nile_test_test_helper(filter_func, test_states=False, test_ll=True)
#
#
# @pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
# def test_loglike_calculation_with_missing(filter_func):
#     nile_test_test_helper(filter_func, n_missing=5)


if __name__ == "__main__":
    unittest.main()
