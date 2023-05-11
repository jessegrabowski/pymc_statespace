import unittest

import pytensor
import pytest

from pymc_statespace.filters import (
    CholeskyFilter,
    SingleTimeseriesFilter,
    StandardFilter,
    UnivariateFilter,
)
from tests.utilities.test_helpers import (
    filter_output_shapes_test_helper,
    initialize_filter,
    make_test_inputs,
    nile_test_test_helper,
    no_missing_outputs_helper,
)

standard_inout = initialize_filter(StandardFilter())
cholesky_inout = initialize_filter(CholeskyFilter())
univariate_inout = initialize_filter(UnivariateFilter())
single_inout = initialize_filter(SingleTimeseriesFilter())

f_standard = pytensor.function(*standard_inout)
f_cholesky = pytensor.function(*cholesky_inout)
f_univariate = pytensor.function(*univariate_inout)
f_single_ts = pytensor.function(*single_inout)

filter_funcs = [f_standard, f_cholesky, f_univariate, f_single_ts]
filter_funcs_nd = [
    f_standard,
    f_cholesky,
    f_univariate,
    pytest.param(f_single_ts, marks=[pytest.mark.xfail]),
]
filter_names = ["StandardFilter", "CholeskyFilter", "UnivariateFilter", "SingleTimeSeriesFilter"]


@pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
def test_output_shapes_one_state_one_observed(filter_func):
    p, m, r, n = 1, 1, 1, 10
    data, *inputs = make_test_inputs(p, m, r, n)

    outputs = filter_func(data, *inputs)
    filter_output_shapes_test_helper(outputs, data, p, m, r, n)


@pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
def test_output_shapes_when_all_states_are_stochastic(filter_func):
    p, m, r, n = 1, 2, 2, 10
    data, *inputs = make_test_inputs(p, m, r, n)

    outputs = filter_func(data, *inputs)
    filter_output_shapes_test_helper(outputs, data, p, m, r, n)


@pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
def test_output_shapes_when_some_states_are_deterministic(filter_func):
    p, m, r, n = 1, 5, 2, 10
    data, *inputs = make_test_inputs(p, m, r, n)

    outputs = filter_func(data, *inputs)
    filter_output_shapes_test_helper(outputs, data, p, m, r, n)


@pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
def test_output_with_deterministic_observation_equation(filter_func):
    p, m, r, n = 1, 5, 1, 10
    data, *inputs = make_test_inputs(p, m, r, n)

    outputs = filter_func(data, *inputs)
    filter_output_shapes_test_helper(outputs, data, p, m, r, n)


@pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
def test_missing_data(filter_func):
    p, m, r, n = 1, 5, 1, 10
    data, *inputs = make_test_inputs(p, m, r, n, missing_data=1)
    outputs = filter_func(data, *inputs)
    no_missing_outputs_helper(outputs)


@pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
def test_state_calculations(filter_func):
    nile_test_test_helper(filter_func, test_ll=False, test_states=True)


@pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
def test_state_calculations_with_missing(filter_func):
    nile_test_test_helper(filter_func, test_ll=False, test_states=True, n_missing=5)

    #
    # def test_loglike_calculation(self):
    #     data = self.nile.copy()
    #     self.nile_test_test_helper(data)
    #
    # def test_loglike_calculation_with_missing(self):
    #     data = self.nile.copy()
    #     missing_idx = np.random.choice(data.shape[0], size=5, replace=False)
    #     data.iloc[missing_idx] = np.nan
    #
    #     self.nile_test_test_helper(data, rtol=1e-2)
    #

    #
    # def test_multiple_observed(self):
    #     if not test_multiple_observed:
    #         return
    #
    #     m, p, r, n = 4, 2, 4, 10
    #
    #     data = np.arange(n).repeat(2).reshape(-1, 2)
    #     a0 = np.zeros((m, 1))
    #     P0 = np.eye(m)
    #
    #     T = np.array(
    #         [
    #             [1.0, 1.0, 0.0, 0.0],
    #             [0.0, 1.0, 0.0, 0.0],
    #             [0.0, 0.0, 1.0, 1.0],
    #             [0.0, 0.0, 0.0, 1.0],
    #         ]
    #     )
    #
    #     Z = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    #     R = np.eye(4)
    #     H = np.eye(2)
    #     Q = np.eye(4)
    #
    #     (
    #         filtered_states,
    #         predicted_states,
    #         smoothed_states,
    #         filtered_covs,
    #         predicted_covs,
    #         smoothed_covs,
    #         log_likelihood,
    #         ll_obs,
    #     ) = self.filter_func(data, a0, P0, T, Z, R, H, Q)
    #
    #     self.assertTrue(filtered_states.shape == (n, m, 1))
    #     self.assertTrue(predicted_states.shape == (n + 1, m, 1))
    #     self.assertTrue(smoothed_states.shape == (n, m, 1))
    #
    #     self.assertTrue(filtered_covs.shape == (n, m, m))
    #     self.assertTrue(predicted_covs.shape == (n + 1, m, m))
    #     self.assertTrue(smoothed_covs.shape == (n, m, m))
    #
    #     self.assertTrue(ll_obs.ravel().shape == data[:, 0].shape)
    #     self.assertTrue(log_likelihood.shape == ())


if __name__ == "__main__":
    unittest.main()
