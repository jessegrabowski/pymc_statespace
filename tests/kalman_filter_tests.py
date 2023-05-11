import unittest

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as at
from statsmodel_local_level import LocalLinearTrend

from pymc_statespace.filters import (
    CholeskyFilter,
    KalmanSmoother,
    SingleTimeseriesFilter,
    StandardFilter,
    UnivariateFilter,
)


def initialize_filter(kfilter):
    ksmoother = KalmanSmoother()
    data = at.matrix()
    a0 = at.matrix()
    P0 = at.matrix()
    Q = at.matrix()
    H = at.matrix()
    T = at.matrix()
    R = at.matrix()
    Z = at.matrix()

    inputs = [data, a0, P0, T, Z, R, H, Q]

    (
        filtered_states,
        predicted_states,
        filtered_covs,
        predicted_covs,
        log_likelihood,
        ll_obs,
    ) = kfilter.build_graph(*inputs)

    smoothed_states, smoothed_covs = ksmoother.build_graph(T, R, Q, filtered_states, filtered_covs)

    outputs = [
        filtered_states,
        predicted_states,
        smoothed_states,
        filtered_covs,
        predicted_covs,
        smoothed_covs,
        log_likelihood,
        ll_obs,
    ]

    return inputs, outputs


def add_missing_data(data, n_missing):
    n = data.shape[0]
    missing_idx = np.random.choice(n, n_missing, replace=False)
    data[missing_idx] = np.nan

    return data


def make_test_inputs(p, m, r, n, missing_data=None):
    data = np.arange(n * p, dtype="float").reshape(-1, p)
    if missing_data is not None:
        data = add_missing_data(data, missing_data)

    a0 = np.zeros((m, 1))
    P0 = np.eye(m)
    Q = np.ones((r, r))
    H = np.ones((p, p))
    T = np.ones((m, m))
    R = np.ones((m, r))
    Z = np.ones((p, m))

    inputs = [data, a0, P0, T, Z, R, H, Q]

    return inputs


class FilterTestBase(unittest.TestCase):
    nile = pd.read_csv("../data/nile.csv")

    def output_shape_test_helper(self, outputs, data, p, m, r, n):
        (
            filtered_states,
            predicted_states,
            smoothed_states,
            filtered_covs,
            predicted_covs,
            smoothed_covs,
            log_likelihood,
            ll_obs,
        ) = outputs

        self.assertTrue(filtered_states.shape == (n, m, 1))
        self.assertTrue(predicted_states.shape == (n + 1, m, 1))
        self.assertTrue(smoothed_states.shape == (n, m, 1))

        self.assertTrue(filtered_covs.shape == (n, m, m))
        self.assertTrue(predicted_covs.shape == (n + 1, m, m))
        self.assertTrue(smoothed_covs.shape == (n, m, m))

        self.assertTrue(ll_obs.ravel().shape == data.ravel().shape)
        self.assertTrue(log_likelihood.shape == ())

    def none_missing_test_helper(self, outputs):
        (
            filtered_states,
            predicted_states,
            smoothed_states,
            filtered_covs,
            predicted_covs,
            smoothed_covs,
            log_likelihood,
            ll_obs,
        ) = outputs

        self.assertTrue(~np.any(np.isnan(filtered_states)))
        self.assertTrue(~np.any(np.isnan(predicted_states)))
        self.assertTrue(~np.any(np.isnan(smoothed_states)))

        self.assertTrue(~np.any(np.isnan(filtered_covs)))
        self.assertTrue(~np.any(np.isnan(predicted_covs)))
        self.assertTrue(~np.any(np.isnan(smoothed_covs)))

        self.assertTrue(~np.any(np.isnan(ll_obs)))

    def nile_test_test_helper(self, data, test_ll=True, test_states=False, **allclose_kwargs):
        a0 = np.zeros((2, 1))
        P0 = np.eye(2) * 1e6
        Q = np.eye(2) * np.array([0.5, 0.01])
        H = np.eye(1) * 0.8
        T = np.array([[1.0, 1.0], [0.0, 1.0]])
        R = np.eye(2)
        Z = np.array([[1.0, 0.0]])

        sm_model = LocalLinearTrend(
            endog=data,
            initialization="known",
            initial_state_cov=P0,
            initial_state=a0.ravel(),
        )

        (
            filtered_states,
            predicted_states,
            smoothed_states,
            filtered_covs,
            predicted_covs,
            smoothed_covs,
            log_likelihood,
            ll_obs,
        ) = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        res = sm_model.fit_constrained(
            constraints={
                "sigma2.measurement": 0.8,
                "sigma2.level": 0.5,
                "sigma2.trend": 0.01,
            }
        )

        if test_ll:
            self.assertTrue(np.allclose(ll_obs.ravel(), res.llf_obs, **allclose_kwargs))
        elif test_states:
            self.assertTrue(np.allclose(filtered_states.squeeze(-1), res.states.filtered.values))
            self.assertTrue(np.allclose(predicted_states.squeeze(-1), res.states.predicted.values))
            self.assertTrue(np.allclose(smoothed_states.squeeze(-1), res.states.smoothed.values))

            self.assertTrue(
                np.allclose(filtered_covs, res.states.filtered_cov.values.reshape(-1, 2, 2))
            )
            self.assertTrue(
                np.allclose(predicted_covs, res.states.predicted_cov.values.reshape(-1, 2, 2))
            )
            self.assertTrue(
                np.allclose(
                    smoothed_covs,
                    res.states.smoothed_cov.values.reshape(-1, 2, 2),
                    atol=1e-2,
                )
            )


def filter_test_class_factory(kfilter, test_multiple_observed=True):
    class FilterTestSuite(FilterTestBase):
        def setUp(self):
            inputs, outputs = initialize_filter(kfilter())
            self.filter_func = pytensor.function(inputs, outputs)

        def test_output_shapes_1d(self):
            p, m, r, n = 1, 1, 1, 10
            data, *inputs = make_test_inputs(p, m, r, n)

            outputs = self.filter_func(data, *inputs)
            self.output_shape_test_helper(outputs, data, p, m, r, n)

        def test_output_shapes_full_rank(self):
            p, m, r, n = 1, 2, 2, 10
            data, *inputs = make_test_inputs(p, m, r, n)

            outputs = self.filter_func(data, *inputs)
            self.output_shape_test_helper(outputs, data, p, m, r, n)

        def test_output_shapes_low_rank(self):
            p, m, r, n = 1, 5, 2, 10
            n = 10
            data, *inputs = make_test_inputs(p, m, r, n)

            outputs = self.filter_func(data, *inputs)
            self.output_shape_test_helper(outputs, data, p, m, r, n)

        def test_output_deterministic_observation(self):
            p, m, r, n = 1, 5, 1, 10
            data, *inputs = make_test_inputs(p, m, r, n)

            outputs = self.filter_func(data, *inputs)
            self.output_shape_test_helper(outputs, data, p, m, r, n)

        def test_missing_data(self):
            p, m, r, n = 1, 5, 1, 10
            data, *inputs = make_test_inputs(p, m, r, n, missing_data=1)
            outputs = self.filter_func(data, *inputs)
            self.none_missing_test_helper(outputs)

        def test_loglike_calculation(self):
            data = self.nile.copy()
            self.nile_test_test_helper(data)

        def test_loglike_calculation_with_missing(self):
            data = self.nile.copy()
            missing_idx = np.random.choice(data.shape[0], size=5, replace=False)
            data.iloc[missing_idx] = np.nan

            self.nile_test_test_helper(data, rtol=1e-2)

        def test_state_calculations(self):
            data = self.nile.copy()
            self.nile_test_test_helper(data, test_ll=False, test_states=True)

        def test_state_calculations_with_missing(self):
            data = self.nile.copy()
            missing_idx = np.random.choice(data.shape[0], size=5, replace=False)
            data.iloc[missing_idx] = np.nan

            self.nile_test_test_helper(data, test_ll=False, test_states=True)

        def test_multiple_observed(self):
            if not test_multiple_observed:
                return

            m, p, r, n = 4, 2, 4, 10

            data = np.arange(n).repeat(2).reshape(-1, 2)
            a0 = np.zeros((m, 1))
            P0 = np.eye(m)

            T = np.array(
                [
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

            Z = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
            R = np.eye(4)
            H = np.eye(2)
            Q = np.eye(4)

            (
                filtered_states,
                predicted_states,
                smoothed_states,
                filtered_covs,
                predicted_covs,
                smoothed_covs,
                log_likelihood,
                ll_obs,
            ) = self.filter_func(data, a0, P0, T, Z, R, H, Q)

            self.assertTrue(filtered_states.shape == (n, m, 1))
            self.assertTrue(predicted_states.shape == (n + 1, m, 1))
            self.assertTrue(smoothed_states.shape == (n, m, 1))

            self.assertTrue(filtered_covs.shape == (n, m, m))
            self.assertTrue(predicted_covs.shape == (n + 1, m, m))
            self.assertTrue(smoothed_covs.shape == (n, m, m))

            self.assertTrue(ll_obs.ravel().shape == data[:, 0].shape)
            self.assertTrue(log_likelihood.shape == ())

    return FilterTestSuite


# StandardFilterBasicFunctionality = filter_test_class_factory(StandardFilter)
# CholeskyFilterBasicFunctionality = filter_test_class_factory(CholeskyFilter)
# UnivariateFilterBasicFunctionality = filter_test_class_factory(UnivariateFilter)
SingleTimeSeriesFilterBasicFunctionality = filter_test_class_factory(SingleTimeseriesFilter)

if __name__ == "__main__":
    unittest.main()
