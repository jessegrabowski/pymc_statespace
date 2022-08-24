import unittest
from pymc_statespace.filters import StandardFilter, UnivariateFilter, CholeskyFilter, KalmanSmoother, \
        SingleTimeseriesFilter
import aesara
import aesara.tensor as at
import numpy as np
import pandas as pd
from statsmodel_local_level import LocalLinearTrend


class StandardFilterBasicFunctionality(unittest.TestCase):

    def setUp(self):
        self.nile = pd.read_csv('../data/nile.csv')

        kfilter = StandardFilter()
        ksmoother = KalmanSmoother()
        data = at.matrix()
        a0 = at.matrix()
        P0 = at.matrix()
        Q = at.matrix()
        H = at.matrix()
        T = at.matrix()
        R = at.matrix()
        Z = at.matrix()

        filtered_states, predicted_states, \
            filtered_covs, predicted_covs, \
            log_likelihood, ll_obs = kfilter.build_graph(data, a0, P0, T, Z, R, H, Q)

        smoothed_states, smoothed_covs = ksmoother.build_graph(T, R, Q, filtered_states, filtered_covs)

        self.filter_func = aesara.function([data, a0, P0, T, Z, R, H, Q],
                                           [filtered_states, predicted_states, smoothed_states,
                                            filtered_covs, predicted_covs, smoothed_covs,
                                            log_likelihood, ll_obs])

    def test_output_shapes_1d(self):
        p, m, r = 1, 1, 1
        n = 10
        data = np.arange(n * p).reshape(-1, p)
        a0 = np.zeros((m, 1))
        P0 = np.eye(m)
        Q = np.ones((r, r))
        H = np.ones((p, p))
        T = np.ones((m, m))
        R = np.ones((m, r))
        Z = np.ones((p, m))

        filtered_states, predicted_states, smoothed_states, \
            filtered_covs, predicted_covs, smoothed_covs, \
            log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        self.assertTrue(filtered_states.shape == (n, m, 1))
        self.assertTrue(predicted_states.shape == (n + 1, m, 1))
        self.assertTrue(smoothed_states.shape == (n, m, 1))

        self.assertTrue(filtered_covs.shape == (n, m, m))
        self.assertTrue(predicted_covs.shape == (n + 1, m, m))
        self.assertTrue(smoothed_covs.shape == (n, m, m))

        self.assertTrue(ll_obs.ravel().shape == data.ravel().shape)
        self.assertTrue(log_likelihood.shape == ())

    def test_output_shapes_full_rank(self):
        p, m, r = 1, 2, 2
        n = 10
        data = np.arange(n * p).reshape(-1, p)
        a0 = np.zeros((m, 1))
        P0 = np.eye(m)
        Q = np.ones((r, r))
        H = np.ones((p, p))
        T = np.ones((m, m))
        R = np.ones((m, r))
        Z = np.ones((p, m))

        filtered_states, predicted_states, smoothed_states, \
            filtered_covs, predicted_covs, smoothed_covs, \
            log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        self.assertTrue(filtered_states.shape == (n, m, 1))
        self.assertTrue(predicted_states.shape == (n + 1, m, 1))
        self.assertTrue(smoothed_states.shape == (n, m, 1))

        self.assertTrue(filtered_covs.shape == (n, m, m))
        self.assertTrue(predicted_covs.shape == (n + 1, m, m))
        self.assertTrue(smoothed_covs.shape == (n, m, m))

        self.assertTrue(ll_obs.ravel().shape == data.ravel().shape)
        self.assertTrue(log_likelihood.shape == ())

    def test_output_shapes_low_rank(self):
        p, m, r = 1, 5, 2
        n = 10
        data = np.arange(n * p).reshape(-1, p)
        a0 = np.zeros((m, 1))
        P0 = np.eye(m)
        Q = np.ones((r, r))
        H = np.ones((p, p))
        T = np.ones((m, m))
        R = np.ones((m, r))
        Z = np.ones((p, m))

        filtered_states, predicted_states, smoothed_states, \
            filtered_covs, predicted_covs, smoothed_covs, \
            log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        self.assertTrue(filtered_states.shape == (n, m, 1))
        self.assertTrue(predicted_states.shape == (n + 1, m, 1))
        self.assertTrue(smoothed_states.shape == (n, m, 1))

        self.assertTrue(filtered_covs.shape == (n, m, m))
        self.assertTrue(predicted_covs.shape == (n + 1, m, m))
        self.assertTrue(smoothed_covs.shape == (n, m, m))

        self.assertTrue(ll_obs.ravel().shape == data.ravel().shape)
        self.assertTrue(log_likelihood.shape == ())

    def test_output_deterministic_observation(self):
        p, m, r = 1, 5, 1
        n = 10
        data = np.arange(n * p).reshape(-1, p)
        a0 = np.zeros((m, 1))
        P0 = np.eye(m)
        Q = np.ones((r, r))
        H = np.zeros((p, p))
        T = np.ones((m, m))
        R = np.zeros((m, r))
        R[0, 0] = 1
        Z = np.ones((p, m))

        filtered_states, predicted_states, smoothed_states, \
            filtered_covs, predicted_covs, smoothed_covs, \
            log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        self.assertTrue(filtered_states.shape == (n, m, 1))
        self.assertTrue(predicted_states.shape == (n + 1, m, 1))
        self.assertTrue(smoothed_states.shape == (n, m, 1))

        self.assertTrue(filtered_covs.shape == (n, m, m))
        self.assertTrue(predicted_covs.shape == (n + 1, m, m))
        self.assertTrue(smoothed_covs.shape == (n, m, m))

        self.assertTrue(ll_obs.ravel().shape == data.ravel().shape)
        self.assertTrue(log_likelihood.shape == ())

    def test_missing_data(self):
        p, m, r = 1, 5, 1
        n = 10
        data = np.arange(n * p, dtype='float64').reshape(-1, p)
        data[n // 2] = np.nan

        a0 = np.zeros((m, 1))
        P0 = np.eye(m)
        Q = np.ones((r, r))
        H = np.zeros((p, p))
        T = np.ones((m, m))
        R = np.zeros((m, r))
        R[0, 0] = 1
        Z = np.ones((p, m))

        filtered_states, predicted_states, smoothed_states, \
            filtered_covs, predicted_covs, smoothed_covs, \
            log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        self.assertTrue(~np.any(np.isnan(filtered_states)))
        self.assertTrue(~np.any(np.isnan(predicted_states)))
        self.assertTrue(~np.any(np.isnan(smoothed_states)))

        self.assertTrue(~np.any(np.isnan(filtered_covs)))
        self.assertTrue(~np.any(np.isnan(predicted_covs)))
        self.assertTrue(~np.any(np.isnan(smoothed_covs)))

        self.assertTrue(~np.any(np.isnan(ll_obs)))

    def test_loglike_calculation(self):
        data = self.nile.copy()

        a0 = np.zeros((2, 1))
        P0 = np.eye(2) * 1e6
        Q = np.eye(2) * np.array([0.5, 0.01])
        H = np.eye(1) * 0.8
        T = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
        R = np.eye(2)
        Z = np.array([[1.0, 0.0]])

        sm_model = LocalLinearTrend(endog=data, initialization='known', initial_state_cov=P0, initial_state=a0.ravel())

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        res = sm_model.fit_constrained(constraints={'sigma2.measurement': 0.8,
                                                    'sigma2.level': 0.5,
                                                    'sigma2.trend': 0.01})

        self.assertTrue(np.allclose(ll_obs.ravel(), res.llf_obs))

    def test_loglike_calculation_with_missing(self):
        data = self.nile.copy()
        missing_idx = np.random.choice(data.shape[0], size=5, replace=False)
        data.iloc[missing_idx] = np.nan

        a0 = np.zeros((2, 1))
        P0 = np.eye(2) * 1e6
        Q = np.eye(2) * np.array([0.5, 0.01])
        H = np.eye(1) * 0.8
        T = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
        R = np.eye(2)
        Z = np.array([[1.0, 0.0]])

        sm_model = LocalLinearTrend(endog=data, initialization='known', initial_state_cov=P0, initial_state=a0.ravel())

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        res = sm_model.fit_constrained(constraints={'sigma2.measurement': 0.8,
                                                    'sigma2.level': 0.5,
                                                    'sigma2.trend': 0.01})

        self.assertTrue(np.allclose(ll_obs.ravel(), res.llf_obs, rtol=1e-2))

    def test_state_calculations(self):
        data = self.nile.copy()

        a0 = np.zeros((2, 1))
        P0 = np.eye(2) * 1e6
        Q = np.eye(2) * np.array([0.5, 0.01])
        H = np.eye(1) * 0.8
        T = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
        R = np.eye(2)
        Z = np.array([[1.0, 0.0]])

        sm_model = LocalLinearTrend(endog=data, initialization='known', initial_state_cov=P0, initial_state=a0.ravel())

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        res = sm_model.fit_constrained(constraints={'sigma2.measurement': 0.8,
                                                    'sigma2.level': 0.5,
                                                    'sigma2.trend': 0.01})

        self.assertTrue(np.allclose(filtered_states.squeeze(-1), res.states.filtered.values))
        self.assertTrue(np.allclose(predicted_states.squeeze(-1), res.states.predicted.values))
        self.assertTrue(np.allclose(smoothed_states.squeeze(-1), res.states.smoothed.values))

        self.assertTrue(np.allclose(filtered_covs, res.states.filtered_cov.values.reshape(-1, 2, 2)))
        self.assertTrue(np.allclose(predicted_covs, res.states.predicted_cov.values.reshape(-1, 2, 2)))

        # TODO: One value at t=4 has an error of 1e-4, the rest are <= 1e-11... why?
        # self.assertTrue(np.allclose(smoothed_covs, res.states.smoothed_cov.values.reshape(-1, 2, 2), atol=1e-4))

    def test_state_calculations_with_missing(self):
        data = self.nile.copy()

        missing_idx = np.random.choice(data.shape[0], size=5, replace=False)
        data.iloc[missing_idx] = np.nan

        a0 = np.zeros((2, 1))
        P0 = np.eye(2) * 1e6
        Q = np.eye(2) * np.array([0.5, 0.01])
        H = np.eye(1) * 0.8
        T = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
        R = np.eye(2)
        Z = np.array([[1.0, 0.0]])

        sm_model = LocalLinearTrend(endog=data, initialization='known', initial_state_cov=P0, initial_state=a0.ravel())

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        res = sm_model.fit_constrained(constraints={'sigma2.measurement': 0.8,
                                                    'sigma2.level': 0.5,
                                                    'sigma2.trend': 0.01})

        self.assertTrue(np.allclose(filtered_states.squeeze(-1), res.states.filtered.values))
        self.assertTrue(np.allclose(predicted_states.squeeze(-1), res.states.predicted.values))
        self.assertTrue(np.allclose(smoothed_states.squeeze(-1), res.states.smoothed.values))

        self.assertTrue(np.allclose(filtered_covs, res.states.filtered_cov.values.reshape(-1, 2, 2)))
        self.assertTrue(np.allclose(predicted_covs, res.states.predicted_cov.values.reshape(-1, 2, 2)))

        # TODO: One value at t=4 has an error of 1e-4, the rest are <= 1e-11... why?
        # self.assertTrue(np.allclose(smoothed_covs, res.states.smoothed_cov.values.reshape(-1, 2, 2), atol=1e-4))

    def test_multiple_observed(self):
        n = 10
        m, p, r = 4, 2, 4

        data = np.arange(n).repeat(2).reshape(-1, 2)
        a0 = np.zeros((m, 1))
        P0 = np.eye(m)

        T = np.array([[1.0, 1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 1.0],
                      [0.0, 0.0, 0.0, 1.0]])

        Z = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0]])
        R = np.eye(4)
        H = np.eye(2)
        Q = np.eye(4)

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        self.assertTrue(filtered_states.shape == (n, m, 1))
        self.assertTrue(predicted_states.shape == (n + 1, m, 1))
        self.assertTrue(smoothed_states.shape == (n, m, 1))

        self.assertTrue(filtered_covs.shape == (n, m, m))
        self.assertTrue(predicted_covs.shape == (n + 1, m, m))
        self.assertTrue(smoothed_covs.shape == (n, m, m))

        self.assertTrue(ll_obs.ravel().shape == data[:, 0].shape)
        self.assertTrue(log_likelihood.shape == ())


class CholeskyFilterBasicFunctionality(unittest.TestCase):

    def setUp(self):
        self.nile = pd.read_csv('../data/nile.csv')

        kfilter = CholeskyFilter()
        ksmoother = KalmanSmoother()
        data = at.matrix()
        a0 = at.matrix()
        P0 = at.matrix()
        Q = at.matrix()
        H = at.matrix()
        T = at.matrix()
        R = at.matrix()
        Z = at.matrix()

        filtered_states, predicted_states, \
        filtered_covs, predicted_covs, \
        log_likelihood, ll_obs = kfilter.build_graph(data, a0, P0, T, Z, R, H, Q)

        smoothed_states, smoothed_covs = ksmoother.build_graph(T, R, Q, filtered_states, filtered_covs)

        self.filter_func = aesara.function([data, a0, P0, T, Z, R, H, Q],
                                           [filtered_states, predicted_states, smoothed_states,
                                            filtered_covs, predicted_covs, smoothed_covs,
                                            log_likelihood, ll_obs])

    def test_output_shapes_1d(self):
        p, m, r = 1, 1, 1
        n = 10
        data = np.arange(n * p).reshape(-1, p)
        a0 = np.zeros((m, 1))
        P0 = np.eye(m)
        Q = np.ones((r, r))
        H = np.ones((p, p))
        T = np.ones((m, m))
        R = np.ones((m, r))
        Z = np.ones((p, m))

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        self.assertTrue(filtered_states.shape == (n, m, 1))
        self.assertTrue(predicted_states.shape == (n + 1, m, 1))
        self.assertTrue(smoothed_states.shape == (n, m, 1))

        self.assertTrue(filtered_covs.shape == (n, m, m))
        self.assertTrue(predicted_covs.shape == (n + 1, m, m))
        self.assertTrue(smoothed_covs.shape == (n, m, m))

        self.assertTrue(ll_obs.ravel().shape == data.ravel().shape)
        self.assertTrue(log_likelihood.shape == ())

    def test_output_shapes_full_rank(self):
        p, m, r = 1, 2, 2
        n = 10
        data = np.arange(n * p).reshape(-1, p)
        a0 = np.zeros((m, 1))
        P0 = np.eye(m)
        Q = np.ones((r, r))
        H = np.ones((p, p))
        T = np.ones((m, m))
        R = np.ones((m, r))
        Z = np.ones((p, m))

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        self.assertTrue(filtered_states.shape == (n, m, 1))
        self.assertTrue(predicted_states.shape == (n + 1, m, 1))
        self.assertTrue(smoothed_states.shape == (n, m, 1))

        self.assertTrue(filtered_covs.shape == (n, m, m))
        self.assertTrue(predicted_covs.shape == (n + 1, m, m))
        self.assertTrue(smoothed_covs.shape == (n, m, m))

        self.assertTrue(ll_obs.ravel().shape == data.ravel().shape)
        self.assertTrue(log_likelihood.shape == ())

    def test_output_shapes_low_rank(self):
        p, m, r = 1, 5, 2
        n = 10
        data = np.arange(n * p).reshape(-1, p)
        a0 = np.zeros((m, 1))
        P0 = np.eye(m)
        Q = np.ones((r, r))
        H = np.ones((p, p))
        T = np.ones((m, m))
        R = np.ones((m, r))
        Z = np.ones((p, m))

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        self.assertTrue(filtered_states.shape == (n, m, 1))
        self.assertTrue(predicted_states.shape == (n + 1, m, 1))
        self.assertTrue(smoothed_states.shape == (n, m, 1))

        self.assertTrue(filtered_covs.shape == (n, m, m))
        self.assertTrue(predicted_covs.shape == (n + 1, m, m))
        self.assertTrue(smoothed_covs.shape == (n, m, m))

        self.assertTrue(ll_obs.ravel().shape == data.ravel().shape)
        self.assertTrue(log_likelihood.shape == ())

    def test_output_deterministic_observation(self):
        p, m, r = 1, 5, 1
        n = 10
        data = np.arange(n * p).reshape(-1, p)
        a0 = np.zeros((m, 1))
        P0 = np.eye(m)
        Q = np.ones((r, r))
        H = np.zeros((p, p))
        T = np.ones((m, m))
        R = np.zeros((m, r))
        R[0, 0] = 1
        Z = np.ones((p, m))

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        self.assertTrue(filtered_states.shape == (n, m, 1))
        self.assertTrue(predicted_states.shape == (n + 1, m, 1))
        self.assertTrue(smoothed_states.shape == (n, m, 1))

        self.assertTrue(filtered_covs.shape == (n, m, m))
        self.assertTrue(predicted_covs.shape == (n + 1, m, m))
        self.assertTrue(smoothed_covs.shape == (n, m, m))

        self.assertTrue(ll_obs.ravel().shape == data.ravel().shape)
        self.assertTrue(log_likelihood.shape == ())

    def test_missing_data(self):
        p, m, r = 1, 5, 1
        n = 10
        data = np.arange(n * p, dtype='float64').reshape(-1, p)
        data[n // 2] = np.nan

        a0 = np.zeros((m, 1))
        P0 = np.eye(m)
        Q = np.ones((r, r))
        H = np.zeros((p, p))
        T = np.ones((m, m))
        R = np.zeros((m, r))
        R[0, 0] = 1
        Z = np.ones((p, m))

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        self.assertTrue(~np.any(np.isnan(filtered_states)))
        self.assertTrue(~np.any(np.isnan(predicted_states)))
        self.assertTrue(~np.any(np.isnan(smoothed_states)))

        self.assertTrue(~np.any(np.isnan(filtered_covs)))
        self.assertTrue(~np.any(np.isnan(predicted_covs)))
        self.assertTrue(~np.any(np.isnan(smoothed_covs)))

        self.assertTrue(~np.any(np.isnan(ll_obs)))

    def test_loglike_calculation(self):
        data = self.nile.copy()

        a0 = np.zeros((2, 1))
        P0 = np.eye(2) * 1e6
        Q = np.eye(2) * np.array([0.5, 0.01])
        H = np.eye(1) * 0.8
        T = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
        R = np.eye(2)
        Z = np.array([[1.0, 0.0]])

        sm_model = LocalLinearTrend(endog=data, initialization='known', initial_state_cov=P0, initial_state=a0.ravel())

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        res = sm_model.fit_constrained(constraints={'sigma2.measurement': 0.8,
                                                    'sigma2.level': 0.5,
                                                    'sigma2.trend': 0.01})

        self.assertTrue(np.allclose(ll_obs.ravel(), res.llf_obs))

    def test_loglike_calculation_with_missing(self):
        data = self.nile.copy()
        missing_idx = np.random.choice(data.shape[0], size=5, replace=False)
        data.iloc[missing_idx] = np.nan

        a0 = np.zeros((2, 1))
        P0 = np.eye(2) * 1e6
        Q = np.eye(2) * np.array([0.5, 0.01])
        H = np.eye(1) * 0.8
        T = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
        R = np.eye(2)
        Z = np.array([[1.0, 0.0]])

        sm_model = LocalLinearTrend(endog=data, initialization='known', initial_state_cov=P0, initial_state=a0.ravel())

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        res = sm_model.fit_constrained(constraints={'sigma2.measurement': 0.8,
                                                    'sigma2.level': 0.5,
                                                    'sigma2.trend': 0.01})

        self.assertTrue(np.allclose(ll_obs.ravel(), res.llf_obs, rtol=1e-2))

    def test_state_calculations(self):
        data = self.nile.copy()

        a0 = np.zeros((2, 1))
        P0 = np.eye(2) * 1e6
        Q = np.eye(2) * np.array([0.5, 0.01])
        H = np.eye(1) * 0.8
        T = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
        R = np.eye(2)
        Z = np.array([[1.0, 0.0]])

        sm_model = LocalLinearTrend(endog=data, initialization='known', initial_state_cov=P0, initial_state=a0.ravel())

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        res = sm_model.fit_constrained(constraints={'sigma2.measurement': 0.8,
                                                    'sigma2.level': 0.5,
                                                    'sigma2.trend': 0.01})

        self.assertTrue(np.allclose(filtered_states.squeeze(-1), res.states.filtered.values))
        self.assertTrue(np.allclose(predicted_states.squeeze(-1), res.states.predicted.values))
        self.assertTrue(np.allclose(smoothed_states.squeeze(-1), res.states.smoothed.values))

        self.assertTrue(np.allclose(filtered_covs, res.states.filtered_cov.values.reshape(-1, 2, 2)))
        self.assertTrue(np.allclose(predicted_covs, res.states.predicted_cov.values.reshape(-1, 2, 2)))

        # TODO: One value at t=4 has an error of 1e-4, the rest are <= 1e-11... why?
        # self.assertTrue(np.allclose(smoothed_covs, res.states.smoothed_cov.values.reshape(-1, 2, 2), atol=1e-4))

    def test_state_calculations_with_missing(self):
        data = self.nile.copy()

        missing_idx = np.random.choice(data.shape[0], size=5, replace=False)
        data.iloc[missing_idx] = np.nan

        a0 = np.zeros((2, 1))
        P0 = np.eye(2) * 1e6
        Q = np.eye(2) * np.array([0.5, 0.01])
        H = np.eye(1) * 0.8
        T = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
        R = np.eye(2)
        Z = np.array([[1.0, 0.0]])

        sm_model = LocalLinearTrend(endog=data, initialization='known', initial_state_cov=P0, initial_state=a0.ravel())

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        res = sm_model.fit_constrained(constraints={'sigma2.measurement': 0.8,
                                                    'sigma2.level': 0.5,
                                                    'sigma2.trend': 0.01})

        self.assertTrue(np.allclose(filtered_states.squeeze(-1), res.states.filtered.values))
        self.assertTrue(np.allclose(predicted_states.squeeze(-1), res.states.predicted.values))
        self.assertTrue(np.allclose(smoothed_states.squeeze(-1), res.states.smoothed.values))

        self.assertTrue(np.allclose(filtered_covs, res.states.filtered_cov.values.reshape(-1, 2, 2)))
        self.assertTrue(np.allclose(predicted_covs, res.states.predicted_cov.values.reshape(-1, 2, 2)))

        # TODO: One value at t=4 has an error of 1e-4, the rest are <= 1e-11... why?
        # self.assertTrue(np.allclose(smoothed_covs, res.states.smoothed_cov.values.reshape(-1, 2, 2), atol=1e-4))

    def test_multiple_observed(self):
        n = 10
        m, p, r = 4, 2, 4

        data = np.arange(n).repeat(2).reshape(-1, 2)
        a0 = np.zeros((m, 1))
        P0 = np.eye(m)

        T = np.array([[1.0, 1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 1.0],
                      [0.0, 0.0, 0.0, 1.0]])

        Z = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0]])
        R = np.eye(4)
        H = np.eye(2)
        Q = np.eye(4)

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        self.assertTrue(filtered_states.shape == (n, m, 1))
        self.assertTrue(predicted_states.shape == (n + 1, m, 1))
        self.assertTrue(smoothed_states.shape == (n, m, 1))

        self.assertTrue(filtered_covs.shape == (n, m, m))
        self.assertTrue(predicted_covs.shape == (n + 1, m, m))
        self.assertTrue(smoothed_covs.shape == (n, m, m))

        self.assertTrue(ll_obs.ravel().shape == data[:, 0].shape)
        self.assertTrue(log_likelihood.shape == ())


class UnivariateFilterBasicFunctionality(unittest.TestCase):

    def setUp(self):
        self.nile = pd.read_csv('../data/nile.csv')

        kfilter = UnivariateFilter()
        ksmoother = KalmanSmoother()

        data = at.matrix()
        a0 = at.matrix()
        P0 = at.matrix()
        Q = at.matrix()
        H = at.matrix()
        T = at.matrix()
        R = at.matrix()
        Z = at.matrix()

        filtered_states, predicted_states, \
            filtered_covs, predicted_covs, \
            log_likelihood, ll_obs = kfilter.build_graph(data, a0, P0, T, Z, R, H, Q)

        smoothed_states, smoothed_covs = ksmoother.build_graph(T, R, Q, filtered_states, filtered_covs)

        self.filter_func = aesara.function([data, a0, P0, T, Z, R, H, Q],
                                           [filtered_states, predicted_states, smoothed_states,
                                            filtered_covs, predicted_covs, smoothed_covs,
                                            log_likelihood, ll_obs])

    def test_output_shapes_1d(self):
        p, m, r = 1, 1, 1
        n = 10
        data = np.arange(n * p).reshape(-1, p)
        a0 = np.zeros((m, 1))
        P0 = np.eye(m)
        Q = np.ones((r, r))
        H = np.ones((p, p))
        T = np.ones((m, m))
        R = np.ones((m, r))
        Z = np.ones((p, m))

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        self.assertTrue(filtered_states.shape == (n, m, 1))
        self.assertTrue(predicted_states.shape == (n + 1, m, 1))
        self.assertTrue(smoothed_states.shape == (n, m, 1))

        self.assertTrue(filtered_covs.shape == (n, m, m))
        self.assertTrue(predicted_covs.shape == (n + 1, m, m))
        self.assertTrue(smoothed_covs.shape == (n, m, m))

        self.assertTrue(ll_obs.ravel().shape == data.ravel().shape)
        self.assertTrue(log_likelihood.shape == ())

    def test_output_shapes_full_rank(self):
        p, m, r = 1, 2, 2
        n = 10
        data = np.arange(n * p).reshape(-1, p)
        a0 = np.zeros((m, 1))
        P0 = np.eye(m)
        Q = np.ones((r, r))
        H = np.ones((p, p))
        T = np.ones((m, m))
        R = np.ones((m, r))
        Z = np.ones((p, m))

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        self.assertTrue(filtered_states.shape == (n, m, 1))
        self.assertTrue(predicted_states.shape == (n + 1, m, 1))
        self.assertTrue(smoothed_states.shape == (n, m, 1))

        self.assertTrue(filtered_covs.shape == (n, m, m))
        self.assertTrue(predicted_covs.shape == (n + 1, m, m))
        self.assertTrue(smoothed_covs.shape == (n, m, m))

        self.assertTrue(ll_obs.ravel().shape == data.ravel().shape)
        self.assertTrue(log_likelihood.shape == ())

    def test_output_shapes_low_rank(self):
        p, m, r = 1, 5, 2
        n = 10
        data = np.arange(n * p).reshape(-1, p)
        a0 = np.zeros((m, 1))
        P0 = np.eye(m)
        Q = np.ones((r, r))
        H = np.ones((p, p))
        T = np.ones((m, m))
        R = np.ones((m, r))
        Z = np.ones((p, m))

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        self.assertTrue(filtered_states.shape == (n, m, 1))
        self.assertTrue(predicted_states.shape == (n + 1, m, 1))
        self.assertTrue(smoothed_states.shape == (n, m, 1))

        self.assertTrue(filtered_covs.shape == (n, m, m))
        self.assertTrue(predicted_covs.shape == (n + 1, m, m))
        self.assertTrue(smoothed_covs.shape == (n, m, m))

        self.assertTrue(ll_obs.ravel().shape == data.ravel().shape)
        self.assertTrue(log_likelihood.shape == ())

    def test_output_deterministic_observation(self):
        p, m, r = 1, 5, 1
        n = 10
        data = np.arange(n * p).reshape(-1, p)
        a0 = np.zeros((m, 1))
        P0 = np.eye(m)
        Q = np.ones((r, r))
        H = np.zeros((p, p))
        T = np.ones((m, m))
        R = np.zeros((m, r))
        R[0, 0] = 1
        Z = np.ones((p, m))

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        self.assertTrue(filtered_states.shape == (n, m, 1))
        self.assertTrue(predicted_states.shape == (n + 1, m, 1))
        self.assertTrue(smoothed_states.shape == (n, m, 1))

        self.assertTrue(filtered_covs.shape == (n, m, m))
        self.assertTrue(predicted_covs.shape == (n + 1, m, m))
        self.assertTrue(smoothed_covs.shape == (n, m, m))

        self.assertTrue(ll_obs.ravel().shape == data.ravel().shape)
        self.assertTrue(log_likelihood.shape == ())

    def test_missing_data(self):
        p, m, r = 1, 5, 1
        n = 10
        data = np.arange(n * p, dtype='float64').reshape(-1, p)
        data[n // 2] = np.nan

        a0 = np.zeros((m, 1))
        P0 = np.eye(m)
        Q = np.ones((r, r))
        H = np.zeros((p, p))
        T = np.ones((m, m))
        R = np.zeros((m, r))
        R[0, 0] = 1
        Z = np.ones((p, m))

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        self.assertTrue(~np.any(np.isnan(filtered_states)))
        self.assertTrue(~np.any(np.isnan(predicted_states)))
        self.assertTrue(~np.any(np.isnan(smoothed_states)))

        self.assertTrue(~np.any(np.isnan(filtered_covs)))
        self.assertTrue(~np.any(np.isnan(predicted_covs)))
        self.assertTrue(~np.any(np.isnan(smoothed_covs)))

        self.assertTrue(~np.any(np.isnan(ll_obs)))

    def test_loglike_calculation(self):
        data = self.nile.copy()

        a0 = np.zeros((2, 1))
        P0 = np.eye(2) * 1e6
        Q = np.eye(2) * np.array([0.5, 0.01])
        H = np.eye(1) * 0.8
        T = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
        R = np.eye(2)
        Z = np.array([[1.0, 0.0]])

        sm_model = LocalLinearTrend(endog=data, initialization='known', initial_state_cov=P0, initial_state=a0.ravel())

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        res = sm_model.fit_constrained(constraints={'sigma2.measurement': 0.8,
                                                    'sigma2.level': 0.5,
                                                    'sigma2.trend': 0.01})

        self.assertTrue(np.allclose(ll_obs.ravel(), res.llf_obs))

    def test_loglike_calculation_with_missing(self):
        data = self.nile.copy()
        missing_idx = np.random.choice(data.shape[0], size=5, replace=False)
        data.iloc[missing_idx] = np.nan

        a0 = np.zeros((2, 1))
        P0 = np.eye(2) * 1e6
        Q = np.eye(2) * np.array([0.5, 0.01])
        H = np.eye(1) * 0.8
        T = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
        R = np.eye(2)
        Z = np.array([[1.0, 0.0]])

        sm_model = LocalLinearTrend(endog=data, initialization='known', initial_state_cov=P0, initial_state=a0.ravel())

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        res = sm_model.fit_constrained(constraints={'sigma2.measurement': 0.8,
                                                    'sigma2.level': 0.5,
                                                    'sigma2.trend': 0.01})

        self.assertTrue(np.allclose(ll_obs.ravel(), res.llf_obs, rtol=1e-2))

    def test_state_calculations(self):
        data = self.nile.copy()

        a0 = np.zeros((2, 1))
        P0 = np.eye(2) * 1e6
        Q = np.eye(2) * np.array([0.5, 0.01])
        H = np.eye(1) * 0.8
        T = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
        R = np.eye(2)
        Z = np.array([[1.0, 0.0]])

        sm_model = LocalLinearTrend(endog=data, initialization='known', initial_state_cov=P0, initial_state=a0.ravel())

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        res = sm_model.fit_constrained(constraints={'sigma2.measurement': 0.8,
                                                    'sigma2.level': 0.5,
                                                    'sigma2.trend': 0.01})

        self.assertTrue(np.allclose(filtered_states.squeeze(-1), res.states.filtered.values))
        self.assertTrue(np.allclose(predicted_states.squeeze(-1), res.states.predicted.values))
        self.assertTrue(np.allclose(smoothed_states.squeeze(-1), res.states.smoothed.values))

        self.assertTrue(np.allclose(filtered_covs, res.states.filtered_cov.values.reshape(-1, 2, 2)))
        self.assertTrue(np.allclose(predicted_covs, res.states.predicted_cov.values.reshape(-1, 2, 2)))

        # TODO: One value at t=4 has an error of 1e-4, the rest are <= 1e-11... why?
        # self.assertTrue(np.allclose(smoothed_covs, res.states.smoothed_cov.values.reshape(-1, 2, 2), atol=1e-4))

    def test_state_calculations_with_missing(self):
        data = self.nile.copy()

        missing_idx = np.random.choice(data.shape[0], size=5, replace=False)
        data.iloc[missing_idx] = np.nan

        a0 = np.zeros((2, 1))
        P0 = np.eye(2) * 1e6
        Q = np.eye(2) * np.array([0.5, 0.01])
        H = np.eye(1) * 0.8
        T = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
        R = np.eye(2)
        Z = np.array([[1.0, 0.0]])

        sm_model = LocalLinearTrend(endog=data, initialization='known', initial_state_cov=P0, initial_state=a0.ravel())

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        res = sm_model.fit_constrained(constraints={'sigma2.measurement': 0.8,
                                                    'sigma2.level': 0.5,
                                                    'sigma2.trend': 0.01})

        self.assertTrue(np.allclose(filtered_states.squeeze(-1), res.states.filtered.values))
        self.assertTrue(np.allclose(predicted_states.squeeze(-1), res.states.predicted.values))
        self.assertTrue(np.allclose(smoothed_states.squeeze(-1), res.states.smoothed.values))

        self.assertTrue(np.allclose(filtered_covs, res.states.filtered_cov.values.reshape(-1, 2, 2)))
        self.assertTrue(np.allclose(predicted_covs, res.states.predicted_cov.values.reshape(-1, 2, 2)))

        # TODO: One value at t=4 has an error of 1e-4, the rest are <= 1e-11... why?
        # self.assertTrue(np.allclose(smoothed_covs, res.states.smoothed_cov.values.reshape(-1, 2, 2), atol=1e-4))


    def test_multiple_observed(self):
        n = 10
        m, p, r = 4, 2, 4

        data = np.arange(n).repeat(2).reshape(-1, 2)
        a0 = np.zeros((m, 1))
        P0 = np.eye(m)

        T = np.array([[1.0, 1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 1.0],
                      [0.0, 0.0, 0.0, 1.0]])

        Z = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0]])
        R = np.eye(4)
        H = np.eye(2)
        Q = np.eye(4)

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        self.assertTrue(filtered_states.shape == (n, m, 1))
        self.assertTrue(predicted_states.shape == (n + 1, m, 1))
        self.assertTrue(smoothed_states.shape == (n, m, 1))

        self.assertTrue(filtered_covs.shape == (n, m, m))
        self.assertTrue(predicted_covs.shape == (n + 1, m, m))
        self.assertTrue(smoothed_covs.shape == (n, m, m))

        self.assertTrue(ll_obs.ravel().shape == data[:, 0].shape)
        self.assertTrue(log_likelihood.shape == ())


class SingleTimeSeriesFilterBasicFunctionality(unittest.TestCase):

    def setUp(self):
        self.nile = pd.read_csv('../data/nile.csv')

        kfilter = SingleTimeseriesFilter()
        ksmoother = KalmanSmoother()

        data = at.matrix()
        a0 = at.matrix()
        P0 = at.matrix()
        Q = at.matrix()
        H = at.matrix()
        T = at.matrix()
        R = at.matrix()
        Z = at.matrix()

        filtered_states, predicted_states, \
        filtered_covs, predicted_covs, \
        log_likelihood, ll_obs = kfilter.build_graph(data, a0, P0, T, Z, R, H, Q)

        smoothed_states, smoothed_covs = ksmoother.build_graph(T, R, Q, filtered_states, filtered_covs)

        self.filter_func = aesara.function([data, a0, P0, T, Z, R, H, Q],
                                           [filtered_states, predicted_states, smoothed_states,
                                            filtered_covs, predicted_covs, smoothed_covs,
                                            log_likelihood, ll_obs])

    def test_output_shapes_1d(self):
        p, m, r = 1, 1, 1
        n = 10
        data = np.arange(n * p).reshape(-1, p)
        a0 = np.zeros((m, 1))
        P0 = np.eye(m)
        Q = np.ones((r, r))
        H = np.ones((p, p))
        T = np.ones((m, m))
        R = np.ones((m, r))
        Z = np.ones((p, m))

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        self.assertTrue(filtered_states.shape == (n, m, 1))
        self.assertTrue(predicted_states.shape == (n + 1, m, 1))
        self.assertTrue(smoothed_states.shape == (n, m, 1))

        self.assertTrue(filtered_covs.shape == (n, m, m))
        self.assertTrue(predicted_covs.shape == (n + 1, m, m))
        self.assertTrue(smoothed_covs.shape == (n, m, m))

        self.assertTrue(ll_obs.ravel().shape == data.ravel().shape)
        self.assertTrue(log_likelihood.shape == ())

    def test_output_shapes_full_rank(self):
        p, m, r = 1, 2, 2
        n = 10
        data = np.arange(n * p).reshape(-1, p)
        a0 = np.zeros((m, 1))
        P0 = np.eye(m)
        Q = np.ones((r, r))
        H = np.ones((p, p))
        T = np.ones((m, m))
        R = np.ones((m, r))
        Z = np.ones((p, m))

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        self.assertTrue(filtered_states.shape == (n, m, 1))
        self.assertTrue(predicted_states.shape == (n + 1, m, 1))
        self.assertTrue(smoothed_states.shape == (n, m, 1))

        self.assertTrue(filtered_covs.shape == (n, m, m))
        self.assertTrue(predicted_covs.shape == (n + 1, m, m))
        self.assertTrue(smoothed_covs.shape == (n, m, m))

        self.assertTrue(ll_obs.ravel().shape == data.ravel().shape)
        self.assertTrue(log_likelihood.shape == ())

    def test_output_shapes_low_rank(self):
        p, m, r = 1, 5, 2
        n = 10
        data = np.arange(n * p).reshape(-1, p)
        a0 = np.zeros((m, 1))
        P0 = np.eye(m)
        Q = np.ones((r, r))
        H = np.ones((p, p))
        T = np.ones((m, m))
        R = np.ones((m, r))
        Z = np.ones((p, m))

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        self.assertTrue(filtered_states.shape == (n, m, 1))
        self.assertTrue(predicted_states.shape == (n + 1, m, 1))
        self.assertTrue(smoothed_states.shape == (n, m, 1))

        self.assertTrue(filtered_covs.shape == (n, m, m))
        self.assertTrue(predicted_covs.shape == (n + 1, m, m))
        self.assertTrue(smoothed_covs.shape == (n, m, m))

        self.assertTrue(ll_obs.ravel().shape == data.ravel().shape)
        self.assertTrue(log_likelihood.shape == ())

    def test_output_deterministic_observation(self):
        p, m, r = 1, 5, 1
        n = 10
        data = np.arange(n * p).reshape(-1, p)
        a0 = np.zeros((m, 1))
        P0 = np.eye(m)
        Q = np.ones((r, r))
        H = np.zeros((p, p))
        T = np.ones((m, m))
        R = np.zeros((m, r))
        R[0, 0] = 1
        Z = np.ones((p, m))

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        self.assertTrue(filtered_states.shape == (n, m, 1))
        self.assertTrue(predicted_states.shape == (n + 1, m, 1))
        self.assertTrue(smoothed_states.shape == (n, m, 1))

        self.assertTrue(filtered_covs.shape == (n, m, m))
        self.assertTrue(predicted_covs.shape == (n + 1, m, m))
        self.assertTrue(smoothed_covs.shape == (n, m, m))

        self.assertTrue(ll_obs.ravel().shape == data.ravel().shape)
        self.assertTrue(log_likelihood.shape == ())

    def test_missing_data(self):
        p, m, r = 1, 5, 1
        n = 10
        data = np.arange(n * p, dtype='float64').reshape(-1, p)
        data[n // 2] = np.nan

        a0 = np.zeros((m, 1))
        P0 = np.eye(m)
        Q = np.ones((r, r))
        H = np.zeros((p, p))
        T = np.ones((m, m))
        R = np.zeros((m, r))
        R[0, 0] = 1
        Z = np.ones((p, m))

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        self.assertTrue(~np.any(np.isnan(filtered_states)))
        self.assertTrue(~np.any(np.isnan(predicted_states)))
        self.assertTrue(~np.any(np.isnan(smoothed_states)))

        self.assertTrue(~np.any(np.isnan(filtered_covs)))
        self.assertTrue(~np.any(np.isnan(predicted_covs)))
        self.assertTrue(~np.any(np.isnan(smoothed_covs)))

        self.assertTrue(~np.any(np.isnan(ll_obs)))

    def test_loglike_calculation(self):
        data = self.nile.copy()

        a0 = np.zeros((2, 1))
        P0 = np.eye(2) * 1e6
        Q = np.eye(2) * np.array([0.5, 0.01])
        H = np.eye(1) * 0.8
        T = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
        R = np.eye(2)
        Z = np.array([[1.0, 0.0]])

        sm_model = LocalLinearTrend(endog=data, initialization='known', initial_state_cov=P0, initial_state=a0.ravel())

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        res = sm_model.fit_constrained(constraints={'sigma2.measurement': 0.8,
                                                    'sigma2.level': 0.5,
                                                    'sigma2.trend': 0.01})

        self.assertTrue(np.allclose(ll_obs.ravel(), res.llf_obs))

    def test_loglike_calculation_with_missing(self):
        data = self.nile.copy()
        missing_idx = np.random.choice(data.shape[0], size=5, replace=False)
        data.iloc[missing_idx] = np.nan

        a0 = np.zeros((2, 1))
        P0 = np.eye(2) * 1e6
        Q = np.eye(2) * np.array([0.5, 0.01])
        H = np.eye(1) * 0.8
        T = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
        R = np.eye(2)
        Z = np.array([[1.0, 0.0]])

        sm_model = LocalLinearTrend(endog=data, initialization='known', initial_state_cov=P0, initial_state=a0.ravel())

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        res = sm_model.fit_constrained(constraints={'sigma2.measurement': 0.8,
                                                    'sigma2.level': 0.5,
                                                    'sigma2.trend': 0.01})

        self.assertTrue(np.allclose(ll_obs.ravel(), res.llf_obs, rtol=1e-2))

    def test_state_calculations(self):
        data = self.nile.copy()

        a0 = np.zeros((2, 1))
        P0 = np.eye(2) * 1e6
        Q = np.eye(2) * np.array([0.5, 0.01])
        H = np.eye(1) * 0.8
        T = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
        R = np.eye(2)
        Z = np.array([[1.0, 0.0]])

        sm_model = LocalLinearTrend(endog=data, initialization='known', initial_state_cov=P0, initial_state=a0.ravel())

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        res = sm_model.fit_constrained(constraints={'sigma2.measurement': 0.8,
                                                    'sigma2.level': 0.5,
                                                    'sigma2.trend': 0.01})

        self.assertTrue(np.allclose(filtered_states.squeeze(-1), res.states.filtered.values))
        self.assertTrue(np.allclose(predicted_states.squeeze(-1), res.states.predicted.values))
        self.assertTrue(np.allclose(smoothed_states.squeeze(-1), res.states.smoothed.values))

        self.assertTrue(np.allclose(filtered_covs, res.states.filtered_cov.values.reshape(-1, 2, 2)))
        self.assertTrue(np.allclose(predicted_covs, res.states.predicted_cov.values.reshape(-1, 2, 2)))

        # TODO: One value at t=4 has an error of 1e-4, the rest are <= 1e-11... why?
        # self.assertTrue(np.allclose(smoothed_covs, res.states.smoothed_cov.values.reshape(-1, 2, 2), atol=1e-4))

    def test_state_calculations_with_missing(self):
        data = self.nile.copy()

        missing_idx = np.random.choice(data.shape[0], size=5, replace=False)
        data.iloc[missing_idx] = np.nan

        a0 = np.zeros((2, 1))
        P0 = np.eye(2) * 1e6
        Q = np.eye(2) * np.array([0.5, 0.01])
        H = np.eye(1) * 0.8
        T = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
        R = np.eye(2)
        Z = np.array([[1.0, 0.0]])

        sm_model = LocalLinearTrend(endog=data, initialization='known', initial_state_cov=P0, initial_state=a0.ravel())

        filtered_states, predicted_states, smoothed_states, \
        filtered_covs, predicted_covs, smoothed_covs, \
        log_likelihood, ll_obs = self.filter_func(data, a0, P0, T, Z, R, H, Q)

        res = sm_model.fit_constrained(constraints={'sigma2.measurement': 0.8,
                                                    'sigma2.level': 0.5,
                                                    'sigma2.trend': 0.01})

        self.assertTrue(np.allclose(filtered_states.squeeze(-1), res.states.filtered.values))
        self.assertTrue(np.allclose(predicted_states.squeeze(-1), res.states.predicted.values))
        self.assertTrue(np.allclose(smoothed_states.squeeze(-1), res.states.smoothed.values))

        self.assertTrue(np.allclose(filtered_covs, res.states.filtered_cov.values.reshape(-1, 2, 2)))
        self.assertTrue(np.allclose(predicted_covs, res.states.predicted_cov.values.reshape(-1, 2, 2)))

        # TODO: One value at t=4 has an error of 1e-4, the rest are <= 1e-11... why?
        # self.assertTrue(np.allclose(smoothed_covs, res.states.smoothed_cov.values.reshape(-1, 2, 2), atol=1e-4))


if __name__ == '__main__':
    unittest.main()
