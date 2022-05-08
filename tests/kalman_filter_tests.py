import unittest
from pymc_statespace.filters.kalman_filter import KalmanFilter
import aesara
import aesara.tensor as at
import numpy as np


class BasicFunctionality(unittest.TestCase):

    def setUp(self):
        filter = KalmanFilter()
        data = at.matrix()
        a0 = at.matrix()
        P0 = at.matrix()
        Q = at.matrix()
        H = at.matrix()
        T = at.matrix()
        R = at.matrix()
        Z = at.matrix()

        filtered_states, filtered_covs, log_likelihood = filter.build_graph(data, a0, P0, Q, H, T, R, Z)

        self.filter_func = aesara.function([data, a0, P0, Q, H, T, R, Z],
                                           [filtered_states, filtered_covs, log_likelihood])

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
        a, b, c = self.filter_func(data, a0, P0, Q, H, T, R, Z)

        self.assertTrue(a.shape == (n + 1, m, 1))
        self.assertTrue(b.shape == (n + 1, r, r))
        self.assertTrue(c.shape == ())

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
        a, b, c = self.filter_func(data, a0, P0, Q, H, T, R, Z)

        self.assertTrue(a.shape == (n + 1, m, 1))
        self.assertTrue(b.shape == (n + 1, r, r))
        self.assertTrue(c.shape == ())

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
        a, b, c = self.filter_func(data, a0, P0, Q, H, T, R, Z)

        self.assertTrue(a.shape == (n + 1, m, 1))
        self.assertTrue(b.shape == (n + 1, m, m))
        self.assertTrue(c.shape == ())

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
        a, b, c = self.filter_func(data, a0, P0, Q, H, T, R, Z)

        self.assertTrue(a.shape == (n + 1, m, 1))
        self.assertTrue(b.shape == (n + 1, m, m))
        self.assertTrue(c.shape == ())



if __name__ == '__main__':
    unittest.main()
