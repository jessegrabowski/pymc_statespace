import unittest

import numpy as np

from pymc_statespace.core.representation import PytensorRepresentation


class BasicFunctionality(unittest.TestCase):
    def setUp(self):
        self.data = np.arange(10)[:, None]

    def test_default_shapes_full_rank(self):
        ssm = PytensorRepresentation(data=self.data, k_states=5, k_posdef=5)
        p = ssm.data.shape[1]
        m = ssm.k_states
        r = ssm.k_posdef

        self.assertTrue(ssm.data.shape == (10, 1))
        self.assertTrue(ssm["design"].eval().shape == (p, m))
        self.assertTrue(ssm["transition"].eval().shape == (m, m))
        self.assertTrue(ssm["selection"].eval().shape == (m, r))
        self.assertTrue(ssm["state_cov"].eval().shape == (r, r))
        self.assertTrue(ssm["obs_cov"].eval().shape == (p, p))

    def test_default_shapes_low_rank(self):
        ssm = PytensorRepresentation(data=self.data, k_states=5, k_posdef=2)
        p = ssm.data.shape[1]
        m = ssm.k_states
        r = ssm.k_posdef

        self.assertTrue(ssm.data.shape == (10, 1))
        self.assertTrue(ssm["design"].eval().shape == (p, m))
        self.assertTrue(ssm["transition"].eval().shape == (m, m))
        self.assertTrue(ssm["selection"].eval().shape == (m, r))
        self.assertTrue(ssm["state_cov"].eval().shape == (r, r))
        self.assertTrue(ssm["obs_cov"].eval().shape == (p, p))

    def test_matrix_assignment(self):
        ssm = PytensorRepresentation(data=self.data, k_states=5, k_posdef=2)

        ssm["design", 0, 0] = 3.0
        ssm["transition", 0, :] = 2.7
        ssm["selection", -1, -1] = 9.9

        self.assertTrue(ssm["design"].eval()[0, 0] == 3.0)
        self.assertTrue(np.all(ssm["transition"].eval()[0, :] == 2.7))
        self.assertTrue(ssm["selection"].eval()[-1, -1] == 9.9)


if __name__ == "__main__":
    unittest.main()
