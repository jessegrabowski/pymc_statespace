import aesara.tensor as at
import aesara
import numpy as np


class KalmanFilter:

    def __init__(self):
        self.matrix_det = at.nlinalg.Det()

    def build_graph(self, data, a0, P0, Q, H, T, R, Z):
        """
        Construct the computation graph for the Kalman filter

        TODO: Add a check for time-varying matrices (ndim > 2) and add matrices to scan sequences if so.
        """

        results, updates = aesara.scan(self._kalman_step,
                                       sequences=[data],
                                       outputs_info=[a0, P0, np.zeros(1)],
                                       non_sequences=[Q, H, T, R, Z])

        states, covariances, log_likelihoods = results

        states = at.concatenate([a0[None], states], axis=0)
        covariances = at.concatenate([P0[None], covariances], axis=0)

        log_likelihood = -data.shape[0] * data.shape[1] / 2 * np.log(2 * np.pi) - 0.5 * log_likelihoods[-1]

        return states, covariances, log_likelihood[0]

    def _kalman_step(self, y, a, P, ll, Q, H, T, R, Z):
        """
        Conjugate update rule for the mean and covariance matrix, with log-likelihood en passant

        TODO: Verify these equations are correct if there are multiple endogenous variables.

        TODO: Add handling for NA values
        """
        v = y - Z.dot(a)
        F = Z.dot(P).dot(Z.T) + H
        F_inv = at.nlinalg.matrix_inverse(F)

        a_update = a + P.dot(Z.T).dot(F_inv).dot(v)
        P_update = P - P.dot(Z.T).dot(F_inv).dot(Z).dot(P)

        a_hat = T.dot(a_update)
        P_hat = T.dot(P_update).dot(T.T) + R.dot(Q).dot(R.T)

        ll += (at.log(self.matrix_det(F)) + (v.T).dot(F_inv).dot(v)).ravel()[0]

        return a_hat, P_hat, ll
