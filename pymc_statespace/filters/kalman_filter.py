import aesara.tensor as at
import aesara
from aesara.ifelse import ifelse
from aesara.tensor.nlinalg import matrix_dot
import numpy as np

from typing import List

PI = at.constant(np.pi, dtype='floatX')
MVN_CONST = at.log(2 * at.constant(np.pi, dtype='float64'))


class KalmanFilter:

    def build_graph(self, data, a0, P0, T, Z, R, H, Q):
        """
        Construct the computation graph for the Kalman filter. [1] recommends taking the mean of the log-likelihood
        rather than the sum for numerical stability.

        References
        ----------
        .. [1] Koopman, Siem Jan, Neil Shephard, and Jurgen A. Doornik. 1999.
           Statistical Algorithms for Models in State Space Using SsfPack 2.2.
           Econometrics Journal 2 (1): 107-60. doi:10.1111/1368-423X.00023.

        TODO: Add a check for time-varying matrices (ndim > 2) and add matrices to scan sequences if so.
        """

        results, updates = aesara.scan(self.kalman_step,
                                       sequences=[data],
                                       outputs_info=[None, a0, None, P0, None, None, None, None],
                                       non_sequences=[T, Z, R, H, Q],
                                       name='forward_kalman_pass')

        filtered_states, predicted_states, \
            filtered_covariances, predicted_covariances, \
            log_likelihoods, \
            v_history, F_inv_history, K_history = results

        # This follows the Statsmodels output, which appends x0 and P0 to the predicted states, but not to the
        # filtered states
        predicted_states = at.concatenate([a0[None], predicted_states], axis=0)
        predicted_covariances = at.concatenate([P0[None], predicted_covariances], axis=0)

        v_history = v_history.reshape((-1, data.shape[1], 1))

        smoother_result, updates = aesara.scan(self.smoother_step,
                                               sequences=[predicted_states[:-1],
                                                          predicted_covariances[:-1],
                                                          v_history, F_inv_history, K_history],
                                               outputs_info=[None, at.zeros_like(a0), None, at.zeros_like(P0)],
                                               non_sequences=[T, Z],
                                               go_backwards=True,
                                               name='backward_kalman_pass')

        smoothed_states, _, smoothed_covariances, _ = smoother_result
        smoothed_states = at.concatenate([smoothed_states[::-1], predicted_states[-1][None]], axis=0)[:-1]
        smoothed_covariances = at.concatenate([smoothed_covariances[::-1],
                                               predicted_covariances[-1][None]], axis=0)[:-1]

        return filtered_states, predicted_states, smoothed_states, \
                    filtered_covariances, predicted_covariances, smoothed_covariances,\
                    log_likelihoods.sum(), log_likelihoods

    @staticmethod
    def predict(a, P, T, R, Q):
        a_hat = T.dot(a)
        P_hat = matrix_dot(T, P, T.T) + matrix_dot(R, Q, R.T)

        return a_hat, P_hat

    @staticmethod
    def update(a, P, y, Z, H, all_nan_flag):
        """
        Conjugate update rule for the mean and covariance matrix, with log-likelihood en passant
        TODO: Verify these equations are correct if there are multiple endogenous variables.
        TODO: Is there a more elegant way to handle nans?
        """

        v = y - Z.dot(a)

        PZT = P.dot(Z.T)
        F = Z.dot(PZT) + H

        F_inv = at.linalg.solve(F + at.eye(F.shape[0]) * all_nan_flag,
                                at.eye(F.shape[0]),
                                assume_a='pos')
        F_inv *= (1 - all_nan_flag)

        K = PZT.dot(F_inv)
        I_KZ = at.eye(K.shape[0]) - K.dot(Z)

        a_filtered = a + K.dot(v)
        P_filtered = matrix_dot(I_KZ, P, I_KZ.T) + matrix_dot(K, H, K.T) #Joseph form

        inner_term = matrix_dot(v.T, F_inv, v)
        ll = at.switch(all_nan_flag,
                       0.0,
                       -0.5 * (MVN_CONST + at.log(at.linalg.det(F)) + inner_term).ravel()[0])

        return a_filtered, P_filtered, ll, v.squeeze(), F_inv, K

    def kalman_step(self, y, a, P, T, Z, R, H, Q):
        '''
        The timing convention follows [1]. a0 and P0 are taken to be predicted states, so we begin
        with an update step rather than a predict step.

        References
        ----------
        .. [1] Durbin, J., and S. J. Koopman. Time Series Analysis by State Space Methods.
               2nd ed, Oxford University Press, 2012.
        '''

        nan_mask = at.isnan(y)
        all_nan_flag = at.all(nan_mask).astype(aesara.config.floatX)

        W = at.set_subtensor(at.eye(y.shape[0])[nan_mask, nan_mask], 0.0)

        Z_masked = W.dot(Z)
        H_masked = W.dot(H)
        y_masked = at.set_subtensor(y[nan_mask], 0.0)

        a_filtered, P_filtered, ll, v, F_inv, K = self.update(y=y_masked, a=a, P=P, Z=Z_masked,
                                                         H=H_masked, all_nan_flag=all_nan_flag)
        a_hat, P_hat = self.predict(a=a_filtered, P=P_filtered, T=T, R=R, Q=Q)

        return a_filtered, a_hat, P_filtered, P_hat, ll, v, F_inv, K

    @staticmethod
    def smoother_step(a, P, v, F_inv, K, r_t, N_t, T, Z):
        L = T - matrix_dot(T, K, Z)
        ZTF_inv = Z.T.dot(F_inv)

        r_next = ZTF_inv.dot(v) + L.T.dot(r_t)
        N_next = ZTF_inv.dot(Z) + matrix_dot(L.T, N_t, L)

        a_smooth = a + P.dot(r_next)
        P_smooth = P - matrix_dot(P, N_next, P)

        return a_smooth, r_next, P_smooth, N_next
