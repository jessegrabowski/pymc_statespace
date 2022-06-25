import aesara.tensor as at
import aesara
from aesara.tensor.nlinalg import matrix_dot
import numpy as np

from typing import Protocol

MVN_CONST = at.log(2 * at.constant(np.pi, dtype='float64'))


class KalmanFilter(Protocol):

    def build_graph(self, data, a0, P0, T, Z, R, H, Q):
        raise NotImplementedError

    @staticmethod
    def predict(a, P, T, R, Q):
        raise NotImplementedError

    @staticmethod
    def update(a, P, y, Z, H, nan_flag):
        raise NotImplementedError

    @staticmethod
    def smoother_step(a, P, v, F, K, r_t, N_t, T, Z):
        raise NotImplementedError

    def kalman_step(self, y, a, P, T, Z, R, H, Q):
        raise NotImplementedError


class StandardFilter:

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

        filter_results = [filtered_states, predicted_states, smoothed_states,
                          filtered_covariances, predicted_covariances, smoothed_covariances,
                          log_likelihoods.sum(), log_likelihoods]

        return [x.squeeze() for x in filter_results]

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
        P_filtered = matrix_dot(I_KZ, P, I_KZ.T) + matrix_dot(K, H, K.T)  # Joseph form

        inner_term = matrix_dot(v.T, F_inv, v)
        ll = at.switch(all_nan_flag,
                       0.0,
                       -0.5 * (MVN_CONST + at.log(at.linalg.det(F)) + inner_term).ravel()[0])

        return a_filtered, P_filtered, ll, v, F_inv, K

    def kalman_step(self, y, a, P, T, Z, R, H, Q):
        """
        The timing convention follows [1]. a0 and P0 are taken to be predicted states, so we begin
        with an update step rather than a predict step.

        References
        ----------
        .. [1] Durbin, J., and S. J. Koopman. Time Series Analysis by State Space Methods.
               2nd ed, Oxford University Press, 2012.
        """

        y = y[:, None]
        nan_mask = at.isnan(y).ravel()
        all_nan_flag = at.all(nan_mask).astype(aesara.config.floatX)

        W = at.set_subtensor(at.eye(y.shape[0])[nan_mask, nan_mask], 0.0)

        Z_masked = W.dot(Z)
        H_masked = W.dot(H)
        y_masked = at.set_subtensor(y[nan_mask, :], 0.0)

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


class UnivariateFilter:

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
        v_history, F_history, K_history = results

        # This follows the Statsmodels output, which appends x0 and P0 to the predicted states, but not to the
        # filtered states
        predicted_states = at.concatenate([a0[None], predicted_states], axis=0)
        predicted_covariances = at.concatenate([P0[None], predicted_covariances], axis=0)

        smoother_result, updates = aesara.scan(self.smoother_step,
                                               sequences=[predicted_states[:-1],
                                                          predicted_covariances[:-1],
                                                          v_history, F_history, K_history],
                                               outputs_info=[None, at.zeros_like(a0), None, at.zeros_like(P0)],
                                               non_sequences=[T, Z],
                                               go_backwards=True,
                                               name='backward_kalman_pass')

        smoothed_states, _, smoothed_covariances, _ = smoother_result
        smoothed_states = at.concatenate([smoothed_states[::-1], predicted_states[-1][None]], axis=0)[:-1]
        smoothed_covariances = at.concatenate([smoothed_covariances[::-1],
                                               predicted_covariances[-1][None]], axis=0)[:-1]

        filter_results = [filtered_states, predicted_states, smoothed_states,
                          filtered_covariances, predicted_covariances, smoothed_covariances,
                          log_likelihoods.sum(), log_likelihoods]

        return [x.squeeze() for x in filter_results]

    @staticmethod
    def predict(a, P, T, R, Q):
        a_hat = T.dot(a)
        P_hat = matrix_dot(T, P, T.T) + matrix_dot(R, Q, R.T)

        return a_hat, P_hat

    @staticmethod
    def _univariate_inner_filter_step(y, Z_row, sigma_H, nan_flag, a, P):
        Z_row = Z_row[None, :]
        v = y - Z_row.dot(a)

        PZT = P.dot(Z_row.T)
        F = Z_row.dot(PZT) + sigma_H

        F_zero_flag = at.or_(at.eq(F, 0), nan_flag)

        # This should easier than trying to dodge the log(F) and 1 / F with a switch
        F = F + 1e-8 * F_zero_flag

        # If F is zero (implies y is NAN or another degenerate case), then we want:
        # K = 0, a = a, P = P, ll = 0
        K = PZT / F * (1 - F_zero_flag)
        a_filtered = a + K * v * (1 - F_zero_flag)
        P_filtered = P - at.outer(K, K) * F * (1 - F_zero_flag)
        ll_inner = (at.log(F) + v ** 2 / F) * (1 - F_zero_flag)

        return a_filtered, P_filtered, ll_inner, v, F, K

    def kalman_step(self, y, a, P, T, Z, R, H, Q):
        '''
        The univariate kalman filter, described in [1], section 6.4.2, avoids inversion of the F matrix, as well as two additonal
        matrix multiplications, at the cost of an additional loop.

        This is useful when states are perfectly observed, because the F matrix can easily become degenerate in these cases.

        References
        ----------
        .. [1] Durbin, J., and S. J. Koopman. Time Series Analysis by State Space Methods.
               2nd ed, Oxford University Press, 2012.
        '''

        y = y[:, None]
        nan_mask = at.isnan(y).ravel()

        W = at.set_subtensor(at.eye(y.shape[0])[nan_mask, nan_mask], 0.0)
        Z_masked = W.dot(Z)
        H_masked = W.dot(H)
        y_masked = at.set_subtensor(y[nan_mask], 0.0)

        result, updates = aesara.scan(self._univariate_inner_filter_step,
                                      sequences=[y_masked, Z_masked, at.diag(H_masked), nan_mask],
                                      outputs_info=[a, P, None, None, None, None])

        a_filtered, P_filtered, ll_inner, v, F, K = result
        a_filtered, P_filtered = a_filtered[-1], P_filtered[-1]

        a_hat, P_hat = self.predict(a=a_filtered, P=P_filtered, T=T, R=R, Q=Q)

        ll = -0.5 * ((at.gt(ll_inner, 0).sum()) * MVN_CONST + ll_inner.sum())

        return a_filtered, a_hat, P_filtered, P_hat, ll, v, F, K

    @staticmethod
    def _univariate_inner_smoother_step(v, F, K, Z, r_p, N_p, T):
        Z_row = Z[None, :]
        L = at.eye(T.shape[0]) - K @ Z_row
        ZTF_inv = Z_row.T / F

        r_next = ZTF_inv * v + L.T @ r_p
        N_next = ZTF_inv.dot(Z_row) + matrix_dot(L.T, N_p, L)

        return r_next, N_next

    def smoother_step(self, a, P, v, F, K, r_t, N_t, T, Z):
        results, updates = aesara.scan(self._univariate_inner_smoother_step,
                                       sequences=[v, F, K, Z],
                                       outputs_info=[r_t, N_t],
                                       non_sequences=[T])

        r_next, N_next = results
        r_next = r_next[-1]
        N_next = N_next[-1]

        a_smooth = a + P.dot(r_next)
        P_smooth = P - matrix_dot(P, N_next, P)

        new_r = T.T.dot(r_next)
        new_N = matrix_dot(T.T, N_next, T)

        return a_smooth, new_r, P_smooth, new_N
