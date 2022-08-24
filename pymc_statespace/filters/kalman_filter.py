from abc import ABC
from typing import List, Tuple

import aesara
import aesara.tensor as at
from aesara.tensor.nlinalg import matrix_dot
from aesara.tensor import TensorVariable

import numpy as np

from pymc_statespace.utils.aesara_scipy import solve_discrete_are
import pymc as pm
from pymc.model import modelcontext

MVN_CONST = at.log(2 * at.constant(np.pi, dtype='float64'))


class BaseFilter(ABC):

    def build_graph(self, data, a0, P0, T, Z, R, H, Q) -> List[TensorVariable]:
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
                                       outputs_info=[None, a0, None, P0, None],
                                       non_sequences=[T, Z, R, H, Q],
                                       name='forward_kalman_pass')

        filter_results = self._postprocess_scan_results(results, a0, P0)

        return filter_results

    @staticmethod
    def _postprocess_scan_results(results, a0, P0) -> List[TensorVariable]:
        filtered_states, predicted_states, filtered_covariances, predicted_covariances, log_likelihoods = results

        # This follows the Statsmodels output, which appends x0 and P0 to the predicted states, but not to the
        # filtered states
        predicted_states = at.concatenate([a0[None], predicted_states], axis=0)
        predicted_covariances = at.concatenate([P0[None], predicted_covariances], axis=0)

        filter_results = [filtered_states, predicted_states,
                          filtered_covariances, predicted_covariances,
                          log_likelihoods.sum(), log_likelihoods]

        return [x.squeeze() for x in filter_results]

    @staticmethod
    def predict(a, P, T, R, Q) -> Tuple[TensorVariable, TensorVariable]:
        a_hat = T.dot(a)
        P_hat = matrix_dot(T, P, T.T) + matrix_dot(R, Q, R.T)

        # Force P_hat to be symmetric
        P_hat = 0.5 * (P_hat + P_hat.T)

        return a_hat, P_hat

    @staticmethod
    def update(a, P, y, Z, H, all_nan_flag) -> Tuple[TensorVariable, TensorVariable, TensorVariable]:
        raise NotImplementedError

    def kalman_step(self, y, a, P, T, Z, R, H, Q) -> Tuple[TensorVariable, TensorVariable, TensorVariable,
                                                           TensorVariable, TensorVariable]:
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

        a_filtered, P_filtered, ll = self.update(y=y_masked, a=a, P=P, Z=Z_masked, H=H_masked,
                                                 all_nan_flag=all_nan_flag)

        a_hat, P_hat = self.predict(a=a_filtered, P=P_filtered, T=T, R=R, Q=Q)

        return a_filtered, a_hat, P_filtered, P_hat, ll


class StandardFilter(BaseFilter):

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

        K = PZT.dot(F_inv)
        I_KZ = at.eye(K.shape[0]) - K.dot(Z)

        a_filtered = a + K.dot(v)
        P_filtered = matrix_dot(I_KZ, P, I_KZ.T) + matrix_dot(K, H, K.T)  # Joseph form

        inner_term = matrix_dot(v.T, F_inv, v)
        ll = at.switch(all_nan_flag,
                       0.0,
                       -0.5 * (MVN_CONST + at.log(at.linalg.det(F)) + inner_term).ravel()[0])

        return a_filtered, P_filtered, ll


class CholeskyFilter(BaseFilter):

    @staticmethod
    def update(a, P, y, Z, H, all_nan_flag):
        """
        Conjugate update rule for the mean and covariance matrix, with log-likelihood en passant
        TODO: Verify these equations are correct if there are multiple endogenous variables.
        TODO: Is there a more elegant way to handle nans?
        """

        v = y - Z.dot(a)

        PZT = P.dot(Z.T)

        # If everything is missing, F will be [[0]] and F_chol will raise an error, so add identity to avoid the error
        F = Z.dot(PZT) + H + at.eye(y.shape[0]) * all_nan_flag

        F_chol = at.linalg.cholesky(F)

        # If everything is missing, K = 0, IKZ = I
        K = (at.linalg.solve_lower_triangular(F_chol.T, at.linalg.solve_lower_triangular(F_chol, PZT.T)).T
             * (1 - all_nan_flag))
        I_KZ = at.eye(K.shape[0]) - K.dot(Z)

        a_filtered = a + K.dot(v)
        P_filtered = matrix_dot(I_KZ, P, I_KZ.T) + matrix_dot(K, H, K.T)  # Joseph form

        inner_term = at.linalg.solve_lower_triangular(F_chol.T, at.linalg.solve_lower_triangular(F_chol, v))
        n = y.shape[0]

        ll = at.switch(all_nan_flag,
                       0.0,
                       -0.5 * (n * MVN_CONST + (v.T @ inner_term).ravel()) - at.log(at.diag(F_chol)).sum())

        return a_filtered, P_filtered, ll


class SingleTimeseriesFilter(BaseFilter):
    def build_graph(self, data, a0, P0, T, Z, R, H, Q) -> List[TensorVariable]:
        """
        Construct the computation graph for the Kalman filter. This differs from the base filter bceause the
        _postprocess_scan_results function does not take Z or H.
        """

        results, updates = aesara.scan(self.kalman_step,
                                       sequences=[data],
                                       outputs_info=[None, a0, None, P0, None],
                                       non_sequences=[T, Z, R, H, Q],
                                       name='forward_kalman_pass')

        filter_results = self._postprocess_scan_results(results, a0, P0)

        return filter_results

    @staticmethod
    def _postprocess_scan_results(results, a0, P0, Z, H) -> List[TensorVariable]:
        filtered_states, predicted_states, filtered_covariances, predicted_covariances, \
            log_likelihoods = results
            # obs_hat, cov_hat,\


        # This follows the Statsmodels output, which appends x0 and P0 to the predicted states, but not to the
        # filtered states
        predicted_states = at.concatenate([a0[None], predicted_states], axis=0)
        predicted_covariances = at.concatenate([P0[None], predicted_covariances], axis=0)

        filter_results = [filtered_states, predicted_states,
                          filtered_covariances, predicted_covariances,
                          # obs_hat, cov_hat,
                          log_likelihoods.sum(), log_likelihoods]

        return [x.squeeze() for x in filter_results]

    def update(self, a, P, y, Z, H, all_nan_flag):
        # y, v are scalar, but a might not be
        y_hat = Z.dot(a).ravel()
        v = y - y_hat

        PZT = P.dot(Z.T)

        # F is scalar, K is a column vector
        F = Z.dot(PZT).ravel() + H
        K = PZT / F

        a_filtered = a + K * v
        P_filtered = P - P @ P / F

        ll = -0.5 * (MVN_CONST + at.log(F) + v ** 2 / F)

        return a_filtered, P_filtered, ll #y_hat, F, ll

    def kalman_step(self, y, a, P, T, Z, R, H, Q):
        # a_filtered, P_filtered, obs_mu, obs_cov, ll = self.update(y=y, a=a, P=P, Z=Z, H=H, all_nan_flag=False)
        a_filtered, P_filtered, ll = self.update(y=y, a=a, P=P, Z=Z, H=H, all_nan_flag=False)

        a_hat, P_hat = self.predict(a=a_filtered, P=P_filtered, T=T, R=R, Q=Q)

        return a_filtered, a_hat, P_filtered, P_hat, ll # obs_mu, obs_cov, ll


class SteadyStateFilter(BaseFilter):
    """
    This filter avoids the need to invert the covariance matrix of innovations at each time step by solving the
    Discrete Algebraic Riccati Equation associated with the filtering problem once and for all at initialization and
    uses the resulting steady-state covariance matrix in each step.

    The innovation covariance matrix will always converge to the steady state value as T -> oo, so this filter will
    only have differences from the standard approach in the early steps (T < 10?). A process of "learning" is lost.
    """

    def build_graph(self, data, a0, P0, T, Z, R, H, Q):
        P_steady = solve_discrete_are(T.T, Z.T, matrix_dot(R, Q, R.T), H)
        F = matrix_dot(Z, P_steady, Z.T) + H
        F_inv = at.linalg.solve(F, at.eye(F.shape[0]), assume_a='pos')

        results, updates = aesara.scan(self.kalman_step,
                                       sequences=[data],
                                       outputs_info=[None, a0, None, P0, None],
                                       non_sequences=[F_inv, T, Z, R, H, Q],
                                       name='forward_kalman_pass')

        return self._postprocess_scan_results(results, a0, P0)

    @staticmethod
    def update(a, P, F_inv, y, Z, H, all_nan_flag):
        """
        Conjugate update rule for the mean and covariance matrix, with log-likelihood en passant
        TODO: Verify these equations are correct if there are multiple endogenous variables.
        TODO: Is there a more elegant way to handle nans?
        """

        v = y - Z.dot(a)
        PZT = P.dot(Z.T)
        F = Z.dot(PZT) + H

        K = PZT @ F_inv
        I_KZ = at.eye(K.shape[0]) - K.dot(Z)

        a_filtered = a + K.dot(v)
        P_filtered = matrix_dot(I_KZ, P, I_KZ.T) + matrix_dot(K, H, K.T)  # Joseph form

        inner_term = matrix_dot(v.T, F_inv, v)
        ll = at.switch(all_nan_flag,
                       0.0,
                       -0.5 * (MVN_CONST + at.log(at.linalg.det(F)) + inner_term).ravel()[0])

        return a_filtered, P_filtered, ll

    def kalman_step(self, y, a, P, F_inv, T, Z, R, H, Q):
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

        a_filtered, P_filtered, ll = self.update(y=y_masked, a=a, P=P, F_inv=F_inv, Z=Z_masked, H=H_masked,
                                                 all_nan_flag=all_nan_flag)

        a_hat, P_hat = self.predict(a=a_filtered, P=P_filtered, T=T, R=R, Q=Q)

        return a_filtered, a_hat, P_filtered, P_hat, ll


class UnivariateFilter(BaseFilter):
    """
    The univariate kalman filter, described in [1], section 6.4.2, avoids inversion of the F matrix, as well as two
    additonal matrix multiplications, at the cost of an additional loop.

    This is useful when states are perfectly observed, because the F matrix can easily become degenerate in these cases.

    References
    ----------
    .. [1] Durbin, J., and S. J. Koopman. Time Series Analysis by State Space Methods.
            2nd ed, Oxford University Press, 2012.

    """
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

        return a_filtered, P_filtered, ll_inner

    def kalman_step(self, y, a, P, T, Z, R, H, Q):

        y = y[:, None]
        nan_mask = at.isnan(y).ravel()

        W = at.set_subtensor(at.eye(y.shape[0])[nan_mask, nan_mask], 0.0)
        Z_masked = W.dot(Z)
        H_masked = W.dot(H)
        y_masked = at.set_subtensor(y[nan_mask], 0.0)

        result, updates = aesara.scan(self._univariate_inner_filter_step,
                                      sequences=[y_masked, Z_masked, at.diag(H_masked), nan_mask],
                                      outputs_info=[a, P, None])

        a_filtered, P_filtered, ll_inner = result
        a_filtered, P_filtered = a_filtered[-1], P_filtered[-1]

        a_hat, P_hat = self.predict(a=a_filtered, P=P_filtered, T=T, R=R, Q=Q)

        ll = -0.5 * ((at.neq(ll_inner, 0).sum()) * MVN_CONST + ll_inner.sum())

        return a_filtered, a_hat, P_filtered, P_hat, ll
