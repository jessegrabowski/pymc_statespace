from abc import ABC
from typing import List, Tuple

import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import get_mode
from pytensor.raise_op import Assert
from pytensor.tensor import TensorVariable
from pytensor.tensor.nlinalg import matrix_dot
from pytensor.tensor.slinalg import SolveTriangular

from pymc_statespace.filters.utilities import split_vars_into_seq_and_nonseq
from pymc_statespace.utils.pytensor_scipy import solve_discrete_are

MVN_CONST = pt.log(2 * pt.constant(np.pi, dtype="float64"))
PARAM_NAMES = ["c", "d", "T", "Z", "R", "H", "Q"]
solve_lower_triangular = SolveTriangular(lower=True)
assert_data_is_1d = Assert("UnivariateTimeSeries filter requires data be at most 1-dimensional")
assert_time_varying_dim_correct = Assert(
    "The first dimension of a time varying matrix (the time dimension) must be "
    "equal to the first dimension of the data (the time dimension)."
)


class BaseFilter(ABC):
    def __init__(self, mode=None):
        self.mode = mode
        self.seq_names = []
        self.non_seq_names = []

        self.eye_states = None
        self.eye_posdef = None
        self.eye_endog = None

    @staticmethod
    def initialize_intercepts(c, d, Z):
        """
        # TODO: Should this be handled in the StateSpaceModel object instead, and the KF should demand they are always
            given?
        """
        m, p, *_ = Z.shape

        if c is None:
            c = pt.zeros((p, 1))
            c.name = "c"
        if d is None:
            d = pt.zeros((m, 1))
            d.name = "d"

        return c, d

    def initialize_eyes(self, R, Z):
        """
        It's surprisingly expensive for pytensor to create an identity matrix every time we need one for e.g. a matrix
        inversion (see [1] for benchmarks). This function creates some identity matrices of useful sizes for the model
        to re-use as a small optimization.

        Also, we're not sure whether R or Z are time-varying when this function is called, so the states are indexed
        from the back.

        References
        ----------
        .. [1] https://gist.github.com/jessegrabowski/acd3235833163943a11654d78a72f04b
        """

        n_states, n_posdef, n_endog = R.shape[-2], R.shape[-1], Z.shape[-2]
        self.eye_states = pt.eye(n_states)
        self.eye_posdef = pt.eye(n_posdef)
        self.eye_endog = pt.eye(n_endog)

    def check_params(self, data, a0, P0, c, d, T, Z, R, H, Q):
        """
        Apply any checks on validity of inputs. For most filters this is just the identity function.
        """
        c, d = self.initialize_intercepts(c, d, Z)
        return data, a0, P0, c, d, T, Z, R, H, Q

    @staticmethod
    def check_time_varying_shapes(data, sequence_params):
        """
        If any matrices are time-varying, make sure the number of matrices is equal to the number of time steps.
        # TODO: The PytensorRepresentation object puts the time dimension last, should the reshaping happen here in
            the Kalman filter, or in the StateSpaceModel, before passing into the KF?
        """
        n_steps = data.shape[0]
        return [
            assert_time_varying_dim_correct(param, pt.eq(param.shape[0], n_steps))
            for param in sequence_params
        ]

    def unpack_args(self, args):
        """
        The order of inputs to the inner scan function is not known, since some, all, or none of the input matrices
        can be time varying. The order arguments are fed to the inner function is sequences, outputs_info,
        non-sequences. This function works out which matrices are where, and returns a standardized order expected
        by the kalman_step function.

        The standard order is: y, a0, P0, c, d, T, Z, R, H, Q
        """
        # If there are no sequence parameters (all params are static),
        # no changes are needed, params will be in order.
        args = list(args)
        n_seq = len(self.seq_names)
        if n_seq == 0:
            return args

        # The first arg is always y
        y = args.pop(0)

        # There are always two outputs_info wedged between the seqs and non_seqs
        seqs, (a0, P0), non_seqs = args[:n_seq], args[n_seq : n_seq + 2], args[n_seq + 2 :]
        return_ordered = []
        for name in ["c", "d", "T", "Z", "R", "H", "Q"]:
            if name in self.seq_names:
                idx = self.seq_names.index(name)
                return_ordered.append(seqs[idx])
            else:
                idx = self.non_seq_names.index(name)
                return_ordered.append(non_seqs[idx])

        c, d, T, Z, R, H, Q = return_ordered

        return y, a0, P0, c, d, T, Z, R, H, Q

    def build_graph(
        self, data, a0, P0, T, Z, R, H, Q, c=None, d=None, mode=None
    ) -> List[TensorVariable]:
        """
        Construct the computation graph for the Kalman filter. See [1] for details.

        References
        ----------
        .. [1] Koopman, Siem Jan, Neil Shephard, and Jurgen A. Doornik. 1999.
           Statistical Algorithms for Models in State Space Using SsfPack 2.2.
           Econometrics Journal 2 (1): 107-60. doi:10.1111/1368-423X.00023.
        """
        self.mode = mode
        self.initialize_eyes(R, Z)

        data, a0, P0, *params = self.check_params(data, a0, P0, c, d, T, Z, R, H, Q)
        sequences, non_sequences, seq_names, non_seq_names = split_vars_into_seq_and_nonseq(
            params, PARAM_NAMES
        )

        self.seq_names = seq_names
        self.non_seq_names = non_seq_names

        if len(sequences) > 0:
            sequences = self.check_time_varying_shapes(data, sequences)

        results, updates = pytensor.scan(
            self.kalman_step,
            sequences=[data] + sequences,
            outputs_info=[None, a0, None, P0, None],
            non_sequences=non_sequences,
            name="forward_kalman_pass",
            mode=get_mode(mode),
        )

        filter_results = self._postprocess_scan_results(results, a0, P0)

        return filter_results

    @staticmethod
    def _postprocess_scan_results(results, a0, P0) -> List[TensorVariable]:
        """
        Transform the values returned by the Kalman Filter scan into a form expected by users. In particular:
            1. Append the initial state and covariance matrix to their respective Kalman predictions. This matches the
                output returned by Statsmodels state space models.
            2. Squeeze away the 3rd dimension from the filtered and predicted states, as well as the likelihoods.
        """
        (
            filtered_states,
            predicted_states,
            filtered_covariances,
            predicted_covariances,
            log_likelihoods,
        ) = results

        predicted_states = pt.concatenate([pt.atleast_3d(a0), predicted_states], axis=0)
        predicted_covariances = pt.concatenate([pt.atleast_3d(P0), predicted_covariances], axis=0)

        filter_results = [
            filtered_states,
            predicted_states,
            filtered_covariances,
            predicted_covariances,
            log_likelihoods.sum(),
            log_likelihoods.squeeze(),
        ]

        return filter_results

    @staticmethod
    def handle_missing_values(y, Z, H):
        """
        Adjust Z and H matrices according to [1] in the presence of missing data. Fill missing values with zeros
        to avoid propagating the NaNs. Return a flag if everything is missing (needed for numerical adjustments in the
        update methods)

        # TODO: Do the intercepts need to be masked as well?
        """
        nan_mask = pt.isnan(y)
        all_nan_flag = pt.all(nan_mask).astype(pytensor.config.floatX)

        W = pt.set_subtensor(pt.eye(y.shape[0])[nan_mask.ravel(), nan_mask.ravel()], 0.0)

        Z_masked = W.dot(Z)
        H_masked = W.dot(H)
        y_masked = pt.set_subtensor(y[nan_mask], 0.0)

        return y_masked, Z_masked, H_masked, all_nan_flag

    @staticmethod
    def predict(a, P, c, T, R, Q) -> Tuple[TensorVariable, TensorVariable]:
        a_hat = T.dot(a) + c
        P_hat = matrix_dot(T, P, T.T) + matrix_dot(R, Q, R.T)

        # Force P_hat to be symmetric
        P_hat = 0.5 * (P_hat + P_hat.T)

        return a_hat, P_hat

    @staticmethod
    def update(
        a, P, y, c, d, Z, H, all_nan_flag
    ) -> Tuple[TensorVariable, TensorVariable, TensorVariable]:
        raise NotImplementedError

    def kalman_step(
        self, *args
    ) -> Tuple[TensorVariable, TensorVariable, TensorVariable, TensorVariable, TensorVariable]:
        """
        The timing convention follows [1]. a0 and P0 are taken to be predicted states, so we begin
        with an update step rather than a predict step.
        References
        ----------
        .. [1] Durbin, J., and S. J. Koopman. Time Series Analysis by State Space Methods.
               2nd ed, Oxford University Press, 2012.
        """
        y, a, P, c, d, T, Z, R, H, Q = self.unpack_args(args)

        y_masked, Z_masked, H_masked, all_nan_flag = self.handle_missing_values(y, Z, H)

        a_filtered, P_filtered, ll = self.update(
            y=y_masked, a=a, c=c, d=d, P=P, Z=Z_masked, H=H_masked, all_nan_flag=all_nan_flag
        )

        a_hat, P_hat = self.predict(a=a_filtered, P=P_filtered, c=c, T=T, R=R, Q=Q)

        return a_filtered, a_hat, P_filtered, P_hat, ll


class StandardFilter(BaseFilter):
    def update(self, a, P, y, c, d, Z, H, all_nan_flag):
        """
        Conjugate update rule for the mean and covariance matrix, with log-likelihood en passant
        TODO: Verify these equations are correct if there are multiple endogenous variables.
        TODO: Is there a more elegant way to handle nans?
        """
        v = y - Z.dot(a) - d

        PZT = P.dot(Z.T)
        F = Z.dot(PZT) + H

        F_inv = pt.linalg.solve(
            F + self.eye_endog * all_nan_flag, self.eye_endog, assume_a="pos", check_finite=False
        )

        K = PZT.dot(F_inv)
        I_KZ = self.eye_states - K.dot(Z)

        a_filtered = a + K.dot(v)
        P_filtered = matrix_dot(I_KZ, P, I_KZ.T) + matrix_dot(K, H, K.T)  # Joseph form

        inner_term = matrix_dot(v.T, F_inv, v)
        ll = pt.switch(
            all_nan_flag,
            0.0,
            -0.5 * (MVN_CONST + pt.log(pt.linalg.det(F)) + inner_term).ravel()[0],
        )

        return a_filtered, P_filtered, ll


class CholeskyFilter(BaseFilter):
    def update(self, a, P, y, c, d, Z, H, all_nan_flag):
        v = y - Z.dot(a) - d

        PZT = P.dot(Z.T)

        # If everything is missing, F will be [[0]] and F_chol will raise an error, so add identity to avoid the error
        F = Z.dot(PZT) + H + self.eye_endog * all_nan_flag

        F_chol = pt.linalg.cholesky(F)

        # If everything is missing, K = 0, IKZ = I
        K = solve_lower_triangular(F_chol.T, solve_lower_triangular(F_chol, PZT.T)).T * (
            1 - all_nan_flag
        )
        I_KZ = self.eye_states - K.dot(Z)

        a_filtered = a + K.dot(v)
        P_filtered = matrix_dot(I_KZ, P, I_KZ.T) + matrix_dot(K, H, K.T)

        inner_term = solve_lower_triangular(F_chol.T, solve_lower_triangular(F_chol, v))
        n = y.shape[0]

        ll = pt.switch(
            all_nan_flag,
            0.0,
            (
                -0.5 * (n * MVN_CONST + (v.T @ inner_term).ravel()) - pt.log(pt.diag(F_chol)).sum()
            ).ravel()[0],
        )

        return a_filtered, P_filtered, ll


class SingleTimeseriesFilter(BaseFilter):
    """
    If there is only a single observed timeseries, regardless of the number of hidden states, there is no need to
    perform a matrix inversion anywhere in the filter.
    """

    def check_params(self, data, a0, P0, c, d, T, Z, R, H, Q):
        c, d = self.initialize_intercepts(c, d, Z)
        data = assert_data_is_1d(data, pt.eq(data.shape[1], 1))

        return data, a0, P0, c, d, T, Z, R, H, Q

    def update(self, a, P, y, c, d, Z, H, all_nan_flag):
        # y, v are scalar, but a might not be
        y_hat = Z.dot(a).ravel() - d
        v = y - y_hat

        PZT = P.dot(Z.T)

        # F is scalar, K is a column vector
        F = (Z.dot(PZT) + H).ravel() + all_nan_flag
        K = PZT / F

        I_KZ = self.eye_states - K.dot(Z)

        a_filtered = a + (K * v)
        P_filtered = matrix_dot(I_KZ, P, I_KZ.T) + matrix_dot(K, H, K.T)

        ll = pt.switch(all_nan_flag, 0.0, -0.5 * (MVN_CONST + pt.log(F) + v**2 / F)).ravel()[0]

        return a_filtered, P_filtered, ll


class SteadyStateFilter(BaseFilter):
    """
    This filter avoids the need to invert the covariance matrix of innovations at each time step by solving the
    Discrete Algebraic Riccati Equation associated with the filtering problem once and for all at initialization and
    uses the resulting steady-state covariance matrix in each step.

    The innovation covariance matrix will always converge to the steady state value as T -> oo, so this filter will
    only have differences from the standard approach in the early steps (T < 10?). A process of "learning" is lost.
    """

    def build_graph(self, data, a0, P0, T, Z, R, H, Q, c=None, d=None, mode=None):
        """
        Need to override the base step to add an argument to self.update, passing F_inv at every step.
        """
        self.mode = mode
        self.initialize_eyes(R, Z)

        data, a0, P0, *params = self.check_params(data, a0, P0, c, d, T, Z, R, H, Q)
        sequences, non_sequences, seq_names, non_seq_names = split_vars_into_seq_and_nonseq(
            params, PARAM_NAMES
        )
        self.seq_names = seq_names
        self.non_seq_names = non_seq_names
        c, d, T, Z, R, H, Q = params

        if len(sequences) > 0:
            assert ValueError(
                "All system matrices must be time-invariant to use the SteadyStateFilter"
            )

        P_steady = solve_discrete_are(T.T, Z.T, matrix_dot(R, Q, R.T), H)
        F = matrix_dot(Z, P_steady, Z.T) + H
        F_inv = pt.linalg.solve(F, pt.eye(F.shape[0]), assume_a="pos", check_finite=False)

        results, updates = pytensor.scan(
            self.kalman_step,
            sequences=[data],
            outputs_info=[None, a0, None, P_steady, None],
            non_sequences=[c, d, F_inv, T, Z, R, H, Q],
            name="forward_kalman_pass",
            mode=get_mode(mode),
        )

        return self._postprocess_scan_results(results, a0, P0)

    def update(self, a, P, c, d, F_inv, y, Z, H, all_nan_flag):
        v = y - Z.dot(a)

        PZT = P.dot(Z.T)

        F = Z.dot(PZT) + H
        K = PZT.dot(F_inv)

        I_KZ = self.eye_states - K.dot(Z)

        a_filtered = a + K.dot(v)
        P_filtered = matrix_dot(I_KZ, P, I_KZ.T) + matrix_dot(K, H, K.T)

        inner_term = matrix_dot(v.T, F_inv, v)
        ll = pt.switch(
            all_nan_flag,
            0.0,
            -0.5 * (MVN_CONST + pt.log(pt.linalg.det(F)) + inner_term).ravel()[0],
        )

        return a_filtered, P_filtered, ll

    def kalman_step(self, y, a, P, c, d, F_inv, T, Z, R, H, Q):
        """
        Need to override the base step to add an argument to self.update, passing F_inv at every step.
        """

        y_masked, Z_masked, H_masked, all_nan_flag = self.handle_missing_values(y, Z, H)
        a_filtered, P_filtered, ll = self.update(
            y=y_masked,
            a=a,
            P=P,
            c=c,
            d=d,
            F_inv=F_inv,
            Z=Z_masked,
            H=H_masked,
            all_nan_flag=all_nan_flag,
        )

        a_hat, P_hat = self.predict(a=a_filtered, P=P_filtered, c=c, T=T, R=R, Q=Q)

        return a_filtered, a_hat, P_filtered, P_hat, ll


class UnivariateFilter(BaseFilter):
    """
    The univariate kalman filter, described in [1], section 6.4.2, avoids inversion of the F matrix, as well as two
    matrix multiplications, at the cost of an additional loop. Note that the name doesn't mean there's only one
    observed time series, that's the SingleTimeSeries filter. This is called univariate because it updates the state
    mean and covariance matrices one variable at a time, using an inner-inner loop.

    This is useful when states are perfectly observed, because the F matrix can easily become degenerate in these cases.

    References
    ----------
    .. [1] Durbin, J., and S. J. Koopman. Time Series Analysis by State Space Methods.
            2nd ed, Oxford University Press, 2012.

    """

    @staticmethod
    def _univariate_inner_filter_step(y, Z_row, d_row, sigma_H, nan_flag, a, P):
        Z_row = Z_row[None, :]
        v = y - Z_row.dot(a) - d_row

        PZT = P.dot(Z_row.T)
        F = Z_row.dot(PZT) + sigma_H

        F_zero_flag = pt.or_(pt.eq(F, 0), nan_flag)

        # This should easier than trying to dodge the log(F) and 1 / F with a switch
        F = F + 1e-8 * F_zero_flag

        # If F is zero (implies y is NAN or another degenerate case), then we want:
        # K = 0, a = a, P = P, ll = 0
        K = PZT / F * (1 - F_zero_flag)
        a_filtered = a + K * v * (1 - F_zero_flag)
        P_filtered = P - pt.outer(K, K) * F * (1 - F_zero_flag)
        ll_inner = (pt.log(F) + v**2 / F) * (1 - F_zero_flag)

        return a_filtered, P_filtered, ll_inner

    def kalman_step(self, y, a, P, c, d, T, Z, R, H, Q):
        y = y[:, None]
        nan_mask = pt.isnan(y).ravel()

        W = pt.set_subtensor(pt.eye(y.shape[0])[nan_mask, nan_mask], 0.0)
        Z_masked = W.dot(Z)
        H_masked = W.dot(H)
        y_masked = pt.set_subtensor(y[nan_mask], 0.0)

        result, updates = pytensor.scan(
            self._univariate_inner_filter_step,
            sequences=[y_masked, Z_masked, d, pt.diag(H_masked), nan_mask],
            outputs_info=[a, P, None],
            mode=get_mode(self.mode),
        )

        a_filtered, P_filtered, ll_inner = result
        a_filtered, P_filtered = a_filtered[-1], P_filtered[-1]

        a_hat, P_hat = self.predict(a=a_filtered, P=P_filtered, c=c, T=T, R=R, Q=Q)

        ll = -0.5 * ((pt.neq(ll_inner, 0).sum()) * MVN_CONST + ll_inner.sum())

        return a_filtered, a_hat, P_filtered, P_hat, ll
