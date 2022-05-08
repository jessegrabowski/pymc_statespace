import aesara.tensor as at
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Tuple, List, Type

from pymc_statespace.filters.kalman_filter import KalmanFilter

KeyLike = Tuple[str | int] | str


class AesaraRepresentation:
    data = at.matrix(name='Data')
    design = at.tensor3(name='design')
    obs_cov = at.tensor3(name='obs_cov')
    transition = at.tensor3(name='transition')
    selection = at.tensor3(name='selection')
    state_cov = at.tensor3(name='state_cov')

    def __init__(self,
                 data: ArrayLike,
                 k_states: int,
                 k_posdef: int,
                 design: Optional[ArrayLike] = None,
                 obs_intercept: Optional[ArrayLike] = None,
                 obs_cov=None,
                 transition=None,
                 state_intercept=None,
                 selection=None,
                 state_cov=None,
                 initial_state=None,
                 initial_state_cov=None) -> None:
        """
        A representation of a State Space model, in Aesara. Shamelessly copied from the Statsmodels.api implementation
        found here:

        https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/statespace/representation.py

        Parameters
        ----------
        data: ArrayLike
            Array of observed data (called exog in statsmodels)
        k_states: int
            Number of hidden states
        k_posdef: int
            Number of states that have exogenous shocks; also the rank of the selection matrix R.
        design: ArrayLike, optional
            Design matrix, denoted 'Z' in [1].
        obs_intercept: ArrayLike, optional
            Constant vector in the observation equation, denoted 'd' in [1]. Currently
            not used.
        obs_cov: ArrayLike, optional
            Covariance matrix for multivariate-normal errors in the observation equation. Denoted 'H' in
            [1].
        transition: ArrayLike, optional
            Transition equation that updates the hidden state between time-steps. Denoted 'T' in [1].
        state_intercept: ArrayLike, optional
            Constant vector for the observation equation, denoted 'c' in [1]. Currently not used.
        selection: ArrayLike, optional
            Selection matrix that matches shocks to hidden states, denoted 'R' in [1]. This is the identity
            matrix when k_posdef = k_states.
        state_cov: ArrayLike, optional
            Covariance matrix for state equations, denoted 'Q' in [1]. Null matrix when there is no observation
            noise.
        initial_state: ArrayLike, optional
            Experimental setting to allow for Bayesian estimation of the initial state, denoted `alpha_0` in [1]. Default
            It should potentially be removed in favor of the closed-form diffuse initialization.
        initial_state_cov: ArrayLike, optional
            Experimental setting to allow for Bayesian estimation of the initial state, denoted `P_0` in [1]. Default
            It should potentially be removed in favor of the closed-form diffuse initialization.

        References
        ----------
        .. [1] Durbin, James, and Siem Jan Koopman. 2012.
            Time Series Analysis by State Space Methods: Second Edition.
            Oxford University Press.
        """

        self.data = data
        self.k_states = k_states
        self.k_posdef = k_posdef if k_posdef is not None else k_states

        self.n_obs, self.k_endog = data.shape

        # The last dimension is for time varying matrices; it could be n_obs. Not thinking about that now.
        self.shapes = {
            'data': (self.k_endog, self.n_obs),
            'design': (self.k_endog, self.k_states, 1),
            'obs_intercept': (self.k_endog, 1),
            'obs_cov': (self.k_endog, self.k_endog, 1),
            'transition': (self.k_states, self.k_states, 1),
            'state_intercept': (self.k_states, 1),
            'selection': (self.k_states, self.k_posdef, 1),
            'state_cov': (self.k_posdef, self.k_posdef, 1),
            'initial_state': (self.k_states, 1, 1),
            'initial_state_cov': (self.k_states, self.k_states, 1)
        }

        # Initialize the representation matrices
        # TODO: Can this API be improved (using scope and setattr seems hacky?)
        scope = locals()
        for name, shape in self.shapes.items():
            if name == 'data':
                continue
            setattr(self, name, at.zeros(shape))
            #             setattr(self, name, np.zeros(shape))

            if scope[name] is not None:
                matrix = self._numpy_to_aesara(name, scope[name])
                setattr(self, name, matrix)

    def _validate_key(self, key: KeyLike) -> None:
        if key not in self.shapes:
            raise IndexError(f'{key} is an invalid state space matrix name')

    def _add_time_dim_to_slice(self, name: str, slice_: List[int] | Tuple[int], n_dim: int) -> Tuple[int]:
        if self.shapes[name][-1] == 1 and len(slice_) <= (n_dim - 1):
            return tuple(slice_) + (0,)

    @staticmethod
    def _validate_key_and_get_type(key: KeyLike) -> Type[str]:
        if isinstance(key, tuple) and not isinstance(key[0], str):
            raise IndexError('First index must the name of a valid state space matrix.')

        return type(key)

    def _validate_matrix_shape(self, name: str, X: ArrayLike) -> None:
        *expected_shape, time_dim = self.shapes[name]
        expected_shape = tuple(expected_shape)

        # Assume X is always 2d, even if there is a time-varying component, i.e. X always gives
        # the constant values over time.
        if expected_shape != X.shape:
            raise ValueError(f'Array provided for {name} has shape {X.shape}, expected {expected_shape}')

    def _numpy_to_aesara(self, name: str, X: ArrayLike) -> at.TensorVariable:
        self._validate_matrix_shape(name, X)
        # Add a time dimension if one isn't provided
        if X.ndim == 2:
            X = X[:, :, None]

        return at.as_tensor(X, name=name)

    def __getitem__(self, key: KeyLike) -> at.TensorVariable:
        _type = self._validate_key_and_get_type(key)

        # Case 1: user asked for an entire matrix by name
        if _type is str:
            self._validate_key(key)
            matrix = getattr(self, key)

            if self.shapes[key][-1] == 1:
                return matrix[(slice(None),) * (matrix.ndim - 1) + (0,)]

            else:
                return matrix

        # Case 2: user asked for a particular matrix and some slices of it
        elif _type is tuple:
            name, *slice_ = key
            self._validate_key(name)

            matrix = getattr(self, name)
            slice_ = self._add_time_dim_to_slice(name, slice_, matrix.ndim)

            return matrix[slice_]

        # Case 3: There is only one slice index, but it's not a string
        else:
            raise IndexError('First index must the name of a valid state space matrix.')

    def __setitem__(self, key: KeyLike, value: float | int | ArrayLike) -> None:
        _type = type(key)
        # Case 1: key is a string: we are setting an entire matrix.
        if _type is str:
            self._validate_key(key)
            if isinstance(value, np.ndarray):
                value = self._numpy_to_aesara(key, value)
            setattr(self, key, value)

        elif _type is tuple:
            name, *slice_ = key
            self._validate_key(name)

            matrix = getattr(self, name)

            slice_ = self._add_time_dim_to_slice(name, slice_, matrix.ndim)

            matrix = at.set_subtensor(matrix[slice_], value)
            #             matrix[slice_] = value
            setattr(self, name, matrix)


class PyMCStateSpace:
    def __init__(self, data, k_states, k_posdef):
        self.data = data
        self.n_obs, self.k_endog = data.shape
        self.k_states = k_states
        self.k_posdef = k_posdef

        # All models contain a state space representation and a Kalman filter
        self.ssm = AesaraRepresentation(data, k_states, k_posdef)
        self.kalman_filter = KalmanFilter()

        # Placeholders for the aesara functions that will return the predicted state, covariance, and log likelihood
        # given parameter vector theta

        self.log_likelihood = None
        self.filtered_states = None
        self.filtered_covarainces = None

    def unpack_statespace(self):
        a0 = self.ssm['initial_state']
        P0 = self.ssm['initial_state_cov']
        Q = self.ssm['state_cov']
        H = self.ssm['obs_cov']
        T = self.ssm['transition']
        R = self.ssm['selection']
        Z = self.ssm['design']

        return a0, P0, Q, H, T, R, Z

    def _clear_existing_graphs(self):
        if self.log_likelihood is not None:
            del self.log_likelihood
            self.log_likelihood = None

        if self.filtered_states is not None:
            del self.filtered_states
            self.filtered_states = None

        if self.filtered_covarainces is not None:
            del self.filtered_covarainces
            self.filtered_covarainces = None

    def update(self, theta: at.TensorVariable) -> None:
        """
        Put parameter values from vector theta into the correct positions in the state space matrices.
        TODO: Can this be done using variable names to avoid the need to ravel and concatenate all RVs in the
              PyMC model?

        Parameters
        ----------
        theta: TensorVariable
            Vector of all variables in the state space model
        """
        raise NotImplementedError

    def build_statespace_graph(self, theta: at.TensorVariable) -> None:
        """
        Given parameter vector theta, constructs the full computational graph describing the state space model.

        Parameters
        ----------
        theta: TensorVariable
            Symbolic tensor varaible representing all unknown parameters among all state space matrices in the model.
        """

        self._clear_existing_graphs()
        self.update(theta)
        states, covariances, log_likelihood = self.kalman_filter.build_graph(self.data, *self.unpack_statespace())

        self.log_likelihood = log_likelihood
        self.filtered_states = states
        self.filtered_covarainces = covariances
