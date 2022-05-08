from pymc_statespace.core.statespace import AesaraRepresentation, PyMCStateSpace
import numpy as np
import aesara.tensor as at
from typing import Tuple


class BayesianARMA(PyMCStateSpace):

    def __init__(self,
                 data,
                 order: Tuple[int, int]):

        # Model order
        self.p, self.q = order

        k_states = max(self.p, self.q+1)
        k_posdef = 1

        super().__init__(data, k_states, k_posdef)

        # Initialize the statespace
        self.ssm = AesaraRepresentation(data, k_states=k_states, k_posdef=k_posdef)

        # Initialize the matrices
        self.ssm['design'] = np.r_[[1.0], np.zeros(k_states-1)][None]

        self.ssm['transition'] = np.eye(k_states, k=1)

        self.ssm['selection'] = np.r_[[[1.0]], np.zeros(k_states-1)[:, None]]

        self.ssm['initial_state'] = np.zeros(k_states)[:, None]

        self.ssm['initial_state_cov'] = np.eye(k_states)

        # Cache some indices
        self._state_cov_idx = ('state_cov',) + np.diag_indices(k_posdef)
        self._ar_param_idx = ('transition',) + (np.arange(self.p, dtype=int), np.zeros(self.p, dtype=int))
        self._ma_param_idx = ('selection',) + (np.arange(1, self.q+1, dtype=int), np.zeros(self.q, dtype=int))

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
        cursor = 0

        # initial states
        param_slice = slice(cursor, cursor + self.k_states)
        cursor += self.k_states
        self.ssm['initial_state', :, 0] = theta[param_slice]

        # initial covariance
        param_slice = slice(cursor, cursor + self.k_states ** 2)
        cursor += self.k_states ** 2
        self.ssm['initial_state_cov', :, :] = theta[param_slice].reshape((self.k_states, self.k_states))

        # State covariance
        param_slice = slice(cursor, cursor + 1)
        cursor += 1
        self.ssm[self._state_cov_idx] = theta[param_slice]

        # AR parameteres
        param_slice = slice(cursor, cursor + self.p)
        cursor += self.p
        self.ssm[self._ar_param_idx] = theta[param_slice]

        # MA parameters
        param_slice = slice(cursor, cursor + self.q)
        cursor += self.q
        self.ssm[self._ma_param_idx] = theta[param_slice]


