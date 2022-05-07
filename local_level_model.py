from kalman_filter import KalmanFilter
from statespace import AesaraRepresentation, PyMCStateSpace
import numpy as np
import aesara
import aesara.tensor as at


class BayesianLocalLevel(PyMCStateSpace):

    def __init__(self, data):
        # Model order
        k_states = k_posdef = 2

        super().__init__(data, k_states, k_posdef)

        # Initialize the statespace
        self.ssm = AesaraRepresentation(data, k_states=k_states, k_posdef=k_posdef)

        # Initialize the matrices
        self.ssm['design'] = np.array([[1.0, 0.0]])
        self.ssm['transition'] = np.array([[1.0, 1.0],
                                           [0.0, 1.0]])
        self.ssm['selection'] = np.eye(k_states)

        self.ssm['initial_state'] = np.array([[0.0],
                                              [0.0]])
        self.ssm['initial_state_cov'] = np.array([[1.0, 0.0],
                                                  [0.0, 1.0]])

        # Cache some indices
        self._state_cov_idx = ('state_cov',) + np.diag_indices(k_posdef)

        self.ssm[self._state_cov_idx] = 1.0

        self.compile_aesara_functions()

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
        # initial states
        self.ssm['initial_state', :, 0] = theta[:2]

        # initial covariance
        self.ssm['initial_state_cov', :, :] = theta[2:6].reshape((2, 2))

        # Observation covariance
        self.ssm['obs_cov', 0, 0] = theta[6]

        # State covariance
        self.ssm[self._state_cov_idx] = theta[7:]

    def compile_aesara_functions(self) -> None:
        theta = at.vector('theta')

        self.update(theta)
        states, covariances, log_likelihood = self.kalman_filter.build_graph(self.data, *self.unpack_statespace())

        self.f_loglike = aesara.function([theta], log_likelihood)
        self.f_loglike_grad = aesara.function([theta], at.grad(log_likelihood, theta))

        self.f_y_hat = aesara.function([theta], states)
        self.f_cov_hat = aesara.function([theta], covariances)

    def print_model_description(self):
        system_matrices = self.unpack_statespace()
        names = ['Initial States', 'Initial State Covariance', 'Hidden State Covariance',
                 'Observed State Covariance', 'Transition Matrix', 'Selection Matrix',
                 'Design Matrix']

        print(
            f'State Space System with {self.k_endog} observed states and {self.k_states - self.k_endog} hidden states')
        print('Current system matrices:')

        with np.printoptions(linewidth=1000, precision=3, suppress=True):
            for name, matrix in zip(names, system_matrices):
                print(name)
                print(matrix)
                print('\n')

        return ''
