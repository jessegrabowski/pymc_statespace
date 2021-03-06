import aesara.tensor as at
import aesara

from pymc_statespace.filters import StandardFilter, UnivariateFilter, SteadyStateFilter, KalmanSmoother, SingleTimeseriesFilter
from pymc_statespace.core.representation import AesaraRepresentation
from pymc_statespace.utils.simulation import conditional_simulation, unconditional_simulations
from warnings import simplefilter, catch_warnings

import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Tuple, List

from pymc.model import modelcontext
import pymc as pm

FILTER_FACTORY = {'standard': StandardFilter, 'univariate': UnivariateFilter, 'steady_state': SteadyStateFilter,
                  'single':SingleTimeseriesFilter}


class PyMCStateSpace:
    def __init__(self, data, k_states, k_posdef, filter_type='standard'):
        self.data = data
        self.n_obs, self.k_endog = data.shape
        self.k_states = k_states
        self.k_posdef = k_posdef

        # All models contain a state space representation and a Kalman filter
        self.ssm = AesaraRepresentation(data, k_states, k_posdef)

        if filter_type.lower() not in FILTER_FACTORY.keys():
            raise NotImplementedError('The following are valid filter types: ' + ', '.join(list(FILTER_FACTORY.keys())))

        if filter_type == 'single' and self.k_endog > 1:
            raise ValueError('Cannot use filter_type = "single" with multiple observed time series')

        self.kalman_filter = FILTER_FACTORY[filter_type.lower()]()
        self.kalman_smoother = KalmanSmoother()

        # Placeholders for the aesara functions that will return the predicted state, covariance, and log likelihood
        # given parameter vector theta

        self.log_likelihood = None
        self.ll_obs = None

        self.filtered_states = None
        self.predicted_states = None

        self.filtered_covariances = None
        self.predicted_covariances = None

    def unpack_statespace(self):
        a0 = self.ssm['initial_state']
        P0 = self.ssm['initial_state_cov']
        Q = self.ssm['state_cov']
        H = self.ssm['obs_cov']
        T = self.ssm['transition']
        R = self.ssm['selection']
        Z = self.ssm['design']

        return a0, P0, T, Z, R, H, Q

    @property
    def param_names(self) -> List[str]:
        return NotImplementedError

    def update(self, theta: at.TensorVariable) -> None:
        """
        Put parameter values from vector theta into the correct positions in the state space matrices.

        Parameters
        ----------
        theta: TensorVariable
            Vector of all variables in the state space model
        """
        raise NotImplementedError

    def gather_required_random_variables(self) -> at.TensorVariable:
        """
        Iterates through random variables in the model on the context stack, matches their names with the statespace
        model's named parameters, and returns a single vector of parameters to pass to the update method.

        Important point is that the *order of the variables matters*. Update will expect that the theta vector will be
        organized as variables are listed in param_names. This could be improved...

        Returns
        ----------
        theta: TensorVariable
            A p x 1 theano vector containing all parameters to be estimated among all state space matrices in the
            system.
        """

        theta = []
        pymc_model = modelcontext(None)
        found_params = []
        with pymc_model:
            for param_name in self.param_names:
                for param in pymc_model.rvs_to_values:
                    if param.name == param_name:
                        theta.append(param.ravel())
                        found_params.append(param.name)
                for param in pymc_model.deterministics:
                    if param.name == param_name:
                        theta.append(param.ravel())
                        found_params.append(param.name)

        missing_params = set(self.param_names) - set(found_params)
        if len(missing_params) > 0:
            raise ValueError("The following required model parameters were not found in the PyMC model:" + ', '.join(
                param for param in list(missing_params)
            ))
        return at.concatenate(theta)

    def build_statespace_graph(self) -> None:
        """
        Given parameter vector theta, constructs the full computational graph describing the state space model.
        Must be called inside a PyMC model context.
        """

        pymc_model = modelcontext(None)
        with pymc_model:
            theta = self.gather_required_random_variables()
            self.update(theta)

            # filtered_states, predicted_states, smoothed_states, \
            # filtered_covariances, predicted_covariances, smoothed_covariances, \
            # log_likelihood, ll_obs = self.kalman_filter.build_graph(at.as_tensor_variable(self.data),
            #                                                         *self.unpack_statespace())

            # filtered_states, predicted_states, \
            #     filtered_covariances, predicted_covariances,\
            #     log_likelihood, ll_obs = self.kalman_filter.build_graph(at.as_tensor_variable(self.data),
            #                                                             *self.unpack_statespace())

            # pm.Deterministic('filtered_states', filtered_states)
            # pm.Deterministic('predicted_states', predicted_states)
            # pm.Deterministic('smoothed_states', smoothed_states)

            # pm.Deterministic('predicted_covariances', predicted_covariances)
            # pm.Deterministic('filtered_covariances', filtered_covariances)
            # pm.Deterministic('smoothed_covariances', smoothed_covariances)

            # pm.Potential('log_likelihood', log_likelihood)

            return self.kalman_filter.build_graph(at.as_tensor_variable(self.data),
                                                  *self.unpack_statespace())

    @staticmethod
    def sample_conditional_prior(filter_output='filtered',
                                 n_simulations=100,
                                 prior_samples=500) -> ArrayLike:
        """
        Sample from the conditional prior; that is, given parameter draws from the prior distribution, compute kalman
        filtered trajectories. Trajectories are drawn from a single multivariate normal with mean and covariance
        computed via either the kalman filter, smoother, or predictions.

        Parameters
        ----------
        filter_output: string, default = 'filtered'
            One of 'filtered', 'smoothed', or 'predicted'. Corresponds to which Kalman filter output you would like to
            sample from.
        n_simulations: int, default = 100
            The number of simulations to run for each prior parameter sample drawn. Total trajectories returned by this
            function will be n_simulations x prior_samples
        prior_samples: int, default = 500
            The number of samples to draw from the prior distribution, passed to pm.sample_prior_predictive. Defaults
            to the PyMC default of 500.

        Returns
        -------
        simulations: ArrayLike
            A numpy array of shape (n_simulations x prior_samples, n_timesteps, n_states) with simulated trajectories.

        """
        if filter_output.lower() not in ['filtered', 'predicted', 'smoothed']:
            raise ValueError(
                f'filter_output should be one of filtered, predicted, or smoothed, recieved {filter_output}')

        pymc_model = modelcontext(None)
        with pymc_model:
            with catch_warnings():
                simplefilter('ignore', category=UserWarning)
                cond_prior = pm.sample_prior_predictive(samples=prior_samples,
                                                        var_names=[f'{filter_output}_states',
                                                                   f'{filter_output}_covariances'])

        _, _, n, k, *_ = cond_prior.prior[f'{filter_output}_states'].shape
        mus = cond_prior.prior[f'{filter_output}_states'].values.squeeze().reshape(-1, n * k)
        covs = cond_prior.prior[f'{filter_output}_covariances'].values.squeeze().reshape(-1, n, k, k)

        simulations = conditional_simulation(mus, covs, n, k, n_simulations)

        return simulations

    @staticmethod
    def sample_conditional_posterior(trace,
                                     filter_output: str = 'filtered',
                                     n_simulations: int = 100,
                                     posterior_samples: Optional[float | int] = None):
        """
        Sample from the conditional posterior; that is, given parameter draws from the posterior distribution,
        compute kalman filtered trajectories. Trajectories are drawn from a single multivariate normal with mean and
        covariance computed via either the kalman filter, smoother, or predictions.

        Parameters
        ----------
        trace: xarray
            PyMC trace idata object. Should be an xarray returned by pm.sample() with return_inferencedata = True.
        filter_output: string, default = 'filtered'
            One of 'filtered', 'smoothed', or 'predicted'. Corresponds to which Kalman filter output you would like to
            sample from.
        n_simulations: int, default = 100
            The number of simulations to run for each prior parameter sample drawn. Total trajectories returned by this
            function will be n_simulations x prior_samples
        posterior_samples: int or float, default = None
            A number of subsamples to draw from the posterior trace. If None, all samples in the trace are used. If an
            integer, that number of samples will be drawn with replacement (from among all chains) from the trace. If a
            float between 0 and 1, that fraction of total draws in the trace will be sampled.

        Returns
        -------
        simulations: ArrayLike
            A numpy array of shape (n_simulations x prior_samples, n_timesteps, n_states) with simulated trajectories.

        """
        if filter_output.lower() not in ['filtered', 'predicted', 'smoothed']:
            raise ValueError(
                f'filter_output should be one of filtered, predicted, or smoothed, recieved {filter_output}')

        chains, draws, n, k, *_ = trace.posterior[f'{filter_output}_states'].shape
        posterior_size = chains * draws
        if isinstance(posterior_samples, float):
            if posterior_samples > 1.0 or posterior_samples < 0.0:
                raise ValueError('If posterior_samples is a float, it should be between 0 and 1, representing the '
                                 'fraction of total posterior samples to re-sample.')
            posterior_samples = int(np.floor(posterior_samples * posterior_size))

        elif posterior_samples is None:
            posterior_samples = posterior_size

        resample_idxs = np.random.randint(0, posterior_size, size=posterior_samples)

        mus = trace.posterior[f'{filter_output}_states'].values.squeeze().reshape(-1, n * k)[resample_idxs]
        covs = trace.posterior[f'{filter_output}_covariances'].values.squeeze().reshape(-1, n, k, k)[resample_idxs]

        simulations = conditional_simulation(mus, covs, n, k, n_simulations)

        return simulations

    def sample_unconditional_prior(self,
                                   n_steps=100,
                                   n_simulations=100,
                                   prior_samples=500) -> Tuple[ArrayLike, ArrayLike]:
        """
        Draw unconditional sample trajectories according to state space dynamics, using random samples from the prior
        distribution over model parameters. The state space update equations are:

            X[t+1] = T @ X[t] + R @ eta[t], eta ~ N(0, Q)
            Y[t] = Z @ X[t] + nu[t], nu ~ N(0, H)

        Parameters
        ----------
        n_steps: int, default = 100
            Number of time steps to simulate
        n_simulations: int, default = 100
            Number of stochastic simulations to run for each parameter draw.
        prior_samples: int, default = 500
            Number of parameter draws from the prior distribution, passed to pm.sample_prior_predictive. Defaults to
            the PyMC default of 500.

        Returns
        -------
        simulated_states: ArrayLike
            Numpy array of shape (prior_samples * n_simulations, n_steps, n_states), corresponding to the unobserved
            states in the state-space system, X in the equations above
        simulated_data: ArrayLike
            Numpy array of shape (prior_samples * n_simulations, n_steps, n_observed), corresponding to the observed
            states in the state-space system, Y in the equations above.
        """
        pymc_model = modelcontext(None)

        with pymc_model:
            with catch_warnings():
                simplefilter('ignore', category=UserWarning)
                prior_params = pm.sample_prior_predictive(var_names=self.param_names, samples=prior_samples)

        rvs_on_graph = []
        with pymc_model:
            for param_name in self.param_names:
                for param in pymc_model.rvs_to_values:
                    if param.name == param_name:
                        rvs_on_graph.append(param)
                for param in pymc_model.deterministics:
                    if param.name == param_name:
                        rvs_on_graph.append(param)

        # TODO: This is pretty hacky, ask on the forums if there is a better solution
        matrix_update_funcs = [aesara.function(rvs_on_graph, [X], on_unused_input='ignore') for X in
                               self.unpack_statespace()]

        # Take the 0th element to remove the chain dimension
        thetas = [prior_params.prior[var].values[0] for var in self.param_names]
        simulated_states, simulated_data = unconditional_simulations(thetas, matrix_update_funcs,
                                                                     n_steps=n_steps, n_simulations=n_simulations)

        return simulated_states, simulated_data

    def sample_unconditional_posterior(self, trace,
                                       n_steps=100,
                                       n_simulations=100,
                                       posterior_samples=None) -> Tuple[ArrayLike, ArrayLike]:
        """
        Draw unconditional sample trajectories according to state space dynamics, using random samples from the
        posterior distribution over model parameters. The state space update equations are:

            X[t+1] = T @ X[t] + R @ eta[t], eta ~ N(0, Q)
            Y[t] = Z @ X[t] + nu[t], nu ~ N(0, H)

        Parameters
        ----------
        trace: xarray
            PyMC trace idata object. Should be an xarray returned by pm.sample() with return_inferencedata = True.
        n_steps: int, default = 100
            Number of time steps to simulate
        n_simulations: int, default = 100
            Number of stochastic simulations to run for each parameter draw.
        posterior_samples: int or float, default = None
            A number of subsamples to draw from the posterior trace. If None, all samples in the trace are used. If an
            integer, that number of samples will be drawn with replacement (from among all chains) from the trace. If a
            float between 0 and 1, that fraction of total draws in the trace will be sampled.

        Returns
        -------
        simulations: ArrayLike
            A numpy array of shape (n_simulations x prior_samples, n_timesteps, n_states) with simulated trajectories.

        """

        chains = trace.posterior.dims['chain']
        draws = trace.posterior.dims['draw']

        posterior_size = chains * draws
        if isinstance(posterior_samples, float):
            if posterior_samples > 1.0 or posterior_samples < 0.0:
                raise ValueError('If posterior_samples is a float, it should be between 0 and 1, representing the '
                                 'fraction of total posterior samples to re-sample.')
            posterior_samples = int(np.floor(posterior_samples * posterior_size))

        elif posterior_samples is None:
            posterior_samples = posterior_size

        resample_idxs = np.random.randint(0, posterior_size, size=posterior_samples)

        pymc_model = modelcontext(None)

        rvs_on_graph = []
        with pymc_model:
            for param_name in self.param_names:
                for param in pymc_model.rvs_to_values:
                    if param.name == param_name:
                        rvs_on_graph.append(param)
                for param in pymc_model.deterministics:
                    if param.name == param_name:
                        rvs_on_graph.append(param)

        matrix_update_funcs = [aesara.function(rvs_on_graph, [X], on_unused_input='ignore') for X in
                               self.unpack_statespace()]

        thetas = [trace.posterior[var].values for var in self.param_names]
        thetas = [arr.reshape(-1, *arr.shape[2:])[resample_idxs] for arr in thetas]

        simulated_states, simulated_data = unconditional_simulations(thetas, matrix_update_funcs,
                                                                     n_steps=n_steps, n_simulations=n_simulations)

        return simulated_states, simulated_data
