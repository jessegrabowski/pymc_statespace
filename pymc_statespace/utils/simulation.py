from numba import njit
from pymc_statespace.utils.numba_linalg import numba_block_diagonal
import numpy as np


@njit
def numba_mvn_draws(mu, cov):
    samples = np.random.randn(*mu.shape)
    L = np.linalg.cholesky(cov)

    return mu + L @ samples


@njit
def conditional_simulation(mus, covs, n, k, n_simulations=100):
    simulations = np.empty((n * n_simulations, n, k))
    for i in range(n):
        for j in range(n_simulations):
            sim = numba_mvn_draws(mus[i], numba_block_diagonal(covs[i]))
            simulations[(i * n_simulations + j), :, :] = sim.reshape(n, k)
    return simulations


@njit
def simulate_statespace(T, Z, R, H, Q, n_steps):
    n_obs, n_states = Z.shape
    state_noise = np.random.randn(n_steps, n_states)
    obs_noise = np.random.randn(n_steps, n_obs)

    state_chol = np.linalg.cholesky(Q)
    obs_chol = np.linalg.cholesky(H)

    state_innovations = state_noise @ state_chol
    obs_innovations = obs_noise @ obs_chol

    simulated_states = np.zeros((n_steps, n_states))
    simulated_obs = np.zeros((n_steps, n_obs))

    for t in range(1, n_steps):
        simulated_states[t] = T @ simulated_states[t - 1] + R @ state_innovations[t]
        simulated_obs[t] = Z @ simulated_states[t - 1] + obs_innovations[t]

    return simulated_states, simulated_obs


def unconditional_simulations(thetas, update_funcs, n_steps=100, n_simulations=100):
    samples, *_ = thetas[0].shape
    _, _, T, Z, R, H, Q = [f(*[theta[0] for theta in thetas])[0] for f in update_funcs]
    _, k = Z.shape

    states = np.empty((samples * n_simulations, n_steps, k))
    observed = np.empty((samples * n_simulations, n_steps, k))

    for i in range(samples):
        theta = [x[i] for x in thetas]
        _, _, T, Z, R, H, Q = [f(*theta)[0] for f in update_funcs]
        for j in range(n_simulations):
            sim_state, sim_obs = simulate_statespace(T, Z, R, H, Q, n_steps=n_steps)
            states[i * n_simulations + j, :, :] = sim_state
            observed[i * n_simulations + j, :, :] = sim_obs

    return states, observed
