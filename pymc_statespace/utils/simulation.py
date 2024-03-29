import numpy as np
from numba import njit

from pymc_statespace.utils.numba_linalg import numba_block_diagonal


@njit
def numba_mvn_draws(mu, cov):
    samples = np.random.randn(*mu.shape)
    k = cov.shape[0]
    jitter = np.eye(k) * np.random.uniform(1e-12, 1e-8)

    L = np.linalg.cholesky(cov + jitter)
    return mu + L @ samples


@njit
def conditional_simulation(mus, covs, n, k, n_simulations=100):
    n_samples = mus.shape[0]
    simulations = np.empty((n_samples * n_simulations, n, k))

    for i in range(n_samples):
        for j in range(n_simulations):
            sim = numba_mvn_draws(mus[i], numba_block_diagonal(covs[i]))
            simulations[(i * n_simulations + j), :, :] = sim.reshape(n, k)
    return simulations


@njit
def simulate_statespace(T, Z, R, H, Q, n_steps, x0=None):
    n_obs, n_states = Z.shape
    k_posdef = R.shape[1]
    k_obs_noise = H.shape[0] * (1 - int(np.all(H == 0)))

    state_noise = np.random.randn(n_steps, k_posdef)
    state_chol = np.linalg.cholesky(Q)
    state_innovations = state_noise @ state_chol

    if k_obs_noise != 0:
        obs_noise = np.random.randn(n_steps, k_obs_noise)
        obs_chol = np.linalg.cholesky(H)
        obs_innovations = obs_noise @ obs_chol

    simulated_states = np.zeros((n_steps, n_states))
    simulated_obs = np.zeros((n_steps, n_obs))

    if x0 is not None:
        simulated_states[0] = x0
        simulated_obs[0] = Z @ x0

    if k_obs_noise != 0:
        for t in range(1, n_steps):
            simulated_states[t] = T @ simulated_states[t - 1] + R @ state_innovations[t]
            simulated_obs[t] = Z @ simulated_states[t - 1] + obs_innovations[t]
    else:
        for t in range(1, n_steps):
            simulated_states[t] = T @ simulated_states[t - 1] + R @ state_innovations[t]
            simulated_obs[t] = Z @ simulated_states[t - 1]

    return simulated_states, simulated_obs


def unconditional_simulations(thetas, update_funcs, n_steps=100, n_simulations=100):
    samples, *_ = thetas[0].shape
    _, _, T, Z, R, H, Q = (f(*[theta[0] for theta in thetas])[0] for f in update_funcs)
    n_obs, n_states = Z.shape

    states = np.empty((samples * n_simulations, n_steps, n_states))
    observed = np.empty((samples * n_simulations, n_steps, n_obs))

    for i in range(samples):
        theta = [x[i] for x in thetas]
        _, _, T, Z, R, H, Q = (f(*theta)[0] for f in update_funcs)
        for j in range(n_simulations):
            sim_state, sim_obs = simulate_statespace(T, Z, R, H, Q, n_steps=n_steps)
            states[i * n_simulations + j, :, :] = sim_state
            observed[i * n_simulations + j, :, :] = sim_obs

    return states, observed
