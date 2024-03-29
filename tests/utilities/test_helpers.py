import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytensor.tensor as pt

from pymc_statespace.filters.kalman_smoother import KalmanSmoother
from tests.utilities.statsmodel_local_level import LocalLinearTrend

ROOT = Path(__file__).parent.parent.absolute()
nile_data = pd.read_csv(os.path.join(ROOT, "test_data/nile.csv"))
nile_data["x"] = nile_data["x"].astype(float)


def initialize_filter(kfilter):
    ksmoother = KalmanSmoother()
    data = pt.dtensor3(name="data")
    a0 = pt.matrix(name="a0")
    P0 = pt.matrix(name="P0")
    Q = pt.matrix(name="Q")
    H = pt.matrix(name="H")
    T = pt.matrix(name="T")
    R = pt.matrix(name="R")
    Z = pt.matrix(name="Z")

    inputs = [data, a0, P0, T, Z, R, H, Q]

    (
        filtered_states,
        predicted_states,
        filtered_covs,
        predicted_covs,
        log_likelihood,
        ll_obs,
    ) = kfilter.build_graph(*inputs)

    smoothed_states, smoothed_covs = ksmoother.build_graph(T, R, Q, filtered_states, filtered_covs)

    outputs = [
        filtered_states,
        predicted_states,
        smoothed_states,
        filtered_covs,
        predicted_covs,
        smoothed_covs,
        log_likelihood,
        ll_obs,
    ]

    return inputs, outputs


def add_missing_data(data, n_missing):
    n = data.shape[0]
    missing_idx = np.random.choice(n, n_missing, replace=False)
    data[missing_idx] = np.nan

    return data


def make_test_inputs(p, m, r, n, missing_data=None, H_is_zero=False):
    data = np.arange(n * p, dtype="float").reshape(-1, p, 1)
    if missing_data is not None:
        data = add_missing_data(data, missing_data)

    a0 = np.zeros((m, 1))
    P0 = np.eye(m)
    Q = np.eye(r)
    H = np.zeros((p, p)) if H_is_zero else np.eye(p)
    T = np.eye(m, k=-1)
    T[0, :] = 1 / m
    R = np.eye(m)[:, :r]
    Z = np.eye(m)[:p, :]

    return data, a0, P0, T, Z, R, H, Q


def get_expected_shape(name, p, m, r, n):
    if name == "log_likelihood":
        return ()
    elif name == "ll_obs":
        return (n,)
    filter_type, variable = name.split("_")
    if filter_type == "predicted":
        n += 1
    if variable == "states":
        return n, m, 1
    if variable == "covs":
        return n, m, m


def get_sm_state_from_output_name(res, name):
    if name == "log_likelihood":
        return res.llf
    elif name == "ll_obs":
        return res.llf_obs

    filter_type, variable = name.split("_")
    sm_states = getattr(res, "states")

    if variable == "states":
        return getattr(sm_states, filter_type)
    if variable == "covs":
        m = res.filter_results.k_states
        # remove the "s" from "covs"
        return getattr(sm_states, name[:-1]).reshape(-1, m, m)


def nile_test_test_helper(n_missing=0):
    a0 = np.zeros((2, 1))
    P0 = np.eye(2) * 1e6
    Q = np.eye(2) * np.array([0.5, 0.01])
    H = np.eye(1) * 0.8
    T = np.array([[1.0, 1.0], [0.0, 1.0]])
    R = np.eye(2)
    Z = np.array([[1.0, 0.0]])

    data = nile_data.values.copy()
    if n_missing > 0:
        data = add_missing_data(data, n_missing)

    sm_model = LocalLinearTrend(
        endog=data,
        initialization="known",
        initial_state_cov=P0,
        initial_state=a0.ravel(),
    )

    res = sm_model.fit_constrained(
        constraints={
            "sigma2.measurement": 0.8,
            "sigma2.level": 0.5,
            "sigma2.trend": 0.01,
        }
    )

    inputs = [data[..., None], a0, P0, T, Z, R, H, Q]

    return res, inputs
