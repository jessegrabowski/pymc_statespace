import warnings
from abc import ABC
from typing import List, Tuple, Union

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import (
    Distribution,
    SymbolicRandomVariable,
    _moment,
    moment,
)
from pymc.distributions.shape_utils import (
    _change_dist_size,
    change_dist_size,
    get_support_shape_1d,
)
from pymc.logprob.abstract import _logprob
from pymc.logprob.basic import logp
from pymc.logprob.utils import ignore_logprob
from pymc.pytensorf import intX
from pymc.util import check_dist_not_registered
from pytensor.compile import get_mode
from pytensor.graph.basic import Node
from pytensor.tensor import TensorVariable
from pytensor.tensor.nlinalg import matrix_dot
from pytensor.tensor.random.op import RandomVariable

from pymc_statespace.core.statespace import FILTER_FACTORY


def _validate_init_dist(init_dist, name=None, max_ndim_support=1) -> None:
    """
    Check that a given inital distribution of a time series is valid

    Parameters
    ----------
    init_dist: unnamed distribution
        User provided initial distribution

    name: str, optional
        Name of the distribution being checked, used for error messages.

    max_ndim_support: int, optional
        Maximum number of expected support (i.e. time) dimensions expected on the distribution. It's valid for an
        initial distribution to have fewer (as its dims can be expanded) but not more (since it will not be clear how
        to slice it down). Default is 1, for a single time dimension.

    Raises
    -------
    ValueError
        If the provided initial distribution is not valid
    """
    if name is None:
        name = init_dist.owner.op.name

    if not isinstance(init_dist, TensorVariable) or not isinstance(
        init_dist.owner.op, (RandomVariable, SymbolicRandomVariable)
    ):
        raise ValueError(
            f"Init dist must be a distribution created via the `.dist()` API, "
            f"got {type(init_dist)}"
        )

    check_dist_not_registered(init_dist)
    if init_dist.owner.op.ndim_supp > max_ndim_support:
        raise ValueError(
            f"Initial distribution {name} must have at most {max_ndim_support} support "
            f"dimension{'s' if max_ndim_support > 0 else ''}. Found ndim_supp={init_dist.owner.op.ndim_supp}.",
        )


class LinearGaussianStateSpaceRV(SymbolicRandomVariable):
    default_output = 1
    _print_name = ("LinearGaussianStateSpace", "\\operatorname{LinearGaussianStateSpace}")

    def __init__(self, *args, filter_type, **kwargs):
        self.filter_type = filter_type
        super().__init__(*args, **kwargs)

    def update(self, node: Node):
        return {node.inputs[-1]: node.outputs[0]}


class LinearGaussianStateSpace(Distribution):
    r"""
    The Kalman Filter is an iterative algorithm for computing the mean and covariance matrix of a multivariate normal
    distribution with recursive dependencies between the distribution's components. Specifically, given a linear
    system:
    .. math::

        \begin{align}x_t &= A x_{t-1} + R \varepsilon_t \quad &\varepsilon &\sim N(0, Q) \\
                     y_t &= Z x_t + \eta_t \quad & \eta_t &\sim N(0, H) \end{align}

    Where :math:`x_t` is a :math:`k \times 1` vector of hidden states and :math:`y_t` is a vector of :math:`p \times 1`
    vector of observed states, with :math:`p \leq k`. Because the system is linear and the errors are normal, it can be
    shown that the posterior mean and covariance of the sequences :math:`\{x_t, y_t\}_0^T`

    References
    ----------
    .. [1] Koopman, Siem Jan, Neil Shephard, and Jurgen A. Doornik. 1999.
       Statistical Algorithms for Models in State Space Using SsfPack 2.2.
       Econometrics Journal 2 (1): 107-60. doi:10.1111/1368-423X.00023.

    Parameters
    ----------
    T: array

    Z: array

    R: array

    H: array

    Q: array

    a0_dist: SymbolicRandomVariable

    P0_dist: SymbolicRandomVariable

    P0_method : str

    filter_type: str

    steps: int

    Notes
    -----
    # TODO: WRITEME
    Examples
    --------
    # TODO: WRITEME
    """

    rv_type = LinearGaussianStateSpaceRV

    def __new__(cls, *args, steps=None, filter_type, **kwargs):
        steps = get_support_shape_1d(
            support_shape=steps,
            shape=None,
            dims=kwargs.get("dims", None),
            observed=kwargs.get("observed", None),
            support_shape_offset=0,
        )

        return super().__new__(cls, *args, steps=steps, filter_type=filter_type, **kwargs)

    @classmethod
    def dist(
        cls,
        T=None,
        Z=None,
        R=None,
        H=None,
        Q=None,
        a0_init_dist=None,
        P0_init_dist=None,
        steps=None,
        P0_method="estimate",
        filter_type="standard",
        **kwargs,
    ):
        steps = get_support_shape_1d(
            support_shape=steps, shape=kwargs.get("shape", None), support_shape_offset=0
        )

        if steps is None:
            raise ValueError("Must specify steps or shape parameter")

        if P0_method not in ["estimate", "steady_state"]:
            raise ValueError(
                f'P0_method must be one of "estimate" or "steady_state", got {P0_method}'
            )

        valid_filter_options = FILTER_FACTORY.keys()
        if filter_type not in valid_filter_options:
            raise ValueError(
                f'filter_type must be one of {", ".join(valid_filter_options)}, got {filter_type}'
            )

        system_mats = [T, Z, R, H, Q]
        for mat, name in zip(system_mats, ["T", "Z", "R", "H", "Q"]):
            if mat is None:
                raise ValueError(f"All system matrices are required inputs. Found None for {name}")

        T, Z, R, H, Q = list(map(pt.as_tensor_variable, system_mats))
        n_states, n_posdef = R.shape
        n_states_obs, _ = Z.shape

        steps = pt.as_tensor_variable(intX(steps))
        _validate_init_dist(a0_init_dist)

        if a0_init_dist is not None:
            _validate_init_dist(a0_init_dist, name="a0_init_dist", max_ndim_support=1)

        else:
            warnings.warn(
                "Initial distribution not specified, defaulting to "
                "`Normal.dist(mu=0, sigma=100), shape=...)`. You can specify an init_dist "
                "manually to suppress this warning.",
                UserWarning,
            )

            a0_init_dist = pm.Normal.dist(mu=0, sigma=100, size=n_states)

        if P0_method == "estimate":
            if P0_init_dist is not None:
                _validate_init_dist(P0_init_dist, name="P0_init_dist", max_ndim_support=1)

            else:
                warnings.warn(
                    "Initial distribution not specified, defaulting to a diffuse initialization, defined as "
                    "`pt.diag(HalfNormal.dist(sigma=1e6), shape=...))`.You can specify an init_dist manually to "
                    "suppress this warning.",
                    UserWarning,
                )
            P0_init_dist = ignore_logprob(P0_init_dist)

        # We can ignore init_dist, as it will be accounted for in the logp term
        a0_init_dist = ignore_logprob(a0_init_dist)

        return super().dist(
            system_mats + [a0_init_dist, P0_init_dist, steps], filter_type=filter_type, **kwargs
        )

    @classmethod
    def rv_op(
        cls,
        T,
        Z,
        R,
        H,
        Q,
        a0_init_dist,
        P0_init_dist,
        steps,
        size=None,
        P0_method="estimate",
        filter_type="standard",
    ):
        a0_init_dist_ = a0_init_dist.type()
        P0_init_dist = P0_init_dist.type()
        T_, Z_, R_, H_, Q_ = map(lambda x: x.type(), [T, Z, R, H, Q])
        steps_ = steps.type()

        kfilter = FILTER_FACTORY[filter_type]()

        pass
