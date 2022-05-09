# PyMC StateSpace
A system for Bayesian estimation of state space models using PyMC 4.0. The API is currently under development, but the basics are essentially identical to those of statsmodels.api's implementation. The `PyMCStateSpace` model just needs to be initialized with all state space matrices, and the `update` method, whichtells the model where to random parameter values into the state space matrices, needs to be implemented. The model can then be compiled along side a PyMC model using the `model.build_statespace_graph(parameter_rvs)` method, allowing PyMC to access the gradients of the Kalman Filter for use in NUTS sampling.

See the example notebook for details. Better documentation coming!
