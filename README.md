# PyMC StateSpace
A system for Bayesian estimation of state space models using PyMC 4.0. The API is currently under development, but the basics are essentially identical to those of statsmodels.api's implementation. The `PyMCStateSpace` model should be extended with `update` and `compile_aesara_functions` methods. These can then be given to the aesara Ops in the `aesara_ops.py` for use inside a PyMC model block. 

See the example notebook for details. Better documentation coming!
