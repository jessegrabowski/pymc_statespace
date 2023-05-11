# PyMC StateSpace
A system for Bayesian estimation of state space models using PyMC 4.0. This package is designed to mirror the functionality of the Statsmodels.api `tsa.statespace` module, except within a Bayesian estimation framework. To accomplish this, PyMC Statespace has a Kalman filter written in Pytensor, allowing the gradients of the iterative Kalman filter likelihood to be computed and provided to the PyMC NUTS sampler.

## State Space Models
This package follows Statsmodels in using the Durbin and Koopman (2012) nomenclature for a linear state space model. Under this nomenclature, the model is written as:

$y_t = Z_t \alpha_t + d_t + \varepsilon_t, \quad \varepsilon_t \sim N(0, H_t)$
$\alpha_{t+1} = T_t \alpha_t + c_t + R_t \eta_t, \quad \eta_t \sim N(0, Q_t)$
$\alpha_1 \sim N(a_1, P_1)$

The objects in the above equation have the following shapes and meanings:

- $y_t, p \times 1$, observation vector
- $\alpha_t, m \times 1$, state vector
- $\varepsilon_t, p \times 1$, observation noise vector
- $\eta_t, r \times 1$, state innovation vector


- $Z_t, p \times m$, the design matrix
- $H_t, p \times p$, observation noise covariance matrix
- $T_t, m \times m$, the transition matrix
- $R_t, m \times r$, the selection matrix
- $Q_t, r \times r$, the state innovation covariance matrix


- $c_t, m \times 1$, the state intercept vector
- $d_t, p \times 1$, the observation intercept vector

The linear state space model is a workhorse in many disciplines, and is flexible enough to represent a wide range of models, including Box-Jenkins SARIMAX class models, time series decompositions, and model of multiple time series (VARMAX) models. Use of a Kalman filter allows for estimation of unobserved and missing variables. Taken together, these are a powerful class of models that can be further augmented by Bayesian estimation. This allows the researcher to integrate uncertainty about the true model when estimating model parameteres.


## Example Usage

Currently, local level and ARMA models are implemented. To initialize a model, simply create a model object, provide data, and any additional arguments unique to that model.
```python
import pymc_statespace as pmss
#make sure data is a 2d numpy array!
arma_model = pmss.BayesianARMA(data = data[:, None], order=(1, 1))
```
You will see a message letting you know the model was created, as well as telling you the expected parameter names you will need to create inside a PyMC model block.
```commandline
Model successfully initialized! The following parameters should be assigned priors inside a PyMC model block: x0, sigma_state, rho, theta
```

Next, a PyMC model is declared as usual, and the parameters can be passed into the state space model. This is done as follows:
```python
with pm.Model() as arma_model:
    x0 = pm.Normal('x0', mu=0, sigma=1.0, shape=mod.k_states)
    sigma_state = pm.HalfNormal('sigma_state', sigma=1)

    rho = pm.TruncatedNormal('rho', mu=0.0, sigma=0.5, lower=-1.0, upper=1.0, shape=p)
    theta = pm.Normal('theta', mu=0.0, sigma=0.5, shape=q)

    arma_model.build_statespace_graph()
    trace_1 = pm.sample(init='jitter+adapt_diag_grad', target_accept=0.9)
```

After you place priors over the requested parameters, call `arma_model.build_statespace_graph()` to automatically put everything together into a state space system. If you are missing any parameters, you will get a error letting you know what else needs to be defined.

And that's it! After this, you can sample the PyMC model as normal.


## Creating your own state space model

Creating a custom state space model isn't complicated. Once again, the API follows the Statsmodels implementation. All models need to subclass the `PyMCStateSpace` class, and pass three values into the class super construction: `data` (from which p is inferred), `k_states` (this is "m" in the shapes above), and `k_posdef` (this is "r" above). The user also needs to declare any required state space matrices. Here is an example of a simple local linear trend model:

```python
def __init__(self, data):
    # Model shapes
    k_states = k_posdef = 2

    super().__init__(data, k_states, k_posdef)

    # Initialize the matrices
    self.ssm['design'] = np.array([[1.0, 0.0]])
    self.ssm['transition'] = np.array([[1.0, 1.0],
                                       [0.0, 1.0]])
    self.ssm['selection'] = np.eye(k_states)

    self.ssm['initial_state'] = np.array([[0.0],
                                          [0.0]])
    self.ssm['initial_state_cov'] = np.array([[1.0, 0.0],
                                              [0.0, 1.0]])
```

You will also need to set up the `param_names` class method, which returns a list of parameter names. This lets users know what priors they need to define, and lets the model gather the required parameters from the PyMC model context.

```python
@property
def param_names(self):
    return ['x0', 'P0', 'sigma_obs', 'sigma_state']
```

`self.ssm` is an `PytensorRepresentation` class object that is created by the super constructor. Every model has a `self.ssm` and a `self.kalman_filter` created after the super constructor is called. All the matrices stored in `self.ssm` are Pytensor tensor variables, but numpy arrays can be passed to them for convenience. Behind the scenes, they will be converted to Pytensor tensors.

Note that the names of the matrices correspond to the names listed above. They are (in the same order):

- Z = design
- H = obs_cov
- T = transition
- R = selection
- Q = state_cov
- c = state_intercept
- d = obs_intercept
- a1 = initial_state
- P1 = initial_state_cov

Indexing by name only will expose the entire matrix. A name can also be followed by the usual numpy slice notation to get a specific element, row, or column.

The user also needs to implement an `update` method, which takes in a single Pytensor tensor as an argument. This method routes the parameters estimated by PyMC into the right spots in the state space matrices. The local level has at least two parameters to estimate: the variance of the level state innovations, and the variance of the trend state innovations. Here is the corresponding update method:

```python
def update(self, theta: at.TensorVariable) -> None:
    # Observation covariance
    self.ssm['obs_cov', 0, 0] = theta[0]

    # State covariance
    self.ssm['state_cov', np.diag_indices(2)] = theta[1:]
```

And that's it! By making a model you also gain access to simulation helper functions, including prior and posterior simulation from the predicted, filtered, and smoothed states, as well as stability analysis and impulse response functions.

This package is very much a work in progress and really needs help! If you're interested in time series analysis and want to contribute, please reach out!
