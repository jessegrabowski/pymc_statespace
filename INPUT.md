# PyMC StateSpace
A system for Bayesian estimation of state space models using PyMC 4.0. This package is designed to mirror the functionality of the Statsmodels.api `tsa.statespace` module, except within a Bayesian estimation framework. To accomplish this, PyMC Statespace has a Kalman filter written in Aesara, allowing the gradients of the iterative Kalman filter likelihood to be computed and provided to the PyMC NUTS sampler.

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

At present, the API is not particularly friendly, but neverthless ARMA models are available out of the box. One needs only import the package and delcare a new model of order `(p, q)`, as follows:

```python
import pymc_statespace as pmss
#make sure data is a 2d numpy array!
model = pmss.BayesianARMA(data = data[:, None], order=(1, 1))
```

Next, a PyMC model is declared as usual, and the parameters can be passed into the state space model. This is done as follows:
```python
with pm.Model() as arma_model:
    # Normal PyMC stuff
    a1 = pm.Normal('initial_states', mu=0.0, sigma=1.0, shape=2) # m = 2
    state_sigmas = pm.HalfNormal('state_sigma', sigma=1.0, shape=1) # r = 1
    
    initial_sigma = pm.HalfNormal('initial_sigma', sigma=1.0, shape=2)

    # note the entire P1 matrix needs to be declared, even r != m. Here I only estimate the diagonal of P1, in principal
    # you can estimate some, none, or all of the initial states and covariances.
    P1 = np.eye(2) * initial_sigma 
    
    # AR parameter
    rhos = pm.TruncatedNormal('rho', mu=0.0, sigma=0.5, lower=-1.0, upper=1.0, shape=1) # p = 1

    # MA parameter
    thetas = pm.Normal('thetas', mu=0.0, sigma=0.5, shape=1) # q = 1
    
    #Special PyMC StateSpace stuff
    params = at.concatenate([x0.ravel(), P0.ravel(), state_sigmas.ravel(), rhos.ravel(), thetas.ravel()])
    model.build_statespace_graph(params)

    likelihood = pm.Potential("likelihood", model.log_likelihood)
    y_hat = pm.Deterministic('y_hat', model.filtered_states)
    cov_hat = pm.Deterministic('cov_hat', model.filtered_covarainces)
```

Priors over the initial model parameters are declared as normal in a PyMC model, then need to be flattened and concatenated into a single vector. Note that the order matters! This isn't a great design, hopefully it can become a bit more user-friendly soon.

After forming the parameter vector, pass it into the `model.build_statespace_graph` method to create the Kalman filter computational graph and allow PyMC access to the model's gradients. Once this method is run, three results from the Kalman filter will be exposed in the model: `model.log_likelihood`, `model.filtered_states`, and `model.filtered_covariances`. The log likelihood needs to be provided to `pm.Potential` to complete the model. The other two can be passed into `pm.Deterministic` to make post-estimation a bit more convenient, but they aren't strictly necessary.

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

`self.ssm` is an `AesaraRepresentation` class object that is created by the super constructor. Every model has a `self.ssm` and a `self.kalman_filter` created after the super constructor is called. All the matrices stored in `self.ssm` are Aesara tensor variables, but numpy arrays can be passed to them for convenience. Behind the scenes, they will be converted to Aesara tensors. 

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

The user also needs to implement an `update` method, which takes in a single Aesara tensor as an argument. This method routes the parameters estimated by PyMC into the right spots in the state space matrices. The local level has at least two parameters to estimate: the variance of the level state innovations, and the variance of the trend state innovations. Here is the corresponding update method:

```python
def update(self, theta: at.TensorVariable) -> None:
    # Observation covariance
    self.ssm['obs_cov', 0, 0] = theta[0]

    # State covariance
    self.ssm['state_cov', np.diag_indices(2)] = theta[1:]
```

This function is why the order matters when flattening and concatenating the random variables inside the PyMC model. In this case, we must first pass `sigma_obs`, followed by `sigma_level`, then `sigma_trend`. 

But that's it! Obviously this API isn't great, and will be subject to change as the package evolves, but it should be enough to get a motivated research going. Happy estimation, and let me know all the bugs you find by opening an issue.