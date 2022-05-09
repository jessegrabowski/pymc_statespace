# PyMC StateSpace
A system for Bayesian estimation of state space models using PyMC 4.0. This package is designed to mirror the functionality of the Statsmodels.api `tsa.statespace` module, except within a Bayesian estimation framework. To accomplish this, PyMC Statespace has a Kalman filter written in Aesara, allowing the gradients of the iterative Kalman filter likelihood to be computed and provided to the PyMC NUTS sampler.

## State Space Models
This package follows Statsmodels in using the Durbin and Koopman (2012) nomenclature for a linear state space model. Under this nomenclature, the model is written as:

<p align="center"><img src="https://rawgit.com/jessegrabowski/pymc_statespace/main/svgs/536ec8a09411a7bf2f91219b52fc0924.svg?invert_in_darkmode" align=middle width=301.76022359999996pt height=65.753424pt/></p>

The objects in the above equation have the following shapes and meanings:

- <img src="https://rawgit.com/jessegrabowski/pymc_statespace/main/svgs/d523a14b8179ebe46f0ed16895ee46f0.svg?invert_in_darkmode" align=middle width=57.733970249999985pt height=21.18721440000001pt/>, observation vector
- <img src="https://rawgit.com/jessegrabowski/pymc_statespace/main/svgs/54221efbfb5e69569dfe8ddea785093a.svg?invert_in_darkmode" align=middle width=66.35271884999999pt height=21.18721440000001pt/>, state vector
- <img src="https://rawgit.com/jessegrabowski/pymc_statespace/main/svgs/523b266d36c270dbbb5daf2c9092ce0f.svg?invert_in_darkmode" align=middle width=57.340042649999994pt height=21.18721440000001pt/>, observation noise vector
- <img src="https://rawgit.com/jessegrabowski/pymc_statespace/main/svgs/17b59c002f249204f24e31507dc4957d.svg?invert_in_darkmode" align=middle width=57.43908884999998pt height=21.18721440000001pt/>, state innovation vector


- <img src="https://rawgit.com/jessegrabowski/pymc_statespace/main/svgs/ff25a8f22c7430ca572d33206c0a9176.svg?invert_in_darkmode" align=middle width=67.10991044999999pt height=22.465723500000017pt/>, the design matrix
- <img src="https://rawgit.com/jessegrabowski/pymc_statespace/main/svgs/a06c0e58d4d162b0e87d32927c9812db.svg?invert_in_darkmode" align=middle width=63.39029894999998pt height=22.465723500000017pt/>, observation noise covariance matrix
- <img src="https://rawgit.com/jessegrabowski/pymc_statespace/main/svgs/3b5e41543d7fc8cedf98ec609b343134.svg?invert_in_darkmode" align=middle width=71.65714874999999pt height=22.465723500000017pt/>, the transition matrix
- <img src="https://rawgit.com/jessegrabowski/pymc_statespace/main/svgs/cac7e81ebde5e530e639eae5389f149e.svg?invert_in_darkmode" align=middle width=67.97229284999999pt height=22.465723500000017pt/>, the selection matrix
- <img src="https://rawgit.com/jessegrabowski/pymc_statespace/main/svgs/edcff444fd5240add1c47d2de50ebd7e.svg?invert_in_darkmode" align=middle width=61.92609719999999pt height=22.465723500000017pt/>, the state innovation covariance matrix


- <img src="https://rawgit.com/jessegrabowski/pymc_statespace/main/svgs/92b8c1194757fb3131cda468a34be85f.svg?invert_in_darkmode" align=middle width=62.950875899999986pt height=21.18721440000001pt/>, the state intercept vector
- <img src="https://rawgit.com/jessegrabowski/pymc_statespace/main/svgs/a13d89295e999545a129b2d412e99f6d.svg?invert_in_darkmode" align=middle width=58.230501449999984pt height=22.831056599999986pt/>, the observation intercept vector

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