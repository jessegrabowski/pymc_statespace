import os
import sys
import unittest
import warnings
from itertools import product
from pathlib import Path

import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose

from pymc_statespace import BayesianVARMAX

ROOT = Path(__file__).parent.absolute()
sys.path.append(ROOT)


@pytest.fixture
def data():
    return pd.read_csv(
        os.path.join(ROOT, "test_data/statsmodels_macrodata_processed.csv"), index_col=0
    )


ps = [0, 1, 2, 3]
qs = [0, 1, 2, 3]
orders = list(product(ps, qs))[1:]
ids = [f"p={x[0]}, q={x[1]}" for x in orders]


@pytest.mark.parametrize("order", orders, ids=ids)
@pytest.mark.parametrize("matrix", ["transition", "selection", "state_cov", "obs_cov", "design"])
def test_VARMAX_shapes_match_statsmodels(data, order, matrix):
    p, q = order
    if p == q == 0:
        pytest.skip("Skipping p = q = 0 case")

    mod = BayesianVARMAX(data, order=(p, q), verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm_var = sm.tsa.VARMAX(data, order=(p, q))
    print(mod.ssm[matrix].eval())
    print(sm_var.ssm[matrix])
    assert_allclose(mod.ssm[matrix].eval(), sm_var.ssm[matrix])
