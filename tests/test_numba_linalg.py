import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymc_statespace.utils.numba_linalg import numba_block_diagonal


def test_numba_block_diagonal():
    stack = np.concatenate([np.eye(3)[None]] * 5, axis=0)
    block_stack = numba_block_diagonal(stack)
    assert_allclose(block_stack, np.eye(15))
