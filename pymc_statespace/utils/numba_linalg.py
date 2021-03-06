from numba import njit
import numpy as np

@njit
def numba_block_diagonal(nd_array):
    n, rows, cols = nd_array.shape

    out = np.zeros((n * rows, n * cols))

    r, c = 0, 0
    for i, (rr, cc) in enumerate([(rows, cols)] * n):
        out[r:r + rr, c:c + cc] = nd_array[i]
        r += rr
        c += cc
    return out

