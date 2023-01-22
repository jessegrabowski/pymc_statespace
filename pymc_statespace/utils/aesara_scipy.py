from typing import Optional
import pytensor
from pytensor.tensor import as_tensor_variable, TensorVariable
import pytensor.tensor as at
from pytensor.tensor.nlinalg import matrix_dot

import scipy


class SolveDiscreteLyapunov(at.Op):
    __props__ = ("method",)

    def __init__(self, method=None):
        self.method = method

    def make_node(self, A, B):
        A = as_tensor_variable(A)
        B = as_tensor_variable(B)

        out_dtype = pytensor.scalar.upcast(A.dtype, B.dtype)
        X = pytensor.tensor.matrix(dtype=out_dtype)

        return pytensor.graph.basic.Apply(self, [A, B], [X])

    def perform(self, node, inputs, output_storage):
        (A, Q) = inputs
        X = output_storage[0]

        X[0] = scipy.linalg.solve_discrete_lyapunov(A, Q, method=self.method)

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]

    def grad(self, inputs, output_grads):
        # Gradient computations come from Kao and Hennequin (2020), https://arxiv.org/pdf/2011.11430.pdf
        A, Q = inputs
        (dX,) = output_grads

        X = self(A, Q)

        # Eq 41, note that it is not written as a proper Lyapunov equation
        S = self(A.conj().T, dX)

        A_bar = matrix_dot(S, A, X.conj().T) + matrix_dot(S.conj().T, A, X)
        Q_bar = S
        return [A_bar, Q_bar]


def solve_discrete_lyapunov(A, Q, method: Optional[str] = None) -> TensorVariable:
    """
    Solve the discrete Lyapunov equation :math:`AXA^H - X + Q = 0`.
    Parameters
    ----------
    A: ArrayLike
        Square matrix of shape N x N; must have the same shape as Q
    Q: ArrayLike
        Square matrix of shape N x N; must have the same shape as A
    method: Optional, string
        Solver method passed to scipy.linalg.solve_discrete_lyapunov, either "bilinear", "direct", or None. "direct"
        scales poorly with size. If None, uses "direct" if N < 10, else "bilinear".

    Returns
    -------
    X: at.matrix
        Square matrix of shape N x N, representing the solution to the Lyapunov equation
    """

    return SolveDiscreteLyapunov(method)(A, Q)


class SolveDiscreteARE(at.Op):
    __props__ = ('enforce_Q_symmetric',)

    def __init__(self, enforce_Q_symmetric=False):
        self.enforce_Q_symmetric = enforce_Q_symmetric

    def make_node(self, A, B, Q, R):
        A = as_tensor_variable(A)
        B = as_tensor_variable(B)
        Q = as_tensor_variable(Q)
        R = as_tensor_variable(R)

        out_dtype = pytensor.scalar.upcast(A.dtype, B.dtype, Q.dtype, R.dtype)
        X = pytensor.tensor.matrix(dtype=out_dtype)

        return pytensor.graph.basic.Apply(self, [A, B, Q, R], [X])

    def perform(self, node, inputs, output_storage):
        A, B, Q, R = inputs
        X = output_storage[0]

        if self.enforce_Q_symmetric:
            Q = 0.5 * (Q + Q.T)
        X[0] = scipy.linalg.solve_discrete_are(A, B, Q, R)

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]

    def grad(self, inputs, output_grads):
        # Gradient computations come from Kao and Hennequin (2020), https://arxiv.org/pdf/2011.11430.pdf
        A, B, Q, R = inputs

        (dX,) = output_grads
        X = self(A, B, Q, R)

        K_inner = R + at.linalg.matrix_dot(B.T, X, B)
        K_inner_inv = at.linalg.solve(K_inner, at.eye(R.shape[0]))
        K = matrix_dot(K_inner_inv, B.T, X, A)

        A_tilde = A - B.dot(K)

        dX_symm = 0.5 * (dX + dX.T)
        S = solve_discrete_lyapunov(A_tilde, dX_symm)

        A_bar = 2 * matrix_dot(X, A_tilde, S)
        B_bar = -2 * matrix_dot(X, A_tilde, S, K.T)
        Q_bar = S
        R_bar = matrix_dot(K, S, K.T)

        return [A_bar, B_bar, Q_bar, R_bar]


def solve_discrete_are(A, B, Q, R) -> TensorVariable:
    """
    Solve the discrete Algebraic Riccati equation :math:`A^TXA - X - (A^TXB)(R + B^TXB)^{-1}(B^TXA) + Q = 0`.
    Parameters
    ----------
    A: ArrayLike
        Square matrix of shape M x M
    B: ArrayLike
        Square matrix of shape M x M
    Q: ArrayLike
        Square matrix of shape M x M
    R: ArrayLike
        Square matrix of shape N x N

    Returns
    -------
    X: at.matrix
        Square matrix of shape M x M, representing the solution to the DARE
    """

    return SolveDiscreteARE()(A, B, Q, R)


def allocate_block(A, out, r_start, c_start, r_stride, c_stride):
    row_slice = slice(r_start, r_start + r_stride)
    col_slice = slice(c_start, c_start + c_stride)

    next_r = r_start + r_stride
    next_c = c_start + c_stride

    return at.set_subtensor(out[row_slice, col_slice], A), next_r, next_c


def block_diag(arr: at.tensor3):
    n, rows, cols = arr.shape
    out = at.zeros((n * rows, n * cols))

    result, _ = pytensor.scan(allocate_block,
                            sequences=[arr],
                            outputs_info=[out, at.zeros(1, dtype='int64'), at.zeros(1, dtype='int64')],
                            non_sequences=[rows.astype('int64'), cols.astype('int64')])
    return result[0][-1]
