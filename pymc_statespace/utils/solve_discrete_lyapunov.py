import aesara
import aesara.tensor as at
from aesara.scan.utils import until as scan_until


def doubling_step(alpha, gamma, tol):
    new_alpha = alpha.dot(alpha)
    new_gamma = gamma + alpha.dot(gamma).dot(alpha.T)

    diff = at.max(at.abs((new_gamma - gamma)))
    return (new_alpha, new_gamma), scan_until(diff < tol)
