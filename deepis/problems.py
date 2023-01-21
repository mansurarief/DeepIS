from typing import List
from typing import Union

import numpy as np
from scipy.stats import multivariate_normal as normal

from .exceptions import InputDimensionError, InfeasibleProblemError

from sympy import Float
from sympy import Symbol
from sympy.stats import ChiSquared
from sympy.stats import cdf


class GaussianTailProbability():
    def __init__(self,
                 mu: Union[List, np.ndarray] = np.array([0., 0.]),
                 sigma: float = 1.,
                 rad: float = 4.,
                 dim: int = 2):
        assert isinstance(mu, (np.ndarray, List))
        if isinstance(mu, List):
            mu = np.array(mu)
        if mu.shape[0] != dim:
            raise InputDimensionError

        self.mu = mu
        self.sigma = sigma
        self.rad = rad
        self.dim = dim
        self.p = normal(mu, np.eye(dim) * sigma)

    def f(self, x: np.ndarray) -> np.ndarray:
        g_values = np.linalg.norm(x - self.mu, axis=-1)
        return np.array(g_values >= self.rad).reshape(-1,)

    def compute_target(self, decimal_place: int = 5) -> float:
        z = Symbol("z")
        p_chi = ChiSquared("x", self.dim)
        val = Float(self.rad**2 / self.sigma)
        out1 = 1 - cdf(p_chi)(z)
        gt_val = float(out1.evalf(decimal_place, subs={z: val}))
        return gt_val
    
class PolySqrt():
    def __init__(self,
                 mu: Union[List, np.ndarray] = np.array([0., 0.]),
                 sigma: float = 1.,
                 T: float = 6.,
                 dim: int = 2):
        assert isinstance(mu, (np.ndarray, List))
        if isinstance(mu, List):
            mu = np.array(mu)
        if mu.shape[0] != dim:
            raise InputDimensionError
            
        if T > 12.98:
            raise InfeasibleProblemError

        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.dim = dim
        self.p = normal(mu, np.eye(dim) * sigma)

    def f(self, x: np.ndarray) -> np.ndarray:
        g_values = - np.sqrt((-x[:, 0]+10)**2 + (x[:, 1] + 7 )**2 + 10 * x.sum(axis=-1)**2) + 14
        return np.array(g_values  >= self.T).reshape(-1,)

    def compute_target(self, decimal_place: int = 5) -> Float or None:
        if self.T == 6.:
            return float(2.35e-6)
        return None
    

class FourBranches():
    def __init__(self,
                 mu: Union[List, np.ndarray] = np.array([0., 0.]),
                 sigma: float = 1.,
                 T: float = 10.,
                 dim: int = 2):
        assert isinstance(mu, (np.ndarray, List))
        if isinstance(mu, List):
            mu = np.array(mu)
        if mu.shape[0] != dim:
            raise InputDimensionError
            
        if T > 12.98:
            raise InfeasibleProblemError

        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.dim = dim
        self.p = normal(mu, np.eye(dim) * sigma)

    def f(self, x: np.ndarray) -> np.ndarray:
        branch1 = 3 + 0.1*np.diff(x, axis=-1)**2 - x.sum(axis=-1).reshape(-1, 1)/np.sqrt(2)
        branch2 = 3 + 0.1*np.diff(x, axis=-1)**2 + x.sum(axis=-1).reshape(-1, 1)/np.sqrt(2)
        branch3 = np.diff(x, axis=-1) + 7/np.sqrt(2)
        branch4 = -1*np.diff(x, axis=-1) + 7/np.sqrt(2)
        g_values = 10 - np.hstack([branch1, branch2, branch3, branch4]).min(axis=-1)
        return np.array(g_values >= self.T).reshape(-1,)

    def compute_target(self, decimal_place: int = 5) -> Float or None:
        if self.T == 6.:
            return float(2.35e-6)
        return None
    
    