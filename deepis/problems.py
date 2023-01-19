from typing import List
from typing import Union

import numpy as np
from scipy.stats import multivariate_normal as normal

from .exceptions import InputDimensionError

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
