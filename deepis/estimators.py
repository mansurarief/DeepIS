from collections import defaultdict

import numpy as np


class Estimator():
    def __init__(self, X, Y, likelihood_orig: np.ndarray = None, likelihood_prop: np.ndarray = None, method: str = 'MC') -> None:
        self.num_samples, self.dimension = X.shape
        self.X = X
        self.Y = Y
        self.method = method
        self.output = {}
        self._prepare_samples(num_instances=self.num_samples)
        self._compute_mean(method, likelihood_orig, likelihood_prop)



    def _prepare_samples(self, num_instances: str = 2):
        assert num_instances >= 2
        self.num_instances = num_instances
        self.samples = self.X[:num_instances]
        self.labels = self.Y[:num_instances]

    def _compute_weights(self, likelihood_orig: np.ndarray, likelihood_prop: np.ndarray) -> np.ndarray:
        assert likelihood_orig.shape == likelihood_prop.shape
        weights = np.divide(likelihood_orig, likelihood_prop).reshape(-1, 1)
        return weights

    def _compute_weighted_labels(self, labels, weights):
        assert labels.shape == weights.shape
        weighted_results = np.multiply(labels, weights)
        return weighted_results

    def _compute_mean(self, method, likelihood_orig=None, likelihood_prop=None):
        if method == 'MC':
            weights = np.ones([self.num_instances, 1])
        if method == 'IS':
            assert likelihood_orig is not None and likelihood_prop is not None
            weights = self._compute_weights(likelihood_orig, likelihood_prop)

        weighted_labels = self._compute_weighted_labels(self.labels, weights)
        self.mean = weighted_labels.mean()
        self.std = weighted_labels.std()
        self.output[str(self.num_instances+1)] = {
            'mean': self.mean, 'std': self.std}

    def compute_estimator(X, Y, proposal_dist, gmm_pdf, p):
        Lq = gmm_pdf(X, proposal_dist)
        Lp = p.pdf(X)
        W = np.divide(Lp, Lq).reshape(-1, 1)
        YW = np.multiply(Y, W)

        num_samples = np.arange(0, X.shape[0]+1, 1)[1:]  # np.arange(1, n2+1)
        mus = np.divide(YW.cumsum()[num_samples-1], num_samples)
        stds = []

        dur_ = []
        re_ = []

        for i in num_samples:
            stds.append(YW[:i].std()/i**0.5)
            re = stds[-1]/YW[:i].mean()
            re_.append(re)
            #print(i, end="\r")

        stds = np.array(stds)
        res = np.divide(stds, mus[-1])
        #num_samples +=n1
