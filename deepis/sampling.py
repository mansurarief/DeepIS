import numpy as np
import sklearn
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal as normal
from .exceptions import InputDimensionError, WeightDimensionError

seed = 123
np.random.seed(seed)


class ProposalDistribution():
    def __init__(self, mu, sigma, weights=None) -> None:
        self.mu = mu
        self.sigma = sigma
        self.num_components = mu.shape[0]
        self.weights = weights
        if self.weights == None:
            self.weights = np.ones([self.num_components, 1])/self.num_components

        if self.weights.shape[0] != self.num_components:
            raise WeightDimensionError

        self.dimension = mu.shape[-1]
        self._build_gmm_model()

    def _build_gmm_model(self):
        GMM_model = GaussianMixture(n_components=self.num_components, covariance_type="diag")            
        GMM_model.fit(self.mu)
        GMM_model.means_ = self.mu
        GMM_model.covariances_ = np.ones([self.num_components, 1])*self.sigma
        self.model = GMM_model

    def pdf(self, X):
        num_instances, dim = X.shape
        if dim != self.dimension:
            raise InputDimensionError

        pdf_values = np.zeros([num_instances, self.num_components])

        for i, mu_ in enumerate(self.mu):
            pdf_values[:, i] = normal.pdf(X, mu_, self.sigma*np.eye(dim))

        return np.dot(pdf_values, self.weights)

    def sample(self, num_samples: int = 1):
        samples = self.model.sample(num_samples)[0]
        np.random.shuffle(samples)
        return samples
