# ml/representation/feature_extraction/rcsp.py

import numpy as np
from scipy.linalg import eigh

from ml.representation.feature_extraction.csp import CSPFeatureExtractor


class RCSPFeatureExtractor(CSPFeatureExtractor):
    def __init__(
        self,
        n_components=4,
        alpha=0.1,
        reg=1e-6,
        log=True,
        pre_scaler=None,
        post_scaler=None,
    ):
        super().__init__(
            n_components=n_components,
            reg=reg,
            log=log,
            pre_scaler=pre_scaler,
            post_scaler=post_scaler,
        )
        self.alpha = alpha

    def _regularize(self, C):
        target = np.trace(C) / C.shape[0] * np.eye(C.shape[0])
        return (1 - self.alpha) * C + self.alpha * target

    def _fit_binary(self, positive, negative):
        C_pos = self._regularize(self._mean_covariance(positive))
        C_neg = self._regularize(self._mean_covariance(negative))
        I = np.eye(C_pos.shape[0])

        eigenvalues, eigenvectors = eigh(
            C_pos + self.reg * I,
            C_pos + C_neg + 2 * self.reg * I,
        )
        return self._select_filters(eigenvectors, eigenvalues)