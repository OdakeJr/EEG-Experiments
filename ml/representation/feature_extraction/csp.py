# ml/representation/feature_extraction/csp.py

import numpy as np
from scipy.linalg import eigh

from ml.representation.feature_extraction.base import FeatureExtractor


class CSPFeatureExtractor(FeatureExtractor):
    def __init__(self, n_components=4, reg=1e-6, log=True, pre_scaler=None, post_scaler=None):
        if pre_scaler is not None:
            raise ValueError("CSP does not support pre_scaling of signal input.")

        super().__init__(pre_scaler=None, post_scaler=post_scaler)
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.classes_ = None
        self.filters_ = None

    def _as_bands(self, X):
        X = np.asarray(X)
        if X.ndim == 3:
            return X[:, None, :, :]
        if X.ndim == 4:
            return X
        raise ValueError(f"CSP expects [N,C,T] or [N,B,C,T], got {X.shape}.")

    def _mean_covariance(self, X):
        covs = []
        for trial in X:
            cov = trial @ trial.T
            cov /= max(np.trace(cov), 1e-12)
            covs.append(cov)
        return np.mean(covs, axis=0)

    def _select_filters(self, eigenvectors, eigenvalues):
        order = np.argsort(eigenvalues)
        n_low = self.n_components // 2
        idx = np.concatenate([order[-(self.n_components - n_low):], order[:n_low]])
        return eigenvectors[:, idx].T

    def _fit_binary(self, positive, negative):
        C_pos = self._mean_covariance(positive)
        C_neg = self._mean_covariance(negative)
        I = np.eye(C_pos.shape[0])

        eigenvalues, eigenvectors = eigh(
            C_pos + self.reg * I,
            C_pos + C_neg + 2 * self.reg * I,
        )
        return self._select_filters(eigenvectors, eigenvalues)

    def _fit(self, X, y=None, domains=None):
        if y is None:
            raise ValueError("CSP requires class labels.")

        X, y = self._as_bands(X), np.asarray(y)
        self.classes_ = np.unique(y)
        self.filters_ = []

        for band in range(X.shape[1]):
            filters = []
            for cls in self.classes_:
                positive, negative = X[y == cls, band], X[y != cls, band]
                if not len(positive) or not len(negative):
                    raise ValueError(f"Cannot fit CSP for class '{cls}'.")
                filters.append(self._fit_binary(positive, negative))
            self.filters_.append(filters)

        return self

    def _transform(self, X, domains=None):
        if self.filters_ is None:
            raise RuntimeError("CSP must be fitted before transform().")

        X = self._as_bands(X)
        features = []

        for band, filters in enumerate(self.filters_):
            for W in filters:
                projected = np.einsum("kc,nct->nkt", W, X[:, band])
                variance = np.var(projected, axis=2)
                variance /= np.maximum(variance.sum(axis=1, keepdims=True), 1e-12)

                if self.log:
                    variance = np.log(np.maximum(variance, 1e-12))

                features.append(variance)

        return np.concatenate(features, axis=1)