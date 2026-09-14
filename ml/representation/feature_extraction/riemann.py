# ml/representation/feature_extraction/riemann.py

import numpy as np

from ml.representation.feature_extraction.base import FeatureExtractor


class RiemannianFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        reg=1e-6,
        normalize_cov=True,
        max_iter=50,
        tol=1e-9,
        pre_scaler=None,
        post_scaler=None,
    ):
        if pre_scaler is not None:
            raise ValueError("Riemannian features do not support pre_scaling of signal input.")

        super().__init__(pre_scaler=None, post_scaler=post_scaler)
        self.reg = reg
        self.normalize_cov = normalize_cov
        self.max_iter = max_iter
        self.tol = tol
        self.references_ = None

    def _as_bands(self, X):
        X = np.asarray(X)
        if X.ndim == 3:
            return X[:, None, :, :]
        if X.ndim == 4:
            return X
        raise ValueError(f"Riemannian features expect [N,C,T] or [N,B,C,T], got {X.shape}.")

    def _matrix_function(self, C, function):
        values, vectors = np.linalg.eigh(C)
        values = np.clip(values, 1e-12, None)
        return (vectors * function(values)) @ vectors.T

    def _covariance(self, X):
        covs = []

        for trial in X:
            C = trial @ trial.T / max(trial.shape[1] - 1, 1)

            if self.normalize_cov:
                C /= max(np.trace(C), 1e-12)

            scale = np.trace(C) / C.shape[0]
            C += self.reg * max(scale, 1e-12) * np.eye(C.shape[0])
            covs.append(C)

        return np.asarray(covs)

    def _riemannian_mean(self, covs):
        G = np.mean(covs, axis=0)

        for _ in range(self.max_iter):
            sqrt_G = self._matrix_function(G, np.sqrt)
            invsqrt_G = self._matrix_function(G, lambda x: 1 / np.sqrt(x))
            logs = [
                self._matrix_function(invsqrt_G @ C @ invsqrt_G, np.log)
                for C in covs
            ]
            delta = np.mean(logs, axis=0)

            if np.linalg.norm(delta, "fro") < self.tol:
                break

            G = sqrt_G @ self._matrix_function(delta, np.exp) @ sqrt_G
            G = (G + G.T) / 2

        return G

    def _vectorize(self, matrices):
        idx = np.triu_indices(matrices.shape[1])
        weights = np.where(idx[0] == idx[1], 1.0, np.sqrt(2.0))
        return matrices[:, idx[0], idx[1]] * weights

    def _fit(self, X, y=None, domains=None):
        X = self._as_bands(X)
        self.references_ = []

        for band in range(X.shape[1]):
            covs = self._covariance(X[:, band])
            self.references_.append(self._riemannian_mean(covs))

        return self

    def _transform(self, X, domains=None):
        if self.references_ is None:
            raise RuntimeError("Riemannian extractor must be fitted before transform().")

        X = self._as_bands(X)

        if X.shape[1] != len(self.references_):
            raise ValueError("Number of frequency bands differs from fitted data.")

        features = []
        for band, reference in enumerate(self.references_):
            covs = self._covariance(X[:, band])
            invsqrt = self._matrix_function(reference, lambda x: 1 / np.sqrt(x))
            tangent = np.asarray([
                self._matrix_function(invsqrt @ C @ invsqrt, np.log)
                for C in covs
            ])
            features.append(self._vectorize(tangent))

        return np.concatenate(features, axis=1)