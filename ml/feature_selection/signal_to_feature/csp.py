import numpy as np
from scipy.linalg import eigh

from ml.feature_selection.base import FeatureTransformer


class CSPTransformer(FeatureTransformer):
    input_representation = "signal"
    output_representation = "features"

    def __init__(
        self,
        n_components=4,
        reg=1e-6,
        log=True,
        pre_scaler=None,
        post_scaler=None,
    ):
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

        raise ValueError(
            f"CSP expects [N,C,T] or [N,B,C,T], got {X.shape}."
        )

    def _mean_covariance(self, X):
        covariances = []

        for trial in X:
            cov = trial @ trial.T
            trace = np.trace(cov)

            if trace > 0:
                cov /= trace

            covariances.append(cov)

        return np.mean(covariances, axis=0)

    def _select_filters(self, eigenvectors, eigenvalues):
        order = np.argsort(eigenvalues)
        low = order[: self.n_components // 2]
        high = order[-(self.n_components - len(low)) :]

        indices = np.concatenate([high, low])
        return eigenvectors[:, indices].T

    def _fit_binary(self, X_positive, X_negative):
        C_pos = self._mean_covariance(X_positive)
        C_neg = self._mean_covariance(X_negative)

        n_channels = C_pos.shape[0]
        identity = np.eye(n_channels)

        C_pos += self.reg * identity
        C_neg += self.reg * identity

        eigenvalues, eigenvectors = eigh(
            C_pos,
            C_pos + C_neg,
        )

        return self._select_filters(
            eigenvectors,
            eigenvalues,
        )

    def _fit(self, X, y=None, domains=None):
        if y is None:
            raise ValueError("CSP requires class labels.")

        X = self._as_bands(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.filters_ = []

        for band in range(X.shape[1]):
            band_filters = []

            for cls in self.classes_:
                positive = X[y == cls, band]
                negative = X[y != cls, band]

                if len(positive) == 0 or len(negative) == 0:
                    raise ValueError(f"Cannot fit CSP for class '{cls}'.")

                band_filters.append(
                    self._fit_binary(positive, negative)
                )

            self.filters_.append(band_filters)

        return self

    def _transform(self, X, domains=None):
        if self.filters_ is None:
            raise RuntimeError("CSP must be fitted before transform().")

        X = self._as_bands(X)
        features = []

        for band, band_filters in enumerate(self.filters_):
            signal = X[:, band]

            for filters in band_filters:
                projected = np.einsum("kc,nct->nkt", filters, signal)
                variance = np.var(projected, axis=2)

                variance /= np.maximum(
                    variance.sum(axis=1, keepdims=True),
                    1e-12,
                )

                if self.log:
                    variance = np.log(np.maximum(variance, 1e-12))

                features.append(variance)

        return np.concatenate(features, axis=1)