import numpy as np

from ml.feature_selection.base import FeatureTransformer


class RandomSelector(FeatureTransformer):
    def __init__(self, ratio=0.5, seed=42, pre_scaler=None, post_scaler=None):
        super().__init__(pre_scaler=pre_scaler, post_scaler=post_scaler)
        self.ratio = ratio
        self.seed = seed
        self.indices = None

    def _fit(self, X, y=None, domains=None):
        if not 0 < self.ratio <= 1:
            raise ValueError("'ratio' must be in (0, 1].")

        k = max(1, int(round(X.shape[1] * self.ratio)))
        rng = np.random.default_rng(self.seed)

        self.indices = np.sort(
            rng.choice(X.shape[1], size=k, replace=False)
        )

        return self

    def _transform(self, X, domains=None):
        if self.indices is None:
            raise RuntimeError("RandomSelector must be fitted before transform().")

        return X[:, self.indices]