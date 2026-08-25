# ml/feature_selection/random_fs.py

import numpy as np

from ml.feature_selection.base import FeatureTransformer


class RandomSelector(FeatureTransformer):

    def __init__(self, k=10, seed=42):
        self.k = k
        self.seed = seed
        self.indices = None

    def fit(self, X, y=None, domains=None):
        if self.k > X.shape[1]:
            raise ValueError(
                f"k={self.k} exceeds the number of features ({X.shape[1]})."
            )

        rng = np.random.default_rng(self.seed)

        self.indices = np.sort(
            rng.choice(X.shape[1], size=self.k, replace=False)
        )

        return self

    def transform(self, X, domains=None):
        if self.indices is None:
            raise RuntimeError("RandomSelector must be fitted before transform().")

        return X[:, self.indices]