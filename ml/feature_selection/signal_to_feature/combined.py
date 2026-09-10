# ml/feature_selection/signal_to_feature/combined.py

import numpy as np

from ml.feature_selection.signal_to_feature.base import SignalToFeatureTransformer
from ml.feature_selection.signal_to_feature.csp import CSPTransformer
from ml.feature_selection.signal_to_feature.rcsp import RCSPTransformer
from ml.feature_selection.signal_to_feature.riemann import RiemannianTransformer


SIGNAL_FEATURES = {
    "csp": CSPTransformer,
    "rcsp": RCSPTransformer,
    "riemann": RiemannianTransformer,
}


class SignalFeatureTransformer(SignalToFeatureTransformer):
    def __init__(self, features, pre_scaler=None, post_scaler=None):
        if pre_scaler is not None:
            raise ValueError("Signal feature extraction does not support pre_scaling.")

        super().__init__(pre_scaler=None, post_scaler=post_scaler)
        self.features = features
        self.transformers_ = {}

    def _build_transformers(self):
        transformers = {}

        for name, params in self.features.items():
            if name not in SIGNAL_FEATURES:
                raise ValueError(
                    f"Unknown signal feature '{name}'. "
                    f"Available: {sorted(SIGNAL_FEATURES)}"
                )

            transformers[name] = SIGNAL_FEATURES[name](**(params or {}))

        return transformers

    def _fit(self, X, y=None, domains=None):
        self.transformers_ = self._build_transformers()

        for transformer in self.transformers_.values():
            transformer.fit(X, y, domains)

        return self

    def _transform(self, X, domains=None):
        if not self.transformers_:
            raise RuntimeError("SignalFeatureTransformer must be fitted before transform().")

        outputs = [
            transformer.transform(X, domains)
            for transformer in self.transformers_.values()
        ]

        return np.concatenate(outputs, axis=1)