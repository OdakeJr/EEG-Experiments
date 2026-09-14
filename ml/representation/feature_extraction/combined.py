# ml/representation/feature_extraction/combined.py

import numpy as np

from ml.representation.feature_extraction.base import FeatureExtractor
from ml.representation.feature_extraction.csp import CSPFeatureExtractor
from ml.representation.feature_extraction.rcsp import RCSPFeatureExtractor
from ml.representation.feature_extraction.riemann import RiemannianFeatureExtractor


FEATURE_EXTRACTORS = {
    "csp": CSPFeatureExtractor,
    "rcsp": RCSPFeatureExtractor,
    "riemann": RiemannianFeatureExtractor,
    "riemannian": RiemannianFeatureExtractor,
}


class CombinedFeatureExtractor(FeatureExtractor):
    def __init__(self, features, pre_scaler=None, post_scaler=None):
        if pre_scaler is not None:
            raise ValueError("Signal feature extraction does not support pre_scaling.")

        super().__init__(pre_scaler=None, post_scaler=post_scaler)
        self.features = features
        self.extractors_ = {}

    def _build_extractors(self):
        extractors = {}

        for name, params in self.features.items():
            if name not in FEATURE_EXTRACTORS:
                raise ValueError(
                    f"Unknown feature extractor '{name}'. "
                    f"Available: {sorted(FEATURE_EXTRACTORS)}"
                )

            extractors[name] = FEATURE_EXTRACTORS[name](**(params or {}))

        return extractors

    def _fit(self, X, y=None, domains=None):
        self.extractors_ = self._build_extractors()

        for extractor in self.extractors_.values():
            extractor.fit(X, y, domains)

        return self

    def _transform(self, X, domains=None):
        if not self.extractors_:
            raise RuntimeError("CombinedFeatureExtractor must be fitted before transform().")

        outputs = [
            extractor.transform(X, domains)
            for extractor in self.extractors_.values()
        ]

        return np.concatenate(outputs, axis=1)