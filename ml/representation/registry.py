# ml/representation/registry.py

from ml.representation.feature_extraction.combined import CombinedFeatureExtractor
from ml.representation.feature_extraction.csp import CSPFeatureExtractor
from ml.representation.feature_extraction.handcrafted import HandcraftedFeatureExtractor
from ml.representation.feature_extraction.rcsp import RCSPFeatureExtractor
from ml.representation.feature_extraction.riemann import RiemannianFeatureExtractor
from ml.representation.feature_selection.classical import (
    ANOVASelector,
    MutualInformationSelector,
    VarianceSelector,
)
from ml.representation.feature_selection.random import RandomSelector
from ml.representation.signal_transform.identity import IdentitySignalTransformer
from ml.representation.signal_transform.standardize import StandardizeSignalTransformer


REPRESENTATION_TRANSFORMERS = {
    "identity_signal": IdentitySignalTransformer,
    "standardize_signal": StandardizeSignalTransformer,
    "handcrafted": HandcraftedFeatureExtractor,
    "csp": CSPFeatureExtractor,
    "rcsp": RCSPFeatureExtractor,
    "riemann": RiemannianFeatureExtractor,
    "riemannian": RiemannianFeatureExtractor,
    "combined": CombinedFeatureExtractor,
    "variance": VarianceSelector,
    "anova": ANOVASelector,
    "mutual_information": MutualInformationSelector,
    "random": RandomSelector,
}


def get_representation_transformer(name, params=None):
    if name not in REPRESENTATION_TRANSFORMERS:
        raise ValueError(
            f"Unknown representation transformer '{name}'. "
            f"Available: {sorted(REPRESENTATION_TRANSFORMERS)}"
        )

    return REPRESENTATION_TRANSFORMERS[name](**(params or {}))