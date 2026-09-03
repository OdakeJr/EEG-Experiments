# ml/feature_selection/registry.py

from ml.feature_selection.feature_to_feature.classical import (
    VarianceSelector,
    ANOVASelector,
    MutualInformationSelector,
)
from ml.feature_selection.feature_to_feature.random_fs import RandomSelector
from ml.feature_selection.signal_to_feature.csp import CSPTransformer


FEATURE_TRANSFORMERS = {
    "variance": VarianceSelector,
    "anova": ANOVASelector,
    "mutual_information": MutualInformationSelector,
    "random": RandomSelector,
    "csp": CSPTransformer,
}


def get_feature_transformer(name, params=None):
    if name not in FEATURE_TRANSFORMERS:
        raise ValueError(
            f"Unknown feature transformer '{name}'. "
            f"Available: {sorted(FEATURE_TRANSFORMERS)}"
        )

    return FEATURE_TRANSFORMERS[name](**(params or {}))