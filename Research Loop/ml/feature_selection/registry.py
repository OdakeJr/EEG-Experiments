# ml/feature_selection/registry.py

from ml.feature_selection.classical import (
    VarianceSelector,
    ANOVASelector,
    MutualInformationSelector,
)

from ml.feature_selection.random_fs import (
    RandomSelector,
)


FEATURE_TRANSFORMERS = {
    "variance": VarianceSelector,
    "anova": ANOVASelector,
    "mutual_information": MutualInformationSelector,
    "random": RandomSelector,
}


def get_feature_transformer(name, params=None):
    if name not in FEATURE_TRANSFORMERS:
        raise ValueError(
            f"Unknown feature transformer '{name}'. "
            f"Available: {sorted(FEATURE_TRANSFORMERS)}"
        )

    params = params or {}

    return FEATURE_TRANSFORMERS[name](**params)