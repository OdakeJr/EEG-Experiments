from ml.models.classical import (
    logistic_regression,
    random_forest,
    svm,
)
from ml.models.mlp import MLP


MODELS = {
    "logistic_regression": logistic_regression,
    "random_forest": random_forest,
    "svm": svm,
    "mlp": MLP,
}


def get_model(name, params=None):
    if name not in MODELS:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {sorted(MODELS)}"
        )

    params = params or {}

    return MODELS[name](**params)