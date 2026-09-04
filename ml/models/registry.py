from ml.models.classical import (
    logistic_regression,
    random_forest,
    svm,
)
from ml.models.mlp import MLP
from ml.models.eegnet import EEGNet


MODELS = {
    "logistic_regression": logistic_regression,
    "random_forest": random_forest,
    "svm": svm,
    "mlp": MLP,
    "eegnet": EEGNet,
}


def get_model(name, params=None, **context):
    if name not in MODELS:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {sorted(MODELS)}"
        )

    resolved_params = {
        **(params or {}),
        **context,
    }

    return MODELS[name](**resolved_params)