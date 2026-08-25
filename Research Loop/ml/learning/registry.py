from ml.learning.classical.sklearn_erm import SklearnERM
from ml.learning.classical.neural_erm import NeuralERM


LEARNING_ALGORITHMS = {
    "sklearn_erm": SklearnERM,
    "neural_erm": NeuralERM,
}


def get_learning_algorithm(name, params=None):
    if name not in LEARNING_ALGORITHMS:
        raise ValueError(
            f"Unknown learning algorithm '{name}'. "
            f"Available: {sorted(LEARNING_ALGORITHMS)}"
        )

    params = params or {}

    return LEARNING_ALGORITHMS[name](**params)