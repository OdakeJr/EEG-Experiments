# plugins/machine_learning/models/trained_model.py

from abc import ABC, abstractmethod

class PredictiveModel(ABC):
    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

class TrainedModel:
    """Standard representation of a trained machine-learning model."""

    def __init__(
        self,
        model,
        classes,
        training_history=None,
        artifacts=None,
    ):
        self.model = model
        self.classes = classes

        # Information produced during training, e.g.
        # loss curves, validation metrics, epochs, etc.
        self.training_history = training_history or {}

        # Flexible method-specific information.
        self.artifacts = artifacts or {}

    def predict(self, X):
        """Generate predictions."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Generate class probabilities."""
        return self.model.predict_proba(X)