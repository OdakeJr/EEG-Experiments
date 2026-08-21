# plugins/machine_learning/models/neural_models.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .trained_models import PredictiveModel


# ============================================================
# MLP
# ============================================================

class MLP(nn.Module, PredictiveModel):
    """Configurable MLP for feature-based classification."""

    def __init__(
        self,
        input_dim,
        hidden_dims=(128, 128),
        output_dim=2,
        activation="relu",
        dropout=0.0,
        batch_norm=False,
        classes=None,
    ):
        super().__init__()

        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "elu": nn.ELU,
            "tanh": nn.Tanh,
        }

        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")

        self.classes = None if classes is None else np.asarray(classes)

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(activations[activation]())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, output_dim)

    def extract_features(self, x):
        return self.feature_extractor(x)

    def forward(self, x):
        return self.classifier(
            self.extract_features(x)
        )

    def predict_proba(self, X):
        """Generate class probabilities."""

        self.eval()

        device = next(self.parameters()).device

        X = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=device
        )

        with torch.no_grad():
            logits = self(X)
            probabilities = torch.softmax(logits, dim=1)

        return probabilities.cpu().numpy()

    def predict(self, X):
        """Generate class predictions."""

        indices = np.argmax(
            self.predict_proba(X),
            axis=1
        )

        if self.classes is None:
            return indices

        return self.classes[indices]


# ============================================================
# Gradient reversal
# ============================================================

class GradientReversal(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


# ============================================================
# Domain discriminator
# ============================================================

class DomainDiscriminator(nn.Module):

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# MCD
# ============================================================

class MCDModel(nn.Module, PredictiveModel):
    """Model with one feature extractor and two classifiers."""

    def __init__(
        self,
        input_dim,
        hidden_dims,
        n_classes,
        activation="relu",
        dropout=0.0,
        batch_norm=False,
        classes=None,
    ):
        super().__init__()

        self.classes = None if classes is None else np.asarray(classes)

        base = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=n_classes,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm
        )

        self.feature_extractor = base.feature_extractor

        feature_dim = base.classifier.in_features

        self.classifier1 = nn.Linear(
            feature_dim,
            n_classes
        )

        self.classifier2 = nn.Linear(
            feature_dim,
            n_classes
        )

    def extract_features(self, x):
        return self.feature_extractor(x)

    def forward(self, x):
        z = self.extract_features(x)

        return (
            self.classifier1(z),
            self.classifier2(z)
        )

    def predict_proba(self, X):
        """Generate probabilities by averaging both classifiers."""

        self.eval()

        device = next(self.parameters()).device

        X = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=device
        )

        with torch.no_grad():
            logits_1, logits_2 = self(X)

            probabilities_1 = torch.softmax(logits_1, dim=1)
            probabilities_2 = torch.softmax(logits_2, dim=1)

            probabilities = (
                probabilities_1 + probabilities_2
            ) / 2

        return probabilities.cpu().numpy()

    def predict(self, X):
        """Generate final class predictions."""

        indices = np.argmax(
            self.predict_proba(X),
            axis=1
        )

        if self.classes is None:
            return indices

        return self.classes[indices]


# ============================================================
# Cosine classifier
# ============================================================

class CosineClassifier(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        temperature=0.05
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(output_dim, input_dim)
        )

        self.temperature = temperature

        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)

        return (x @ w.T) / self.temperature


# ============================================================
# Environment predictor
# ============================================================

class EnvironmentPredictor(nn.Module):

    def __init__(
        self,
        input_dim,
        n_classes
    ):
        super().__init__()

        self.source_classifier = nn.Linear(
            input_dim,
            n_classes
        )

        self.target_classifier = nn.Linear(
            input_dim,
            n_classes
        )

    def forward(self, x, domain):

        if domain == "source":
            return self.source_classifier(x)

        if domain == "target":
            return self.target_classifier(x)

        raise ValueError(
            f"Unknown domain: {domain}"
        )