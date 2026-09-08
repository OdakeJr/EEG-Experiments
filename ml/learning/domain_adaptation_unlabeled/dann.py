# ml/learning/domain_adaptation_unlabeled/dann.py

import pickle
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.learning.base import BaseLearningAlgorithm


class _GradientReversalFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, strength):
        ctx.strength = strength
        return X.view_as(X)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


class _GradientReversal(nn.Module):

    def __init__(self, strength=1.0):
        super().__init__()
        self.strength = strength

    def forward(self, X):
        return _GradientReversalFunction.apply(X, self.strength)


class _DomainDiscriminator(nn.Module):

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, X):
        return self.network(X)


class DANN(BaseLearningAlgorithm):

    def __init__(self):
        self.model = None
        self.domain_discriminator = None
        self.classes_ = None
        self.device = "cpu"

    def fit(
        self,
        model,
        source=None,
        target_super_domain=None,
        target_elementary_domain=None,
        **training_params,
    ):
        X_source, y_source = self._get_source_data(source)
        X_target = self._get_target_data(
            target_super_domain,
            target_elementary_domain,
        )

        epochs = training_params.get("epochs", 100)
        batch_size = training_params.get("batch_size", 64)
        learning_rate = training_params.get("learning_rate", 1e-3)
        weight_decay = training_params.get("weight_decay", 0.0)
        dann_lambda = training_params.get("dann_lambda", 1.0)
        domain_hidden_dim = training_params.get("domain_hidden_dim", 64)
        self.device = training_params.get("device", "cpu")
        seed = training_params.get("seed", 42)

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model.apply(self._reset_parameters)
        model = model.to(self.device)

        if not hasattr(model, "extract_features"):
            raise ValueError("DANN requires a model with extract_features().")

        self.classes_ = np.unique(y_source)
        class_to_index = {
            label: index
            for index, label in enumerate(self.classes_)
        }
        y_encoded = np.array([class_to_index[label] for label in y_source])

        source_dataset = TensorDataset(
            torch.as_tensor(X_source, dtype=torch.float32),
            torch.as_tensor(y_encoded, dtype=torch.long),
        )
        target_dataset = TensorDataset(
            torch.as_tensor(X_target, dtype=torch.float32)
        )

        generator = torch.Generator()
        generator.manual_seed(seed)

        source_loader = DataLoader(
            source_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        target_loader = DataLoader(
            target_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        model.eval()
        with torch.no_grad():
            X_probe = torch.as_tensor(
                X_source[:1],
                dtype=torch.float32,
                device=self.device,
            )
            feature_dim = model.extract_features(X_probe).shape[1]

        domain_discriminator = _DomainDiscriminator(
            feature_dim, domain_hidden_dim
        ).to(self.device)

        gradient_reversal = _GradientReversal(dann_lambda)

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(domain_discriminator.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        classification_criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.CrossEntropyLoss()

        model.train()
        domain_discriminator.train()

        for _ in range(epochs):
            for (X_source_batch, y_source_batch), (X_target_batch,) in zip(
                source_loader, cycle(target_loader)
            ):
                X_source_batch = X_source_batch.to(self.device)
                y_source_batch = y_source_batch.to(self.device)
                X_target_batch = X_target_batch.to(self.device)

                optimizer.zero_grad()

                source_logits = model(X_source_batch)
                classification_loss = classification_criterion(
                    source_logits, y_source_batch
                )

                source_features = model.extract_features(X_source_batch)
                target_features = model.extract_features(X_target_batch)

                source_domain_logits = domain_discriminator(
                    gradient_reversal(source_features)
                )
                target_domain_logits = domain_discriminator(
                    gradient_reversal(target_features)
                )

                source_domain_labels = torch.zeros(
                    len(X_source_batch), dtype=torch.long, device=self.device
                )
                target_domain_labels = torch.ones(
                    len(X_target_batch), dtype=torch.long, device=self.device
                )

                source_domain_loss = domain_criterion(
                    source_domain_logits, source_domain_labels
                )
                target_domain_loss = domain_criterion(
                    target_domain_logits, target_domain_labels
                )

                domain_loss = (source_domain_loss + target_domain_loss) / 2.0
                loss = classification_loss + domain_loss

                loss.backward()
                optimizer.step()

        self.model = model
        self.domain_discriminator = domain_discriminator
        return self

    def predict(self, X, domains=None, super_domains=None):
        probabilities = self.predict_proba(X, domains, super_domains)
        indices = np.argmax(probabilities, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X, domains=None, super_domains=None):
        self._check_fitted()
        self.model.eval()

        batch_size = 256
        probabilities = []

        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                X_batch = torch.as_tensor(
                    X[start:start + batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                probabilities.append(
                    torch.softmax(self.model(X_batch), dim=1).cpu().numpy()
                )

        return np.concatenate(probabilities, axis=0)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def _get_source_data(source):
        if source is None:
            raise ValueError("DANN requires source data.")

        mask = source.partitions == "train"

        if not np.any(mask):
            raise ValueError("No source training samples found.")

        return source.X[mask], source.y[mask]

    @staticmethod
    def _get_target_data(target_super_domain, target_elementary_domain):
        X_parts = []

        if target_super_domain is not None:
            mask = np.isin(
                target_super_domain.partitions,
                ["train", "calibration"],
            )
            if np.any(mask):
                X_parts.append(target_super_domain.X[mask])

        if target_elementary_domain is not None:
            mask = target_elementary_domain.partitions == "calibration"
            if np.any(mask):
                X_parts.append(target_elementary_domain.X[mask])

        if not X_parts:
            raise ValueError(
                "DANN requires unlabeled target adaptation data."
            )

        return np.concatenate(X_parts)

    @staticmethod
    def _reset_parameters(module):
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    def _check_fitted(self):
        if self.model is None:
            raise RuntimeError("Learning algorithm has not been fitted.")