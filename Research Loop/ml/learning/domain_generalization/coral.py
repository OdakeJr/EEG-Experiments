# ml/learning/domain_generalization/coral.py

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ml.learning.base import BaseLearningAlgorithm


class CORAL(BaseLearningAlgorithm):

    def __init__(self):
        self.model = None
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
        X, y, domains = self._get_training_data(
            source
        )

        epochs = training_params.get(
            "epochs",
            100,
        )
        learning_rate = training_params.get(
            "learning_rate",
            1e-3,
        )
        weight_decay = training_params.get(
            "weight_decay",
            0.0,
        )
        coral_lambda = training_params.get(
            "coral_lambda",
            1.0,
        )
        self.device = training_params.get(
            "device",
            "cpu",
        )
        seed = training_params.get(
            "seed",
            42,
        )

        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model.apply(self._reset_parameters)
        model = model.to(self.device)

        if not hasattr(model, "extract_features"):
            raise ValueError(
                "CORAL requires a model with extract_features()."
            )

        self.classes_ = np.unique(y)

        class_to_index = {
            label: index
            for index, label in enumerate(self.classes_)
        }

        y_encoded = np.array([
            class_to_index[label]
            for label in y
        ])

        X_tensor = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=self.device,
        )

        y_tensor = torch.as_tensor(
            y_encoded,
            dtype=torch.long,
            device=self.device,
        )

        domains = np.asarray(domains)
        unique_domains = np.unique(domains)

        domain_indices = {
            domain: torch.as_tensor(
                np.where(domains == domain)[0],
                dtype=torch.long,
                device=self.device,
            )
            for domain in unique_domains
        }

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        criterion = nn.CrossEntropyLoss()

        model.train()

        for _ in range(epochs):

            optimizer.zero_grad()

            logits = model(X_tensor)

            classification_loss = criterion(
                logits,
                y_tensor,
            )

            features = model.extract_features(
                X_tensor
            )

            domain_features = [
                features[indices]
                for indices in domain_indices.values()
            ]

            coral_penalties = []

            for i in range(len(domain_features)):
                for j in range(i + 1, len(domain_features)):

                    if (
                        len(domain_features[i]) < 2
                        or len(domain_features[j]) < 2
                    ):
                        continue

                    coral_penalties.append(
                        self._coral_loss(
                            domain_features[i],
                            domain_features[j],
                        )
                    )

            if coral_penalties:
                coral_penalty = torch.stack(
                    coral_penalties
                ).mean()
            else:
                coral_penalty = torch.tensor(
                    0.0,
                    device=self.device,
                )

            loss = (
                classification_loss
                + coral_lambda * coral_penalty
            )

            loss.backward()
            optimizer.step()

        self.model = model

        return self

    def predict(
        self,
        X,
        domains=None,
        super_domains=None,
    ):
        probabilities = self.predict_proba(
            X,
            domains,
            super_domains,
        )

        indices = np.argmax(
            probabilities,
            axis=1,
        )

        return self.classes_[indices]

    def predict_proba(
        self,
        X,
        domains=None,
        super_domains=None,
    ):
        self._check_fitted()

        self.model.eval()

        X_tensor = torch.as_tensor(
            X,
            dtype=torch.float32,
        ).to(self.device)

        with torch.no_grad():

            logits = self.model(
                X_tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

        return probabilities.cpu().numpy()

    @staticmethod
    def _coral_loss(X, Y):
        covariance_X = CORAL._covariance(X)
        covariance_Y = CORAL._covariance(Y)

        dimension = X.shape[1]

        return (
            (covariance_X - covariance_Y)
            .pow(2)
            .sum()
            / (4.0 * dimension * dimension)
        )

    @staticmethod
    def _covariance(X):
        X = X - X.mean(
            dim=0,
            keepdim=True,
        )

        return (
            X.T @ X
        ) / (X.shape[0] - 1)

    @staticmethod
    def _get_training_data(source):
        if source is None:
            raise ValueError(
                "CORAL requires source data."
            )

        mask = (
            source.partitions == "train"
        )

        if not np.any(mask):
            raise ValueError(
                "No source training samples found."
            )

        return (
            source.X[mask],
            source.y[mask],
            source.elementary_domains[mask],
        )

    @staticmethod
    def _reset_parameters(module):
        if hasattr(
            module,
            "reset_parameters",
        ):
            module.reset_parameters()

    def _check_fitted(self):
        if self.model is None:
            raise RuntimeError(
                "Learning algorithm has not been fitted."
            )