# ml/learning/domain_generalization/irm.py

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ml.learning.base import BaseLearningAlgorithm


class IRM(BaseLearningAlgorithm):

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
        irm_lambda = training_params.get(
            "irm_lambda",
            100.0,
        )
        penalty_anneal_epochs = training_params.get(
            "penalty_anneal_epochs",
            10,
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

        for epoch in range(epochs):

            optimizer.zero_grad()

            risks = []
            penalties = []

            for indices in domain_indices.values():

                logits = model(
                    X_tensor[indices]
                )

                labels = y_tensor[indices]

                risk = criterion(
                    logits,
                    labels,
                )

                penalty = self._irm_penalty(
                    logits,
                    labels,
                )

                risks.append(risk)
                penalties.append(penalty)

            risk = torch.stack(
                risks
            ).mean()

            penalty = torch.stack(
                penalties
            ).mean()

            penalty_weight = (
                irm_lambda
                if epoch >= penalty_anneal_epochs
                else 1.0
            )

            loss = (
                risk
                + penalty_weight * penalty
            )

            if penalty_weight > 1.0:
                loss = loss / penalty_weight

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

    def save(self, path):
        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def _irm_penalty(
        logits,
        labels,
    ):
        scale = torch.tensor(
            1.0,
            device=logits.device,
            requires_grad=True,
        )

        loss = nn.functional.cross_entropy(
            logits * scale,
            labels,
        )

        gradient = torch.autograd.grad(
            loss,
            scale,
            create_graph=True,
        )[0]

        return gradient.pow(2)

    @staticmethod
    def _get_training_data(source):
        if source is None:
            raise ValueError(
                "IRM requires source data."
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