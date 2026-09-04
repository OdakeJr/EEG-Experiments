# ml/learning/domain_generalization/groupdro.py

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ml.learning.base import BaseLearningAlgorithm


class GroupDRO(BaseLearningAlgorithm):

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
        eta = training_params.get(
            "eta",
            0.01,
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

        group_weights = torch.ones(
            len(unique_domains),
            device=self.device,
        )

        group_weights /= group_weights.sum()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        criterion = nn.CrossEntropyLoss()

        model.train()

        for _ in range(epochs):

            optimizer.zero_grad()

            losses = []

            for indices in domain_indices.values():

                logits = model(
                    X_tensor[indices]
                )

                loss = criterion(
                    logits,
                    y_tensor[indices],
                )

                losses.append(loss)

            losses = torch.stack(losses)

            with torch.no_grad():
                group_weights *= torch.exp(
                    eta * losses.detach()
                )

                group_weights /= (
                    group_weights.sum()
                )

            loss = torch.sum(
                group_weights * losses
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
    def _get_training_data(source):
        if source is None:
            raise ValueError(
                "GroupDRO requires source data."
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