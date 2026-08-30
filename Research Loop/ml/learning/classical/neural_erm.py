import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.learning.base import BaseLearningAlgorithm


class NeuralERM(BaseLearningAlgorithm):

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
        X, y = self._get_training_data(
            source,
            target_super_domain,
            target_elementary_domain,
        )

        epochs = training_params.get("epochs", 100)
        batch_size = training_params.get("batch_size", 64)
        learning_rate = training_params.get("learning_rate", 1e-3)
        weight_decay = training_params.get("weight_decay", 0.0)
        self.device = training_params.get("device", "cpu")
        seed = training_params.get("seed", 42)

        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model.apply(self._reset_parameters)

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
        )

        y_tensor = torch.as_tensor(
            y_encoded,
            dtype=torch.long,
        )

        generator = torch.Generator()
        generator.manual_seed(seed)

        loader = DataLoader(
            TensorDataset(
                X_tensor,
                y_tensor,
            ),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        model = model.to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        criterion = nn.CrossEntropyLoss()

        model.train()

        for _ in range(epochs):

            for X_batch, y_batch in loader:

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                logits = model(X_batch)

                loss = criterion(
                    logits,
                    y_batch,
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

            logits = self.model(X_tensor)

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
    def _get_training_data(
        source,
        target_super_domain,
        target_elementary_domain,
    ):
        X_parts = []
        y_parts = []

        for group in [
            source,
            target_super_domain,
            target_elementary_domain,
        ]:
            if group is None:
                continue

            mask = group.partitions == "train"

            if np.any(mask):
                X_parts.append(group.X[mask])
                y_parts.append(group.y[mask])

        if not X_parts:
            raise ValueError(
                "No training samples found."
            )

        return (
            np.concatenate(X_parts),
            np.concatenate(y_parts),
        )

    @staticmethod
    def _reset_parameters(module):
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    def _check_fitted(self):
        if self.model is None:
            raise RuntimeError(
                "Learning algorithm has not been fitted."
            )