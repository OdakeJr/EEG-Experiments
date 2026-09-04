# ml/learning/domain_adaptation_labeled/joint_supervised.py

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.learning.base import BaseLearningAlgorithm


class JointSupervised(BaseLearningAlgorithm):

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

        epochs = training_params.get(
            "epochs",
            100,
        )
        batch_size = training_params.get(
            "batch_size",
            64,
        )
        learning_rate = training_params.get(
            "learning_rate",
            1e-3,
        )
        weight_decay = training_params.get(
            "weight_decay",
            0.0,
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

        model.apply(
            self._reset_parameters
        )

        model = model.to(
            self.device
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

        dataset = TensorDataset(
            torch.as_tensor(
                X,
                dtype=torch.float32,
            ),
            torch.as_tensor(
                y_encoded,
                dtype=torch.long,
            ),
        )

        generator = torch.Generator()
        generator.manual_seed(seed)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        criterion = nn.CrossEntropyLoss()

        model.train()

        for _ in range(epochs):

            for X_batch, y_batch in loader:

                X_batch = X_batch.to(
                    self.device
                )
                y_batch = y_batch.to(
                    self.device
                )

                optimizer.zero_grad()

                logits = model(
                    X_batch
                )

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

        X = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():

            logits = self.model(X)

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

        if source is not None:

            mask = (
                source.partitions
                == "train"
            )

            if np.any(mask):
                X_parts.append(
                    source.X[mask]
                )
                y_parts.append(
                    source.y[mask]
                )

        if target_super_domain is not None:

            mask = np.isin(
                target_super_domain.partitions,
                ["train", "calibration"],
            )

            if np.any(mask):
                X_parts.append(
                    target_super_domain.X[mask]
                )
                y_parts.append(
                    target_super_domain.y[mask]
                )

        if target_elementary_domain is not None:

            mask = (
                target_elementary_domain.partitions
                == "calibration"
            )

            if np.any(mask):
                X_parts.append(
                    target_elementary_domain.X[mask]
                )
                y_parts.append(
                    target_elementary_domain.y[mask]
                )

        if not X_parts:
            raise ValueError(
                "JointSupervised requires labeled training data."
            )

        return (
            np.concatenate(X_parts),
            np.concatenate(y_parts),
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