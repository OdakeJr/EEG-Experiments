# ml/learning/domain_adaptation_unlabeled/deep_coral.py

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ml.learning.base import BaseLearningAlgorithm


class DeepCORAL(BaseLearningAlgorithm):

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
        X_source, y_source = self._get_source_data(
            source
        )

        X_target = self._get_target_data(
            target_super_domain,
            target_elementary_domain,
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

        model.apply(
            self._reset_parameters
        )

        model = model.to(
            self.device
        )

        if not hasattr(
            model,
            "extract_features",
        ):
            raise ValueError(
                "DeepCORAL requires a model with extract_features()."
            )

        self.classes_ = np.unique(
            y_source
        )

        class_to_index = {
            label: index
            for index, label in enumerate(self.classes_)
        }

        y_encoded = np.array([
            class_to_index[label]
            for label in y_source
        ])

        X_source = torch.as_tensor(
            X_source,
            dtype=torch.float32,
            device=self.device,
        )

        y_source = torch.as_tensor(
            y_encoded,
            dtype=torch.long,
            device=self.device,
        )

        X_target = torch.as_tensor(
            X_target,
            dtype=torch.float32,
            device=self.device,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        criterion = nn.CrossEntropyLoss()

        model.train()

        for _ in range(epochs):

            optimizer.zero_grad()

            source_logits = model(
                X_source
            )

            source_loss = criterion(
                source_logits,
                y_source,
            )

            source_features = (
                model.extract_features(
                    X_source
                )
            )

            target_features = (
                model.extract_features(
                    X_target
                )
            )

            coral_loss = self._coral_loss(
                source_features,
                target_features,
            )

            loss = (
                source_loss
                + coral_lambda * coral_loss
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

            logits = self.model(
                X
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

        return (
            probabilities
            .cpu()
            .numpy()
        )

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
    def _coral_loss(
        source,
        target,
    ):
        if (
            source.shape[0] < 2
            or target.shape[0] < 2
        ):
            return torch.tensor(
                0.0,
                device=source.device,
            )

        source_covariance = (
            DeepCORAL._covariance(
                source
            )
        )

        target_covariance = (
            DeepCORAL._covariance(
                target
            )
        )

        dimension = source.shape[1]

        return (
            (
                source_covariance
                - target_covariance
            )
            .pow(2)
            .sum()
            / (
                4.0
                * dimension
                * dimension
            )
        )

    @staticmethod
    def _covariance(X):
        X = (
            X
            - X.mean(
                dim=0,
                keepdim=True,
            )
        )

        return (
            X.T @ X
        ) / (
            X.shape[0] - 1
        )

    @staticmethod
    def _get_source_data(source):
        if source is None:
            raise ValueError(
                "DeepCORAL requires source data."
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
        )

    @staticmethod
    def _get_target_data(
        target_super_domain,
        target_elementary_domain,
    ):
        X_parts = []

        if target_super_domain is not None:

            mask = np.isin(
                target_super_domain.partitions,
                ["train", "calibration"],
            )

            if np.any(mask):
                X_parts.append(
                    target_super_domain.X[mask]
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

        if not X_parts:
            raise ValueError(
                "DeepCORAL requires unlabeled target adaptation data."
            )

        return np.concatenate(
            X_parts
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