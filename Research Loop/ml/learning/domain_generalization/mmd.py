# ml/learning/domain_generalization/mmd.py

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ml.learning.base import BaseLearningAlgorithm


class MMD(BaseLearningAlgorithm):

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
        mmd_lambda = training_params.get(
            "mmd_lambda",
            1.0,
        )
        gamma = training_params.get(
            "gamma",
            None,
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
                "MMD requires a model with extract_features()."
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

            mmd_penalties = []

            for i in range(len(domain_features)):
                for j in range(i + 1, len(domain_features)):

                    penalty = self._mmd(
                        domain_features[i],
                        domain_features[j],
                        gamma,
                    )

                    mmd_penalties.append(
                        penalty
                    )

            if mmd_penalties:
                mmd_penalty = torch.stack(
                    mmd_penalties
                ).mean()
            else:
                mmd_penalty = torch.tensor(
                    0.0,
                    device=self.device,
                )

            loss = (
                classification_loss
                + mmd_lambda * mmd_penalty
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
    def _mmd(
        X,
        Y,
        gamma=None,
    ):
        if gamma is None:
            gamma = 1.0 / X.shape[1]

        K_xx = MMD._rbf_kernel(
            X,
            X,
            gamma,
        )

        K_yy = MMD._rbf_kernel(
            Y,
            Y,
            gamma,
        )

        K_xy = MMD._rbf_kernel(
            X,
            Y,
            gamma,
        )

        return (
            K_xx.mean()
            + K_yy.mean()
            - 2.0 * K_xy.mean()
        )

    @staticmethod
    def _rbf_kernel(
        X,
        Y,
        gamma,
    ):
        distances = torch.cdist(
            X,
            Y,
        ).pow(2)

        return torch.exp(
            -gamma * distances
        )

    @staticmethod
    def _get_training_data(source):
        if source is None:
            raise ValueError(
                "MMD requires source data."
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