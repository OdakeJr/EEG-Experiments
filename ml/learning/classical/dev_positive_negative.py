import copy
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ml.learning.base import BaseLearningAlgorithm


class PositiveNegativeLearning(BaseLearningAlgorithm):

    def __init__(self):
        self.positive_model = None
        self.negative_model = None

        self.classes_ = None
        self.device = "cpu"

        self.fusion_lambda = 1.0

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

        beta = training_params.get("beta", 1.0)
        self.fusion_lambda = training_params.get(
            "fusion_lambda",
            1.0,
        )

        self.device = training_params.get("device", "cpu")
        seed = training_params.get("seed", 42)

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

        dataset = TensorDataset(
            X_tensor,
            y_tensor,
        )

        # ----------------------------------------------------
        # Create independent positive and negative models
        # ----------------------------------------------------

        self.positive_model = copy.deepcopy(model)
        self.negative_model = copy.deepcopy(model)

        # ----------------------------------------------------
        # Train positive model
        # ----------------------------------------------------

        self._set_seed(seed)

        self.positive_model.apply(
            self._reset_parameters
        )

        self.positive_model = self.positive_model.to(
            self.device
        )

        positive_loader = self._make_loader(
            dataset,
            batch_size,
            seed,
        )

        optimizer = torch.optim.Adam(
            self.positive_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.positive_model.train()

        for _ in range(epochs):

            for X_batch, y_batch in positive_loader:

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                logits = self.positive_model(
                    X_batch
                )

                loss = F.cross_entropy(
                    logits,
                    y_batch,
                )

                loss.backward()
                optimizer.step()

        # ----------------------------------------------------
        # Train negative model
        # ----------------------------------------------------

        self._set_seed(seed + 1)

        self.negative_model.apply(
            self._reset_parameters
        )

        self.negative_model = self.negative_model.to(
            self.device
        )

        negative_loader = self._make_loader(
            dataset,
            batch_size,
            seed + 1,
        )

        optimizer = torch.optim.Adam(
            self.negative_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.positive_model.eval()
        self.negative_model.train()

        for _ in range(epochs):

            for X_batch, y_batch in negative_loader:

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Positive model identifies the difficult
                # incorrect alternatives.
                with torch.no_grad():

                    positive_logits = (
                        self.positive_model(X_batch)
                    )

                    positive_probs = F.softmax(
                        positive_logits,
                        dim=1,
                    )

                optimizer.zero_grad()

                negative_logits = (
                    self.negative_model(X_batch)
                )

                loss = self._negative_loss(
                    negative_logits,
                    y_batch,
                    positive_probs,
                    beta=beta,
                )

                loss.backward()
                optimizer.step()

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

        self.positive_model.eval()
        self.negative_model.eval()

        X_tensor = torch.as_tensor(
            X,
            dtype=torch.float32,
        ).to(self.device)

        with torch.no_grad():

            positive_logits = self.positive_model(
                X_tensor
            )

            negative_logits = self.negative_model(
                X_tensor
            )

            positive_probs = F.softmax(
                positive_logits,
                dim=1,
            )

            negative_probs = F.softmax(
                negative_logits,
                dim=1,
            )

            # Positive-negative fusion
            scores = (
                positive_probs
                * (1.0 - negative_probs).pow(
                    self.fusion_lambda
                )
            )

            probabilities = (
                scores
                / scores.sum(
                    dim=1,
                    keepdim=True,
                )
            )

        return probabilities.cpu().numpy()

    @staticmethod
    def _negative_loss(
        logits,
        y,
        positive_probs,
        beta=1.0,
        eps=1e-8,
    ):
        negative_probs = F.softmax(
            logits,
            dim=1,
        )

        # ---------------------------------------------
        # 1. Suppress the true class
        # ---------------------------------------------

        q_true = negative_probs.gather(
            1,
            y.unsqueeze(1),
        ).squeeze(1)

        exclusion_loss = -torch.log(
            1.0 - q_true + eps
        )

        # ---------------------------------------------
        # 2. Focus on hard incorrect alternatives
        # ---------------------------------------------

        mask = torch.ones_like(
            negative_probs
        )

        mask.scatter_(
            1,
            y.unsqueeze(1),
            0.0,
        )

        # Wrong classes considered plausible by
        # the positive classifier.
        hard_targets = (
            positive_probs * mask
        )

        hard_targets = (
            hard_targets
            / (
                hard_targets.sum(
                    dim=1,
                    keepdim=True,
                )
                + eps
            )
        )

        # Negative-model distribution restricted
        # to incorrect classes.
        q_wrong = (
            negative_probs * mask
        )

        q_wrong = (
            q_wrong
            / (
                q_wrong.sum(
                    dim=1,
                    keepdim=True,
                )
                + eps
            )
        )

        hard_negative_loss = -(
            hard_targets
            * torch.log(
                q_wrong + eps
            )
        ).sum(dim=1)

        loss = (
            exclusion_loss
            + beta * hard_negative_loss
        )

        return loss.mean()

    @staticmethod
    def _make_loader(
        dataset,
        batch_size,
        seed,
    ):
        generator = torch.Generator()
        generator.manual_seed(seed)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

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

    @staticmethod
    def _set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _check_fitted(self):
        if (
            self.positive_model is None
            or self.negative_model is None
        ):
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