# ml/learning/source_free/source_free_unlabeled/shot.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.learning.source_free.base import BaseSourceFree


class SHOT(BaseSourceFree):

    def fit(
        self,
        model,
        source=None,
        target_super_domain=None,
        target_elementary_domain=None,
        **training_params,
    ):
        if (
            not hasattr(model, "extract_features")
            or not hasattr(model, "features")
            or not hasattr(model, "classifier")
        ):
            raise ValueError(
                "SHOT requires a model with "
                "features, classifier, and extract_features()."
            )

        # ====================================================
        # Stage 1: common source pretraining
        # ====================================================

        model = self._pretrain_source_model(
            model,
            source,
            **training_params,
        )

        # ====================================================
        # Stage 2: source-free target adaptation
        # ====================================================

        X_target = self._get_target_data(
            target_super_domain,
            target_elementary_domain,
        )

        adaptation_epochs = training_params.get(
            "adaptation_epochs",
            50,
        )
        adaptation_learning_rate = training_params.get(
            "adaptation_learning_rate",
            1e-4,
        )
        weight_decay = training_params.get(
            "weight_decay",
            0.0,
        )
        pseudo_label_weight = training_params.get(
            "pseudo_label_weight",
            1.0,
        )
        entropy_weight = training_params.get(
            "entropy_weight",
            1.0,
        )
        diversity_weight = training_params.get(
            "diversity_weight",
            1.0,
        )
        seed = training_params.get(
            "seed",
            42,
        )

        self._set_seed(
            seed + 1
        )

        X_target = torch.as_tensor(
            X_target,
            dtype=torch.float32,
            device=self.device,
        )

        self._set_requires_grad(
            model.classifier,
            False,
        )

        optimizer = torch.optim.Adam(
            model.features.parameters(),
            lr=adaptation_learning_rate,
            weight_decay=weight_decay,
        )

        criterion = nn.CrossEntropyLoss()

        for _ in range(adaptation_epochs):

            pseudo_labels = self._get_pseudo_labels(
                model,
                X_target,
            )

            model.train()

            optimizer.zero_grad()

            features = model.extract_features(
                X_target
            )

            logits = model.classifier(
                features
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            pseudo_loss = criterion(
                logits,
                pseudo_labels,
            )

            entropy_loss = (
                -probabilities
                * torch.log(
                    probabilities + 1e-8
                )
            ).sum(
                dim=1
            ).mean()

            mean_probability = probabilities.mean(
                dim=0
            )

            diversity_loss = (
                mean_probability
                * torch.log(
                    mean_probability + 1e-8
                )
            ).sum()

            loss = (
                pseudo_label_weight
                * pseudo_loss
                + entropy_weight
                * entropy_loss
                + diversity_weight
                * diversity_loss
            )

            loss.backward()
            optimizer.step()

        self._set_requires_grad(
            model.classifier,
            True,
        )

        self.model = model

        return self

    @staticmethod
    def _get_pseudo_labels(
        model,
        X,
    ):
        model.eval()

        with torch.no_grad():

            features = model.extract_features(
                X
            )

            logits = model.classifier(
                features
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            features = F.normalize(
                features,
                dim=1,
            )

            centroids = (
                probabilities.T
                @ features
            )

            centroids = centroids / (
                probabilities.sum(
                    dim=0
                )[:, None]
                + 1e-8
            )

            centroids = F.normalize(
                centroids,
                dim=1,
            )

            similarities = (
                features
                @ centroids.T
            )

            pseudo_labels = torch.argmax(
                similarities,
                dim=1,
            )

        return pseudo_labels

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
                "SHOT requires unlabeled "
                "target adaptation data."
            )

        return np.concatenate(
            X_parts
        )