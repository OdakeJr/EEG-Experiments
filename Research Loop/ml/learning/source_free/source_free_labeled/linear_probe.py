# ml/learning/source_free/source_free_labeled/linear_probe.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.learning.source_free.base import BaseSourceFree


class LinearProbe(BaseSourceFree):

    def fit(
        self,
        model,
        source=None,
        target_super_domain=None,
        target_elementary_domain=None,
        **training_params,
    ):
        if (
            not hasattr(model, "features")
            or not hasattr(model, "classifier")
        ):
            raise ValueError(
                "LinearProbe requires a model with "
                "features and classifier."
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
        # Stage 2: labeled target linear probe
        # ====================================================

        X_target, y_target = self._get_target_data(
            target_super_domain,
            target_elementary_domain,
        )

        adaptation_epochs = training_params.get(
            "adaptation_epochs",
            50,
        )
        batch_size = training_params.get(
            "batch_size",
            64,
        )
        adaptation_learning_rate = training_params.get(
            "adaptation_learning_rate",
            1e-3,
        )
        weight_decay = training_params.get(
            "weight_decay",
            0.0,
        )
        seed = training_params.get(
            "seed",
            42,
        )

        self._set_seed(
            seed + 1
        )

        y_encoded = self._encode_labels(
            y_target
        )

        # Freeze source representation
        self._set_requires_grad(
            model.features,
            False,
        )

        # New linear classifier for target
        model.classifier.apply(
            self._reset_parameters
        )

        dataset = TensorDataset(
            torch.as_tensor(
                X_target,
                dtype=torch.float32,
            ),
            torch.as_tensor(
                y_encoded,
                dtype=torch.long,
            ),
        )

        generator = torch.Generator()
        generator.manual_seed(
            seed + 1
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        optimizer = torch.optim.Adam(
            model.classifier.parameters(),
            lr=adaptation_learning_rate,
            weight_decay=weight_decay,
        )

        criterion = nn.CrossEntropyLoss()

        model.train()

        for _ in range(adaptation_epochs):

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

        self._set_requires_grad(
            model.features,
            True,
        )

        self.model = model

        return self

    @staticmethod
    def _get_target_data(
        target_super_domain,
        target_elementary_domain,
    ):
        X_parts = []
        y_parts = []

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
                "LinearProbe requires labeled "
                "target adaptation data."
            )

        return (
            np.concatenate(X_parts),
            np.concatenate(y_parts),
        )