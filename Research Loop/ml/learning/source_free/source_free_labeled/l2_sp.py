# ml/learning/source_free/source_free_labeled/l2_sp.py

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.learning.source_free.base import BaseSourceFree


class L2SP(BaseSourceFree):

    def fit(
        self,
        model,
        source=None,
        target_super_domain=None,
        target_elementary_domain=None,
        **training_params,
    ):
        # ====================================================
        # Stage 1: common source pretraining
        # ====================================================

        model = self._pretrain_source_model(
            model,
            source,
            **training_params,
        )

        source_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

        # ====================================================
        # Stage 2: labeled target adaptation
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
            1e-4,
        )
        weight_decay = training_params.get(
            "weight_decay",
            0.0,
        )
        l2sp_lambda = training_params.get(
            "l2sp_lambda",
            1e-3,
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
            model.parameters(),
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

                classification_loss = criterion(
                    logits,
                    y_batch,
                )

                l2sp_penalty = torch.zeros(
                    (),
                    device=self.device,
                )

                for name, parameter in model.named_parameters():
                    l2sp_penalty += (
                        parameter
                        - source_parameters[name]
                    ).pow(2).sum()

                loss = (
                    classification_loss
                    + l2sp_lambda
                    * l2sp_penalty
                )

                loss.backward()
                optimizer.step()

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
                "L2SP requires labeled "
                "target adaptation data."
            )

        return (
            np.concatenate(X_parts),
            np.concatenate(y_parts),
        )