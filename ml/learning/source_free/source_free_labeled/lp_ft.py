# ml/learning/source_free/source_free_labeled/lp_ft.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.learning.source_free.base import BaseSourceFree


class LPFT(BaseSourceFree):

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
                "LPFT requires a model with "
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
        # Target data
        # ====================================================

        X_target, y_target = self._get_target_data(
            target_super_domain,
            target_elementary_domain,
        )

        batch_size = training_params.get(
            "batch_size",
            64,
        )
        probe_epochs = training_params.get(
            "probe_epochs",
            25,
        )
        fine_tune_epochs = training_params.get(
            "fine_tune_epochs",
            25,
        )
        probe_learning_rate = training_params.get(
            "probe_learning_rate",
            1e-3,
        )
        fine_tune_learning_rate = training_params.get(
            "fine_tune_learning_rate",
            1e-4,
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

        criterion = nn.CrossEntropyLoss()

        # ====================================================
        # Stage 2: linear probe
        # ====================================================

        self._set_requires_grad(
            model.features,
            False,
        )

        model.classifier.apply(
            self._reset_parameters
        )

        probe_generator = torch.Generator()
        probe_generator.manual_seed(
            seed + 1
        )

        probe_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=probe_generator,
        )

        optimizer = torch.optim.Adam(
            model.classifier.parameters(),
            lr=probe_learning_rate,
            weight_decay=weight_decay,
        )

        model.train()

        for _ in range(probe_epochs):

            for X_batch, y_batch in probe_loader:

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

        # ====================================================
        # Stage 3: full fine-tuning
        # ====================================================

        self._set_requires_grad(
            model.features,
            True,
        )

        fine_tune_generator = torch.Generator()
        fine_tune_generator.manual_seed(
            seed + 2
        )

        fine_tune_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=fine_tune_generator,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=fine_tune_learning_rate,
            weight_decay=weight_decay,
        )

        model.train()

        for _ in range(fine_tune_epochs):

            for X_batch, y_batch in fine_tune_loader:

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
                "LPFT requires labeled "
                "target adaptation data."
            )

        return (
            np.concatenate(X_parts),
            np.concatenate(y_parts),
        )