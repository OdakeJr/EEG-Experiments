# ml/learning/source_free/source_free_unlabeled/nrc.py

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ml.learning.source_free.base import BaseSourceFree


class NRC(BaseSourceFree):

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
            or not hasattr(model, "classifier")
        ):
            raise ValueError(
                "NRC requires a model with "
                "classifier and extract_features()."
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
        k = training_params.get(
            "k",
            5,
        )
        kk = training_params.get(
            "kk",
            5,
        )
        nonreciprocal_weight = training_params.get(
            "nonreciprocal_weight",
            0.1,
        )
        self_weight = training_params.get(
            "self_weight",
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

        if len(X_target) < 2:
            raise ValueError(
                "NRC requires at least two "
                "target adaptation samples."
            )

        X_target_tensor = torch.as_tensor(
            X_target,
            dtype=torch.float32,
        )

        target_indices = torch.arange(
            len(X_target_tensor),
            dtype=torch.long,
        )

        target_dataset = TensorDataset(
            X_target_tensor,
            target_indices,
        )

        generator = torch.Generator()
        generator.manual_seed(
            seed + 1
        )

        target_loader = DataLoader(
            target_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        k = min(
            int(k),
            len(X_target) - 1,
        )

        kk = min(
            int(kk),
            len(X_target) - 1,
        )

        feature_bank, score_bank = (
            self._initialize_memory_banks(
                model,
                X_target_tensor,
            )
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=adaptation_learning_rate,
            weight_decay=weight_decay,
        )

        for _ in range(adaptation_epochs):

            model.train()

            for X_batch, indices in target_loader:

                X_batch = X_batch.to(
                    self.device
                )

                indices = indices.to(
                    self.device
                )

                optimizer.zero_grad()

                features = model.extract_features(
                    X_batch
                )

                logits = model.classifier(
                    features
                )

                probabilities = torch.softmax(
                    logits,
                    dim=1,
                )

                normalized_features = F.normalize(
                    features,
                    dim=1,
                )

                # --------------------------------------------
                # Update memory banks
                # --------------------------------------------

                with torch.no_grad():

                    feature_bank[
                        indices
                    ] = normalized_features.detach()

                    score_bank[
                        indices
                    ] = probabilities.detach()

                    # ----------------------------------------
                    # Nearest neighbors
                    # ----------------------------------------

                    similarity = (
                        normalized_features.detach()
                        @ feature_bank.T
                    )

                    similarity[
                        torch.arange(
                            len(indices),
                            device=self.device,
                        ),
                        indices,
                    ] = -torch.inf

                    nearest_indices = torch.topk(
                        similarity,
                        k=k,
                        dim=1,
                    ).indices

                    nearest_scores = score_bank[
                        nearest_indices
                    ]

                    nearest_features = feature_bank[
                        nearest_indices
                    ]

                    # ----------------------------------------
                    # Neighbors of neighbors
                    # ----------------------------------------

                    second_similarity = (
                        nearest_features
                        @ feature_bank.T
                    )

                    second_similarity.scatter_(
                        2,
                        nearest_indices.unsqueeze(-1),
                        -torch.inf,
                    )

                    expanded_indices = torch.topk(
                        second_similarity,
                        k=kk,
                        dim=2,
                    ).indices

                    expanded_scores = score_bank[
                        expanded_indices
                    ]

                    # ----------------------------------------
                    # Reciprocal neighbors
                    # ----------------------------------------

                    current_indices = (
                        indices.view(-1, 1, 1)
                    )

                    reciprocal = (
                        expanded_indices
                        == current_indices
                    ).any(
                        dim=2
                    )

                    neighbor_weights = torch.where(
                        reciprocal,
                        torch.ones_like(
                            reciprocal,
                            dtype=torch.float32,
                        ),
                        torch.full_like(
                            reciprocal,
                            nonreciprocal_weight,
                            dtype=torch.float32,
                        ),
                    )

                    self_scores = score_bank[
                        indices
                    ]

                # --------------------------------------------
                # Neighbor consistency
                # --------------------------------------------

                neighbor_similarity = (
                    nearest_scores
                    * probabilities.unsqueeze(1)
                ).sum(
                    dim=2
                )

                neighbor_loss = -(
                    neighbor_weights
                    * neighbor_similarity
                ).sum(
                    dim=1
                ).mean()

                # --------------------------------------------
                # Expanded neighborhood
                # --------------------------------------------

                expanded_similarity = (
                    expanded_scores
                    * probabilities[
                        :,
                        None,
                        None,
                        :
                    ]
                ).sum(
                    dim=3
                )

                expanded_mask = (
                    expanded_indices
                    != current_indices
                ).float()

                expanded_loss = -(
                    nonreciprocal_weight
                    * expanded_mask
                    * expanded_similarity
                ).sum(
                    dim=(1, 2)
                ).mean()

                # --------------------------------------------
                # Self regularization
                # --------------------------------------------

                self_loss = -(
                    self_scores
                    * probabilities
                ).sum(
                    dim=1
                ).mean()

                # --------------------------------------------
                # Diversity
                # --------------------------------------------

                mean_probability = (
                    probabilities.mean(
                        dim=0
                    )
                )

                diversity_loss = (
                    mean_probability
                    * torch.log(
                        mean_probability
                        + 1e-8
                    )
                ).sum()

                loss = (
                    neighbor_loss
                    + expanded_loss
                    + self_weight
                    * self_loss
                    + diversity_weight
                    * diversity_loss
                )

                loss.backward()
                optimizer.step()

        self.model = model

        return self

    def _initialize_memory_banks(
        self,
        model,
        X_target,
    ):
        model.eval()

        X_target = X_target.to(
            self.device
        )

        with torch.no_grad():

            features = model.extract_features(
                X_target
            )

            feature_bank = F.normalize(
                features,
                dim=1,
            )

            logits = model.classifier(
                features
            )

            score_bank = torch.softmax(
                logits,
                dim=1,
            )

        return (
            feature_bank.detach().clone(),
            score_bank.detach().clone(),
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
                "NRC requires unlabeled "
                "target adaptation data."
            )

        return np.concatenate(
            X_parts
        )