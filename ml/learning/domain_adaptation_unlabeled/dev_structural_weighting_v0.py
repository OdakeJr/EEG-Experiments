# ml/learning/domain_adaptation_unlabeled/dev_structural_weighting_v0.py

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.learning.base import BaseLearningAlgorithm


# ============================================================
# Structural helpers
# ============================================================

def _center_domains(
    X_source,
    X_target,
    eps=1e-12,
):
    """
    Center source and target independently and use a shared scale.

    This reduces the influence of simple mean shifts and emphasizes
    differences in second-order feature interactions.
    """

    X_source = np.asarray(
        X_source,
        dtype=float,
    )

    X_target = np.asarray(
        X_target,
        dtype=float,
    )

    if X_source.ndim != 2 or X_target.ndim != 2:
        raise ValueError(
            "Structural weighting expects arrays with shape "
            "(n_samples, n_features)."
        )

    if X_source.shape[1] != X_target.shape[1]:
        raise ValueError(
            "Source and target must have the same number of features."
        )

    source_mean = X_source.mean(
        axis=0,
        keepdims=True,
    )

    target_mean = X_target.mean(
        axis=0,
        keepdims=True,
    )

    X_source = (
        X_source
        - source_mean
    )

    X_target = (
        X_target
        - target_mean
    )

    pooled = np.concatenate(
        [
            X_source,
            X_target,
        ],
        axis=0,
    )

    scale = pooled.std(
        axis=0,
        keepdims=True,
    )

    scale = np.where(
        scale < eps,
        1.0,
        scale,
    )

    return (
        X_source / scale,
        X_target / scale,
    )


def _target_structure(
    X_target,
    include_diagonal=False,
):
    """
    Estimate the average second-order structure of the target domain.
    """

    denominator = max(
        len(X_target) - 1,
        1,
    )

    structure = (
        X_target.T
        @ X_target
    ) / denominator

    if not include_diagonal:

        structure = structure.copy()

        np.fill_diagonal(
            structure,
            0.0,
        )

    return structure


def _sample_structure_distances(
    X_source,
    target_structure,
    include_diagonal=False,
    normalize_structure=True,
    eps=1e-12,
):
    """
    Compare each source sample's second-order interaction matrix

        S_i = x_i x_i^T

    with the target structural matrix.

    Note that S_i is an interaction matrix, not a conventional
    sample covariance matrix.
    """

    target_structure = np.asarray(
        target_structure,
        dtype=float,
    )

    if normalize_structure:

        target_norm = np.linalg.norm(
            target_structure,
            ord="fro",
        )

        if target_norm > eps:

            target_structure = (
                target_structure
                / target_norm
            )

    distances = np.empty(
        len(X_source),
        dtype=float,
    )

    for index, sample in enumerate(
        X_source
    ):

        sample_structure = np.outer(
            sample,
            sample,
        )

        if not include_diagonal:

            np.fill_diagonal(
                sample_structure,
                0.0,
            )

        if normalize_structure:

            sample_norm = np.linalg.norm(
                sample_structure,
                ord="fro",
            )

            if sample_norm > eps:

                sample_structure = (
                    sample_structure
                    / sample_norm
                )

        distances[index] = np.linalg.norm(
            sample_structure
            - target_structure,
            ord="fro",
        )

    return distances


def _distances_to_weights(
    distances,
    gamma=1.0,
    scale="median",
    min_weight=None,
    max_weight=None,
    normalize_weights=True,
    eps=1e-12,
):
    """
    Convert structural distances into compatibility weights:

        alpha_i = exp(-gamma * d_i / scale)
    """

    distances = np.asarray(
        distances,
        dtype=float,
    )

    if scale == "median":

        positive = distances[
            distances > eps
        ]

        if len(positive) == 0:
            distance_scale = 1.0

        else:
            distance_scale = float(
                np.median(
                    positive
                )
            )

    elif scale == "mean":

        distance_scale = max(
            float(
                distances.mean()
            ),
            eps,
        )

    elif isinstance(
        scale,
        (int, float),
    ):

        distance_scale = max(
            float(scale),
            eps,
        )

    else:

        raise ValueError(
            "distance_scale must be "
            "'median', 'mean', or numeric."
        )

    weights = np.exp(
        -float(gamma)
        * distances
        / distance_scale
    )

    if min_weight is not None:

        weights = np.maximum(
            weights,
            float(min_weight),
        )

    if max_weight is not None:

        weights = np.minimum(
            weights,
            float(max_weight),
        )

    if normalize_weights:

        weights = (
            weights
            / (
                weights.mean()
                + eps
            )
        )

    if not np.all(
        np.isfinite(weights)
    ):

        raise ValueError(
            "Structural weights contain non-finite values."
        )

    return (
        weights,
        distance_scale,
    )


# ============================================================
# Structural Weighting
# ============================================================

class StructuralWeighting(BaseLearningAlgorithm):

    def __init__(self):

        self.model = None
        self.classes_ = None
        self.device = "cpu"

        self.structural_weights = None
        self.structural_distances = None

        self.target_structure = None
        self.distance_scale = None

    def fit(
        self,
        model,
        source=None,
        target_super_domain=None,
        target_elementary_domain=None,
        **training_params,
    ):
        X_source, y_source = (
            self._get_source_data(
                source
            )
        )

        X_target = (
            self._get_target_data(
                target_super_domain,
                target_elementary_domain,
            )
        )

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

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

        gamma = training_params.get(
            "gamma",
            1.0,
        )

        distance_scale = training_params.get(
            "distance_scale",
            "median",
        )

        include_diagonal = training_params.get(
            "include_diagonal",
            False,
        )

        normalize_structure = training_params.get(
            "normalize_structure",
            True,
        )

        normalize_weights = training_params.get(
            "normalize_weights",
            True,
        )

        min_weight = training_params.get(
            "min_weight",
            None,
        )

        max_weight = training_params.get(
            "max_weight",
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

        np.random.seed(
            seed
        )

        torch.manual_seed(
            seed
        )

        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(
                seed
            )

        # ----------------------------------------------------
        # Structural weighting
        # ----------------------------------------------------

        (
            X_source_structure,
            X_target_structure,
        ) = _center_domains(
            X_source,
            X_target,
        )

        target_structure = (
            _target_structure(
                X_target_structure,
                include_diagonal=(
                    include_diagonal
                ),
            )
        )

        distances = (
            _sample_structure_distances(
                X_source_structure,
                target_structure,
                include_diagonal=(
                    include_diagonal
                ),
                normalize_structure=(
                    normalize_structure
                ),
            )
        )

        weights, fitted_scale = (
            _distances_to_weights(
                distances,
                gamma=gamma,
                scale=distance_scale,
                min_weight=min_weight,
                max_weight=max_weight,
                normalize_weights=(
                    normalize_weights
                ),
            )
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        self.classes_ = np.unique(
            y_source
        )

        class_to_index = {
            label: index
            for index, label
            in enumerate(
                self.classes_
            )
        }

        y_encoded = np.array([
            class_to_index[label]
            for label in y_source
        ])

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        dataset = TensorDataset(
            torch.as_tensor(
                X_source,
                dtype=torch.float32,
            ),
            torch.as_tensor(
                y_encoded,
                dtype=torch.long,
            ),
            torch.as_tensor(
                weights,
                dtype=torch.float32,
            ),
        )

        generator = torch.Generator()

        generator.manual_seed(
            seed
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model.apply(
            self._reset_parameters
        )

        model = model.to(
            self.device
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        criterion = nn.CrossEntropyLoss(
            reduction="none"
        )

        model.train()

        # ----------------------------------------------------
        # Structurally weighted ERM
        # ----------------------------------------------------

        for _ in range(
            epochs
        ):

            for (
                X_batch,
                y_batch,
                weight_batch,
            ) in loader:

                X_batch = X_batch.to(
                    self.device
                )

                y_batch = y_batch.to(
                    self.device
                )

                weight_batch = (
                    weight_batch.to(
                        self.device
                    )
                )

                optimizer.zero_grad()

                logits = model(
                    X_batch
                )

                losses = criterion(
                    logits,
                    y_batch,
                )

                loss = (
                    weight_batch
                    * losses
                ).sum() / (
                    weight_batch.sum()
                    .clamp_min(
                        1e-12
                    )
                )

                loss.backward()

                optimizer.step()

        # ----------------------------------------------------
        # Save fitted state
        # ----------------------------------------------------

        self.model = model

        self.structural_weights = (
            weights
        )

        self.structural_distances = (
            distances
        )

        self.target_structure = (
            target_structure
        )

        self.distance_scale = (
            fitted_scale
        )

        return self

    def predict(
        self,
        X,
        domains=None,
        super_domains=None,
    ):
        probabilities = (
            self.predict_proba(
                X,
                domains,
                super_domains,
            )
        )

        indices = np.argmax(
            probabilities,
            axis=1,
        )

        return self.classes_[
            indices
        ]

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

    def save(
        self,
        path,
    ):
        Path(
            path
        ).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            path,
            "wb",
        ) as file:

            pickle.dump(
                self,
                file,
            )

    @classmethod
    def load(
        cls,
        path,
    ):
        with open(
            path,
            "rb",
        ) as file:

            return pickle.load(
                file
            )

    # ========================================================
    # Data helpers
    # ========================================================

    @staticmethod
    def _get_source_data(
        source,
    ):
        if source is None:

            raise ValueError(
                "Structural weighting requires source data."
            )

        mask = (
            source.partitions
            == "train"
        )

        if not np.any(
            mask
        ):

            raise ValueError(
                "No source training samples found."
            )

        return (
            source.X[
                mask
            ],
            source.y[
                mask
            ],
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
                [
                    "train",
                    "calibration",
                ],
            )

            if np.any(
                mask
            ):

                X_parts.append(
                    target_super_domain.X[
                        mask
                    ]
                )

        if target_elementary_domain is not None:

            mask = (
                target_elementary_domain.partitions
                == "calibration"
            )

            if np.any(
                mask
            ):

                X_parts.append(
                    target_elementary_domain.X[
                        mask
                    ]
                )

        if not X_parts:

            raise ValueError(
                "Structural weighting requires "
                "unlabeled target adaptation data."
            )

        return np.concatenate(
            X_parts
        )

    @staticmethod
    def _reset_parameters(
        module,
    ):
        if hasattr(
            module,
            "reset_parameters",
        ):

            module.reset_parameters()

    def _check_fitted(
        self,
    ):
        if self.model is None:

            raise RuntimeError(
                "Learning algorithm has not been fitted."
            )