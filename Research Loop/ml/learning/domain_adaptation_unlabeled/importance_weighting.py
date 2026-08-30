# ml/learning/domain_adaptation_unlabeled/importance_weighting.py

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from torch.utils.data import DataLoader, TensorDataset

from ml.learning.base import BaseLearningAlgorithm


# ============================================================
# Density-ratio helpers
# ============================================================

def _rbf_kernel(X, centers, sigma):
    distances = (
        np.sum(X ** 2, axis=1)[:, None]
        + np.sum(centers ** 2, axis=1)[None, :]
        - 2.0 * X @ centers.T
    )

    distances = np.maximum(
        distances,
        0.0,
    )

    return np.exp(
        -distances
        / (2.0 * sigma ** 2)
    )


def _median_sigma(
    X,
    seed=42,
    max_samples=500,
):
    rng = np.random.default_rng(seed)

    if len(X) > max_samples:
        indices = rng.choice(
            len(X),
            size=max_samples,
            replace=False,
        )

        X = X[indices]

    distances = (
        np.sum(X ** 2, axis=1)[:, None]
        + np.sum(X ** 2, axis=1)[None, :]
        - 2.0 * X @ X.T
    )

    distances = np.sqrt(
        np.maximum(
            distances,
            0.0,
        )
    )

    values = distances[
        np.triu_indices(
            len(X),
            k=1,
        )
    ]

    values = values[
        values > 0
    ]

    if len(values) == 0:
        return 1.0

    return float(
        np.median(values)
    )


# ============================================================
# KLIEP
# ============================================================

class _KLIEP:

    def __init__(
        self,
        sigma=None,
        n_centers=100,
        max_iter=500,
        seed=42,
    ):
        self.sigma = sigma
        self.n_centers = n_centers
        self.max_iter = max_iter
        self.seed = seed

        self.centers = None
        self.alpha = None

    def fit(
        self,
        X_source,
        X_target,
    ):
        rng = np.random.default_rng(
            self.seed
        )

        n_centers = min(
            self.n_centers,
            len(X_target),
        )

        indices = rng.choice(
            len(X_target),
            size=n_centers,
            replace=False,
        )

        self.centers = X_target[
            indices
        ]

        if self.sigma is None:
            self.sigma = _median_sigma(
                np.concatenate([
                    X_source,
                    X_target,
                ]),
                seed=self.seed,
            )

        K_source = _rbf_kernel(
            X_source,
            self.centers,
            self.sigma,
        )

        K_target = _rbf_kernel(
            X_target,
            self.centers,
            self.sigma,
        )

        source_mean = (
            K_source.mean(axis=0)
        )

        alpha0 = np.ones(
            n_centers
        )

        normalization = (
            source_mean @ alpha0
        )

        alpha0 /= max(
            normalization,
            1e-12,
        )

        def objective(alpha):
            ratios = (
                K_target @ alpha
            )

            return -np.mean(
                np.log(
                    ratios + 1e-12
                )
            )

        def gradient(alpha):
            ratios = (
                K_target @ alpha
            )

            return -np.mean(
                K_target
                / (
                    ratios[:, None]
                    + 1e-12
                ),
                axis=0,
            )

        constraint = {
            "type": "eq",
            "fun": lambda alpha: (
                source_mean @ alpha
                - 1.0
            ),
            "jac": lambda alpha: (
                source_mean
            ),
        }

        result = minimize(
            objective,
            alpha0,
            jac=gradient,
            bounds=[
                (0.0, None)
                for _ in range(
                    n_centers
                )
            ],
            constraints=[
                constraint
            ],
            method="SLSQP",
            options={
                "maxiter": self.max_iter,
            },
        )

        if not result.success:
            raise RuntimeError(
                "KLIEP optimization failed: "
                f"{result.message}"
            )

        self.alpha = result.x

        return self

    def predict(self, X):
        weights = (
            _rbf_kernel(
                X,
                self.centers,
                self.sigma,
            )
            @ self.alpha
        )

        return np.maximum(
            weights,
            0.0,
        )

    def fit_predict(
        self,
        X_source,
        X_target,
    ):
        self.fit(
            X_source,
            X_target,
        )

        return self.predict(
            X_source
        )


# ============================================================
# uLSIF
# ============================================================

class _ULSIF:

    def __init__(
        self,
        sigma=None,
        regularization=1e-3,
        n_centers=100,
        seed=42,
    ):
        self.sigma = sigma
        self.regularization = regularization
        self.n_centers = n_centers
        self.seed = seed

        self.centers = None
        self.alpha = None

    def fit(
        self,
        X_source,
        X_target,
    ):
        rng = np.random.default_rng(
            self.seed
        )

        n_centers = min(
            self.n_centers,
            len(X_target),
        )

        indices = rng.choice(
            len(X_target),
            size=n_centers,
            replace=False,
        )

        self.centers = X_target[
            indices
        ]

        if self.sigma is None:
            self.sigma = _median_sigma(
                np.concatenate([
                    X_source,
                    X_target,
                ]),
                seed=self.seed,
            )

        K_source = _rbf_kernel(
            X_source,
            self.centers,
            self.sigma,
        )

        K_target = _rbf_kernel(
            X_target,
            self.centers,
            self.sigma,
        )

        H = (
            K_source.T
            @ K_source
        ) / len(
            X_source
        )

        h = K_target.mean(
            axis=0
        )

        regularized = (
            H
            + self.regularization
            * np.eye(
                n_centers
            )
        )

        self.alpha = np.linalg.solve(
            regularized,
            h,
        )

        return self

    def predict(self, X):
        weights = (
            _rbf_kernel(
                X,
                self.centers,
                self.sigma,
            )
            @ self.alpha
        )

        return np.maximum(
            weights,
            0.0,
        )

    def fit_predict(
        self,
        X_source,
        X_target,
    ):
        self.fit(
            X_source,
            X_target,
        )

        return self.predict(
            X_source
        )


def _get_density_ratio_estimator(
    name,
    params,
    seed,
):
    name = name.lower()
    params = dict(params)

    params.setdefault(
        "seed",
        seed,
    )

    if name == "kliep":
        return _KLIEP(
            **params
        )

    if name == "ulsif":
        return _ULSIF(
            **params
        )

    raise ValueError(
        "Unknown density-ratio estimator "
        f"'{name}'. Available: "
        "['kliep', 'ulsif']"
    )


# ============================================================
# Importance Weighting
# ============================================================

class ImportanceWeighting(BaseLearningAlgorithm):

    def __init__(self):
        self.model = None
        self.classes_ = None
        self.device = "cpu"

        self.density_ratio_estimator = None
        self.importance_weights = None

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

        estimator_name = training_params.get(
            "estimator",
            "kliep",
        )

        estimator_params = training_params.get(
            "estimator_params",
            {},
        )

        normalize_weights = training_params.get(
            "normalize_weights",
            True,
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

        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                seed
            )

        # ----------------------------------------------------
        # Density ratio
        # ----------------------------------------------------

        ratio_estimator = (
            _get_density_ratio_estimator(
                estimator_name,
                estimator_params,
                seed,
            )
        )

        weights = (
            ratio_estimator.fit_predict(
                X_source,
                X_target,
            )
        )

        if max_weight is not None:
            weights = np.clip(
                weights,
                0.0,
                max_weight,
            )

        if normalize_weights:
            weights = (
                weights
                / (
                    weights.mean()
                    + 1e-12
                )
            )

        if not np.all(
            np.isfinite(weights)
        ):
            raise ValueError(
                "Importance weights contain "
                "non-finite values."
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
        # Weighted ERM
        # ----------------------------------------------------

        for _ in range(epochs):

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

        self.model = model
        self.density_ratio_estimator = (
            ratio_estimator
        )
        self.importance_weights = weights

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

    def save(self, path):
        Path(path).parent.mkdir(
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
    def load(cls, path):
        with open(
            path,
            "rb",
        ) as file:

            return pickle.load(
                file
            )

    @staticmethod
    def _get_source_data(source):
        if source is None:
            raise ValueError(
                "Importance weighting "
                "requires source data."
            )

        mask = (
            source.partitions
            == "train"
        )

        if not np.any(mask):
            raise ValueError(
                "No source training "
                "samples found."
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
                "Importance weighting requires "
                "unlabeled target adaptation data."
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
                "Learning algorithm "
                "has not been fitted."
            )