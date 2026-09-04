# ml/learning/domain_adaptation_unlabeled/dev_structural_weighting_v1.py

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from torch.utils.data import DataLoader, TensorDataset

from ml.learning.base import BaseLearningAlgorithm


# ============================================================
# Structural kernel helpers
# ============================================================

def _fit_reference_transform(
    X_source,
    X_target,
    standardize=False,
    eps=1e-12,
):
    """
    Fit the common reference transformation used by the
    structural representation.

    The same mean must be used for source and target:

        x_tilde = (x - mu) / scale

    By default only centering is used, matching the initial
    mathematical formulation.
    """

    X_all = np.concatenate(
        [
            X_source,
            X_target,
        ],
        axis=0,
    )

    mean = X_all.mean(
        axis=0,
        keepdims=True,
    )

    if standardize:

        scale = X_all.std(
            axis=0,
            keepdims=True,
        )

        scale = np.where(
            scale < eps,
            1.0,
            scale,
        )

    else:

        scale = np.ones(
            (
                1,
                X_all.shape[1],
            ),
            dtype=float,
        )

    return mean, scale


def _transform(
    X,
    mean,
    scale,
):
    return (
        X - mean
    ) / scale


def _structural_distances_squared(
    X,
    centers,
    mean,
    scale,
):
    """
    Compute squared Frobenius distances between implicit
    second-order representations

        phi(x) = x_tilde x_tilde^T

    without explicitly constructing the d x d matrices.

    We use

        ||aa^T - bb^T||_F^2
        =
        ||a||^4
        + ||b||^4
        - 2(a^T b)^2.
    """

    X = _transform(
        X,
        mean,
        scale,
    )

    centers = _transform(
        centers,
        mean,
        scale,
    )

    X_norm_squared = np.sum(
        X ** 2,
        axis=1,
    )

    center_norm_squared = np.sum(
        centers ** 2,
        axis=1,
    )

    inner_products = (
        X
        @ centers.T
    )

    distances_squared = (
        X_norm_squared[:, None] ** 2
        + center_norm_squared[None, :] ** 2
        - 2.0 * inner_products ** 2
    )

    return np.maximum(
        distances_squared,
        0.0,
    )


def _structural_rbf_kernel(
    X,
    centers,
    sigma,
    mean,
    scale,
):
    """
    RBF kernel in the implicit second-order structural space:

        k(x, x')
        =
        exp(
            -||phi(x)-phi(x')||_F^2
            /(2 sigma^2)
        )
    """

    distances_squared = (
        _structural_distances_squared(
            X,
            centers,
            mean,
            scale,
        )
    )

    return np.exp(
        -distances_squared
        / (
            2.0
            * sigma ** 2
        )
    )


def _median_structural_sigma(
    X,
    mean,
    scale,
    seed=42,
    max_samples=500,
):
    """
    Median heuristic computed directly in the implicit
    structural space.
    """

    rng = np.random.default_rng(
        seed
    )

    if len(X) > max_samples:

        indices = rng.choice(
            len(X),
            size=max_samples,
            replace=False,
        )

        X = X[
            indices
        ]

    distances_squared = (
        _structural_distances_squared(
            X,
            X,
            mean,
            scale,
        )
    )

    distances = np.sqrt(
        distances_squared
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
        np.median(
            values
        )
    )


# ============================================================
# Structural KLIEP
# ============================================================

class _StructuralKLIEP:

    def __init__(
        self,
        sigma=None,
        n_centers=100,
        max_iter=500,
        standardize=False,
        seed=42,
    ):
        self.sigma = sigma
        self.n_centers = n_centers
        self.max_iter = max_iter
        self.standardize = standardize
        self.seed = seed

        self.centers = None
        self.alpha = None

        self.mean = None
        self.scale = None

    def fit(
        self,
        X_source,
        X_target,
    ):
        rng = np.random.default_rng(
            self.seed
        )

        # ----------------------------------------------------
        # Common structural reference
        # ----------------------------------------------------

        (
            self.mean,
            self.scale,
        ) = _fit_reference_transform(
            X_source,
            X_target,
            standardize=self.standardize,
        )

        # ----------------------------------------------------
        # Kernel centers
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Structural kernel bandwidth
        # ----------------------------------------------------

        if self.sigma is None:

            self.sigma = (
                _median_structural_sigma(
                    np.concatenate(
                        [
                            X_source,
                            X_target,
                        ],
                        axis=0,
                    ),
                    self.mean,
                    self.scale,
                    seed=self.seed,
                )
            )

        # ----------------------------------------------------
        # Structural kernels
        # ----------------------------------------------------

        K_source = (
            _structural_rbf_kernel(
                X_source,
                self.centers,
                self.sigma,
                self.mean,
                self.scale,
            )
        )

        K_target = (
            _structural_rbf_kernel(
                X_target,
                self.centers,
                self.sigma,
                self.mean,
                self.scale,
            )
        )

        source_mean = (
            K_source.mean(
                axis=0
            )
        )

        alpha0 = np.ones(
            n_centers
        )

        normalization = (
            source_mean
            @ alpha0
        )

        alpha0 /= max(
            normalization,
            1e-12,
        )

        # ----------------------------------------------------
        # KLIEP optimization
        # ----------------------------------------------------

        def objective(
            alpha,
        ):
            ratios = (
                K_target
                @ alpha
            )

            return -np.mean(
                np.log(
                    ratios
                    + 1e-12
                )
            )

        def gradient(
            alpha,
        ):
            ratios = (
                K_target
                @ alpha
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
                source_mean
                @ alpha
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
                "Structural KLIEP "
                "optimization failed: "
                f"{result.message}"
            )

        self.alpha = result.x

        return self

    def predict(
        self,
        X,
    ):
        weights = (
            _structural_rbf_kernel(
                X,
                self.centers,
                self.sigma,
                self.mean,
                self.scale,
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
# Structural uLSIF
# ============================================================

class _StructuralULSIF:

    def __init__(
        self,
        sigma=None,
        regularization=1e-3,
        n_centers=100,
        standardize=False,
        seed=42,
    ):
        self.sigma = sigma
        self.regularization = regularization
        self.n_centers = n_centers
        self.standardize = standardize
        self.seed = seed

        self.centers = None
        self.alpha = None

        self.mean = None
        self.scale = None

    def fit(
        self,
        X_source,
        X_target,
    ):
        rng = np.random.default_rng(
            self.seed
        )

        # ----------------------------------------------------
        # Common structural reference
        # ----------------------------------------------------

        (
            self.mean,
            self.scale,
        ) = _fit_reference_transform(
            X_source,
            X_target,
            standardize=self.standardize,
        )

        # ----------------------------------------------------
        # Kernel centers
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Structural kernel bandwidth
        # ----------------------------------------------------

        if self.sigma is None:

            self.sigma = (
                _median_structural_sigma(
                    np.concatenate(
                        [
                            X_source,
                            X_target,
                        ],
                        axis=0,
                    ),
                    self.mean,
                    self.scale,
                    seed=self.seed,
                )
            )

        # ----------------------------------------------------
        # Structural kernels
        # ----------------------------------------------------

        K_source = (
            _structural_rbf_kernel(
                X_source,
                self.centers,
                self.sigma,
                self.mean,
                self.scale,
            )
        )

        K_target = (
            _structural_rbf_kernel(
                X_target,
                self.centers,
                self.sigma,
                self.mean,
                self.scale,
            )
        )

        # ----------------------------------------------------
        # uLSIF
        # ----------------------------------------------------

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

    def predict(
        self,
        X,
    ):
        weights = (
            _structural_rbf_kernel(
                X,
                self.centers,
                self.sigma,
                self.mean,
                self.scale,
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
# Estimator factory
# ============================================================

def _get_structural_density_ratio_estimator(
    name,
    params,
    seed,
):
    name = name.lower()

    params = dict(
        params
    )

    params.setdefault(
        "seed",
        seed,
    )

    if name == "kliep":

        return _StructuralKLIEP(
            **params
        )

    if name == "ulsif":

        return _StructuralULSIF(
            **params
        )

    raise ValueError(
        "Unknown structural density-ratio "
        f"estimator '{name}'. "
        "Available: ['kliep', 'ulsif']"
    )


# ============================================================
# Structural Weighting V1
# ============================================================

class StructuralWeightingV1(
    BaseLearningAlgorithm
):

    def __init__(self):

        self.model = None
        self.classes_ = None
        self.device = "cpu"

        self.density_ratio_estimator = None
        self.structural_weights = None

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
        # Structural density-ratio estimation
        # ----------------------------------------------------

        ratio_estimator = (
            _get_structural_density_ratio_estimator(
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

        # ----------------------------------------------------
        # Weight processing
        # ----------------------------------------------------

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
            np.isfinite(
                weights
            )
        ):

            raise ValueError(
                "Structural weights contain "
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
            class_to_index[
                label
            ]
            for label in y_source
        ])

        # ----------------------------------------------------
        # Dataset
        #
        # IMPORTANT:
        # prediction still uses the ORIGINAL feature vectors.
        # Only the importance weights were estimated in the
        # implicit structural space.
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

        generator = (
            torch.Generator()
        )

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

        self.density_ratio_estimator = (
            ratio_estimator
        )

        self.structural_weights = (
            weights
        )

        return self

    # ========================================================
    # Prediction
    # ========================================================

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

            probabilities = (
                torch.softmax(
                    logits,
                    dim=1,
                )
            )

        return (
            probabilities
            .cpu()
            .numpy()
        )

    # ========================================================
    # Persistence
    # ========================================================

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
                "Structural weighting "
                "requires source data."
            )

        mask = (
            source.partitions
            == "train"
        )

        if not np.any(
            mask
        ):

            raise ValueError(
                "No source training "
                "samples found."
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

        if (
            target_super_domain
            is not None
        ):

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

        if (
            target_elementary_domain
            is not None
        ):

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
            X_parts,
            axis=0,
        )

    # ========================================================
    # Utilities
    # ========================================================

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
                "Learning algorithm "
                "has not been fitted."
            )