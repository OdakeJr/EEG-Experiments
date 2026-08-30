# ml/learning/source_free/source_free_unlabeled/sfda_de.py

import numpy as np
import torch
import torch.nn.functional as F

from ml.learning.source_free.base import BaseSourceFree


class SFDADE(BaseSourceFree):

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
            or not hasattr(model.classifier, "weight")
        ):
            raise ValueError(
                "SFDA-DE requires a model with "
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
        confidence_threshold = training_params.get(
            "confidence_threshold",
            0.5,
        )
        covariance_gamma = training_params.get(
            "covariance_gamma",
            1.0,
        )
        kernel_gamma = training_params.get(
            "kernel_gamma",
            None,
        )
        samples_per_class = training_params.get(
            "samples_per_class",
            16,
        )
        steps_per_epoch = training_params.get(
            "steps_per_epoch",
            10,
        )
        kmeans_iterations = training_params.get(
            "kmeans_iterations",
            10,
        )
        covariance_eps = training_params.get(
            "covariance_eps",
            1e-5,
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

        anchors = (
            model.classifier.weight
            .detach()
            .clone()
        )

        for _ in range(adaptation_epochs):

            # ------------------------------------------------
            # Pseudo-labeling and distribution estimation
            # ------------------------------------------------

            model.eval()

            with torch.no_grad():

                target_features = (
                    model.extract_features(
                        X_target
                    )
                )

                pseudo_labels, distances = (
                    self._spherical_kmeans(
                        target_features,
                        anchors,
                        kmeans_iterations,
                    )
                )

                confident_mask = (
                    distances
                    < confidence_threshold
                )

                means, covariances = (
                    self._estimate_distributions(
                        target_features,
                        pseudo_labels,
                        confident_mask,
                        anchors,
                        covariance_gamma,
                        covariance_eps,
                    )
                )

            # ------------------------------------------------
            # CDD adaptation
            # ------------------------------------------------

            model.train()

            for _ in range(steps_per_epoch):

                optimizer.zero_grad()

                target_features = (
                    model.extract_features(
                        X_target
                    )
                )

                surrogate_features = []
                target_class_features = []

                for class_index in range(
                    len(self.classes_)
                ):

                    class_mask = (
                        (pseudo_labels == class_index)
                        & confident_mask
                    )

                    if not torch.any(class_mask):
                        class_mask = (
                            pseudo_labels
                            == class_index
                        )

                    if not torch.any(class_mask):
                        continue

                    class_target = (
                        target_features[
                            class_mask
                        ]
                    )

                    class_target = self._sample_rows(
                        class_target,
                        samples_per_class,
                    )

                    class_surrogate = (
                        self._sample_gaussian(
                            means[class_index],
                            covariances[class_index],
                            samples_per_class,
                        )
                    )

                    target_class_features.append(
                        class_target
                    )

                    surrogate_features.append(
                        class_surrogate
                    )

                if len(
                    target_class_features
                ) < 2:
                    raise ValueError(
                        "SFDA-DE requires at least "
                        "two represented target classes."
                    )

                loss = self._cdd_loss(
                    surrogate_features,
                    target_class_features,
                    kernel_gamma,
                )

                loss.backward()
                optimizer.step()

        self._set_requires_grad(
            model.classifier,
            True,
        )

        self.model = model

        return self

    # ========================================================
    # Pseudo-labeling
    # ========================================================

    @staticmethod
    def _spherical_kmeans(
        features,
        anchors,
        iterations,
    ):
        normalized_features = F.normalize(
            features,
            dim=1,
        )

        centers = F.normalize(
            anchors,
            dim=1,
        )

        for _ in range(iterations):

            similarity = (
                normalized_features
                @ centers.T
            )

            labels = torch.argmax(
                similarity,
                dim=1,
            )

            new_centers = centers.clone()

            for class_index in range(
                len(centers)
            ):

                mask = (
                    labels
                    == class_index
                )

                if torch.any(mask):

                    center = (
                        normalized_features[
                            mask
                        ]
                        .mean(dim=0)
                    )

                    new_centers[
                        class_index
                    ] = F.normalize(
                        center.unsqueeze(0),
                        dim=1,
                    ).squeeze(0)

            if torch.allclose(
                centers,
                new_centers,
                atol=1e-5,
            ):
                centers = new_centers
                break

            centers = new_centers

        similarity = (
            normalized_features
            @ centers.T
        )

        labels = torch.argmax(
            similarity,
            dim=1,
        )

        assigned_similarity = similarity[
            torch.arange(
                len(features),
                device=features.device,
            ),
            labels,
        ]

        distances = (
            0.5
            * (
                1.0
                - assigned_similarity
            )
        )

        return (
            labels,
            distances,
        )

    # ========================================================
    # Distribution estimation
    # ========================================================

    @staticmethod
    def _estimate_distributions(
        features,
        labels,
        confident_mask,
        anchors,
        covariance_gamma,
        covariance_eps,
    ):
        means = []
        covariances = []

        feature_dim = features.shape[1]

        identity = torch.eye(
            feature_dim,
            device=features.device,
        )

        for class_index in range(
            len(anchors)
        ):

            mask = (
                (labels == class_index)
                & confident_mask
            )

            if not torch.any(mask):
                mask = (
                    labels
                    == class_index
                )

            if torch.any(mask):

                class_features = (
                    features[mask]
                )

            else:

                normalized_anchor = F.normalize(
                    anchors[
                        class_index
                    ].unsqueeze(0),
                    dim=1,
                ).squeeze(0)

                similarities = (
                    F.normalize(
                        features,
                        dim=1,
                    )
                    @ normalized_anchor
                )

                index = torch.argmax(
                    similarities
                )

                class_features = (
                    features[
                        index:index + 1
                    ]
                )

            target_mean = (
                class_features.mean(
                    dim=0
                )
            )

            anchor = anchors[
                class_index
            ]

            source_mean = (
                torch.linalg.vector_norm(
                    target_mean
                )
                * anchor
                / (
                    torch.linalg.vector_norm(
                        anchor
                    )
                    + 1e-8
                )
            )

            centered = (
                class_features
                - target_mean
            )

            covariance = (
                centered.T
                @ centered
            ) / max(
                len(class_features),
                1,
            )

            covariance = (
                covariance_gamma
                * covariance
                + covariance_eps
                * identity
            )

            means.append(
                source_mean.detach()
            )

            covariances.append(
                covariance.detach()
            )

        return (
            means,
            covariances,
        )

    # ========================================================
    # Sampling
    # ========================================================

    @staticmethod
    def _sample_gaussian(
        mean,
        covariance,
        n_samples,
    ):
        distribution = (
            torch.distributions.MultivariateNormal(
                mean,
                covariance_matrix=covariance,
            )
        )

        return distribution.sample(
            (n_samples,)
        )

    @staticmethod
    def _sample_rows(
        X,
        n_samples,
    ):
        indices = torch.randint(
            low=0,
            high=len(X),
            size=(n_samples,),
            device=X.device,
        )

        return X[indices]

    # ========================================================
    # Contrastive Domain Discrepancy
    # ========================================================

    @staticmethod
    def _cdd_loss(
        surrogate_features,
        target_features,
        gamma=None,
    ):
        intra_losses = []
        inter_losses = []

        n_classes = len(
            target_features
        )

        for i in range(n_classes):

            intra_losses.append(
                SFDADE._mmd(
                    surrogate_features[i],
                    target_features[i],
                    gamma,
                )
            )

            for j in range(n_classes):

                if i == j:
                    continue

                inter_losses.append(
                    SFDADE._mmd(
                        surrogate_features[i],
                        target_features[j],
                        gamma,
                    )
                )

        intra = torch.stack(
            intra_losses
        ).mean()

        inter = torch.stack(
            inter_losses
        ).mean()

        return (
            intra
            - inter
        )

    @staticmethod
    def _mmd(
        X,
        Y,
        gamma=None,
    ):
        if gamma is None:
            gamma = (
                1.0
                / X.shape[1]
            )

        K_xx = SFDADE._rbf_kernel(
            X,
            X,
            gamma,
        )

        K_yy = SFDADE._rbf_kernel(
            Y,
            Y,
            gamma,
        )

        K_xy = SFDADE._rbf_kernel(
            X,
            Y,
            gamma,
        )

        return (
            K_xx.mean()
            + K_yy.mean()
            - 2.0
            * K_xy.mean()
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
            -gamma
            * distances
        )

    # ========================================================
    # Target data
    # ========================================================

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
                "SFDA-DE requires unlabeled "
                "target adaptation data."
            )

        return np.concatenate(
            X_parts
        )