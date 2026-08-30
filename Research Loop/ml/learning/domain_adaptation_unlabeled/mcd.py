# ml/learning/domain_adaptation_unlabeled/mcd.py

import copy
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ml.learning.base import BaseLearningAlgorithm


class MCD(BaseLearningAlgorithm):

    def __init__(self):
        self.model = None
        self.classifier2 = None
        self.classes_ = None
        self.device = "cpu"

    def fit(
        self,
        model,
        source=None,
        target_super_domain=None,
        target_elementary_domain=None,
        **training_params,
    ):
        X_source, y_source = self._get_source_data(
            source
        )

        X_target = self._get_target_data(
            target_super_domain,
            target_elementary_domain,
        )

        epochs = training_params.get(
            "epochs",
            100,
        )
        learning_rate = training_params.get(
            "learning_rate",
            1e-3,
        )
        weight_decay = training_params.get(
            "weight_decay",
            0.0,
        )
        mcd_lambda = training_params.get(
            "mcd_lambda",
            1.0,
        )
        generator_steps = training_params.get(
            "generator_steps",
            4,
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
            torch.cuda.manual_seed_all(seed)

        if (
            not hasattr(model, "extract_features")
            or not hasattr(model, "features")
            or not hasattr(model, "classifier")
        ):
            raise ValueError(
                "MCD requires a model with "
                "features, classifier, and extract_features()."
            )

        model.apply(
            self._reset_parameters
        )

        model = model.to(
            self.device
        )

        classifier2 = copy.deepcopy(
            model.classifier
        )

        classifier2.apply(
            self._reset_parameters
        )

        classifier2 = classifier2.to(
            self.device
        )

        self.classes_ = np.unique(
            y_source
        )

        class_to_index = {
            label: index
            for index, label in enumerate(self.classes_)
        }

        y_encoded = np.array([
            class_to_index[label]
            for label in y_source
        ])

        X_source = torch.as_tensor(
            X_source,
            dtype=torch.float32,
            device=self.device,
        )

        y_source = torch.as_tensor(
            y_encoded,
            dtype=torch.long,
            device=self.device,
        )

        X_target = torch.as_tensor(
            X_target,
            dtype=torch.float32,
            device=self.device,
        )

        optimizer_generator = torch.optim.Adam(
            model.features.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        optimizer_classifiers = torch.optim.Adam(
            list(model.classifier.parameters())
            + list(classifier2.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        criterion = nn.CrossEntropyLoss()

        model.train()
        classifier2.train()

        for _ in range(epochs):

            # ------------------------------------------------
            # Step A: source supervision
            # ------------------------------------------------

            optimizer_generator.zero_grad()
            optimizer_classifiers.zero_grad()

            source_features = (
                model.extract_features(
                    X_source
                )
            )

            source_logits1 = (
                model.classifier(
                    source_features
                )
            )

            source_logits2 = (
                classifier2(
                    source_features
                )
            )

            source_loss = (
                criterion(
                    source_logits1,
                    y_source,
                )
                + criterion(
                    source_logits2,
                    y_source,
                )
            )

            source_loss.backward()

            optimizer_generator.step()
            optimizer_classifiers.step()

            # ------------------------------------------------
            # Step B: maximize target discrepancy
            # ------------------------------------------------

            optimizer_classifiers.zero_grad()

            source_features = (
                model.extract_features(
                    X_source
                )
                .detach()
            )

            target_features = (
                model.extract_features(
                    X_target
                )
                .detach()
            )

            source_logits1 = (
                model.classifier(
                    source_features
                )
            )

            source_logits2 = (
                classifier2(
                    source_features
                )
            )

            target_prob1 = torch.softmax(
                model.classifier(
                    target_features
                ),
                dim=1,
            )

            target_prob2 = torch.softmax(
                classifier2(
                    target_features
                ),
                dim=1,
            )

            discrepancy = (
                self._discrepancy(
                    target_prob1,
                    target_prob2,
                )
            )

            classifier_loss = (
                criterion(
                    source_logits1,
                    y_source,
                )
                + criterion(
                    source_logits2,
                    y_source,
                )
                - mcd_lambda * discrepancy
            )

            classifier_loss.backward()
            optimizer_classifiers.step()

            # ------------------------------------------------
            # Step C: minimize target discrepancy
            # ------------------------------------------------

            self._set_requires_grad(
                model.classifier,
                False,
            )

            self._set_requires_grad(
                classifier2,
                False,
            )

            for _ in range(
                generator_steps
            ):

                optimizer_generator.zero_grad()

                target_features = (
                    model.extract_features(
                        X_target
                    )
                )

                target_prob1 = torch.softmax(
                    model.classifier(
                        target_features
                    ),
                    dim=1,
                )

                target_prob2 = torch.softmax(
                    classifier2(
                        target_features
                    ),
                    dim=1,
                )

                discrepancy = (
                    self._discrepancy(
                        target_prob1,
                        target_prob2,
                    )
                )

                discrepancy.backward()
                optimizer_generator.step()

            self._set_requires_grad(
                model.classifier,
                True,
            )

            self._set_requires_grad(
                classifier2,
                True,
            )

        self.model = model
        self.classifier2 = classifier2

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

        return self.classes_[indices]

    def predict_proba(
        self,
        X,
        domains=None,
        super_domains=None,
    ):
        self._check_fitted()

        self.model.eval()
        self.classifier2.eval()

        X = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():

            features = (
                self.model.extract_features(
                    X
                )
            )

            probabilities1 = torch.softmax(
                self.model.classifier(
                    features
                ),
                dim=1,
            )

            probabilities2 = torch.softmax(
                self.classifier2(
                    features
                ),
                dim=1,
            )

            probabilities = (
                probabilities1
                + probabilities2
            ) / 2.0

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

        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def _discrepancy(
        probabilities1,
        probabilities2,
    ):
        return torch.mean(
            torch.abs(
                probabilities1
                - probabilities2
            )
        )

    @staticmethod
    def _get_source_data(source):
        if source is None:
            raise ValueError(
                "MCD requires source data."
            )

        mask = (
            source.partitions == "train"
        )

        if not np.any(mask):
            raise ValueError(
                "No source training samples found."
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
                "MCD requires unlabeled target adaptation data."
            )

        return np.concatenate(
            X_parts
        )

    @staticmethod
    def _set_requires_grad(
        module,
        value,
    ):
        for parameter in module.parameters():
            parameter.requires_grad = value

    @staticmethod
    def _reset_parameters(module):
        if hasattr(
            module,
            "reset_parameters",
        ):
            module.reset_parameters()

    def _check_fitted(self):
        if (
            self.model is None
            or self.classifier2 is None
        ):
            raise RuntimeError(
                "Learning algorithm has not been fitted."
            )