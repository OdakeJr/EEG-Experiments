import pickle
from pathlib import Path

import numpy as np

from ml.learning.base import BaseLearningAlgorithm


class SklearnERM(BaseLearningAlgorithm):

    def __init__(self):
        self.model = None

    def fit(
        self,
        model,
        source=None,
        target_super_domain=None,
        target_elementary_domain=None,
        **training_params,
    ):
        X, y = self._get_training_data(
            source,
            target_super_domain,
            target_elementary_domain,
        )

        seed = training_params.get(
            "seed",
            42,
        )

        fit_params = training_params.get(
            "fit_params",
            {},
        )

        if (
            hasattr(model, "get_params")
            and "random_state" in model.get_params()
        ):
            model.set_params(
                random_state=seed
            )

        model.fit(
            X,
            y,
            **fit_params,
        )

        self.model = model

        return self

    def predict(
        self,
        X,
        domains=None,
        super_domains=None,
    ):
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(
        self,
        X,
        domains=None,
        super_domains=None,
    ):
        self._check_fitted()

        if not hasattr(
            self.model,
            "predict_proba",
        ):
            raise NotImplementedError(
                "This model does not support predict_proba()."
            )

        return self.model.predict_proba(X)

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
    def _get_training_data(
        source,
        target_super_domain,
        target_elementary_domain,
    ):
        X_parts = []
        y_parts = []

        for group in [
            source,
            target_super_domain,
            target_elementary_domain,
        ]:
            if group is None:
                continue

            mask = group.partitions == "train"

            if np.any(mask):
                X_parts.append(
                    group.X[mask]
                )
                y_parts.append(
                    group.y[mask]
                )

        if not X_parts:
            raise ValueError(
                "No training samples found."
            )

        return (
            np.concatenate(X_parts),
            np.concatenate(y_parts),
        )

    def _check_fitted(self):
        if self.model is None:
            raise RuntimeError(
                "Learning algorithm has not been fitted."
            )