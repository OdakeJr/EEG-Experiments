# ml/learning/source_free/base.py

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.learning.base import BaseLearningAlgorithm


class BaseSourceFree(BaseLearningAlgorithm):

    def __init__(self):
        self.model = None
        self.classes_ = None
        self.device = "cpu"

    def _pretrain_source_model(
        self,
        model,
        source,
        **training_params,
    ):
        X_source, y_source = self._get_source_data(
            source
        )

        source_epochs = training_params.get(
            "source_epochs",
            100,
        )
        batch_size = training_params.get(
            "batch_size",
            64,
        )
        source_learning_rate = training_params.get(
            "source_learning_rate",
            1e-3,
        )
        weight_decay = training_params.get(
            "weight_decay",
            0.0,
        )
        self.device = training_params.get(
            "device",
            "cpu",
        )
        seed = training_params.get(
            "seed",
            42,
        )

        self._set_seed(seed)

        model.apply(
            self._reset_parameters
        )

        model = model.to(
            self.device
        )

        self.classes_ = np.unique(
            y_source
        )

        y_encoded = self._encode_labels(
            y_source
        )

        dataset = TensorDataset(
            torch.as_tensor(
                X_source,
                dtype=torch.float32,
            ),
            torch.as_tensor(
                y_encoded,
                dtype=torch.long,
            ),
        )

        generator = torch.Generator()
        generator.manual_seed(seed)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=source_learning_rate,
            weight_decay=weight_decay,
        )

        criterion = nn.CrossEntropyLoss()

        model.train()

        for _ in range(source_epochs):

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

        self.model = model

        return model

    def _encode_labels(self, y):
        if self.classes_ is None:
            raise RuntimeError(
                "Source classes have not been defined."
            )

        class_to_index = {
            label: index
            for index, label in enumerate(self.classes_)
        }

        unknown = (
            set(np.unique(y))
            - set(self.classes_)
        )

        if unknown:
            raise ValueError(
                f"Unknown classes: {sorted(unknown)}"
            )

        return np.array([
            class_to_index[label]
            for label in y
        ])

    def predict(
        self,
        X,
        domains=None,
        super_domains=None,
    ):
        probabilities = self.predict_proba(
            X,
            domains,
            super_domains,
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

        return probabilities.cpu().numpy()

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
    def _get_source_data(source):
        if source is None:
            raise ValueError(
                "Source-free learning requires source data "
                "for source-model pretraining."
            )

        mask = (
            source.partitions
            == "train"
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
    def _set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

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
        if self.model is None:
            raise RuntimeError(
                "Learning algorithm has not been fitted."
            )