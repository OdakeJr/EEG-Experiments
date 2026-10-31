
# ml/learning/classical/neural_erm.py

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize

from ml.learning.base import BaseLearningAlgorithm
from ml.learning.lib.data import class_mapping, split_indices, make_loader
from ml.learning.lib.neural import (
    make_optimizer, validation_loss, EarlyStopping,
    predict_proba as batched_predict_proba,
)
from models.scenario_data import DataGroup


class NeuralERM(BaseLearningAlgorithm):

    def __init__(self):
        self.model = None
        self.classes_ = None
        self.device = "cpu"
        self.eval_transform = None
        self.prediction_batch_size = 256
        self.prediction_num_workers = 0

    def _prepare_loaders(self, source, target_super_domain,
                         target_elementary_domain, params):
        X, y = self._get_training_data(
            source, target_super_domain, target_elementary_domain
        )
        self.classes_, mapping = class_mapping(y)

        train_idx, val_idx = split_indices(
            y, params.get("validation_fraction", 0.2),
            params.get("seed", 42)
        )

        partitions = np.full(len(y), "train", dtype=object)
        if val_idx is not None:
            partitions[val_idx] = "validation"

        group = DataGroup(
            X=X, y=y,
            elementary_domains=np.full(len(y), "source"),
            partitions=partitions,
        )

        common = {
            "group": group,
            "class_to_idx": mapping,
            "batch_size": params.get("batch_size", 64),
            "seed": params.get("seed", 42),
        }

        train_loader = make_loader(
            **common, partition="train", shuffle=True
        )
        val_loader = (
            make_loader(**common, partition="validation")
            if val_idx is not None else None
        )
        return train_loader, val_loader

    def fit(self, model, source=None, target_super_domain=None,
            target_elementary_domain=None, **params):

        seed = params.get("seed", 42)
        self.device = params.get("device", "cpu")

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # ----------------------------------------------------
        # Data loading
        # ----------------------------------------------------

        train_loader = params.get("train_loader")
        val_loader = params.get("val_loader")

        if train_loader is not None:
            mapping = params["class_to_idx"]
            self.classes_ = np.asarray(sorted(mapping, key=mapping.get))
            self.eval_transform = params.get("eval_transform")
            self.prediction_batch_size = params.get(
                "prediction_batch_size", params.get("batch_size", 64)
            )
            self.prediction_num_workers = params.get("num_workers", 0)

            if self.eval_transform is None:
                raise ValueError("Image training requires eval_transform.")
        else:
            if getattr(model, "input_representation", None) == "image":
                raise ValueError("Image training requires train_loader.")

            train_loader, val_loader = self._prepare_loaders(
                source, target_super_domain, target_elementary_domain, params
            )
            self.eval_transform = None
            self.prediction_batch_size = 256
            self.prediction_num_workers = 0

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        model = model.to(self.device)
        optimizer = make_optimizer(
            model,
            name=params.get("optimizer", "adamw"),
            learning_rate=params.get("learning_rate", 1e-3),
            weight_decay=params.get("weight_decay", 1e-4),
        )
        early_stop = EarlyStopping(
            patience=params.get("patience", 20),
            min_delta=params.get("min_delta", 0.0),
        )

        for _ in range(params.get("epochs", 100)):
            model.train()

            for batch in train_loader:
                xb, yb = (v.to(self.device) for v in batch[:2])

                optimizer.zero_grad()
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                optimizer.step()

            val_loss = validation_loss(model, val_loader, self.device)
            if early_stop.step(val_loss, model):
                break

        self.model = early_stop.restore(model)
        return self

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    def predict_proba(self, X, domains=None, super_domains=None):
        self._check_fitted()
        return batched_predict_proba(
            self.model, X, device=self.device,
            batch_size=getattr(self, "prediction_batch_size", 256),
            transform=getattr(self, "eval_transform", None),
            num_workers=getattr(self, "prediction_num_workers", 0),
        )

    def predict(self, X, domains=None, super_domains=None):
        return self.classes_[
            self.predict_proba(X, domains, super_domains).argmax(axis=1)
        ]

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        for module in self.model.modules():
            if parametrize.is_parametrized(module):
                for name in list(module.parametrizations):
                    parametrize.remove_parametrizations(
                        module, name, leave_parametrized=True
                    )

        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def _get_training_data(source, target_super_domain, target_elementary_domain):
        X_parts, y_parts = [], []

        for group in (source, target_super_domain, target_elementary_domain):
            if group is None:
                continue

            mask = group.partitions == "train"
            if mask.any():
                X_parts.append(group.X[mask])
                y_parts.append(group.y[mask])

        if not X_parts:
            raise ValueError("No training samples found.")

        return np.concatenate(X_parts), np.concatenate(y_parts)

    def _check_fitted(self):
        if self.model is None:
            raise RuntimeError("Learning algorithm has not been fitted.")
