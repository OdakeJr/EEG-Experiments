
# ml/learning/domain_generalization/coral.py

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader

from ml.learning.base import BaseLearningAlgorithm
from ml.learning.lib.data import (
    class_mapping, split_indices, make_loader, DomainBalancedBatchSampler
)
from ml.learning.lib.neural import (
    make_optimizer, validation_loss, EarlyStopping,
    predict_proba as batched_predict_proba,
)
from models.scenario_data import DataGroup


class CORAL(BaseLearningAlgorithm):
    def __init__(self):
        self.model = None
        self.classes_ = None
        self.device = "cpu"
        self.eval_transform = None
        self.prediction_batch_size = 256
        self.prediction_num_workers = 0

    def _prepare_loaders(self, source, params):
        if source is None:
            raise ValueError("CORAL requires source data.")

        mask = source.partitions == "train"
        if not mask.any():
            raise ValueError("No source training samples found.")

        X, y = source.X[mask], source.y[mask]
        domains = source.elementary_domains[mask]
        self.classes_, mapping = class_mapping(y)

        _, val_idx = split_indices(
            y, params.get("validation_fraction", 0.2),
            params.get("seed", 42)
        )
        partitions = np.full(len(y), "train", dtype=object)
        if val_idx is not None:
            partitions[val_idx] = "validation"

        group = DataGroup(X, y, domains, partitions)
        common = {
            "group": group,
            "class_to_idx": mapping,
            "batch_size": params.get("batch_size", 64),
            "seed": params.get("seed", 42),
        }

        train_loader = make_loader(
            **common, partition="train", domain_balanced=True,
            domains_per_batch=params.get("domains_per_batch", 4)
        )
        val_loader = (
            make_loader(**common, partition="validation")
            if val_idx is not None else None
        )
        return train_loader, val_loader

    def fit(self, model, source=None, target_super_domain=None,
            target_elementary_domain=None, **params):

        seed = params.get("seed", 42)
        batch_size = params.get("batch_size", 64)
        self.device = params.get("device", "cpu")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if not callable(getattr(model, "extract_features", None)):
            raise ValueError("CORAL requires model.extract_features().")

        train_loader = params.get("train_loader")
        val_loader = params.get("val_loader")

        if train_loader is None:
            if getattr(model, "input_representation", None) == "image":
                raise ValueError("Image training requires train_loader.")

            train_loader, val_loader = self._prepare_loaders(source, params)
            self.eval_transform = None
        else:
            mapping = params["class_to_idx"]
            self.classes_ = np.asarray(sorted(mapping, key=mapping.get))
            self.eval_transform = params["eval_transform"]
            self.prediction_batch_size = params.get(
                "prediction_batch_size", batch_size
            )
            self.prediction_num_workers = params.get("num_workers", 0)

            sampler = DomainBalancedBatchSampler(
                train_loader.dataset.domains, batch_size,
                params.get("domains_per_batch", 4), seed
            )
            train_loader = DataLoader(
                train_loader.dataset, batch_sampler=sampler,
                num_workers=params.get("num_workers", 0)
            )

        n_domains = train_loader.batch_sampler.n_domains
        per_domain = train_loader.batch_sampler.per_domain

        # ----------------------------------------------------
        # Model and optimizer
        # ----------------------------------------------------

        model = model.to(self.device)
        optimizer = make_optimizer(
            model,
            name=params.get("optimizer", "adam"),
            learning_rate=params.get("learning_rate", 1e-3),
            weight_decay=params.get("weight_decay", 0.0),
        )
        early_stop = EarlyStopping(
            patience=params.get("patience", 20),
            min_delta=params.get("min_delta", 0.0),
        )
        coral_lambda = params.get("coral_lambda", 1.0)

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        for _ in range(params.get("epochs", 100)):
            model.train()

            for batch in train_loader:
                xb, yb = (v.to(self.device) for v in batch[:2])
                optimizer.zero_grad()

                if all(callable(getattr(model, name, None)) for name in (
                    "extract_feature_map", "classify_feature_map"
                )):
                    feature_map = model.extract_feature_map(xb)
                    logits = model.classify_feature_map(feature_map)
                    features = feature_map.flatten(1)
                else:
                    logits = model(xb)
                    features = model.extract_features(xb)

                covariances = torch.stack([
                    self._covariance(group)
                    for group in features.split(per_domain)
                ])

                d = features.shape[1]
                penalty = (
                    torch.pdist(covariances.flatten(1)).square().mean()
                    / (4 * d * d)
                )

                loss = F.cross_entropy(logits, yb) + coral_lambda * penalty
                loss.backward()
                optimizer.step()

            val_loss = validation_loss(model, val_loader, self.device)
            if early_stop.step(val_loss, model):
                break

        self.model = early_stop.restore(model)
        return self

    @staticmethod
    def _covariance(X):
        X = X - X.mean(dim=0, keepdim=True)
        return X.T @ X / (len(X) - 1)

    # ----------------------------------------------------
    # Prediction
    # ----------------------------------------------------

    def predict_proba(self, X, domains=None, super_domains=None):
        if self.model is None:
            raise RuntimeError("CORAL has not been fitted.")

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

    # ----------------------------------------------------
    # Persistence
    # ----------------------------------------------------

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
