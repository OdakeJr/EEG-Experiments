
# ml/learning/domain_generalization/residual_moments.py

import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, Sampler

from ml.learning.base import BaseLearningAlgorithm
from ml.learning.lib.data import (
    class_mapping, split_indices, make_loader, DomainBalancedBatchSampler
)
from ml.learning.lib.neural import (
    make_optimizer, validation_loss, EarlyStopping,
    predict_proba as batched_predict_proba,
)
from models.scenario_data import DataGroup


class _OneDomainBatchSampler(Sampler):
    def __init__(self, domains, batch_size, seed):
        domains = np.asarray(domains)
        self.groups = [
            np.flatnonzero(domains == d)
            for d in np.unique(domains)
        ]
        self.n_domains = 1
        self.per_domain = batch_size
        self.n_batches = math.ceil(len(domains) / batch_size)
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for _ in range(self.n_batches):
            d = self.rng.choice(len(self.groups))
            group = self.groups[d]
            yield self.rng.choice(
                group, self.per_domain,
                replace=len(group) < self.per_domain
            ).tolist()

    def __len__(self):
        return self.n_batches


class ResidualMoments(BaseLearningAlgorithm):
    def __init__(self):
        self.model = None
        self.classes_ = None
        self.device = "cpu"
        self.history_ = []
        self.eval_transform = None
        self.prediction_batch_size = 128
        self.prediction_num_workers = 0

    # ----------------------------------------------------
    # Statistical moments
    # ----------------------------------------------------

    @staticmethod
    def _compute_predicates(features, predicates, normalize):
        z = features.flatten(1)
        if normalize:
            z = F.normalize(z, dim=1, eps=1e-8) * z.shape[1] ** 0.5

        parts = []
        for name in predicates:
            if name == "constant":
                parts.append(torch.ones_like(z[:, :1]))
            elif name == "linear":
                parts.append(z)
            elif name == "quadratic":
                parts.append(z.square())
            else:
                raise ValueError(f"Unknown predicate: {name}")

        return torch.cat(parts, dim=1)

    @classmethod
    def _statistical_moments(cls, features, logits, labels,
                             predicates, normalize):
        psi = cls._compute_predicates(features, predicates, normalize)
        residuals = (
            logits.softmax(dim=1)
            - F.one_hot(labels, logits.shape[1]).to(logits.dtype)
        )
        return psi.T @ residuals / len(labels)

    # ----------------------------------------------------
    # Data preparation
    # ----------------------------------------------------

    def _prepare_loaders(self, source, params):
        if source is None:
            raise ValueError("ResidualMoments requires source data.")

        mask = np.asarray(source.partitions) == "train"
        if not mask.any():
            raise ValueError("No source training samples found.")

        X, y = source.X[mask], source.y[mask]
        domains = source.elementary_domains[mask]
        self.classes_, mapping = class_mapping(y)

        _, val_idx = split_indices(
            y, params.get("validation_fraction", 0.0),
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
            **common, partition="train"
        )
        val_loader = (
            make_loader(**common, partition="validation")
            if val_idx is not None else None
        )
        return train_loader, val_loader

    # ----------------------------------------------------
    # Training
    # ----------------------------------------------------

    def fit(self, model, source=None, target_super_domain=None,
            target_elementary_domain=None, **params):

        if not all(callable(getattr(model, name, None)) for name in (
            "extract_feature_map", "classify_feature_map"
        )):
            raise ValueError(
                "Model must expose extract_feature_map() "
                "and classify_feature_map()."
            )

        risk_weighting = params.get("risk_weighting", "pooled")
        constraint_scope = params.get("constraint_scope", "domain")

        if risk_weighting not in ("pooled", "balanced"):
            raise ValueError("Invalid risk_weighting.")
        if constraint_scope not in ("pooled", "domain"):
            raise ValueError("Invalid constraint_scope.")

        if "predicates" in params:
            predicates = params["predicates"]
        else:
            kind = params.get("moment_type", "linear")
            if kind not in ("linear", "quadratic"):
                raise ValueError(f"Unknown moment_type: {kind}")
            predicates = ["constant", "linear"]
            if kind == "quadratic":
                predicates.append("quadratic")

        supported = {"constant", "linear", "quadratic"}
        if (
            not isinstance(predicates, (list, tuple))
            or not predicates
            or len(set(predicates)) != len(predicates)
            or set(predicates) - supported
        ):
            raise ValueError("Invalid predicates configuration.")

        seed = params.get("seed", 42)
        batch_size = int(params.get("batch_size", 64))
        lam = params.get(
            "constraint_lambda", params.get("moment_lambda", 1.0)
        )
        normalize = params.get("normalize_features", True)
        self.device = params.get("device", "cpu")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # ----------------------------------------------------
        # DataLoaders
        # ----------------------------------------------------

        train_loader = params.get("train_loader")
        val_loader = params.get("val_loader")
        image_mode = train_loader is not None

        if not image_mode:
            if getattr(model, "input_representation", None) == "image":
                raise ValueError("Image training requires train_loader.")

            train_loader, val_loader = self._prepare_loaders(
                source, params
            )
            self.eval_transform = None
        else:
            mapping = params["class_to_idx"]
            self.classes_ = np.asarray(sorted(mapping, key=mapping.get))
            self.eval_transform = params["eval_transform"]

        self.prediction_batch_size = int(params.get(
            "prediction_batch_size",
            batch_size if image_mode else 128
        ))
        self.prediction_num_workers = params.get("num_workers", 0)

        if lam < 0 or batch_size < 2 or self.prediction_batch_size < 1:
            raise ValueError("Invalid regularization strength or batch size.")

        dataset = train_loader.dataset
        domains = (
            np.asarray(dataset.domains)
            if hasattr(dataset, "domains")
            else dataset.tensors[2].numpy()
        )

        domain_names, counts = np.unique(domains, return_counts=True)
        domain_to_idx = {
            str(d): i for i, d in enumerate(domain_names)
        }

        requested = int(params.get("domains_per_batch", 4))
        if requested < 1:
            raise ValueError("domains_per_batch must be positive.")

        n_domains = min(requested, len(domain_names))
        per_domain = batch_size // n_domains

        if per_domain < 2:
            raise ValueError(
                "At least two samples per selected domain are required."
            )

        if n_domains == 1:
            sampler = _OneDomainBatchSampler(
                domains, batch_size, seed
            )
        else:
            sampler = DomainBalancedBatchSampler(
                domains, batch_size, requested, seed
            )

        train_loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=self.prediction_num_workers,
        )

        n_steps = len(train_loader)

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

        self.history_ = []

        # ----------------------------------------------------
        # Training loop
        # ----------------------------------------------------

        for _ in range(int(params.get("epochs", 100))):
            model.train()
            total_risk = 0.0
            total_penalty = 0.0

            for batch in train_loader:
                xb, yb = (v.to(self.device) for v in batch[:2])

                selected = np.asarray([
                    domain_to_idx[str(
                        d.item() if torch.is_tensor(d) else d
                    )]
                    for d in batch[2][::per_domain]
                ])

                optimizer.zero_grad()

                feature_map = model.extract_feature_map(xb)
                logits = model.classify_feature_map(feature_map)

                risks, moments = [], []

                for j in range(n_domains):
                    s = slice(j * per_domain, (j + 1) * per_domain)

                    risks.append(
                        F.cross_entropy(logits[s], yb[s])
                    )
                    moments.append(self._statistical_moments(
                        feature_map[s], logits[s], yb[s],
                        predicates, normalize
                    ))

                # Risk weighting
                weights = torch.as_tensor(
                    counts[selected] / counts.sum()
                    * len(counts) / n_domains,
                    dtype=logits.dtype,
                    device=self.device,
                )

                risk_weights = (
                    weights if risk_weighting == "pooled"
                    else torch.ones_like(weights) / n_domains
                )
                risk = (
                    torch.stack(risks) * risk_weights
                ).sum()

                # Moment constraints
                moments = torch.stack(moments)

                if constraint_scope == "pooled":
                    penalty = (
                        (moments * weights[:, None, None])
                        .sum(0).square().mean()
                    )
                else:
                    penalty = moments.square().mean()

                loss = risk + lam * penalty
                loss.backward()
                optimizer.step()

                total_risk += risk.detach().item()
                total_penalty += penalty.detach().item()

            self.history_.append((
                total_risk / n_steps,
                total_penalty / n_steps,
            ))

            val_loss = validation_loss(
                model, val_loader, self.device
            )
            if early_stop.step(val_loss, model):
                break

        self.model = early_stop.restore(model)
        return self

    # ----------------------------------------------------
    # Prediction
    # ----------------------------------------------------

    def predict_proba(self, X, domains=None, super_domains=None):
        if self.model is None:
            raise RuntimeError("Learning algorithm has not been fitted.")

        return batched_predict_proba(
            self.model, X,
            device=self.device,
            batch_size=getattr(self, "prediction_batch_size", 128),
            transform=getattr(self, "eval_transform", None),
            num_workers=getattr(self, "prediction_num_workers", 0),
        )

    def predict(self, X, domains=None, super_domains=None):
        return self.classes_[
            self.predict_proba(
                X, domains, super_domains
            ).argmax(axis=1)
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
