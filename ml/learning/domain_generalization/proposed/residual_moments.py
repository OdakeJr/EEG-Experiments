
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ml.learning.base import BaseLearningAlgorithm


class ResidualMoments(BaseLearningAlgorithm):
    def __init__(self):
        self.model = None
        self.classes_ = None
        self.device = "cpu"
        self.history_ = []

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
    def _statistical_moments(cls, features, logits, labels, predicates, normalize):
        psi = cls._compute_predicates(features, predicates, normalize)
        residuals = logits.softmax(dim=1) - F.one_hot(labels, logits.shape[1]).to(logits.dtype)
        return psi.T @ residuals / labels.numel()

    def fit(self, model, source=None, target_super_domain=None,
            target_elementary_domain=None, **training_params):
        if source is None:
            raise ValueError("ResidualMoments requires source data.")
        if not all(hasattr(model, name) for name in ("extract_feature_map", "classify_feature_map")):
            raise ValueError("Model must expose extract_feature_map() and classify_feature_map().")

        mask = np.asarray(source.partitions) == "train"
        if not mask.any():
            raise ValueError("No source training samples found.")

        risk_weighting = training_params.get("risk_weighting", "pooled")
        constraint_scope = training_params.get("constraint_scope", "domain")
        if risk_weighting not in ("pooled", "balanced"):
            raise ValueError(f"Unknown risk_weighting: {risk_weighting}")
        if constraint_scope not in ("pooled", "domain"):
            raise ValueError(f"Unknown constraint_scope: {constraint_scope}")

        if "predicates" in training_params:
            predicates = training_params["predicates"]
        else:
            old_type = training_params.get("moment_type", "linear")
            if old_type not in ("linear", "quadratic"):
                raise ValueError(f"Unknown moment_type: {old_type}")
            predicates = ["constant", "linear"] + (["quadratic"] if old_type == "quadratic" else [])
        if not isinstance(predicates, (list, tuple)) or not predicates or len(set(predicates)) != len(predicates):
            raise ValueError("predicates must be a nonempty list of unique names.")
        if set(predicates) - {"constant", "linear", "quadratic"}:
            raise ValueError("Unknown predicate names.")

        epochs = training_params.get("epochs", 100)
        lr = training_params.get("learning_rate", 1e-3)
        weight_decay = training_params.get("weight_decay", 0.0)
        lam = training_params.get("constraint_lambda", training_params.get("moment_lambda", 1.0))
        normalize = training_params.get("normalize_features", True)
        self.device = training_params.get("device", "cpu")
        seed = training_params.get("seed", 42)

        if lam < 0:
            raise ValueError("constraint_lambda must be nonnegative.")

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model.apply(self._reset_parameters)
        model = model.to(self.device)

        self.classes_, encoded_y = np.unique(source.y[mask], return_inverse=True)
        X = torch.as_tensor(source.X[mask], dtype=torch.float32)
        y = torch.as_tensor(encoded_y, dtype=torch.long)
        domains = np.asarray(source.elementary_domains)[mask]
        domain_indices = [
            torch.as_tensor(np.flatnonzero(domains == domain), dtype=torch.long)
            for domain in np.unique(domains)
        ]

        counts = torch.tensor([len(idx) for idx in domain_indices], dtype=torch.float32, device=self.device)
        moment_weights = counts / counts.sum()
        risk_weights = moment_weights if risk_weighting == "pooled" else torch.ones_like(counts) / len(counts)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.history_ = []

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            risks, moments = [], []

            for indices in domain_indices:
                x_e, y_e = X[indices].to(self.device), y[indices].to(self.device)
                feature_map = model.extract_feature_map(x_e)
                logits = model.classify_feature_map(feature_map)

                risks.append(F.cross_entropy(logits, y_e))
                moments.append(self._statistical_moments(
                    feature_map, logits, y_e, predicates, normalize
                ))

            risk = (torch.stack(risks) * risk_weights).sum()
            moments = torch.stack(moments)

            if constraint_scope == "pooled":
                penalty = (moments * moment_weights[:, None, None]).sum(dim=0).square().mean()
            else:
                penalty = moments.square().mean()

            (risk + lam * penalty).backward()
            optimizer.step()
            self.history_.append((risk.item(), penalty.item()))

        self.model = model
        return self

    def predict_proba(self, X, domains=None, super_domains=None):
        if self.model is None:
            raise RuntimeError("Learning algorithm has not been fitted.")
        self.model.eval()
        with torch.no_grad():
            X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            return self.model(X).softmax(dim=1).cpu().numpy()

    def predict(self, X, domains=None, super_domains=None):
        return self.classes_[self.predict_proba(X, domains, super_domains).argmax(axis=1)]

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def _reset_parameters(module):
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
