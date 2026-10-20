
# ml/learning/domain_generalization/residual_moments.py

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
    def _moment_penalty(features, logits, labels, moment_type, normalize):
        features = features.flatten(1)
        if normalize:
            features = F.normalize(features, dim=1, eps=1e-8) * (features.shape[1] ** 0.5)

        tests = [torch.ones_like(features[:, :1]), features]
        if moment_type == "quadratic":
            tests.append(features.square())
        elif moment_type != "linear":
            raise ValueError(f"Unknown moment_type: {moment_type}")

        psi = torch.cat(tests, dim=1)
        residuals = logits.softmax(dim=1) - F.one_hot(labels, logits.shape[1]).to(logits.dtype)
        moments = psi.T @ residuals / labels.numel()
        return moments.square().mean()

    def fit(self, model, source=None, target_super_domain=None,
            target_elementary_domain=None, **training_params):
        if source is None:
            raise ValueError("ResidualMoments requires source data.")
        if not all(hasattr(model, method) for method in ("extract_feature_map", "classify_feature_map")):
            raise ValueError("Model must expose extract_feature_map() and classify_feature_map().")

        mask = np.asarray(source.partitions) == "train"
        if not mask.any():
            raise ValueError("No source training samples found.")

        epochs = training_params.get("epochs", 100)
        lr = training_params.get("learning_rate", 1e-3)
        weight_decay = training_params.get("weight_decay", 0.0)
        lam = training_params.get("moment_lambda", 1.0)
        moment_type = training_params.get("moment_type", "linear")
        normalize = training_params.get("normalize_features", True)
        self.device = training_params.get("device", "cpu")
        seed = training_params.get("seed", 42)

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model.apply(self._reset_parameters)
        model = model.to(self.device)

        self.classes_, y = np.unique(source.y[mask], return_inverse=True)
        X = torch.as_tensor(source.X[mask], dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y, dtype=torch.long, device=self.device)
        domains = np.asarray(source.elementary_domains)[mask]
        domain_indices = [
            torch.as_tensor(np.flatnonzero(domains == domain), device=self.device)
            for domain in np.unique(domains)
        ]

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.history_ = []

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            risks, penalties = [], []

            for indices in domain_indices:
                feature_map = model.extract_feature_map(X[indices])
                logits = model.classify_feature_map(feature_map)

                risks.append(F.cross_entropy(logits, y[indices]))
                penalties.append(self._moment_penalty(
                    feature_map, logits, y[indices], moment_type, normalize
                ))

            risk = torch.stack(risks).mean()
            penalty = torch.stack(penalties).mean()
            (risk + lam * penalty).backward()
            optimizer.step()

            self.history_.append((risk.detach().item(), penalty.detach().item()))

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
