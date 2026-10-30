
# ml/learning/domain_generalization/residual_moments.py

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, TensorDataset

from ml.learning.base import BaseLearningAlgorithm


class ResidualMoments(BaseLearningAlgorithm):
    def __init__(self):
        self.model = None
        self.classes_ = None
        self.device = "cpu"
        self.history_ = []
        self.prediction_batch_size = 128

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
        return psi.T @ residuals / len(labels)

    def fit(self, model, source=None, target_super_domain=None,
            target_elementary_domain=None, **params):
        if source is None:
            raise ValueError("ResidualMoments requires source data.")
        if not all(callable(getattr(model, name, None)) for name in
                   ("extract_feature_map", "classify_feature_map")):
            raise ValueError("Model must expose extract_feature_map() and classify_feature_map().")

        mask = np.asarray(source.partitions) == "train"
        if not mask.any():
            raise ValueError("No source training samples found.")

        risk_weighting = params.get("risk_weighting", "pooled")
        constraint_scope = params.get("constraint_scope", "domain")
        if risk_weighting not in ("pooled", "balanced") or constraint_scope not in ("pooled", "domain"):
            raise ValueError("Invalid risk_weighting or constraint_scope.")

        if "predicates" in params:
            predicates = params["predicates"]
        else:
            kind = params.get("moment_type", "linear")
            if kind not in ("linear", "quadratic"):
                raise ValueError(f"Unknown moment_type: {kind}")
            predicates = ["constant", "linear"] + (["quadratic"] if kind == "quadratic" else [])

        if (not isinstance(predicates, (list, tuple)) or not predicates
                or len(set(predicates)) != len(predicates)
                or set(predicates) - {"constant", "linear", "quadratic"}):
            raise ValueError("predicates must be a nonempty list of unique supported names.")

        epochs = int(params.get("epochs", 100))
        batch_size = int(params.get("batch_size", 64))
        lam = params.get("constraint_lambda", params.get("moment_lambda", 1.0))
        normalize = params.get("normalize_features", True)
        self.device = params.get("device", "cpu")
        self.prediction_batch_size = int(params.get("prediction_batch_size", 128))
        seed = params.get("seed", 42)

        if lam < 0 or batch_size < 1 or self.prediction_batch_size < 1:
            raise ValueError("Invalid regularization strength or batch size.")

        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.classes_, encoded_y = np.unique(source.y[mask], return_inverse=True)
        domains = np.asarray(source.elementary_domains)[mask]
        X = torch.as_tensor(source.X[mask], dtype=torch.float32)
        y = torch.as_tensor(encoded_y, dtype=torch.long)

        indices = np.arange(len(y))
        val_fraction = params.get("validation_fraction", 0.0)
        train_idx, val_idx = train_test_split(
            indices, test_size=val_fraction, stratify=encoded_y, random_state=seed
        ) if val_fraction > 0 else (indices, None)

        domain_indices = [
            train_idx[domains[train_idx] == e]
            for e in np.unique(domains[train_idx])
        ]
        counts = np.array([len(idx) for idx in domain_indices])
        n_domains = min(int(params.get("domains_per_batch", 4)), len(domain_indices))
        per_domain = batch_size // n_domains

        if n_domains < 1 or per_domain < 2:
            raise ValueError("At least two samples per selected domain are required.")

        val_loader = None if val_idx is None else DataLoader(
            TensorDataset(X[val_idx], y[val_idx]), batch_size=batch_size
        )

        model = model.to(self.device)
        optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
        optimizer_name = params.get("optimizer", "adam")
        if optimizer_name not in optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        optimizer = optimizers[optimizer_name](
            model.parameters(),
            lr=params.get("learning_rate", 1e-3),
            weight_decay=params.get("weight_decay", 0.0)
        )

        best_loss, best_state, stale = float("inf"), None, 0
        patience = params.get("patience", 20)
        min_delta = params.get("min_delta", 0.0)
        n_steps = int(np.ceil(len(train_idx) / (per_domain * n_domains)))
        self.history_ = []

        for _ in range(epochs):
            model.train()
            total_risk = total_penalty = 0.0

            for _ in range(n_steps):
                selected = rng.choice(len(domain_indices), n_domains, replace=False)
                batch_idx = np.concatenate([
                    rng.choice(domain_indices[d], size=per_domain,
                               replace=len(domain_indices[d]) < per_domain)
                    for d in selected
                ])
                xb, yb = X[batch_idx].to(self.device), y[batch_idx].to(self.device)

                optimizer.zero_grad()
                feature_map = model.extract_feature_map(xb)
                logits = model.classify_feature_map(feature_map)

                risks, moments = [], []
                for j in range(n_domains):
                    s = slice(j * per_domain, (j + 1) * per_domain)
                    risks.append(F.cross_entropy(logits[s], yb[s]))
                    moments.append(self._statistical_moments(
                        feature_map[s], logits[s], yb[s], predicates, normalize
                    ))

                weights = torch.as_tensor(
                    counts[selected] / counts.sum() * len(counts) / n_domains,
                    dtype=logits.dtype, device=self.device
                )
                risk = (torch.stack(risks) * (
                    weights if risk_weighting == "pooled"
                    else torch.ones_like(weights) / n_domains
                )).sum()

                moments = torch.stack(moments)
                penalty = (
                    (moments * weights[:, None, None]).sum(0).square().mean()
                    if constraint_scope == "pooled"
                    else moments.square().mean()
                )

                (risk + lam * penalty).backward()
                optimizer.step()
                total_risk += risk.detach().item()
                total_penalty += penalty.detach().item()

            self.history_.append((total_risk / n_steps, total_penalty / n_steps))

            if val_loader is None:
                continue

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    val_loss += F.cross_entropy(model(xb), yb, reduction="sum").item()

            val_loss /= len(val_idx)
            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model
        return self

    def predict_proba(self, X, domains=None, super_domains=None):
        if self.model is None:
            raise RuntimeError("Learning algorithm has not been fitted.")

        self.model.eval()
        outputs = []
        with torch.no_grad():
            for i in range(0, len(X), self.prediction_batch_size):
                xb = torch.as_tensor(
                    X[i:i + self.prediction_batch_size],
                    dtype=torch.float32, device=self.device
                )
                outputs.append(self.model(xb).softmax(dim=1).cpu().numpy())
        return np.concatenate(outputs)

    def predict(self, X, domains=None, super_domains=None):
        return self.classes_[self.predict_proba(X, domains, super_domains).argmax(axis=1)]

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        for module in self.model.modules():
            if parametrize.is_parametrized(module):
                for name in list(module.parametrizations):
                    parametrize.remove_parametrizations(module, name, leave_parametrized=True)
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            return pickle.load(file)
