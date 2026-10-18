
# ml/learning/domain_generalization/mmd.py

import copy
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, TensorDataset

from ml.learning.base import BaseLearningAlgorithm


class MMD(BaseLearningAlgorithm):
    def __init__(self):
        self.model = None
        self.classes_ = None
        self.device = "cpu"

    def fit(self, model, source=None, target_super_domain=None,
            target_elementary_domain=None, **params):
        if source is None:
            raise ValueError("MMD requires source data.")
        mask = source.partitions == "train"
        if not np.any(mask):
            raise ValueError("No source training samples found.")
        if not callable(getattr(model, "extract_features", None)):
            raise ValueError("MMD requires model.extract_features().")

        X, y, domains = source.X[mask], source.y[mask], source.elementary_domains[mask]
        self.classes_, y = np.unique(y, return_inverse=True)

        seed = params.get("seed", 42)
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        epochs = params.get("epochs", 100)
        batch_size = params.get("batch_size", 64)
        val_fraction = params.get("validation_fraction", 0.2)
        patience = params.get("patience", 20)
        min_delta = params.get("min_delta", 0.0)
        mmd_lambda = params.get("mmd_lambda", 1.0)
        gamma = params.get("gamma")
        self.device = params.get("device", "cpu")

        indices = np.arange(len(y))
        if val_fraction > 0:
            train_idx, val_idx = train_test_split(
                indices, test_size=val_fraction, stratify=y, random_state=seed
            )
        else:
            train_idx, val_idx = indices, None

        domain_indices = [
            train_idx[domains[train_idx] == domain]
            for domain in np.unique(domains[train_idx])
        ]
        n_domains = min(int(params.get("domains_per_batch", 4)), len(domain_indices))
        if n_domains < 2 or batch_size // n_domains < 2:
            raise ValueError("MMD requires at least 2 domains and 2 samples per domain.")
        per_domain = batch_size // n_domains

        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.long)

        val_loader = None
        if val_idx is not None:
            val_loader = DataLoader(
                TensorDataset(X[val_idx], y[val_idx]), batch_size=batch_size
            )

        model = model.to(self.device)
        optimizer_name = params.get("optimizer", "adam")
        optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
        if optimizer_name not in optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        optimizer = optimizers[optimizer_name](
            model.parameters(),
            lr=params.get("learning_rate", 1e-3),
            weight_decay=params.get("weight_decay", 0.0),
        )

        best_loss, best_state, stale_epochs = float("inf"), None, 0
        n_steps = int(np.ceil(len(train_idx) / batch_size))

        for _ in range(epochs):
            model.train()

            for _ in range(n_steps):
                selected = rng.choice(
                    len(domain_indices), size=n_domains, replace=False
                )
                batch_indices = np.concatenate([
                    rng.choice(
                        domain_indices[d], size=per_domain,
                        replace=len(domain_indices[d]) < per_domain
                    )
                    for d in selected
                ])

                xb = X[batch_indices].to(self.device)
                yb = y[batch_indices].to(self.device)

                optimizer.zero_grad()
                logits = model(xb)
                features = model.extract_features(xb)

                groups = features.split(per_domain)
                penalties = [
                    self._mmd(groups[i], groups[j], gamma)
                    for i in range(n_domains)
                    for j in range(i + 1, n_domains)
                ]

                loss = (
                    F.cross_entropy(logits, yb)
                    + mmd_lambda * torch.stack(penalties).mean()
                )
                loss.backward()
                optimizer.step()

            if val_loader is None:
                continue

            model.eval()
            total_loss = 0.0

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    total_loss += F.cross_entropy(
                        model(xb), yb, reduction="sum"
                    ).item()

            val_loss = total_loss / len(val_idx)

            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        self.model = model
        return self

    @staticmethod
    def _mmd(X, Y, gamma=None):
        if gamma is None:
            gamma = 1.0 / X.shape[1]

        K_xx = MMD._rbf_kernel(X, X, gamma)
        K_yy = MMD._rbf_kernel(Y, Y, gamma)
        K_xy = MMD._rbf_kernel(X, Y, gamma)

        return K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()

    @staticmethod
    def _rbf_kernel(X, Y, gamma):
        return torch.exp(-gamma * torch.cdist(X, Y).square())

    def predict_proba(self, X, domains=None, super_domains=None):
        if self.model is None:
            raise RuntimeError("MMD has not been fitted.")

        self.model.eval()
        probabilities = []

        with torch.no_grad():
            for start in range(0, len(X), 256):
                xb = torch.as_tensor(
                    X[start:start + 256], dtype=torch.float32,
                    device=self.device
                )
                probabilities.append(
                    F.softmax(self.model(xb), dim=1).cpu().numpy()
                )

        return np.concatenate(probabilities)

    def predict(self, X, domains=None, super_domains=None):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        for module in self.model.modules():
            if parametrize.is_parametrized(module):
                for name in list(module.parametrizations.keys()):
                    parametrize.remove_parametrizations(
                        module, name, leave_parametrized=True
                    )

        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            return pickle.load(file)
