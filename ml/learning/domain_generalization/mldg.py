
# ml/learning/domain_generalization/mldg.py

import copy
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.func import functional_call
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, TensorDataset

from ml.learning.base import BaseLearningAlgorithm


class MLDG(BaseLearningAlgorithm):
    def __init__(self):
        self.model = None
        self.classes_ = None
        self.device = "cpu"

    def fit(self, model, source=None, target_super_domain=None,
            target_elementary_domain=None, **params):
        if source is None:
            raise ValueError("MLDG requires source data.")
        mask = source.partitions == "train"
        if not np.any(mask):
            raise ValueError("No source training samples found.")

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
        inner_lr = params.get("mldg_alpha", 1e-3)
        meta_weight = params.get("mldg_beta", 1.0)
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
            raise ValueError("MLDG requires at least 2 domains and 2 samples per domain.")

        per_domain = batch_size // n_domains
        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.long)

        val_loader = None if val_idx is None else DataLoader(
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
                chosen = rng.choice(
                    len(domain_indices), size=n_domains, replace=False
                )
                groups = [
                    rng.choice(
                        domain_indices[d], size=per_domain,
                        replace=len(domain_indices[d]) < per_domain
                    )
                    for d in chosen
                ]

                train_batch = np.concatenate(groups[:-1])
                test_batch = groups[-1]

                x_train = X[train_batch].to(self.device)
                y_train = y[train_batch].to(self.device)
                x_test = X[test_batch].to(self.device)
                y_test = y[test_batch].to(self.device)

                optimizer.zero_grad()

                # Meta-train: temporary adaptation on source domains.
                train_loss = F.cross_entropy(model(x_train), y_train)
                weights = dict(model.named_parameters())

                inner_grad = torch.autograd.grad(
                    train_loss, tuple(weights.values()), retain_graph=True
                )
                fast_weights = {
                    name: weight - inner_lr * grad
                    for (name, weight), grad in zip(weights.items(), inner_grad)
                }

                # Meta-test: evaluate adapted weights on another source domain.
                buffers = {
                    name: value.clone()
                    for name, value in model.named_buffers()
                }
                meta_logits = functional_call(
                    model, (fast_weights, buffers), (x_test,)
                )
                meta_loss = F.cross_entropy(meta_logits, y_test)

                # First-order MLDG update.
                loss = train_loss + meta_weight * meta_loss
                loss.backward()
                optimizer.step()

            if val_loader is None:
                continue

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    val_loss += F.cross_entropy(
                        model(xb), yb, reduction="sum"
                    ).item()

            val_loss /= len(val_idx)

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

    def predict_proba(self, X, domains=None, super_domains=None):
        if self.model is None:
            raise RuntimeError("MLDG has not been fitted.")

        self.model.eval()
        probabilities = []

        with torch.no_grad():
            for start in range(0, len(X), 256):
                xb = torch.as_tensor(
                    X[start:start + 256],
                    dtype=torch.float32,
                    device=self.device,
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
