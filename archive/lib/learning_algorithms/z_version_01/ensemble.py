import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath("../../../"))
PROJECT_ROOT = "../../../"

import archieve.lib.learning_algorithms.z_version_01.benchmark_architectures as my_models


# ==========================================================
# Domain Agreement Model
# ==========================================================
class DomainAgreementModel:
    def __init__(
        self,
        input_dim,
        n_classes,
        model_class,
        model_kwargs=None,
        lr=1e-3,
        weight_decay=0.0,
        batch_size=64,
        device=None,
        random_state=42
    ):
        if model_kwargs is None:
            model_kwargs = {}

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.random_state = random_state

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.models = {}
        self.optimizers = {}
        self.unique_domains = None

        self.criterion = nn.CrossEntropyLoss()

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

    # ======================================================
    # Public API
    # ======================================================
    def fit(
        self,
        X,
        y,
        domains,
        erm_epochs=20,
        align_epochs=5,
        sequential_rounds=1,
        lambda_agree=1.0,
        disagreement_mode="consensus",   # "consensus" | "pairwise"
        disagreement_distance="l2",      # "l2" | "kl"
        do_alignment=True
    ):
        if domains is None:
            raise ValueError("domains must be provided.")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        domains = np.asarray(domains)

        self.unique_domains = np.unique(domains)
        domain_indices = self._get_domain_indices(domains)

        self._initialize_models()

        # --------------------------------------------------
        # Stage 1: independent ERM
        # --------------------------------------------------
        for domain in self.unique_domains:
            self._train_erm(
                indices=domain_indices[domain],
                X=X,
                y=y,
                domain=domain,
                epochs=erm_epochs
            )

        # --------------------------------------------------
        # Stage 2: sequential alignment
        # --------------------------------------------------
        if do_alignment:
            for _ in range(sequential_rounds):
                for domain in self.unique_domains:
                    self._train_alignment(
                        indices=domain_indices[domain],
                        X=X,
                        y=y,
                        domain=domain,
                        epochs=align_epochs,
                        lambda_agree=lambda_agree,
                        disagreement_mode=disagreement_mode,
                        disagreement_distance=disagreement_distance
                    )

        return self

    def predict_proba(self, X):
        X_tensor = self._to_tensor(X)

        for model in self.models.values():
            model.eval()

        outputs = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i:i + self.batch_size]

                probs_per_domain = []
                for domain in self.unique_domains:
                    logits = self.models[domain](batch)
                    probs = self._softmax(logits)
                    probs_per_domain.append(probs)

                mean_probs = torch.stack(probs_per_domain, dim=0).mean(dim=0)
                outputs.append(mean_probs.cpu().numpy())

        return np.vstack(outputs)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    # ======================================================
    # Training internals
    # ======================================================
    def _initialize_models(self):
        self.models = {}
        self.optimizers = {}

        for domain in self.unique_domains:
            model = self.model_class(
                input_dim=self.input_dim,
                **self.model_kwargs
            ).to(self.device)

            optimizer = optim.Adam(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )

            self.models[domain] = model
            self.optimizers[domain] = optimizer

    def _train_erm(self, indices, X, y, domain, epochs):
        loader = self._make_loader(X[indices], y[indices])

        model = self.models[domain]
        optimizer = self.optimizers[domain]

        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()

                logits = model(xb)
                loss = self.criterion(logits, yb)

                loss.backward()
                optimizer.step()

    def _train_alignment(
        self,
        indices,
        X,
        y,
        domain,
        epochs,
        lambda_agree,
        disagreement_mode,
        disagreement_distance
    ):
        loader = self._make_loader(X[indices], y[indices])

        current_model = self.models[domain]
        optimizer = self.optimizers[domain]

        other_domains = [d for d in self.unique_domains if d != domain]

        for other_domain in other_domains:
            self.models[other_domain].eval()

        current_model.train()

        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()

                logits_current = current_model(xb)
                p_current = self._softmax(logits_current)

                erm_loss = self.criterion(logits_current, yb)

                with torch.no_grad():
                    p_others = []
                    for other_domain in other_domains:
                        logits_other = self.models[other_domain](xb)
                        p_other = self._softmax(logits_other)
                        p_others.append(p_other)

                agree_loss = self._disagreement(
                    p_current=p_current,
                    p_others=p_others,
                    mode=disagreement_mode,
                    distance=disagreement_distance
                )

                total_loss = erm_loss + lambda_agree * agree_loss
                total_loss.backward()
                optimizer.step()

    # ======================================================
    # Utilities
    # ======================================================
    def _to_tensor(self, X):
        if isinstance(X, np.ndarray):
            return torch.tensor(X, dtype=torch.float32, device=self.device)
        return X.to(self.device, dtype=torch.float32)

    def _make_loader(self, X, y):
        X_tensor = self._to_tensor(X)
        y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _get_domain_indices(self, domains):
        return {
            domain: np.where(domains == domain)[0]
            for domain in np.unique(domains)
        }

    def _softmax(self, logits):
        return torch.softmax(logits, dim=1)

    def _disagreement(self, p_current, p_others, mode="consensus", distance="l2"):
        if len(p_others) == 0:
            return torch.tensor(0.0, device=self.device)

        if mode == "consensus":
            target = torch.stack(p_others, dim=0).mean(dim=0)
            return self._distance(p_current, target, distance)

        if mode == "pairwise":
            losses = [
                self._distance(p_current, p_other, distance)
                for p_other in p_others
            ]
            return torch.stack(losses).mean()

        raise ValueError(f"Unknown disagreement mode: {mode}")

    def _distance(self, p, q, distance="l2"):
        if distance == "l2":
            return torch.mean((p - q) ** 2)

        if distance == "kl":
            eps = 1e-8
            p = torch.clamp(p, eps, 1.0)
            q = torch.clamp(q, eps, 1.0)
            return torch.mean(torch.sum(p * torch.log(p / q), dim=1))

        raise ValueError(f"Unknown disagreement distance: {distance}")


# ==========================================================
# Training wrapper following your pipeline pattern
# ==========================================================
def train_domain_agreement(
    X_train,
    y_train,
    domains_train=None,
    model_class=my_models.SimpleNN,
    model_kwargs=None,
    erm_epochs=20,
    align_epochs=5,
    sequential_rounds=1,
    lambda_agree=1.0,
    disagreement_mode="consensus",
    disagreement_distance="l2",
    do_alignment=True,
    lr=1e-3,
    weight_decay=0.0,
    batch_size=64,
    device=None,
    random_state=42,
    **kwargs
):
    if domains_train is None:
        raise ValueError("domains_train must be provided.")

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train)

    unique_classes = np.unique(y_train)
    n_classes = len(unique_classes)

    if not np.array_equal(unique_classes, np.arange(n_classes)):
        raise ValueError(
            "y_train must be encoded as 0, 1, ..., n_classes-1."
        )

    model = DomainAgreementModel(
        input_dim=X_train.shape[1],
        n_classes=n_classes,
        model_class=model_class,
        model_kwargs=model_kwargs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        device=device,
        random_state=random_state
    )

    model.fit(
        X=X_train,
        y=y_train,
        domains=domains_train,
        erm_epochs=erm_epochs,
        align_epochs=align_epochs,
        sequential_rounds=sequential_rounds,
        lambda_agree=lambda_agree,
        disagreement_mode=disagreement_mode,
        disagreement_distance=disagreement_distance,
        do_alignment=do_alignment
    )

    return {
        "model": model,
        "predict_proba": lambda X: model.predict_proba(X),
        "predict": lambda X: model.predict(X)
    }
    
    
    
# ==========================================================
# Simple Ensemble
# ==========================================================
class StandardEnsembleModel:
    def __init__(
        self,
        input_dim,
        n_classes,
        n_models,
        model_class,
        model_kwargs=None,
        lr=1e-3,
        weight_decay=0.0,
        batch_size=64,
        device=None,
        random_state=42
    ):
        if model_kwargs is None:
            model_kwargs = {}

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_models = n_models
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.random_state = random_state

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.models = []
        self.optimizers = []
        self.criterion = nn.CrossEntropyLoss()

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

    def fit(self, X, y, epochs=20):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        self._initialize_models()
        loader = self._make_loader(X, y)

        for model, optimizer in zip(self.models, self.optimizers):
            model.train()

            for _ in range(epochs):
                for xb, yb in loader:
                    optimizer.zero_grad()

                    logits = model(xb)
                    loss = self.criterion(logits, yb)

                    loss.backward()
                    optimizer.step()

        return self

    def predict_proba(self, X):
        X_tensor = self._to_tensor(X)

        for model in self.models:
            model.eval()

        outputs = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i:i + self.batch_size]

                probs_list = []
                for model in self.models:
                    logits = model(batch)
                    probs = torch.softmax(logits, dim=1)
                    probs_list.append(probs)

                mean_probs = torch.stack(probs_list, dim=0).mean(dim=0)
                outputs.append(mean_probs.cpu().numpy())

        return np.vstack(outputs)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def _initialize_models(self):
        self.models = []
        self.optimizers = []

        for i in range(self.n_models):
            torch.manual_seed(self.random_state + i)

            model = self.model_class(
                input_dim=self.input_dim,
                **self.model_kwargs
            ).to(self.device)

            optimizer = optim.Adam(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )

            self.models.append(model)
            self.optimizers.append(optimizer)

    def _to_tensor(self, X):
        if isinstance(X, np.ndarray):
            return torch.tensor(X, dtype=torch.float32, device=self.device)
        return X.to(self.device, dtype=torch.float32)

    def _make_loader(self, X, y):
        X_tensor = self._to_tensor(X)
        y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


def train_standard_ensemble(
    X_train,
    y_train,
    domains_train=None,
    model_class=my_models.SimpleNN,
    model_kwargs=None,
    epochs=20,
    n_models=None,
    lr=1e-3,
    weight_decay=0.0,
    batch_size=64,
    device=None,
    random_state=42,
    **kwargs
):
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train)

    unique_classes = np.unique(y_train)
    n_classes = len(unique_classes)

    if not np.array_equal(unique_classes, np.arange(n_classes)):
        raise ValueError("y_train must be encoded as 0, 1, ..., n_classes-1.")

    if n_models is None:
        if domains_train is None:
            raise ValueError("Either n_models or domains_train must be provided.")
        n_models = len(np.unique(domains_train))

    model = StandardEnsembleModel(
        input_dim=X_train.shape[1],
        n_classes=n_classes,
        n_models=n_models,
        model_class=model_class,
        model_kwargs=model_kwargs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        device=device,
        random_state=random_state
    )

    model.fit(X_train, y_train, epochs=epochs)

    return {
        "model": model,
        "predict_proba": lambda X: model.predict_proba(X),
        "predict": lambda X: model.predict(X)
    }