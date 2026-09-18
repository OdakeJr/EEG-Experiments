# ml/learning/classical/neural_erm.py

import copy
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, TensorDataset

from ml.learning.base import BaseLearningAlgorithm


class NeuralERM(BaseLearningAlgorithm):

    def __init__(self):
        self.model = None
        self.classes_ = None
        self.device = "cpu"

    def fit(self, model, source=None, target_super_domain=None,
            target_elementary_domain=None, **training_params):

        X, y = self._get_training_data(
            source, target_super_domain, target_elementary_domain
        )

        # ----------------------------------------------------
        # Training parameters
        # ----------------------------------------------------

        epochs = training_params.get("epochs", 100)
        batch_size = training_params.get("batch_size", 64)
        learning_rate = training_params.get("learning_rate", 1e-3)
        weight_decay = training_params.get("weight_decay", 1e-4)
        optimizer_name = training_params.get("optimizer", "adamw")
        self.device = training_params.get("device", "cpu")
        seed = training_params.get("seed", 42)

        validation_fraction = training_params.get("validation_fraction", 0.2)
        patience = training_params.get("patience", 20)
        min_delta = training_params.get("min_delta", 0.0)

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------

        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        self.classes_ = np.unique(y)
        class_to_index = {
            label: index for index, label in enumerate(self.classes_)
        }
        y_encoded = np.array([class_to_index[label] for label in y])

        # ----------------------------------------------------
        # Train / validation split
        # ----------------------------------------------------

        indices = np.arange(len(y_encoded))

        if validation_fraction > 0:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=validation_fraction,
                stratify=y_encoded,
                random_state=seed,
            )
        else:
            train_idx, val_idx = indices, None

        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_encoded, dtype=torch.long)

        generator = torch.Generator().manual_seed(seed)

        train_loader = DataLoader(
            TensorDataset(X_tensor[train_idx], y_tensor[train_idx]),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        val_loader = None
        if val_idx is not None:
            val_loader = DataLoader(
                TensorDataset(X_tensor[val_idx], y_tensor[val_idx]),
                batch_size=batch_size,
                shuffle=False,
            )

        # ----------------------------------------------------
        # Model / optimizer / loss
        # ----------------------------------------------------

        model = model.to(self.device)

        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(
                f"Unknown optimizer '{optimizer_name}'. "
                "Available: ['adam', 'adamw']."
            )

        criterion = nn.CrossEntropyLoss()

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        best_loss = np.inf
        best_state = None
        epochs_without_improvement = 0

        for _ in range(epochs):
            model.train()

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

            if val_loader is None:
                continue

            model.eval()
            val_loss = 0.0
            n_samples = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    loss = criterion(model(X_batch), y_batch)
                    val_loss += loss.item() * len(y_batch)
                    n_samples += len(y_batch)

            val_loss /= n_samples

            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        self.model = model
        return self

    def predict(self, X, domains=None, super_domains=None):
        probabilities = self.predict_proba(X, domains, super_domains)
        indices = np.argmax(probabilities, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X, domains=None, super_domains=None):
        self._check_fitted()
        self.model.eval()

        batch_size = 256
        probabilities = []

        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                X_batch = torch.as_tensor(
                    X[start:start + batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                logits = self.model(X_batch)
                probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())

        return np.concatenate(probabilities, axis=0)

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

    @staticmethod
    def _get_training_data(
        source, target_super_domain, target_elementary_domain
    ):
        X_parts, y_parts = [], []

        for group in [
            source,
            target_super_domain,
            target_elementary_domain,
        ]:
            if group is None:
                continue

            mask = group.partitions == "train"

            if np.any(mask):
                X_parts.append(group.X[mask])
                y_parts.append(group.y[mask])

        if not X_parts:
            raise ValueError("No training samples found.")

        return np.concatenate(X_parts), np.concatenate(y_parts)

    def _check_fitted(self):
        if self.model is None:
            raise RuntimeError(
                "Learning algorithm has not been fitted."
            )