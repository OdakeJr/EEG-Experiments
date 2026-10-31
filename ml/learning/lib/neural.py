
# ml/learning/lib/neural.py

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def make_optimizer(model, name="adam", learning_rate=1e-3, weight_decay=0.0):
    optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")

    return optimizers[name](
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


@torch.no_grad()
def validation_loss(model, loader, device):
    if loader is None:
        return None

    model.eval()
    total, count = 0.0, 0

    for batch in loader:
        xb, yb = batch[:2]
        xb, yb = xb.to(device), yb.to(device)
        total += F.cross_entropy(model(xb), yb, reduction="sum").item()
        count += len(yb)

    return total / count if count else None


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_state = None
        self.stale = 0

    def step(self, loss, model):
        if loss is None:
            return False

        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            self.stale = 0
        else:
            self.stale += 1

        return self.patience is not None and self.stale >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model


@torch.no_grad()
def predict_proba(model, X, device="cpu", batch_size=256,
                  transform=None, num_workers=0):
    if isinstance(X, DataLoader):
        loader = X
    else:
        if transform is not None:
            from vision.lib.dataset import ImageDataset
            n = len(X)
            dataset = ImageDataset(
                X, np.zeros(n, dtype=int),
                np.full(n, "prediction"), {"0": 0}, transform
            )
        else:
            dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32))

        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers
        )

    model.eval()
    probabilities = []

    for batch in loader:
        xb = batch[0].to(device)
        probabilities.append(
            F.softmax(model(xb), dim=1).cpu().numpy()
        )

    return np.concatenate(probabilities)
