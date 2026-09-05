import sys
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath("../../../../"))

from archieve.lib.learning_algorithms.helper import wrap_model

def train_linear_probe(
    X_source=None, y_source=None, source_domains=None,
    X_target=None, y_target=None, target_domains=None,
    pretrained_model=None,
    epochs=50,
    lr=1e-3,
    batch_size=128,
    weight_decay=0.0,
    random_state=42,
    **kwargs
):
    """Linear probing using labeled target data."""

    if pretrained_model is None:
        raise ValueError("Linear probing requires a pretrained source model.")
    if X_target is None or y_target is None:
        raise ValueError("Linear probing requires labeled target data.")

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = copy.deepcopy(pretrained_model["model"]).to(device)

    if not hasattr(model, "feature_extractor") or not hasattr(model, "classifier"):
        raise ValueError("Linear probing requires feature_extractor and classifier.")

    # Freeze pretrained representation
    for p in model.feature_extractor.parameters():
        p.requires_grad = False

    classes = np.unique(y_target)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([class_to_idx[y] for y in y_target], dtype=np.int64)

    # New target-specific linear classifier
    model.classifier = nn.Linear(
        model.classifier.in_features,
        len(classes)
    ).to(device)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_target, dtype=torch.float32),
        torch.tensor(y_encoded, dtype=torch.long)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    history = {"loss": []}

    for _ in range(epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)

        history["loss"].append(total_loss / len(dataset))

    def predict_proba(X):
        model.eval()
        X = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            return torch.softmax(model(X), dim=1).cpu().numpy()

    def predict(X):
        return classes[np.argmax(predict_proba(X), axis=1)]

    return wrap_model(
        model,
        predict,
        predict_proba,
        classes,
        training_history=history,
        artifacts={
            "scenario": "source_free_labeled",
            "method": "linear_probe",
            "target_labels_used": True,
            "n_target_samples": len(X_target),
        }
    )
    
def train_fine_tuning(
    X_source=None, y_source=None, source_domains=None,
    X_target=None, y_target=None, target_domains=None,
    pretrained_model=None,
    epochs=50,
    lr=1e-4,
    batch_size=128,
    weight_decay=0.0,
    random_state=42,
    **kwargs
):
    """Full fine-tuning using labeled target data."""

    if pretrained_model is None:
        raise ValueError("Fine-tuning requires a pretrained source model.")
    if X_target is None or y_target is None:
        raise ValueError("Fine-tuning requires labeled target data.")

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = copy.deepcopy(pretrained_model["model"]).to(device)
    classes = np.asarray(pretrained_model["classes"])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([class_to_idx[y] for y in y_target], dtype=np.int64)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_target, dtype=torch.float32),
        torch.tensor(y_encoded, dtype=torch.long)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    history = {"loss": []}

    for _ in range(epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)

        history["loss"].append(total_loss / len(dataset))

    def predict_proba(X):
        model.eval()
        X = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            return torch.softmax(model(X), dim=1).cpu().numpy()

    def predict(X):
        return classes[np.argmax(predict_proba(X), axis=1)]

    return wrap_model(
        model,
        predict,
        predict_proba,
        classes,
        training_history=history,
        artifacts={
            "scenario": "source_free_labeled",
            "method": "fine_tuning",
            "target_labels_used": True,
            "n_target_samples": len(X_target),
        }
    )

def train_lp_ft(
    X_source=None, y_source=None, source_domains=None,
    X_target=None, y_target=None, target_domains=None,
    pretrained_model=None,
    probe_epochs=20,
    finetune_epochs=30,
    probe_lr=1e-3,
    finetune_lr=1e-4,
    batch_size=128,
    weight_decay=0.0,
    random_state=42,
    **kwargs
):
    """Linear probing followed by full fine-tuning."""

    if pretrained_model is None:
        raise ValueError("LP-FT requires a pretrained source model.")
    if X_target is None or y_target is None:
        raise ValueError("LP-FT requires labeled target data.")

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = copy.deepcopy(pretrained_model["model"]).to(device)
    classes = np.asarray(pretrained_model["classes"])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([class_to_idx[y] for y in y_target], dtype=np.int64)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_target, dtype=torch.float32),
        torch.tensor(y_encoded, dtype=torch.long)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    criterion = nn.CrossEntropyLoss()
    history = {"probe_loss": [], "finetune_loss": []}

    # --------------------------------------------------------
    # Stage 1: Linear probing
    # --------------------------------------------------------

    for p in model.feature_extractor.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=probe_lr,
        weight_decay=weight_decay
    )

    for _ in range(probe_epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)

        history["probe_loss"].append(total_loss / len(dataset))

    # --------------------------------------------------------
    # Stage 2: Full fine-tuning
    # --------------------------------------------------------

    for p in model.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(
        model.parameters(),
        lr=finetune_lr,
        weight_decay=weight_decay
    )

    for _ in range(finetune_epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)

        history["finetune_loss"].append(total_loss / len(dataset))

    def predict_proba(X):
        model.eval()
        X = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            return torch.softmax(model(X), dim=1).cpu().numpy()

    def predict(X):
        return classes[np.argmax(predict_proba(X), axis=1)]

    return wrap_model(
        model,
        predict,
        predict_proba,
        classes,
        training_history=history,
        artifacts={
            "scenario": "source_free_labeled",
            "method": "lp_ft",
            "target_labels_used": True,
            "n_target_samples": len(X_target),
        }
    )

def train_l2_sp(
    X_source=None, y_source=None, source_domains=None,
    X_target=None, y_target=None, target_domains=None,
    pretrained_model=None,
    epochs=50,
    lr=1e-4,
    batch_size=128,
    l2_sp_lambda=1e-3,
    random_state=42,
    **kwargs
):
    """Fine-tuning with L2-SP regularization toward source parameters."""

    if pretrained_model is None:
        raise ValueError("L2-SP requires a pretrained source model.")
    if X_target is None or y_target is None:
        raise ValueError("L2-SP requires labeled target data.")

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = copy.deepcopy(pretrained_model["model"]).to(device)
    classes = np.asarray(pretrained_model["classes"])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([class_to_idx[y] for y in y_target], dtype=np.int64)

    # Store source parameters as fixed reference
    source_params = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
    }

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_target, dtype=torch.float32),
        torch.tensor(y_encoded, dtype=torch.long)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "classification_loss": [], "l2_sp_penalty": []}

    for _ in range(epochs):
        model.train()
        total_loss = total_cls = total_sp = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            cls_loss = criterion(logits, yb)

            sp_penalty = sum(
                (param - source_params[name]).pow(2).sum()
                for name, param in model.named_parameters()
            )

            loss = cls_loss + l2_sp_lambda * sp_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)
            total_cls += cls_loss.item() * len(xb)
            total_sp += sp_penalty.item() * len(xb)

        n = len(dataset)
        history["loss"].append(total_loss / n)
        history["classification_loss"].append(total_cls / n)
        history["l2_sp_penalty"].append(total_sp / n)

    def predict_proba(X):
        model.eval()
        X = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            return torch.softmax(model(X), dim=1).cpu().numpy()

    def predict(X):
        return classes[np.argmax(predict_proba(X), axis=1)]

    return wrap_model(
        model,
        predict,
        predict_proba,
        classes,
        training_history=history,
        artifacts={
            "scenario": "source_free_labeled",
            "method": "l2_sp",
            "target_labels_used": True,
            "n_target_samples": len(X_target),
        }
    )


