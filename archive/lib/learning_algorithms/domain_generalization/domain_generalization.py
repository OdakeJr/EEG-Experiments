import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath("../../../../"))

import archieve.lib.learning_algorithms.benchmark_models.baseline_nn as my_models
from archieve.lib.learning_algorithms.helper import wrap_model

def train_vrex(
    X_source, y_source, source_domains=None,
    X_target=None, y_target=None, target_domains=None,
    pretrained_model=None,
    hidden_dims=(128, 128),
    activation="relu",
    dropout=0.0,
    batch_norm=False,
    epochs=50,
    lr=1e-3,
    batch_size=128,
    weight_decay=0.0,
    vrex_lambda=1.0,
    penalty_anneal_epochs=0,
    random_state=42,
    **kwargs
):
    """Variance Risk Extrapolation (VREx) for domain generalization."""

    if source_domains is None:
        raise ValueError("VREx requires source-domain labels.")

    X_source = np.asarray(X_source)
    y_source = np.asarray(y_source)
    source_domains = np.asarray(source_domains)

    domains = np.unique(source_domains)
    if len(domains) < 2:
        raise ValueError("VREx requires at least two source domains.")

    np.random.seed(random_state)
    torch.manual_seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Encode class labels
    classes = np.unique(y_source)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([class_to_idx[y] for y in y_source], dtype=np.int64)

    # One loader per source domain
    loaders = {}
    for domain in domains:
        idx = np.where(source_domains == domain)[0]
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_source[idx], dtype=torch.float32),
            torch.tensor(y_encoded[idx], dtype=torch.long)
        )
        loaders[domain] = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

    model = my_models.MLP(
        input_dim=X_source.shape[1],
        hidden_dims=hidden_dims,
        output_dim=len(classes),
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    history = {"loss": [], "erm_loss": [], "vrex_penalty": []}

    for epoch in range(epochs):
        model.train()
        iterators = {d: iter(loader) for d, loader in loaders.items()}
        n_steps = max(len(loader) for loader in loaders.values())

        epoch_loss = epoch_erm = epoch_penalty = 0.0

        for _ in range(n_steps):
            domain_risks = []

            for domain in domains:
                try:
                    xb, yb = next(iterators[domain])
                except StopIteration:
                    iterators[domain] = iter(loaders[domain])
                    xb, yb = next(iterators[domain])

                xb, yb = xb.to(device), yb.to(device)
                domain_risks.append(criterion(model(xb), yb))

            risks = torch.stack(domain_risks)
            erm_loss = risks.mean()
            vrex_penalty = torch.var(risks, unbiased=False)

            penalty_weight = (
                vrex_lambda if epoch >= penalty_anneal_epochs else 0.0
            )
            loss = erm_loss + penalty_weight * vrex_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_erm += erm_loss.item()
            epoch_penalty += vrex_penalty.item()

        history["loss"].append(epoch_loss / n_steps)
        history["erm_loss"].append(epoch_erm / n_steps)
        history["vrex_penalty"].append(epoch_penalty / n_steps)

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
            "scenario": "domain_generalization",
            "method": "vrex",
            "n_source_domains": len(domains),
        }
    )
    
def train_groupdro(
    X_source, y_source, source_domains=None,
    X_target=None, y_target=None, target_domains=None,
    pretrained_model=None,
    hidden_dims=(128, 128),
    activation="relu",
    dropout=0.0,
    batch_norm=False,
    epochs=50,
    lr=1e-3,
    batch_size=128,
    weight_decay=0.0,
    groupdro_eta=0.01,
    random_state=42,
    **kwargs
):
    """Group Distributionally Robust Optimization (GroupDRO)."""

    if source_domains is None:
        raise ValueError("GroupDRO requires source-domain labels.")

    X_source = np.asarray(X_source)
    y_source = np.asarray(y_source)
    source_domains = np.asarray(source_domains)

    domains = np.unique(source_domains)
    if len(domains) < 2:
        raise ValueError("GroupDRO requires at least two source domains.")

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = np.unique(y_source)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([class_to_idx[y] for y in y_source], dtype=np.int64)

    loaders = {}
    for domain in domains:
        idx = np.where(source_domains == domain)[0]
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_source[idx], dtype=torch.float32),
            torch.tensor(y_encoded[idx], dtype=torch.long)
        )
        loaders[domain] = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

    model = my_models.MLP(
        input_dim=X_source.shape[1],
        hidden_dims=hidden_dims,
        output_dim=len(classes),
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    group_weights = torch.ones(
        len(domains), device=device
    ) / len(domains)

    history = {"loss": [], "group_weights": []}

    for _ in range(epochs):
        model.train()
        iterators = {d: iter(loader) for d, loader in loaders.items()}
        n_steps = max(len(loader) for loader in loaders.values())
        epoch_loss = 0.0

        for _ in range(n_steps):
            domain_risks = []

            for domain in domains:
                try:
                    xb, yb = next(iterators[domain])
                except StopIteration:
                    iterators[domain] = iter(loaders[domain])
                    xb, yb = next(iterators[domain])

                xb, yb = xb.to(device), yb.to(device)
                domain_risks.append(criterion(model(xb), yb))

            risks = torch.stack(domain_risks)

            with torch.no_grad():
                group_weights *= torch.exp(groupdro_eta * risks.detach())
                group_weights /= group_weights.sum()

            loss = torch.sum(group_weights * risks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        history["loss"].append(epoch_loss / n_steps)
        history["group_weights"].append(
            group_weights.detach().cpu().numpy().copy()
        )

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
            "scenario": "domain_generalization",
            "method": "groupdro",
            "n_source_domains": len(domains),
            "final_group_weights": group_weights.detach().cpu().numpy(),
        }
    )
    
def train_irm(
    X_source, y_source, source_domains=None,
    X_target=None, y_target=None, target_domains=None,
    pretrained_model=None,
    hidden_dims=(128, 128),
    activation="relu",
    dropout=0.0,
    batch_norm=False,
    epochs=50,
    lr=1e-3,
    batch_size=128,
    weight_decay=0.0,
    irm_lambda=100.0,
    penalty_anneal_epochs=10,
    random_state=42,
    **kwargs
):
    """Invariant Risk Minimization (IRM)."""

    if source_domains is None:
        raise ValueError("IRM requires source-domain labels.")

    X_source = np.asarray(X_source)
    y_source = np.asarray(y_source)
    source_domains = np.asarray(source_domains)

    domains = np.unique(source_domains)
    if len(domains) < 2:
        raise ValueError("IRM requires at least two source domains.")

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = np.unique(y_source)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([class_to_idx[y] for y in y_source], dtype=np.int64)

    loaders = {}
    for domain in domains:
        idx = np.where(source_domains == domain)[0]
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_source[idx], dtype=torch.float32),
            torch.tensor(y_encoded[idx], dtype=torch.long)
        )
        loaders[domain] = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

    model = my_models.MLP(
        input_dim=X_source.shape[1],
        hidden_dims=hidden_dims,
        output_dim=len(classes),
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "erm_loss": [], "irm_penalty": []}

    def irm_penalty(logits, labels):
        scale = torch.tensor(1.0, device=device, requires_grad=True)
        loss = criterion(logits * scale, labels)
        grad = torch.autograd.grad(loss, scale, create_graph=True)[0]
        return grad.pow(2)

    for epoch in range(epochs):
        model.train()
        iterators = {d: iter(loader) for d, loader in loaders.items()}
        n_steps = max(len(loader) for loader in loaders.values())

        epoch_loss = epoch_erm = epoch_penalty = 0.0

        for _ in range(n_steps):
            risks, penalties = [], []

            for domain in domains:
                try:
                    xb, yb = next(iterators[domain])
                except StopIteration:
                    iterators[domain] = iter(loaders[domain])
                    xb, yb = next(iterators[domain])

                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)

                risks.append(criterion(logits, yb))
                penalties.append(irm_penalty(logits, yb))

            erm_loss = torch.stack(risks).mean()
            penalty = torch.stack(penalties).mean()

            penalty_weight = (
                irm_lambda if epoch >= penalty_anneal_epochs else 1.0
            )

            loss = erm_loss + penalty_weight * penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_erm += erm_loss.item()
            epoch_penalty += penalty.item()

        history["loss"].append(epoch_loss / n_steps)
        history["erm_loss"].append(epoch_erm / n_steps)
        history["irm_penalty"].append(epoch_penalty / n_steps)

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
            "scenario": "domain_generalization",
            "method": "irm",
            "n_source_domains": len(domains),
        }
    )
    
def train_coral_dg(
    X_source, y_source, source_domains=None,
    X_target=None, y_target=None, target_domains=None,
    pretrained_model=None,
    hidden_dims=(128, 128),
    activation="relu",
    dropout=0.0,
    batch_norm=False,
    epochs=50,
    lr=1e-3,
    batch_size=128,
    weight_decay=0.0,
    coral_lambda=1.0,
    random_state=42,
    **kwargs
):
    """Multi-source CORAL for domain generalization."""

    if source_domains is None:
        raise ValueError("CORAL DG requires source-domain labels.")

    X_source = np.asarray(X_source)
    y_source = np.asarray(y_source)
    source_domains = np.asarray(source_domains)

    domains = np.unique(source_domains)
    if len(domains) < 2:
        raise ValueError("CORAL DG requires at least two source domains.")

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = np.unique(y_source)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([class_to_idx[y] for y in y_source], dtype=np.int64)

    loaders = {}
    for domain in domains:
        idx = np.where(source_domains == domain)[0]
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_source[idx], dtype=torch.float32),
            torch.tensor(y_encoded[idx], dtype=torch.long)
        )
        loaders[domain] = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

    model = my_models.MLP(
        input_dim=X_source.shape[1],
        hidden_dims=hidden_dims,
        output_dim=len(classes),
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "erm_loss": [], "coral_penalty": []}

    def covariance(features):
        if len(features) < 2:
            return torch.zeros(
                features.shape[1], features.shape[1], device=device
            )
        centered = features - features.mean(dim=0, keepdim=True)
        return centered.T @ centered / (len(features) - 1)

    for _ in range(epochs):
        model.train()
        iterators = {d: iter(loader) for d, loader in loaders.items()}
        n_steps = max(len(loader) for loader in loaders.values())

        epoch_loss = epoch_erm = epoch_coral = 0.0

        for _ in range(n_steps):
            risks, covariances = [], []

            for domain in domains:
                try:
                    xb, yb = next(iterators[domain])
                except StopIteration:
                    iterators[domain] = iter(loaders[domain])
                    xb, yb = next(iterators[domain])

                xb, yb = xb.to(device), yb.to(device)

                features = model.extract_features(xb)
                logits = model.classifier(features)

                risks.append(criterion(logits, yb))
                covariances.append(covariance(features))

            erm_loss = torch.stack(risks).mean()

            coral_penalty = 0.0
            n_pairs = 0
            for i in range(len(domains)):
                for j in range(i + 1, len(domains)):
                    coral_penalty += torch.mean(
                        (covariances[i] - covariances[j]) ** 2
                    )
                    n_pairs += 1

            coral_penalty = coral_penalty / n_pairs
            loss = erm_loss + coral_lambda * coral_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_erm += erm_loss.item()
            epoch_coral += coral_penalty.item()

        history["loss"].append(epoch_loss / n_steps)
        history["erm_loss"].append(epoch_erm / n_steps)
        history["coral_penalty"].append(epoch_coral / n_steps)

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
            "scenario": "domain_generalization",
            "method": "coral",
            "n_source_domains": len(domains),
        }
    )
    
def train_mmd_dg(
    X_source, y_source, source_domains=None,
    X_target=None, y_target=None, target_domains=None,
    pretrained_model=None,
    hidden_dims=(128, 128),
    activation="relu",
    dropout=0.0,
    batch_norm=False,
    epochs=50,
    lr=1e-3,
    batch_size=128,
    weight_decay=0.0,
    mmd_lambda=1.0,
    mmd_gamma=None,
    random_state=42,
    **kwargs
):
    """Multi-source MMD alignment for domain generalization."""

    if source_domains is None:
        raise ValueError("MMD DG requires source-domain labels.")

    X_source = np.asarray(X_source)
    y_source = np.asarray(y_source)
    source_domains = np.asarray(source_domains)

    domains = np.unique(source_domains)
    if len(domains) < 2:
        raise ValueError("MMD DG requires at least two source domains.")

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = np.unique(y_source)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([class_to_idx[y] for y in y_source], dtype=np.int64)

    loaders = {}
    for domain in domains:
        idx = np.where(source_domains == domain)[0]
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_source[idx], dtype=torch.float32),
            torch.tensor(y_encoded[idx], dtype=torch.long)
        )
        loaders[domain] = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

    model = my_models.MLP(
        input_dim=X_source.shape[1],
        hidden_dims=hidden_dims,
        output_dim=len(classes),
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "erm_loss": [], "mmd_penalty": []}

    def mmd_rbf(x, y):
        gamma = mmd_gamma or (1.0 / x.shape[1])

        k_xx = torch.exp(-gamma * torch.cdist(x, x).pow(2))
        k_yy = torch.exp(-gamma * torch.cdist(y, y).pow(2))
        k_xy = torch.exp(-gamma * torch.cdist(x, y).pow(2))

        return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

    for _ in range(epochs):
        model.train()
        iterators = {d: iter(loader) for d, loader in loaders.items()}
        n_steps = max(len(loader) for loader in loaders.values())

        epoch_loss = epoch_erm = epoch_mmd = 0.0

        for _ in range(n_steps):
            risks, features = [], []

            for domain in domains:
                try:
                    xb, yb = next(iterators[domain])
                except StopIteration:
                    iterators[domain] = iter(loaders[domain])
                    xb, yb = next(iterators[domain])

                xb, yb = xb.to(device), yb.to(device)

                z = model.extract_features(xb)
                logits = model.classifier(z)

                risks.append(criterion(logits, yb))
                features.append(z)

            erm_loss = torch.stack(risks).mean()

            penalties = [
                mmd_rbf(features[i], features[j])
                for i in range(len(domains))
                for j in range(i + 1, len(domains))
            ]

            mmd_penalty = torch.stack(penalties).mean()
            loss = erm_loss + mmd_lambda * mmd_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_erm += erm_loss.item()
            epoch_mmd += mmd_penalty.item()

        history["loss"].append(epoch_loss / n_steps)
        history["erm_loss"].append(epoch_erm / n_steps)
        history["mmd_penalty"].append(epoch_mmd / n_steps)

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
            "scenario": "domain_generalization",
            "method": "mmd",
            "n_source_domains": len(domains),
        }
    )