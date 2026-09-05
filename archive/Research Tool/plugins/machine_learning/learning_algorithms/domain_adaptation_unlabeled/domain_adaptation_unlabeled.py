import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath("../../../../"))

import archieve.lib.learning_algorithms.benchmark_models.baseline_nn as my_models
import lib.learning_algorithms.density_ratio_estimator.density_ratio_estimator as iw_estimators
from archieve.lib.learning_algorithms.helper import wrap_model

def train_deep_coral(
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
    """Deep CORAL for unsupervised domain adaptation."""

    if X_target is None:
        raise ValueError("Deep CORAL requires unlabeled target data.")

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_source, y_source = np.asarray(X_source), np.asarray(y_source)
    X_target = np.asarray(X_target)

    classes = np.unique(y_source)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([class_to_idx[y] for y in y_source], dtype=np.int64)

    source_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_source, dtype=torch.float32),
            torch.tensor(y_encoded, dtype=torch.long)
        ),
        batch_size=batch_size, shuffle=True
    )

    target_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_target, dtype=torch.float32)
        ),
        batch_size=batch_size, shuffle=True
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
    history = {"loss": [], "classification_loss": [], "coral_loss": []}

    def covariance(x):
        if len(x) < 2:
            return torch.zeros(x.shape[1], x.shape[1], device=x.device)
        x = x - x.mean(dim=0, keepdim=True)
        return x.T @ x / (len(x) - 1)

    def coral_loss(source, target):
        cs, ct = covariance(source), covariance(target)
        d = source.shape[1]
        return ((cs - ct) ** 2).sum() / (4 * d ** 2)

    for _ in range(epochs):
        model.train()
        source_iter, target_iter = iter(source_loader), iter(target_loader)
        n_steps = max(len(source_loader), len(target_loader))
        totals = np.zeros(3)

        for _ in range(n_steps):
            try:
                xs, ys = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                xs, ys = next(source_iter)

            try:
                xt, = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                xt, = next(target_iter)

            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)

            fs = model.extract_features(xs)
            ft = model.extract_features(xt)

            cls_loss = criterion(model.classifier(fs), ys)
            align_loss = coral_loss(fs, ft)
            loss = cls_loss + coral_lambda * align_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totals += [loss.item(), cls_loss.item(), align_loss.item()]

        totals /= n_steps
        history["loss"].append(totals[0])
        history["classification_loss"].append(totals[1])
        history["coral_loss"].append(totals[2])

    def predict_proba(X):
        model.eval()
        X = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            return torch.softmax(model(X), dim=1).cpu().numpy()

    def predict(X):
        return classes[np.argmax(predict_proba(X), axis=1)]

    return wrap_model(
        model, predict, predict_proba, classes,
        training_history=history,
        artifacts={
            "scenario": "domain_adaptation_unlabeled",
            "method": "deep_coral",
            "target_labels_used": False,
            "n_target_samples": len(X_target),
        }
    )

def train_dann(
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
    dann_lambda=1.0,
    discriminator_hidden_dim=128,
    random_state=42,
    **kwargs
):
    """Domain-Adversarial Neural Network (DANN)."""

    if X_target is None:
        raise ValueError("DANN requires unlabeled target data.")

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_source, y_source = np.asarray(X_source), np.asarray(y_source)
    X_target = np.asarray(X_target)

    classes = np.unique(y_source)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([class_to_idx[y] for y in y_source], dtype=np.int64)

    source_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_source, dtype=torch.float32),
            torch.tensor(y_encoded, dtype=torch.long)
        ),
        batch_size=batch_size, shuffle=True
    )

    target_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_target, dtype=torch.float32)
        ),
        batch_size=batch_size, shuffle=True
    )

    model = my_models.MLP(
        input_dim=X_source.shape[1],
        hidden_dims=hidden_dims,
        output_dim=len(classes),
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm
    ).to(device)

    discriminator = my_models.DomainDiscriminator(
        model.classifier.in_features,
        discriminator_hidden_dim
    ).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(discriminator.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()
    history = {"loss": [], "classification_loss": [], "domain_loss": []}

    for _ in range(epochs):
        model.train()
        discriminator.train()

        source_iter, target_iter = iter(source_loader), iter(target_loader)
        n_steps = max(len(source_loader), len(target_loader))
        totals = np.zeros(3)

        for _ in range(n_steps):
            try:
                xs, ys = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                xs, ys = next(source_iter)

            try:
                xt, = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                xt, = next(target_iter)

            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)

            fs = model.extract_features(xs)
            ft = model.extract_features(xt)

            cls_loss = class_criterion(model.classifier(fs), ys)

            features = torch.cat([fs, ft], dim=0)
            reversed_features = my_models.GradientReversal.apply(
                features, dann_lambda
            )

            domain_logits = discriminator(reversed_features).squeeze(1)
            domain_labels = torch.cat([
                torch.zeros(len(fs), device=device),
                torch.ones(len(ft), device=device)
            ])

            domain_loss = domain_criterion(
                domain_logits, domain_labels
            )

            loss = cls_loss + domain_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totals += [loss.item(), cls_loss.item(), domain_loss.item()]

        totals /= n_steps
        history["loss"].append(totals[0])
        history["classification_loss"].append(totals[1])
        history["domain_loss"].append(totals[2])

    def predict_proba(X):
        model.eval()
        X = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            return torch.softmax(model(X), dim=1).cpu().numpy()

    def predict(X):
        return classes[np.argmax(predict_proba(X), axis=1)]

    return wrap_model(
        model, predict, predict_proba, classes,
        training_history=history,
        artifacts={
            "scenario": "domain_adaptation_unlabeled",
            "method": "dann",
            "domain_discriminator": discriminator,
            "target_labels_used": False,
            "n_target_samples": len(X_target),
        }
    )

def train_mcd(
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
    discrepancy_steps=4,
    random_state=42,
    **kwargs
):
    """Maximum Classifier Discrepancy (MCD)."""

    if X_target is None:
        raise ValueError("MCD requires unlabeled target data.")

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_source, y_source = np.asarray(X_source), np.asarray(y_source)
    X_target = np.asarray(X_target)

    classes = np.unique(y_source)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([class_to_idx[y] for y in y_source], dtype=np.int64)

    source_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_source, dtype=torch.float32),
            torch.tensor(y_encoded, dtype=torch.long)
        ),
        batch_size=batch_size, shuffle=True
    )

    target_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_target, dtype=torch.float32)
        ),
        batch_size=batch_size, shuffle=True
    )

    model = my_models.MCDModel(
        X_source.shape[1], hidden_dims, len(classes),
        activation, dropout, batch_norm
    ).to(device)

    optimizer_g = optim.Adam(
        model.feature_extractor.parameters(),
        lr=lr, weight_decay=weight_decay
    )

    classifier_params = (
        list(model.classifier1.parameters())
        + list(model.classifier2.parameters())
    )

    optimizer_c = optim.Adam(
        classifier_params, lr=lr, weight_decay=weight_decay
    )

    criterion = nn.CrossEntropyLoss()
    history = {
        "classification_loss": [],
        "classifier_discrepancy": []
    }

    def discrepancy(p1, p2):
        p1 = torch.softmax(p1, dim=1)
        p2 = torch.softmax(p2, dim=1)
        return torch.mean(torch.abs(p1 - p2))

    for _ in range(epochs):
        model.train()
        source_iter, target_iter = iter(source_loader), iter(target_loader)
        n_steps = max(len(source_loader), len(target_loader))
        total_cls = total_disc = 0.0

        for _ in range(n_steps):
            try:
                xs, ys = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                xs, ys = next(source_iter)

            try:
                xt, = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                xt, = next(target_iter)

            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)

            # A: train feature extractor + classifiers on source
            y1, y2 = model(xs)
            cls_loss = criterion(y1, ys) + criterion(y2, ys)

            optimizer_g.zero_grad()
            optimizer_c.zero_grad()
            cls_loss.backward()
            optimizer_g.step()
            optimizer_c.step()

            # B: maximize classifier discrepancy on target
            fs = model.extract_features(xs).detach()
            ft = model.extract_features(xt).detach()

            ys1, ys2 = model.classifier1(fs), model.classifier2(fs)
            yt1, yt2 = model.classifier1(ft), model.classifier2(ft)

            source_loss = criterion(ys1, ys) + criterion(ys2, ys)
            disc = discrepancy(yt1, yt2)

            optimizer_c.zero_grad()
            (source_loss - disc).backward()
            optimizer_c.step()

            # C: adapt feature extractor to minimize discrepancy
            for p in classifier_params:
                p.requires_grad = False

            for _ in range(discrepancy_steps):
                ft = model.extract_features(xt)
                yt1 = model.classifier1(ft)
                yt2 = model.classifier2(ft)

                disc = discrepancy(yt1, yt2)

                optimizer_g.zero_grad()
                disc.backward()
                optimizer_g.step()

            for p in classifier_params:
                p.requires_grad = True

            total_cls += cls_loss.item()
            total_disc += disc.item()

        history["classification_loss"].append(total_cls / n_steps)
        history["classifier_discrepancy"].append(total_disc / n_steps)

    def predict_proba(X):
        model.eval()
        X = torch.tensor(X, dtype=torch.float32, device=device)

        with torch.no_grad():
            y1, y2 = model(X)
            p1 = torch.softmax(y1, dim=1)
            p2 = torch.softmax(y2, dim=1)

        return ((p1 + p2) / 2).cpu().numpy()

    def predict(X):
        return classes[np.argmax(predict_proba(X), axis=1)]

    return wrap_model(
        model,
        predict,
        predict_proba,
        classes,
        training_history=history,
        artifacts={
            "scenario": "domain_adaptation_unlabeled",
            "method": "mcd",
            "target_labels_used": False,
            "n_target_samples": len(X_target),
        }
    )

def train_importance_weighting(
    X_source, y_source, source_domains=None,
    X_target=None, y_target=None, target_domains=None,
    pretrained_model=None,
    estimator="kliep",
    estimator_params=None,
    hidden_dims=(128, 128),
    activation="relu",
    dropout=0.0,
    batch_norm=False,
    epochs=50,
    lr=1e-3,
    batch_size=128,
    weight_decay=0.0,
    normalize_weights=True,
    max_weight=None,
    random_state=42,
    **kwargs
):
    """Importance-weighted ERM using an estimated target/source density ratio."""

    if X_target is None:
        raise ValueError("Importance weighting requires unlabeled target data.")

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_source, y_source = np.asarray(X_source), np.asarray(y_source)
    X_target = np.asarray(X_target)

    classes = np.unique(y_source)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([class_to_idx[y] for y in y_source], dtype=np.int64)

    # Estimate w(x) = p_target(x) / p_source(x)
    ratio_estimator = iw_estimators.get_density_ratio_estimator(
        estimator,
        **(estimator_params or {})
    )

    weights = ratio_estimator.fit_predict(X_source, X_target)

    if max_weight is not None:
        weights = np.clip(weights, 0, max_weight)

    if normalize_weights:
        weights = weights / (weights.mean() + 1e-12)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_source, dtype=torch.float32),
        torch.tensor(y_encoded, dtype=torch.long),
        torch.tensor(weights, dtype=torch.float32)
    )

    loader = torch.utils.data.DataLoader(
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

    criterion = nn.CrossEntropyLoss(reduction="none")
    history = {"loss": []}

    for _ in range(epochs):
        model.train()
        total_loss = 0.0

        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)

            losses = criterion(model(xb), yb)
            loss = (wb * losses).sum() / wb.sum().clamp_min(1e-12)

            optimizer.zero_grad()
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
            "scenario": "domain_adaptation_unlabeled",
            "method": "importance_weighting",
            "estimator": estimator,
            "density_ratio_estimator": ratio_estimator,
            "importance_weights": weights,
            "weight_mean": float(weights.mean()),
            "weight_std": float(weights.std()),
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
            "target_labels_used": False,
            "n_target_samples": len(X_target),
        }
    )
