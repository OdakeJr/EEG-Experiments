import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath("../../../../"))

import lib.learning_algorithms.benchmark_models.baseline_nn as my_models
from lib.learning_algorithms.helper import wrap_model

def train_joint_supervised(
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
    target_weight=1.0,
    random_state=42,
    **kwargs
):
    """Joint supervised learning from labeled source and target data."""

    if X_target is None or y_target is None:
        raise ValueError("Joint supervised learning requires labeled target data.")

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_source, y_source = np.asarray(X_source), np.asarray(y_source)
    X_target, y_target = np.asarray(X_target), np.asarray(y_target)

    classes = np.unique(y_source)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    if not set(np.unique(y_target)).issubset(set(classes)):
        raise ValueError("Target contains classes absent from the source data.")

    ys = np.array([class_to_idx[y] for y in y_source], dtype=np.int64)
    yt = np.array([class_to_idx[y] for y in y_target], dtype=np.int64)

    source_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_source, dtype=torch.float32),
            torch.tensor(ys, dtype=torch.long)
        ),
        batch_size=batch_size, shuffle=True
    )

    target_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_target, dtype=torch.float32),
            torch.tensor(yt, dtype=torch.long)
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

    history = {
        "loss": [],
        "source_loss": [],
        "target_loss": []
    }

    for _ in range(epochs):
        model.train()

        source_iter, target_iter = iter(source_loader), iter(target_loader)
        n_steps = max(len(source_loader), len(target_loader))
        totals = np.zeros(3)

        for _ in range(n_steps):
            try:
                xs, ys_batch = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                xs, ys_batch = next(source_iter)

            try:
                xt, yt_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                xt, yt_batch = next(target_iter)

            xs, ys_batch = xs.to(device), ys_batch.to(device)
            xt, yt_batch = xt.to(device), yt_batch.to(device)

            source_loss = criterion(model(xs), ys_batch)
            target_loss = criterion(model(xt), yt_batch)

            loss = source_loss + target_weight * target_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totals += [
                loss.item(),
                source_loss.item(),
                target_loss.item()
            ]

        totals /= n_steps

        history["loss"].append(totals[0])
        history["source_loss"].append(totals[1])
        history["target_loss"].append(totals[2])

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
            "scenario": "domain_adaptation_labeled",
            "method": "joint_supervised",
            "target_labels_used": True,
            "n_target_samples": len(X_target),
            "target_weight": target_weight,
        }
    )

def train_target_supervised_dann(
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
    target_weight=1.0,
    discriminator_hidden_dim=128,
    random_state=42,
    **kwargs
):
    """Supervised DANN using labeled source and target data."""

    if X_target is None or y_target is None:
        raise ValueError("Supervised DANN requires labeled target data.")

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_source, y_source = np.asarray(X_source), np.asarray(y_source)
    X_target, y_target = np.asarray(X_target), np.asarray(y_target)

    classes = np.unique(y_source)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    ys = np.array([class_to_idx[y] for y in y_source], dtype=np.int64)
    yt = np.array([class_to_idx[y] for y in y_target], dtype=np.int64)

    source_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_source, dtype=torch.float32),
            torch.tensor(ys, dtype=torch.long)
        ),
        batch_size=batch_size, shuffle=True
    )

    target_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_target, dtype=torch.float32),
            torch.tensor(yt, dtype=torch.long)
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

    history = {
        "loss": [],
        "source_loss": [],
        "target_loss": [],
        "domain_loss": []
    }

    for _ in range(epochs):
        model.train()
        discriminator.train()

        source_iter, target_iter = iter(source_loader), iter(target_loader)
        n_steps = max(len(source_loader), len(target_loader))
        totals = np.zeros(4)

        for _ in range(n_steps):
            try:
                xs, ys_batch = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                xs, ys_batch = next(source_iter)

            try:
                xt, yt_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                xt, yt_batch = next(target_iter)

            xs, ys_batch = xs.to(device), ys_batch.to(device)
            xt, yt_batch = xt.to(device), yt_batch.to(device)

            fs = model.extract_features(xs)
            ft = model.extract_features(xt)

            source_loss = class_criterion(model.classifier(fs), ys_batch)
            target_loss = class_criterion(model.classifier(ft), yt_batch)

            features = torch.cat([fs, ft], dim=0)
            reversed_features = my_models.GradientReversal.apply(
                features, dann_lambda
            )

            domain_logits = discriminator(reversed_features).squeeze(1)
            domain_labels = torch.cat([
                torch.zeros(len(fs), device=device),
                torch.ones(len(ft), device=device)
            ])

            domain_loss = domain_criterion(domain_logits, domain_labels)

            loss = (
                source_loss
                + target_weight * target_loss
                + domain_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totals += [
                loss.item(),
                source_loss.item(),
                target_loss.item(),
                domain_loss.item()
            ]

        totals /= n_steps

        for key, value in zip(history, totals):
            history[key].append(value)

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
            "scenario": "domain_adaptation_labeled",
            "method": "target_supervised_dann",
            "domain_discriminator": discriminator,
            "target_labels_used": True,
            "n_target_samples": len(X_target),
        }
    )

def train_mme(
    X_source, y_source, source_domains=None,
    X_target=None, y_target=None, target_domains=None,
    pretrained_model=None,
    X_target_unlabeled=None,
    hidden_dims=(128, 128),
    activation="relu",
    dropout=0.0,
    batch_norm=False,
    epochs=50,
    lr=1e-3,
    batch_size=128,
    weight_decay=0.0,
    mme_lambda=0.1,
    temperature=0.05,
    random_state=42,
    **kwargs
):
    """Minimax Entropy (MME) for semi-supervised domain adaptation."""

    if X_target is None or y_target is None:
        raise ValueError("MME requires labeled target data.")
    if X_target_unlabeled is None:
        raise ValueError("MME also requires unlabeled target data.")

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_source, y_source = np.asarray(X_source), np.asarray(y_source)
    X_target, y_target = np.asarray(X_target), np.asarray(y_target)
    X_target_unlabeled = np.asarray(X_target_unlabeled)

    classes = np.unique(y_source)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    ys = np.array([class_to_idx[y] for y in y_source], dtype=np.int64)
    yt = np.array([class_to_idx[y] for y in y_target], dtype=np.int64)

    def make_loader(X, y=None):
        tensors = [torch.tensor(X, dtype=torch.float32)]
        if y is not None:
            tensors.append(torch.tensor(y, dtype=torch.long))
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*tensors),
            batch_size=batch_size, shuffle=True
        )

    source_loader = make_loader(X_source, ys)
    target_loader = make_loader(X_target, yt)
    unlabeled_loader = make_loader(X_target_unlabeled)

    model = my_models.MLP(
        input_dim=X_source.shape[1],
        hidden_dims=hidden_dims,
        output_dim=len(classes),
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm
    ).to(device)

    feature_dim = model.classifier.in_features
    model.classifier = my_models.CosineClassifier(
        feature_dim, len(classes), temperature
    ).to(device)

    optimizer_f = optim.Adam(
        model.feature_extractor.parameters(),
        lr=lr, weight_decay=weight_decay
    )
    optimizer_c = optim.Adam(
        model.classifier.parameters(),
        lr=lr, weight_decay=weight_decay
    )

    criterion = nn.CrossEntropyLoss()
    history = {"supervised_loss": [], "target_entropy": []}

    def entropy(logits):
        p = torch.softmax(logits, dim=1)
        return -(p * torch.log(p + 1e-8)).sum(dim=1).mean()

    for _ in range(epochs):
        model.train()

        loaders = [source_loader, target_loader, unlabeled_loader]
        iters = [iter(x) for x in loaders]
        n_steps = max(len(x) for x in loaders)

        total_sup = total_ent = 0.0

        for _ in range(n_steps):
            batches = []
            for i, loader in enumerate(loaders):
                try:
                    batch = next(iters[i])
                except StopIteration:
                    iters[i] = iter(loader)
                    batch = next(iters[i])
                batches.append(batch)

            (xs, ys_b), (xt, yt_b), (xu,) = batches
            xs, ys_b = xs.to(device), ys_b.to(device)
            xt, yt_b = xt.to(device), yt_b.to(device)
            xu = xu.to(device)

            # 1. Supervised source + labeled-target learning
            sup_loss = (
                criterion(model(xs), ys_b)
                + criterion(model(xt), yt_b)
            )

            optimizer_f.zero_grad()
            optimizer_c.zero_grad()
            sup_loss.backward()
            optimizer_f.step()
            optimizer_c.step()

            # 2. Classifier maximizes entropy on unlabeled target
            with torch.no_grad():
                zu = model.extract_features(xu)

            ent = entropy(model.classifier(zu))

            optimizer_c.zero_grad()
            (-mme_lambda * ent).backward()
            optimizer_c.step()

            # 3. Feature extractor minimizes target entropy
            for p in model.classifier.parameters():
                p.requires_grad = False

            zu = model.extract_features(xu)
            ent = entropy(model.classifier(zu))

            optimizer_f.zero_grad()
            (mme_lambda * ent).backward()
            optimizer_f.step()

            for p in model.classifier.parameters():
                p.requires_grad = True

            total_sup += sup_loss.item()
            total_ent += ent.item()

        history["supervised_loss"].append(total_sup / n_steps)
        history["target_entropy"].append(total_ent / n_steps)

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
            "scenario": "domain_adaptation_labeled",
            "method": "mme",
            "target_labels_used": True,
            "n_labeled_target": len(X_target),
            "n_unlabeled_target": len(X_target_unlabeled),
        }
    )
    
def train_lirr(
    X_source, y_source, source_domains=None,
    X_target=None, y_target=None, target_domains=None,
    pretrained_model=None,
    X_target_unlabeled=None,
    hidden_dims=(128, 128),
    activation="relu",
    dropout=0.0,
    batch_norm=False,
    epochs=50,
    lr=1e-3,
    batch_size=128,
    weight_decay=0.0,
    representation_lambda=1.0,
    risk_lambda=0.1,
    random_state=42,
    **kwargs
):
    """Learning Invariant Representations and Risks (LIRR)."""

    if X_target is None or y_target is None:
        raise ValueError("LIRR requires labeled target data.")
    if X_target_unlabeled is None:
        raise ValueError("LIRR requires unlabeled target data.")

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_source, y_source = np.asarray(X_source), np.asarray(y_source)
    X_target, y_target = np.asarray(X_target), np.asarray(y_target)
    X_target_unlabeled = np.asarray(X_target_unlabeled)

    classes = np.unique(y_source)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    ys = np.array([class_to_idx[y] for y in y_source], dtype=np.int64)
    yt = np.array([class_to_idx[y] for y in y_target], dtype=np.int64)

    def make_loader(X, y=None):
        tensors = [torch.tensor(X, dtype=torch.float32)]
        if y is not None:
            tensors.append(torch.tensor(y, dtype=torch.long))

        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*tensors),
            batch_size=batch_size,
            shuffle=True
        )

    source_loader = make_loader(X_source, ys)
    target_loader = make_loader(X_target, yt)
    unlabeled_loader = make_loader(X_target_unlabeled)

    model = my_models.MLP(
        input_dim=X_source.shape[1],
        hidden_dims=hidden_dims,
        output_dim=len(classes),
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm
    ).to(device)

    feature_dim = model.classifier.in_features

    environment_predictor = my_models.EnvironmentPredictor(
        feature_dim, len(classes)
    ).to(device)

    discriminator = my_models.DomainDiscriminator(
        feature_dim
    ).to(device)

    optimizer = optim.Adam(
        list(model.parameters())
        + list(environment_predictor.parameters())
        + list(discriminator.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    history = {
        "loss": [],
        "invariant_loss": [],
        "environment_loss": [],
        "risk_penalty": [],
        "transfer_loss": [],
    }

    for _ in range(epochs):
        model.train()
        environment_predictor.train()
        discriminator.train()

        loaders = [source_loader, target_loader, unlabeled_loader]
        iterators = [iter(loader) for loader in loaders]
        n_steps = max(len(loader) for loader in loaders)

        totals = np.zeros(5)

        for _ in range(n_steps):
            batches = []

            for i, loader in enumerate(loaders):
                try:
                    batch = next(iterators[i])
                except StopIteration:
                    iterators[i] = iter(loader)
                    batch = next(iterators[i])

                batches.append(batch)

            (xs, ys_b), (xt, yt_b), (xu,) = batches

            xs, ys_b = xs.to(device), ys_b.to(device)
            xt, yt_b = xt.to(device), yt_b.to(device)
            xu = xu.to(device)

            fs = model.extract_features(xs)
            ft = model.extract_features(xt)
            fu = model.extract_features(xu)

            # Domain-invariant prediction risk
            source_loss = criterion(model.classifier(fs), ys_b)
            target_loss = criterion(model.classifier(ft), yt_b)

            invariant_loss = 0.5 * (
                source_loss + target_loss
            )

            # Domain/environment-specific prediction risk
            env_source = criterion(
                environment_predictor(fs, "source"), ys_b
            )
            env_target = criterion(
                environment_predictor(ft, "target"), yt_b
            )

            environment_loss = 0.5 * (
                env_source + env_target
            )

            risk_penalty = torch.abs(
                invariant_loss - environment_loss
            )

            # Domain-invariant representation learning
            features = torch.cat([fs, fu], dim=0)

            reversed_features = my_models.GradientReversal.apply(
                features, representation_lambda
            )

            domain_logits = discriminator(
                reversed_features
            ).squeeze(1)

            domain_labels = torch.cat([
                torch.zeros(len(fs), device=device),
                torch.ones(len(fu), device=device)
            ])

            transfer_loss = nn.functional.binary_cross_entropy_with_logits(
                domain_logits,
                domain_labels
            )

            loss = (
                invariant_loss
                + risk_lambda * risk_penalty
                + transfer_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totals += [
                loss.item(),
                invariant_loss.item(),
                environment_loss.item(),
                risk_penalty.item(),
                transfer_loss.item(),
            ]

        totals /= n_steps

        for key, value in zip(history, totals):
            history[key].append(value)

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
            "scenario": "domain_adaptation_labeled",
            "method": "lirr",
            "environment_predictor": environment_predictor,
            "domain_discriminator": discriminator,
            "target_labels_used": True,
            "n_labeled_target": len(X_target),
            "n_unlabeled_target": len(X_target_unlabeled),
        }
    )


