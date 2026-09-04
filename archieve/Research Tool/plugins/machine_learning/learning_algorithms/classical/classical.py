import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

sys.path.append(os.path.abspath("../../../../"))

import lib.learning_algorithms.benchmark_models.baseline_nn as my_models
from lib.learning_algorithms.helper import wrap_model


# ============================================================
# 1. Logistic Regression
# ============================================================

def train_logistic_regression(
    X_source, y_source, source_domains=None,
    X_target=None, y_target=None, target_domains=None,
    pretrained_model=None,
    max_iter=1000, C=1.0, **kwargs
):
    model = LogisticRegression(max_iter=max_iter, C=C)
    model.fit(X_source, y_source)

    return wrap_model(
        model, model.predict, model.predict_proba, model.classes_,
        artifacts={"scenario": "classical", "method": "logistic_regression"}
    )


# ============================================================
# 2. Random Forest
# ============================================================

def train_random_forest(
    X_source, y_source, source_domains=None,
    X_target=None, y_target=None, target_domains=None,
    pretrained_model=None,
    n_estimators=200, max_depth=None, random_state=42, **kwargs
):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state
    )
    model.fit(X_source, y_source)

    return wrap_model(
        model, model.predict, model.predict_proba, model.classes_,
        artifacts={"scenario": "classical", "method": "random_forest"}
    )


# ============================================================
# 3. RBF-SVM
# ============================================================

def train_svm(
    X_source, y_source, source_domains=None,
    X_target=None, y_target=None, target_domains=None,
    pretrained_model=None,
    C=1.0, gamma="scale", **kwargs
):
    model = SVC(kernel="rbf", C=C, gamma=gamma, probability=True)
    model.fit(X_source, y_source)

    return wrap_model(
        model, model.predict, model.predict_proba, model.classes_,
        artifacts={"scenario": "classical", "method": "rbf_svm"}
    )


# ============================================================
# 4. Multilayer Perceptron (ERM)
# ============================================================

def train_mlp(
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
    **kwargs
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = np.unique(y_source)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([class_to_idx[y] for y in y_source], dtype=np.int64)

    X = torch.tensor(X_source, dtype=torch.float32)
    y = torch.tensor(y_encoded, dtype=torch.long)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=True
    )

    model = my_models.MLP(
        input_dim=X_source.shape[1],
        hidden_dims=hidden_dims,
        output_dim=len(classes),
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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

        history["loss"].append(total_loss / len(X))

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
        artifacts={"scenario": "classical", "method": "mlp"}
    )