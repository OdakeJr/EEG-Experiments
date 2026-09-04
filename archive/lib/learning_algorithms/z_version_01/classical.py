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
PROJECT_ROOT = "../../../../"

import lib.learning_algorithms.z_version_01.benchmark_architectures as my_models

# ==========================================================
# 1) Logistic Regression
# ==========================================================

def train_logistic_regression(
    X_train, y_train, domains_train=None,
    max_iter=1000,
    C=1.0,
    **kwargs
):

    model = LogisticRegression(
        max_iter=max_iter,
        C=C
    )
    model.fit(X_train, y_train)

    return {
        "model": model,
        "predict_proba": lambda X: model.predict_proba(X),
        "predict": lambda X: model.predict(X)
    }


# ==========================================================
# 2) Random Forest
# ==========================================================

def train_random_forest(
    X_train, y_train, domains_train=None,
    n_estimators=200,
    max_depth=None,
    random_state=42,
    **kwargs
):

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    return {
        "model": model,
        "predict_proba": lambda X: model.predict_proba(X),
        "predict": lambda X: model.predict(X)
    }


# ==========================================================
# 3) SVM (RBF)
# ==========================================================

def train_svm(
    X_train, y_train, domains_train=None,
    C=1.0,
    gamma="scale",
    **kwargs
):

    model = SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        probability=True
    )
    model.fit(X_train, y_train)

    return {
        "model": model,
        "predict_proba": lambda X: model.predict_proba(X),
        "predict": lambda X: model.predict(X)
    }


# ==========================================================
# 4) Deep Neural Network (ERM)
# ==========================================================

def train_nn_erm(
    X_train, y_train, domains_train=None,
    hidden_dim=128,
    epochs=50,
    lr=1e-3,
    batch_size=128,
    **kwargs
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = np.unique(y_train)
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([class_to_idx[y] for y in y_train], dtype=np.int64)

    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long).to(device)

    model = my_models.SimpleNN(
        X_train.shape[1],
        hidden_dim=hidden_dim,
        output_dim=n_classes
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    idx_to_class = np.array(classes)

    def predict_proba(X):
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(X):
        probs = predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        return idx_to_class[pred_idx]

    return {
        "model": model,
        "predict_proba": predict_proba,
        "predict": predict
    }


# ==========================================================
# L2 Regularized NN
# ==========================================================

def train_nn_l2(
    X_train, y_train, domains_train=None,
    hidden_dim=128,
    epochs=50,
    lr=1e-3,
    batch_size=128,
    weight_decay=1e-4,
    **kwargs
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = np.unique(y_train)
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([class_to_idx[y] for y in y_train], dtype=np.int64)

    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long).to(device)

    model = my_models.SimpleNN(
        input_dim=X_train.shape[1],
        hidden_dim=hidden_dim,
        output_dim=n_classes
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    idx_to_class = np.array(classes)

    def predict_proba(X):
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(X):
        probs = predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        return idx_to_class[pred_idx]

    return {
        "model": model,
        "predict_proba": predict_proba,
        "predict": predict
    }


# ==========================================================
# Dropout Network
# ==========================================================

def train_nn_dropout(
    X_train, y_train, domains_train=None,
    hidden_dim=128,
    epochs=50,
    lr=1e-3,
    batch_size=128,
    dropout_p=0.5,
    **kwargs
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = np.unique(y_train)
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([class_to_idx[y] for y in y_train], dtype=np.int64)

    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long).to(device)

    model = my_models.DropoutNN(
        input_dim=X_train.shape[1],
        hidden_dim=hidden_dim,
        dropout_p=dropout_p,
        output_dim=n_classes
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    idx_to_class = np.array(classes)

    def predict_proba(X):
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(X):
        probs = predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        return idx_to_class[pred_idx]

    return {
        "model": model,
        "predict_proba": predict_proba,
        "predict": predict
    }