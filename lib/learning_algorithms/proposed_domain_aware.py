import benchmark_architectures as my_models

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
from scipy.stats import mode


class DomainMajoritySVM:
    def __init__(self):
        self.models = []
        self.unique_domains = None

    def fit(self, X_train, y_train, domains_train):
        self.unique_domains = np.unique(domains_train)
        self.models = []

        for d in self.unique_domains:
            mask = domains_train == d
            model_d = SVC(kernel="rbf", probability=True)
            model_d.fit(X_train[mask], y_train[mask])
            self.models.append(model_d)

    def predict(self, X):
        all_preds = np.array([model.predict(X) for model in self.models])
        majority_preds = mode(all_preds, axis=0).mode[0]
        return majority_preds

    def predict_proba(self, X):
        all_probs = np.array([model.predict_proba(X) for model in self.models])
        mean_probs = all_probs.mean(axis=0)
        return mean_probs[:, 1]


def train_domain_majority_svm(
    X_train, y_train, domains_train,
    **kwargs
):

    model = DomainMajoritySVM()
    model.fit(X_train, y_train, domains_train)

    return {
        "model": model,
        "predict_proba": lambda X: model.predict_proba(X),
        "predict": lambda X: model.predict(X)
    }
        

# =========================
# Train SVM Region
# =========================
def train_svm_region(x1, x2):
    if x2.ndim == 1:
        x2 = x2.reshape(-1, 1)
    X = np.hstack([x1, x2])
    svm = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
    svm.fit(X)
    return svm


# =========================
# Differentiable SVM Wrapper
# =========================

class SVMPremise(nn.Module):
    def __init__(self, svm_model):
        super().__init__()

        self.support_vectors = torch.tensor(
            svm_model.support_vectors_, dtype=torch.float32
        )
        self.alpha = torch.tensor(
            svm_model.dual_coef_.flatten(), dtype=torch.float32
        )
        self.rho = torch.tensor(
            svm_model.intercept_[0], dtype=torch.float32
        )
        self.gamma = svm_model._gamma

    def rbf_kernel(self, X1, X2):
        sq1 = (X1**2).sum(dim=1, keepdim=True)
        sq2 = (X2**2).sum(dim=1, keepdim=True)
        sqdist = sq1 + sq2.T - 2 * X1 @ X2.T
        return torch.exp(-self.gamma * sqdist)

    def raw_decision(self, x1, x2_pred):
        X = torch.cat([x1, x2_pred], dim=1)
        K = self.rbf_kernel(self.support_vectors, X)
        f = self.alpha @ K + self.rho
        return f

    def forward(self, x1, x2_pred):
        f = self.raw_decision(x1, x2_pred)
        return torch.relu(-f)   # penalty

def train_nn_svm_premise_multidomain(
    X_train, y_train, domains_train,
    mode="combined",  # "plain", "combined", "premise_only"
    lambda_premise=0.1,
    hidden_dim=128,
    epochs=50,
    lr=1e-3,
    batch_size=128,
    **kwargs
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = my_models.SimpleNN(X_train.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # -------------------------
    # One SVM per domain
    # -------------------------
    premises = []
    unique_domains = np.unique(domains_train)

    if mode in ["combined", "premise_only"]:
        for d in unique_domains:
            mask = domains_train == d
            svm_d = train_svm_region(X_train[mask], y_train[mask])
            premises.append(SVMPremise(svm_d).to(device))

    n_domains = len(premises)

    model.train()

    for _ in range(epochs):
        for xb, yb in loader:

            optimizer.zero_grad()
            logits = model(xb)

            if mode == "plain":
                loss = criterion(logits, yb)

            else:
                total_premise_loss = 0.0

                for premise in premises:
                    total_premise_loss += premise(xb, logits).mean()

                total_premise_loss /= n_domains

                if mode == "combined":
                    task_loss = criterion(logits, yb)
                    loss = task_loss + lambda_premise * total_premise_loss
                else:
                    loss = total_premise_loss

            loss.backward()
            optimizer.step()

    def predict_proba(X):
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs

    def predict(X):
        probs = predict_proba(X)
        return (probs >= 0.5).astype(int)

    return {
        "model": model,
        "predict_proba": predict_proba,
        "predict": predict
    }