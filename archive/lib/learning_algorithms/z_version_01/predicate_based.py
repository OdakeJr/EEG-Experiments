import sys
import os
import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath("../../../"))
PROJECT_ROOT = "../../../"

import archieve.lib.learning_algorithms.z_version_01.benchmark_architectures as my_models


# ============================================================
# Base predicate interface
# ============================================================

class BasePredicate:
    def fit(self, X, y=None, domains=None):
        return self

    def transform(self, X):
        raise NotImplementedError

    def penalty(self, probs, y_onehot, X_batch=None, batch_indices=None):
        return None

    @property
    def name(self):
        return self.__class__.__name__


# ============================================================
# Vapnik-style predicates
# These generate Phi blocks
# ============================================================

class BiasPredicate(BasePredicate):
    def transform(self, X):
        return np.ones((X.shape[0], 1), dtype=float)


class LinearPredicate(BasePredicate):
    def transform(self, X):
        return np.asarray(X, dtype=float)


class QuadraticPredicate(BasePredicate):
    def __init__(self, include_diagonal=True):
        self.include_diagonal = include_diagonal

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        cols = []

        for j in range(d):
            start_k = j if self.include_diagonal else j + 1
            for k in range(start_k, d):
                cols.append((X[:, j] * X[:, k]).reshape(-1, 1))

        if not cols:
            return np.empty((n, 0), dtype=float)

        return np.hstack(cols)


class SquaredOnlyPredicate(BasePredicate):
    def transform(self, X):
        return np.asarray(X, dtype=float) ** 2


class PairwiseProductsPredicate(BasePredicate):
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        cols = []

        for j in range(d):
            for k in range(j + 1, d):
                cols.append((X[:, j] * X[:, k]).reshape(-1, 1))

        if not cols:
            return np.empty((n, 0), dtype=float)

        return np.hstack(cols)


# ============================================================
# Data-driven class-conditional Gaussian predicate
# penalty: encourages model probabilities to align with
# Gaussian class scores estimated from training data
# ============================================================
class GaussianFeatureClassPredicate(BasePredicate):
    """
    One Gaussian per feature, per class.

    For sample x_i with true class y_i = c:
        phi_i = phi_c(x_i)

    Predicate penalty:
        (phi_i * p_theta(y_i|x_i) - phi_i)^2
    which is equivalent to:
        phi_i^2 * (p_theta(y_i|x_i) - 1)^2
    """

    def __init__(self, eps=1e-6, use_log=True, reduce="mean"):
        self.eps = eps
        self.use_log = use_log
        self.reduce = reduce

        self.classes_ = None
        self.class_to_idx_ = None
        self.means_ = None   # [C, d]
        self.vars_ = None    # [C, d]

    def fit(self, X, y=None, domains=None):
        if y is None:
            raise ValueError("GaussianFeatureClassPredicate requires y in fit().")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}

        means = []
        vars_ = []

        for c in self.classes_:
            Xc = X[y == c]
            mu = Xc.mean(axis=0)
            var = Xc.var(axis=0) + self.eps
            means.append(mu)
            vars_.append(var)

        self.means_ = np.stack(means, axis=0)   # [C, d]
        self.vars_ = np.stack(vars_, axis=0)    # [C, d]
        return self

    def _per_feature_logpdf_for_class(self, X, class_idx):
        """
        X: [n, d]
        returns: [n, d]
        """
        mu = self.means_[class_idx]   # [d]
        var = self.vars_[class_idx]   # [d]

        return -0.5 * (
            np.log(2.0 * np.pi * var) + ((X - mu) ** 2) / var
        )

    def transform(self, X):
        """
        Returns one predicate score per class:
            scores[i, c] = aggregate_j log p(x_ij | class=c)
        shape: [n, C]
        """
        X = np.asarray(X, dtype=float)
        n, _ = X.shape
        C = self.means_.shape[0]

        scores = np.zeros((n, C), dtype=float)

        for c in range(C):
            logpdf = self._per_feature_logpdf_for_class(X, c)  # [n, d]

            if self.reduce == "sum":
                scores[:, c] = np.sum(logpdf, axis=1)
            else:
                scores[:, c] = np.mean(logpdf, axis=1)

        return scores

    def true_class_predicate(self, X, y_labels):
        """
        For each sample i, use only the predicate of its true class.
        returns: [n]
        """
        X = np.asarray(X, dtype=float)
        y_labels = np.asarray(y_labels)

        n = X.shape[0]
        phi = np.zeros(n, dtype=float)

        for i in range(n):
            c_idx = self.class_to_idx_[y_labels[i]]
            logpdf = self._per_feature_logpdf_for_class(X[i:i+1], c_idx)  # [1, d]

            if self.reduce == "sum":
                phi[i] = np.sum(logpdf, axis=1)[0]
            else:
                phi[i] = np.mean(logpdf, axis=1)[0]

        return phi

    def penalty(self, probs, y_onehot, X_batch=None, batch_indices=None):
        """
        Vapnik-like true-class predicate penalty:

            mean_i ( phi_i * p_i,true - phi_i * 1 )^2

        = mean_i ( phi_i * p_i,true - phi_i )^2
        """
        if X_batch is None:
            raise ValueError("GaussianFeatureClassPredicate.penalty requires X_batch.")

        X_np = X_batch.detach().cpu().numpy()

        y_idx = torch.argmax(y_onehot, dim=1)              # [B]
        y_idx_np = y_idx.detach().cpu().numpy()
        y_labels = self.classes_[y_idx_np]

        phi_np = self.true_class_predicate(X_np, y_labels)  # [B]
        phi = torch.tensor(phi_np, dtype=probs.dtype, device=probs.device)

        p_true = torch.sum(probs * y_onehot, dim=1)         # [B]

        loss = torch.mean((phi * p_true - phi) ** 2)
        return loss


# ============================================================
# Predicate collection helpers
# ============================================================

def fit_predicates(predicate_list, X, y=None, domains=None):
    if predicate_list is None:
        return []

    fitted = []
    for pred in predicate_list:
        fitted.append(pred.fit(X, y=y, domains=domains))
    return fitted


def build_predicate_matrix_from_objects(
    X,
    predicate_list,
    normalize_columns=True,
    return_info=False
):
    if predicate_list is None:
        return (None, []) if return_info else None

    blocks = []
    info = []

    for pred in predicate_list:
        block = pred.transform(X)

        if block is None:
            continue

        block = np.asarray(block, dtype=float)

        if block.ndim == 1:
            block = block.reshape(-1, 1)

        if block.shape[0] != X.shape[0]:
            raise ValueError(
                f"Predicate '{pred.name}' returned {block.shape[0]} rows, "
                f"but X has {X.shape[0]} samples."
            )

        if block.shape[1] == 0:
            continue

        blocks.append(block)
        info.append((pred.name, block.shape[1]))

    if not blocks:
        Phi = np.empty((X.shape[0], 0), dtype=float)
    else:
        Phi = np.hstack(blocks)

    if normalize_columns and Phi.shape[1] > 0:
        norms = np.linalg.norm(Phi, axis=0, keepdims=True)
        Phi = Phi / (norms + 1e-12)

    if return_info:
        return Phi, info
    return Phi


def split_predicates(predicate_list):
    matrix_preds = []
    custom_preds = []

    if predicate_list is None:
        return matrix_preds, custom_preds

    for pred in predicate_list:
        has_custom_penalty = pred.__class__.penalty is not BasePredicate.penalty
        if has_custom_penalty:
            custom_preds.append(pred)
        else:
            matrix_preds.append(pred)

    return matrix_preds, custom_preds


DEFAULT_VAPNIK_PREDICATES = [
    BiasPredicate(),
    LinearPredicate(),
    QuadraticPredicate()
]


# ============================================================
# Predicate-Regularized SVM
# Only matrix-style predicates are used here
# ============================================================

class PredicateSVM:
    def __init__(self, C=1.0, tau=0.0, kernel="rbf", gamma=1.0):
        self.C = C
        self.tau = tau
        self.kernel = kernel
        self.gamma = gamma

        self.alpha = None
        self.b = None
        self.X_train = None
        self.y_train = None

    def _compute_kernel(self, X1, X2):
        if self.kernel == "linear":
            return X1 @ X2.T

        if self.kernel == "rbf":
            X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
            X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
            sq_dists = X1_sq - 2 * X1 @ X2.T + X2_sq.T
            return np.exp(-self.gamma * sq_dists)

        raise ValueError("Unsupported kernel.")

    def fit(self, X, y, Phi=None):
        self.X_train = X
        self.y_train = y

        n = X.shape[0]
        K = self._compute_kernel(X, X)
        Q = np.outer(y, y) * K

        if Phi is not None and Phi.shape[1] > 0 and self.tau > 0:
            P = (Phi @ Phi.T) / Phi.shape[1]
            Q = Q + self.tau * P

        Q = (Q + Q.T) / 2
        Q = cp.psd_wrap(Q)

        alpha = cp.Variable(n)

        objective = cp.Minimize(0.5 * cp.quad_form(alpha, Q) - cp.sum(alpha))
        constraints = [
            alpha >= 0,
            alpha <= self.C,
            y @ alpha == 0
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)

        self.alpha = alpha.value

        support = self.alpha > 1e-5
        K_support = self._compute_kernel(X[support], X)
        self.b = np.mean(y[support] - K_support @ (self.alpha * y))

        return self

    def decision_function(self, X_new):
        K_new = self._compute_kernel(X_new, self.X_train)
        return K_new @ (self.alpha * self.y_train) + self.b

    def predict(self, X_new):
        return np.sign(self.decision_function(X_new))


def train_predicate_svm(
    X_train,
    y_train,
    domains_train=None,
    C=1.0,
    tau=0.0,
    kernel="rbf",
    gamma=1.0,
    predicates=None,
    normalize_predicates=True,
    **kwargs
):
    matrix_preds, custom_preds = split_predicates(predicates)

    if len(custom_preds) > 0:
        raise ValueError("Custom-penalty predicates are not supported in train_predicate_svm.")

    fitted_preds = fit_predicates(matrix_preds, X_train, y=y_train, domains=domains_train)

    Phi, predicate_info = build_predicate_matrix_from_objects(
        X_train,
        fitted_preds,
        normalize_columns=normalize_predicates,
        return_info=True
    )

    y_unique = np.unique(y_train)
    if len(y_unique) != 2:
        raise ValueError("train_predicate_svm currently supports only binary classification.")

    y_svm = 2 * y_train - 1

    model = PredicateSVM(C=C, tau=tau, kernel=kernel, gamma=gamma)
    model.fit(X_train, y_svm, Phi=Phi if tau > 0 else None)

    def predict_proba(X):
        decision = model.decision_function(X)
        return 1 / (1 + np.exp(-decision))

    def predict(X):
        return (predict_proba(X) >= 0.5).astype(int)

    return {
        "model": model,
        "predict_proba": predict_proba,
        "predict": predict,
        "predicate_info": predicate_info,
        "predicates": fitted_preds
    }


# ============================================================
# Neural net trainer
# Supports:
# - matrix predicates (Vapnik style)
# - custom penalty predicates (Gaussian class style)
# ============================================================

def train_predicate_nn_pipeline(
    X_train, y_train, domains_train=None,
    hidden_dim=128,
    epochs=50,
    lr=1e-3,
    batch_size=128,
    tau=0.0,
    predicates=None,
    normalize_predicates=True,
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

    matrix_preds, custom_preds = split_predicates(predicates)

    fitted_matrix_preds = fit_predicates(matrix_preds, X_train, y=y_train, domains=domains_train)
    fitted_custom_preds = fit_predicates(custom_preds, X_train, y=y_train, domains=domains_train)

    Phi, matrix_info = build_predicate_matrix_from_objects(
        X_train,
        fitted_matrix_preds,
        normalize_columns=normalize_predicates,
        return_info=True
    )

    predicate_info = matrix_info + [(pred.name, "custom_penalty") for pred in fitted_custom_preds]

    Phi_tensor = None
    Y_onehot_tensor = None

    if Phi is not None and Phi.shape[1] > 0 and tau > 0:
        Phi_tensor = torch.tensor(Phi, dtype=torch.float32).to(device)

        Y_onehot = np.zeros((len(y_encoded), n_classes), dtype=np.float32)
        Y_onehot[np.arange(len(y_encoded)), y_encoded] = 1.0
        Y_onehot_tensor = torch.tensor(Y_onehot, dtype=torch.float32).to(device)

    idx_tensor = torch.arange(len(X_train), dtype=torch.long).to(device)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, idx_tensor)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model.train()
    for _ in range(epochs):
        for xb, yb, ib in loader:
            optimizer.zero_grad()

            logits = model(xb)
            loss_task = criterion(logits, yb)
            loss = loss_task

            if tau > 0:
                probs = torch.softmax(logits, dim=1)

                if Phi_tensor is not None:
                    Phi_b = Phi_tensor[ib]
                    if Y_onehot_tensor is None:
                        Y_onehot = np.zeros((len(y_encoded), n_classes), dtype=np.float32)
                        Y_onehot[np.arange(len(y_encoded)), y_encoded] = 1.0
                        Y_onehot_tensor = torch.tensor(Y_onehot, dtype=torch.float32).to(device)

                    Y_b = Y_onehot_tensor[ib]

                    pred_stats = Phi_b.T @ probs
                    true_stats = Phi_b.T @ Y_b

                    loss_matrix = torch.mean((pred_stats - true_stats) ** 2)
                    loss = loss + tau * loss_matrix

                for pred in fitted_custom_preds:
                    Y_b = torch.nn.functional.one_hot(yb, num_classes=n_classes).float()
                    loss_custom = pred.penalty(
                        probs=probs,
                        y_onehot=Y_b,
                        X_batch=xb,
                        batch_indices=ib
                    )
                    if loss_custom is not None:
                        loss = loss + tau * loss_custom

            loss.backward()
            optimizer.step()

    idx_to_class = np.array(classes)

    def predict_proba(X):
        model.eval()
        X_eval = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(X_eval)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(X):
        probs = predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        return idx_to_class[pred_idx]

    return {
        "model": model,
        "predict_proba": predict_proba,
        "predict": predict,
        "predicate_info": predicate_info,
        "predicates": fitted_matrix_preds + fitted_custom_preds
    }