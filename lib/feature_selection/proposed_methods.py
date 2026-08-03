from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import (
    linear_kernel,
    rbf_kernel,
    polynomial_kernel,
    cosine_similarity
)
import numpy as np

# ==========================================================
# Matrices
# ==========================================================
def compute_feature_matrix(
        X,
        y=None,
        domains=None,
        mode="global",                 # global | per_state | per_subject | per_subject_state
        method="correlation",          # correlation | covariance | linear | rbf | poly | cosine
        normalize=True,
        states=None,
        kernel_params=None,
        kernel_normalize=True,
        kernel_center=False
):
    
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    results = {}

    # ----------------------------------------------------------
    # Kernel utilities
    # ----------------------------------------------------------
    def normalize_kernel(K):
        diag = np.sqrt(np.diag(K) + 1e-12)
        return K / (diag[:, None] * diag[None, :])

    def center_kernel(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    # ----------------------------------------------------------
    # Core matrix computation
    # ----------------------------------------------------------
    def compute_matrix(X_subset):

        # ---------------------------
        # Linear statistics
        # ---------------------------
        if method == "correlation":
            return np.corrcoef(X_subset, rowvar=False)

        elif method == "covariance":
            return np.cov(X_subset, rowvar=False)

        # ---------------------------
        # Kernel methods (feature space)
        # ---------------------------
        else:
            Xf = X_subset.T   # shape: (n_features, n_samples)

            params = kernel_params or {}

            if method == "linear":
                K = linear_kernel(Xf, Xf)

            elif method == "rbf":
                gamma = params.get("gamma", 1.0 / Xf.shape[1])
                K = rbf_kernel(Xf, Xf, gamma=gamma)

            elif method == "poly":
                degree = params.get("degree", 3)
                gamma = params.get("gamma", 1.0 / Xf.shape[1])
                coef0 = params.get("coef0", 1.0)
                K = polynomial_kernel(Xf, Xf, degree=degree, gamma=gamma, coef0=coef0)

            elif method == "cosine":
                K = cosine_similarity(Xf, Xf)

            else:
                raise ValueError(f"Unknown method: {method}")

            # Optional normalization (recommended)
            if kernel_normalize:
                K = normalize_kernel(K)

            # Optional centering (more rigorous)
            if kernel_center:
                K = center_kernel(K)

            return K

    # ==========================================================
    # GLOBAL
    # ==========================================================
    if mode == "global":
        results["global"] = compute_matrix(X)

    # ==========================================================
    # PER STATE
    # ==========================================================
    elif mode == "per_state":

        unique_states = states if states is not None else np.unique(y)

        for s in unique_states:
            mask = (y == s)
            if np.sum(mask) < 2:
                continue

            results[f"state_{s}"] = compute_matrix(X[mask])

    # ==========================================================
    # PER SUBJECT
    # ==========================================================
    elif mode == "per_subject":

        unique_domains = np.unique(domains)

        for d in unique_domains:
            mask = (domains == d)
            if np.sum(mask) < 2:
                continue

            results[f"subject_{d}"] = compute_matrix(X[mask])

    # ==========================================================
    # PER SUBJECT + STATE
    # ==========================================================
    elif mode == "per_subject_state":

        unique_domains = np.unique(domains)
        unique_states = states if states is not None else np.unique(y)

        for d in unique_domains:
            for s in unique_states:

                mask = (domains == d) & (y == s)

                if np.sum(mask) < 2:
                    continue

                key = f"subject_{d}_state_{s}"
                results[key] = compute_matrix(X[mask])

    else:
        raise ValueError("Unknown mode")

    return results


# ==========================================================
# Prunning
# ==========================================================
def progressive_feature_pruning_generic(corr_dict, metric_fn, N_remove=10, K_keep=50):

    corr_list = list(corr_dict.values())
    d = corr_list[0].shape[0]

    kept = np.arange(d)

    # --- one-shot mode ---
    if N_remove == -1:
        scores = metric_fn(corr_dict, kept)
        ranked = np.argsort(scores)[::-1]
        return ranked[:K_keep]

    # --- progressive mode ---
    while len(kept) > K_keep:

        scores = metric_fn(corr_dict, kept)

        remaining = len(kept) - K_keep
        n_remove = min(N_remove, remaining)

        worst = np.argsort(scores)[:n_remove]
        kept = np.delete(kept, worst)

    return kept


# ==========================================================
# 1) DIFS (Domain Invariance)
# ==========================================================
def metric_dispersion(corr_dict, kept):
    """
    Requires: mode="per_subject" or compatible
    Goal: low variance across domains
    """
    scores = []

    for i in kept:
        vals = []
        for key, C in corr_dict.items():
            if "subject_" not in key:
                continue
            vals.extend(C[i, kept])

        scores.append(-np.var(vals))

    return np.array(scores)


# ==========================================================
# 2) SSFS (State Separation)
# ==========================================================
def metric_state_separation(corr_dict, kept, eps=1e-12):
    """
    Multi-class SSFS using only class/state matrices.

    Requires:
        mode="per_state"

    Idea:
        - one matrix per class
        - compute pairwise class separation
        - normalize each pairwise difference
        - sum separations across all class pairs
    """
    states = [k for k in corr_dict if "state_" in k and "subject_" not in k]

    if len(states) < 2:
        raise ValueError("SSFS requires at least two states.")

    scores = np.zeros(len(kept), dtype=float)

    for a in range(len(states)):
        for b in range(a + 1, len(states)):
            A = corr_dict[states[a]]
            B = corr_dict[states[b]]

            A_sub = A[np.ix_(kept, kept)]
            B_sub = B[np.ix_(kept, kept)]

            diff = A_sub - B_sub
            #scale = np.abs(A_sub) + np.abs(B_sub) + eps

            #scores += np.linalg.norm(diff / scale, axis=1)
            scores += np.linalg.norm(diff, axis=1)

    return scores

# ==========================================================
# 3) Mean Separation
# ==========================================================
def metric_mean_separation(corr_dict, kept, eps=1e-12):
    """
    Requires: mode="per_subject_state"

    Idea:
        1. Group matrices by state
        2. Average subject/domain matrices inside each state
        3. Compute normalized pairwise separation among all states
    """
    groups = {}

    for key, C in corr_dict.items():
        if "state_" not in key or "subject_" not in key:
            continue
        state = key.split("state_")[-1]
        groups.setdefault(state, []).append(C)

    if len(groups) < 2:
        raise ValueError("Mean separation requires at least 2 states.")

    states = list(groups.keys())
    mean_mats = {s: np.mean(groups[s], axis=0) for s in states}

    scores = np.zeros(len(kept), dtype=float)

    for a in range(len(states)):
        for b in range(a + 1, len(states)):
            A = mean_mats[states[a]][np.ix_(kept, kept)]
            B = mean_mats[states[b]][np.ix_(kept, kept)]

            diff = A - B
            #scale = np.abs(A) + np.abs(B) + eps

            #scores += np.linalg.norm(diff / scale, axis=1)
            scores += np.linalg.norm(diff, axis=1)

    return scores


# ==========================================================
# 4) Wasserstein-based separation
# ==========================================================
def metric_wasserstein(corr_dict, kept):
    """
    Requires: mode="per_subject_state"
    Multiclass Wasserstein-like separation (no normalization).
    """
    groups = {}

    for key, C in corr_dict.items():
        if "state_" not in key or "subject_" not in key:
            continue
        state = key.split("state_")[-1]
        groups.setdefault(state, []).append(C)

    if len(groups) < 2:
        raise ValueError("Wasserstein requires at least 2 states.")

    states = list(groups.keys())
    state_arrays = {s: np.array(groups[s]) for s in states}

    scores = []

    for i in kept:
        score_i = 0.0

        for a in range(len(states)):
            for b in range(a + 1, len(states)):
                A = state_arrays[states[a]]
                B = state_arrays[states[b]]

                mu_a = A[:, i][:, kept].mean(axis=0)
                mu_b = B[:, i][:, kept].mean(axis=0)

                std_a = np.sqrt(A[:, i][:, kept].var(axis=0))
                std_b = np.sqrt(B[:, i][:, kept].var(axis=0))

                d = np.sqrt((mu_a - mu_b)**2 + (std_a - std_b)**2)

                score_i += np.mean(d)

        scores.append(score_i)

    return np.array(scores)


# ==========================================================
# 5) Bhattacharyya-based separation
# ==========================================================
def metric_bhattacharyya(corr_dict, kept):
    """
    Requires: mode="per_subject_state"
    Multiclass Bhattacharyya separation (no extra normalization).
    """
    groups = {}

    for key, C in corr_dict.items():
        if "state_" not in key or "subject_" not in key:
            continue
        state = key.split("state_")[-1]
        groups.setdefault(state, []).append(C)

    if len(groups) < 2:
        raise ValueError("Bhattacharyya requires at least 2 states.")

    states = list(groups.keys())
    state_arrays = {s: np.array(groups[s]) for s in states}

    scores = []

    for i in kept:
        score_i = 0.0

        for a in range(len(states)):
            for b in range(a + 1, len(states)):
                A = state_arrays[states[a]]
                B = state_arrays[states[b]]

                mu_a = A[:, i][:, kept].mean(axis=0)
                mu_b = B[:, i][:, kept].mean(axis=0)

                var_a = A[:, i][:, kept].var(axis=0) + 1e-12
                var_b = B[:, i][:, kept].var(axis=0) + 1e-12

                sigma = 0.5 * (var_a + var_b)

                term1 = 0.125 * ((mu_a - mu_b)**2) / sigma
                term2 = 0.5 * np.log(sigma / np.sqrt(var_a * var_b))

                score_i += np.mean(term1 + term2)

        scores.append(score_i)

    return np.array(scores)

# ==========================================================
# Feature Selector
# ==========================================================
def fs_corr(
    X_train, y_train, domains_train,
    X_test, domains_test=None,
    mode="per_subject",          # global | per_state | per_subject | per_subject_state
    method="correlation",        # correlation | covariance | linear | rbf | poly | cosine
    normalize=True,
    n_features=128,
    N_remove=10,
    metric_fn=None,
    states=None,
    kernel_params=None,
    kernel_normalize=True,
    kernel_center=False
):
    # 1) compute structures from TRAIN only
    corr_struct = compute_feature_matrix(
        X_train,
        y=y_train,
        domains=domains_train,
        mode=mode,
        method=method,
        normalize=normalize,
        states=states,
        kernel_params=kernel_params,
        kernel_normalize=kernel_normalize,
        kernel_center=kernel_center
    )

    # 2) prune/select features
    selected = progressive_feature_pruning_generic(
        corr_struct,
        metric_fn=metric_fn,
        N_remove=N_remove,
        K_keep=n_features
    )

    # 3) apply selection
    return (
        X_train[:, selected],
        X_test[:, selected],
        {
            "selected_features": selected,
            "mode": mode,
            "method": method
        }
    )