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
# 1) C/K-DIFS 
# ==========================================================
def metric_dispersion(corr_dict, kept):

    corr_list = list(corr_dict.values())
    scores = []

    for i in kept:
        vals = []

        for C in corr_list:
            vals.extend(C[i, kept])

        scores.append(-np.var(vals))   # lower variance = better

    return np.array(scores)

# ==========================================================
# 2) C/K-SSFS
# ==========================================================
def metric_state_separation(corr_dict, kept):

    keys = list(corr_dict.keys())
    corr_list = list(corr_dict.values())

    if len(corr_list) != 2:
        raise ValueError("State separation requires exactly two matrices.")

    A, B = corr_list

    diff = A[np.ix_(kept, kept)] - B[np.ix_(kept, kept)]

    scores = np.linalg.norm(diff, axis=1)

    return scores

# ==========================================================
# 3) C/K-HDSFS
# ==========================================================
def metric_mean_separation(corr_dict, kept):

    groups = {}

    for k, C in corr_dict.items():
        if "state_" not in k:
            raise ValueError("Mean separation requires state information.")

        state = k.split("state_")[-1]
        groups.setdefault(state, []).append(C)

    if len(groups) != 2:
        raise ValueError("Exactly two states required.")

    states = list(groups.keys())

    A = np.mean(groups[states[0]], axis=0)
    B = np.mean(groups[states[1]], axis=0)

    diff = A[np.ix_(kept, kept)] - B[np.ix_(kept, kept)]

    scores = np.linalg.norm(diff, axis=1)

    return scores

def metric_wasserstein(corr_dict, kept):

    groups = {}

    for k, C in corr_dict.items():
        state = k.split("state_")[-1]
        groups.setdefault(state, []).append(C)

    states = list(groups.keys())

    A = np.array(groups[states[0]])
    B = np.array(groups[states[1]])

    scores = []

    for i in kept:

        mu_a = A[:, i][:, kept].mean(axis=0)
        mu_b = B[:, i][:, kept].mean(axis=0)

        var_a = A[:, i][:, kept].var(axis=0)
        var_b = B[:, i][:, kept].var(axis=0)

        d = np.sqrt((mu_a - mu_b)**2 + (np.sqrt(var_a) - np.sqrt(var_b))**2)

        scores.append(np.mean(d))

    return np.array(scores)

def metric_bhattacharyya(corr_dict, kept):

    groups = {}

    for k, C in corr_dict.items():
        state = k.split("state_")[-1]
        groups.setdefault(state, []).append(C)

    states = list(groups.keys())

    A = np.array(groups[states[0]])
    B = np.array(groups[states[1]])

    scores = []

    for i in kept:

        mu_a = A[:, i][:, kept].mean(axis=0)
        mu_b = B[:, i][:, kept].mean(axis=0)

        var_a = A[:, i][:, kept].var(axis=0)
        var_b = B[:, i][:, kept].var(axis=0)

        sigma = (var_a + var_b) / 2

        term1 = 0.125 * ((mu_a - mu_b)**2) / (sigma + 1e-12)
        term2 = 0.5 * np.log((sigma + 1e-12) / np.sqrt(var_a * var_b + 1e-12))

        scores.append(np.mean(term1 + term2))

    return np.array(scores)

# ==========================================================
# Feature Selector
# ==========================================================
def fs_corr(
    X_train, y_train, domains_train,
    X_test,  domains_test,
    mode="per_subject",          # global | per_state | per_subject | per_subject_state
    use_correlation=True,
    normalize=True,
    n_features=128,
    N_remove=10,
    metric_fn=None               # we'll define later
):
    # 1) compute correlation/cov structures from TRAIN only
    corr_struct = compute_feature_matrix(
        X_train,
        y=y_train,
        domains=domains_train,
        mode=mode,
        use_correlation=use_correlation,
        normalize=normalize
    )

    # 2) prune/select features (we’ll implement later)
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
        {"selected_features": selected, "mode": mode}
    )