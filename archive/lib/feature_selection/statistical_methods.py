import numpy as np

from sklearn.feature_selection import (
    mutual_info_classif, f_classif, chi2,
    SequentialFeatureSelector, RFE
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr


# ==========================================================
# FS: none
# ==========================================================
def fs_none(X_train, y_train, domains_train, X_test, domains_test, **kwargs):
    return X_train, X_test, {"selected_features": None}


# ==========================================================
# 1) Mutual Information
# ==========================================================
def fs_mi(X_train, y_train, domains_train, X_test, domains_test, n_features=128):
    mi = mutual_info_classif(X_train, y_train, discrete_features=False)
    idx = np.argsort(mi)[::-1][:n_features]
    return X_train[:, idx], X_test[:, idx], {"selected_features": idx}


# ==========================================================
# 2) ANOVA F-test
# ==========================================================
def fs_anova(X_train, y_train, domains_train, X_test, domains_test, n_features=128):
    f_vals, _ = f_classif(X_train, y_train)
    f_vals = np.nan_to_num(f_vals, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    idx = np.argsort(f_vals)[::-1][:n_features]
    return X_train[:, idx], X_test[:, idx], {"selected_features": idx}


# ==========================================================
# 3) Variance
# ==========================================================
def fs_variance(X_train, y_train, domains_train, X_test, domains_test, n_features=128):
    vars_ = np.var(X_train, axis=0)
    idx = np.argsort(vars_)[::-1][:n_features]
    return X_train[:, idx], X_test[:, idx], {"selected_features": idx}


# ==========================================================
# 4) Pearson correlation (abs)
# ==========================================================
def fs_pearson(X_train, y_train, domains_train, X_test, domains_test, n_features=128):
    corrs = np.array([np.corrcoef(X_train[:, i], y_train)[0, 1] for i in range(X_train.shape[1])])
    corrs = np.nan_to_num(corrs)
    idx = np.argsort(np.abs(corrs))[::-1][:n_features]
    return X_train[:, idx], X_test[:, idx], {"selected_features": idx}


# ==========================================================
# 5) Chi-squared (shift to nonnegative; shift based on TRAIN only)
# ==========================================================
def fs_chi2(X_train, y_train, domains_train, X_test, domains_test, n_features=128):
    shift = X_train.min(axis=0)
    X_train_nn = X_train - shift
    X_test_nn = X_test - shift

    chi2_vals, _ = chi2(X_train_nn, y_train)
    chi2_vals = np.nan_to_num(chi2_vals, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    idx = np.argsort(chi2_vals)[::-1][:n_features]

    return X_train[:, idx], X_test[:, idx], {"selected_features": idx}


# ==========================================================
# 6) Relief (binary only) – uses TRAIN only for scoring
# ==========================================================
def fs_relief(X_train, y_train, domains_train, X_test, domains_test, n_features=128):
    classes = np.unique(y_train)
    if len(classes) != 2:
        raise ValueError("Relief here supports only binary classification.")

    # normalize using TRAIN only (no leakage)
    x_min = X_train.min(axis=0)
    x_ptp = X_train.ptp(axis=0) + 1e-12
    Xn = (X_train - x_min) / x_ptp

    nbrs = NearestNeighbors(n_neighbors=2).fit(Xn)
    _, idxs = nbrs.kneighbors(Xn)

    scores = np.zeros(X_train.shape[1], dtype=float)
    for i, (x_i, y_i) in enumerate(zip(Xn, y_train)):
        nn_idx = idxs[i][1]
        x_nn = Xn[nn_idx]
        if y_train[nn_idx] == y_i:
            scores -= np.abs(x_i - x_nn)
        else:
            scores += np.abs(x_i - x_nn)

    idx = np.argsort(scores)[::-1][:n_features]
    return X_train[:, idx], X_test[:, idx], {"selected_features": idx}


# ==========================================================
# 7) L1 Logistic (sparse)
# ==========================================================
def fs_l1(X_train, y_train, domains_train, X_test, domains_test, n_features=128, C=0.1):
    model = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000)
    model.fit(X_train, y_train)
    scores = np.abs(model.coef_).ravel()
    idx = np.argsort(scores)[::-1][:n_features]
    return X_train[:, idx], X_test[:, idx], {"selected_features": idx}


# ==========================================================
# 8) Random Forest importance
# ==========================================================
def fs_rf(X_train, y_train, domains_train, X_test, domains_test, n_features=128, n_estimators=200):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    importances = np.nan_to_num(model.feature_importances_)
    idx = np.argsort(importances)[::-1][:n_features]
    return X_train[:, idx], X_test[:, idx], {"selected_features": idx}


# ==========================================================
# 9) Spearman correlation (abs)
# ==========================================================
def fs_spearman(X_train, y_train, domains_train, X_test, domains_test, n_features=128):
    corrs = np.array([spearmanr(X_train[:, i], y_train).correlation for i in range(X_train.shape[1])])
    corrs = np.nan_to_num(corrs)
    idx = np.argsort(np.abs(corrs))[::-1][:n_features]
    return X_train[:, idx], X_test[:, idx], {"selected_features": idx}


# ==========================================================
# 10) Entropy-based relevance (your IG-like heuristic)
# ==========================================================
def fs_entropy(X_train, y_train, domains_train, X_test, domains_test, n_features=128, bins=10):
    def entropy_1d(a):
        hist, _ = np.histogram(a, bins=bins, density=True)
        p = hist[hist > 0]
        return -np.sum(p * np.log(p))

    H_y = entropy_1d(y_train)
    ig = []
    for i in range(X_train.shape[1]):
        H_x = entropy_1d(X_train[:, i])
        H_xy = entropy_1d(np.concatenate([X_train[:, i], y_train]))
        ig.append(H_x + H_y - H_xy)

    ig = np.nan_to_num(np.array(ig))
    idx = np.argsort(ig)[::-1][:n_features]
    return X_train[:, idx], X_test[:, idx], {"selected_features": idx}


# ==========================================================
# 11) mRMR (simple)
# ==========================================================
def fs_mrmr(X_train, y_train, domains_train, X_test, domains_test, n_features=128):
    mi = mutual_info_classif(X_train, y_train, discrete_features=False)
    selected = []
    remaining = list(range(X_train.shape[1]))

    while len(selected) < n_features and remaining:
        scores = []
        for f in remaining:
            if selected:
                redundancy = np.mean([np.corrcoef(X_train[:, f], X_train[:, s])[0, 1] for s in selected])
                redundancy = 0.0 if np.isnan(redundancy) else redundancy
            else:
                redundancy = 0.0
            scores.append(mi[f] - redundancy)

        best = remaining[int(np.argmax(scores))]
        selected.append(best)
        remaining.remove(best)

    idx = np.array(selected, dtype=int)
    return X_train[:, idx], X_test[:, idx], {"selected_features": idx}


# ==========================================================
# 12) OneR (simple)
# ==========================================================
def fs_oner(X_train, y_train, domains_train, X_test, domains_test, n_features=128, bins=10):
    y_train = np.asarray(y_train).astype(int)
    scores = []

    for i in range(X_train.shape[1]):
        feature = X_train[:, i]
        if np.all(feature == feature[0]):
            scores.append(0.0)
            continue

        cuts = np.linspace(np.min(feature), np.max(feature), bins + 1)
        digitized = np.digitize(feature, cuts)

        preds = np.zeros_like(y_train)
        for b in np.unique(digitized):
            mask = digitized == b
            if np.any(mask):
                preds[mask] = np.argmax(np.bincount(y_train[mask]))
        scores.append(np.mean(preds == y_train))

    scores = np.nan_to_num(np.array(scores))
    idx = np.argsort(scores)[::-1][:n_features]
    return X_train[:, idx], X_test[:, idx], {"selected_features": idx}


# ==========================================================
# 13) SFS (Sequential Forward Selection with Logistic)
# ==========================================================
def fs_sfs(X_train, y_train, domains_train, X_test, domains_test, n_features=128, max_iter=1000):
    n_features = min(n_features, X_train.shape[1])
    model = LogisticRegression(max_iter=max_iter)

    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=n_features,
        direction="forward",
        n_jobs=-1
    )
    sfs.fit(X_train, y_train)
    idx = np.where(sfs.get_support())[0]
    return X_train[:, idx], X_test[:, idx], {"selected_features": idx}


# ==========================================================
# 14) SVM-RFE
# ==========================================================
def fs_svmrfe(X_train, y_train, domains_train, X_test, domains_test, n_features=128):
    n_features = min(n_features, X_train.shape[1])
    svm = SVC(kernel="linear")
    rfe = RFE(estimator=svm, n_features_to_select=n_features, step=1)
    rfe.fit(X_train, y_train)
    idx = np.where(rfe.get_support())[0]
    return X_train[:, idx], X_test[:, idx], {"selected_features": idx}


# ==========================================================
# 15) GA feature selection (can be slow; defaults are modest)
# ==========================================================
def fs_ga(
    X_train, y_train, domains_train, X_test, domains_test,
    n_features=128, pop_size=20, generations=10, mutation_rate=0.05, cv=3, random_state=42
):
    rng = np.random.RandomState(random_state)
    n_total = X_train.shape[1]
    clf = LogisticRegression(max_iter=1000)

    pop = rng.randint(0, 2, size=(pop_size, n_total))

    def fitness(mask):
        k = int(mask.sum())
        if k == 0:
            return 0.0
        X_sel = X_train[:, mask == 1]
        # tiny subsets can be unstable; still ok
        return float(np.mean(cross_val_score(clf, X_sel, y_train, cv=cv, n_jobs=-1)))

    for _ in range(generations):
        scores = np.array([fitness(ind) for ind in pop])

        # keep top half
        idx = scores.argsort()[::-1][: max(2, pop_size // 2)]
        parents = pop[idx]

        # crossover
        children = []
        while len(children) < pop_size - len(parents):
            p1 = parents[rng.randint(len(parents))]
            p2 = parents[rng.randint(len(parents))]
            cross = rng.randint(1, n_total - 1)
            child = np.concatenate([p1[:cross], p2[cross:]])
            children.append(child)
        children = np.array(children)

        # mutation
        mut = rng.rand(*children.shape) < mutation_rate
        children = np.logical_xor(children, mut).astype(int)

        pop = np.vstack([parents, children])

    scores = np.array([fitness(ind) for ind in pop])
    best_mask = pop[int(scores.argmax())].astype(int)

    selected = np.where(best_mask == 1)[0]

    # enforce exactly n_features if too many selected (tie-break by variance)
    if len(selected) > n_features:
        vars_ = np.var(X_train[:, selected], axis=0)
        selected = selected[np.argsort(vars_)[::-1][:n_features]]

    # if too few selected, pad with top-variance among remaining
    if len(selected) < min(n_features, n_total):
        remaining = np.setdiff1d(np.arange(n_total), selected)
        vars_rem = np.var(X_train[:, remaining], axis=0)
        need = min(n_features - len(selected), len(remaining))
        add = remaining[np.argsort(vars_rem)[::-1][:need]]
        selected = np.concatenate([selected, add])

    idx = np.array(selected, dtype=int)
    return X_train[:, idx], X_test[:, idx], {"selected_features": idx}


# ==========================================================
# 16) CFS (your merit-based implementation)
# ==========================================================
def fs_cfs(X_train, y_train, domains_train, X_test, domains_test, k=32):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    n_samples, n_features_total = X_train.shape
    k = min(k, n_features_total)

    y0 = y_train - y_train.mean()
    X0 = X_train - X_train.mean(axis=0)

    r_fc = np.abs((X0 * y0[:, None]).mean(axis=0) / (X0.std(axis=0) * y0.std() + 1e-8))
    r_fc = np.nan_to_num(r_fc)

    corr_ff = np.abs(np.corrcoef(X_train, rowvar=False))
    corr_ff = np.nan_to_num(corr_ff)

    selected = []
    remaining = np.arange(n_features_total)

    sum_r_cf = 0.0
    sum_r_ff = 0.0

    while len(selected) < k and len(remaining) > 0:
        best_feat = None
        best_score = -np.inf

        for f in remaining:
            kc = len(selected) + 1
            new_sum_r_cf = sum_r_cf + r_fc[f]

            if selected:
                new_sum_r_ff = sum_r_ff + 2.0 * np.sum(corr_ff[f, selected])
            else:
                new_sum_r_ff = 0.0

            r_cf_mean = new_sum_r_cf / kc
            r_ff_mean = new_sum_r_ff / (kc * (kc - 1) + 1e-8)

            merit = (kc * r_cf_mean) / np.sqrt(kc + kc * (kc - 1) * r_ff_mean + 1e-8)

            if merit > best_score:
                best_score = merit
                best_feat = int(f)
                best_sum_r_cf = new_sum_r_cf
                best_sum_r_ff = new_sum_r_ff

        selected.append(best_feat)
        remaining = remaining[remaining != best_feat]
        sum_r_cf = best_sum_r_cf
        sum_r_ff = best_sum_r_ff

    idx = np.array(selected, dtype=int)
    return X_train[:, idx], X_test[:, idx], {"selected_features": idx}