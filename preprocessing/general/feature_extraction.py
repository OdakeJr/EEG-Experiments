import numpy as np
import scipy.stats
import scipy.linalg
import scipy.fft
from tqdm import tqdm
import pandas as pd


# New 
def extract_logvar(trial, prefix="logvar_"):
    vals = np.log(np.var(trial, axis=1) + 1e-12)
    names = [f"{prefix}{i}" for i in range(len(vals))]
    return vals, names


def extract_bandpower(trial, sfreq=250.0, bands=None, prefix="bp_"):
    if bands is None:
        bands = [(8, 12), (12, 30)]

    n_samples = trial.shape[1]
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    psd = np.abs(np.fft.rfft(trial, axis=1)) ** 2

    vals = []
    names = []

    for low, high in bands:
        mask = (freqs >= low) & (freqs <= high)
        bp = psd[:, mask].sum(axis=1)
        vals.append(bp)
        names += [f"{prefix}{low}_{high}_{i}" for i in range(trial.shape[0])]

    return np.concatenate(vals), names


# ===========================================================================
# TODO ======================================================================
# ===========================================================================
def fit_csp_binary(X, y, class_a, class_b, n_components=4):
    Xa = X[y == class_a]
    Xb = X[y == class_b]

    Ca = np.mean([np.cov(trial) for trial in Xa], axis=0)
    Cb = np.mean([np.cov(trial) for trial in Xb], axis=0)

    C = Ca + Cb
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 1e-12)

    P = np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    Sa = P @ Ca @ P.T

    d, B = np.linalg.eigh(Sa)
    idx = np.argsort(d)[::-1]
    B = B[:, idx]

    W = B.T @ P
    if 2 * n_components <= W.shape[0]:
        W = np.vstack([W[:n_components], W[-n_components:]])
    else:
        W = W[: 2 * n_components]

    return W


def apply_csp(trial, W):
    return W @ trial


def extract_csp_logvar(trial, W, prefix="csp_logvar_"):
    Z = apply_csp(trial, W)
    vals = np.log(np.var(Z, axis=1) + 1e-12)
    names = [f"{prefix}{i}" for i in range(len(vals))]
    return vals, names


# ===========================================================================
# ===========================================================================
# ===========================================================================

# Old
def extract_mean(trial, prefix="mean_"):
    vals = np.mean(trial, axis=1)
    names = [f"{prefix}{i}" for i in range(len(vals))]
    return vals, names


def extract_std(trial, prefix="std_"):
    vals = np.std(trial, axis=1, ddof=1)
    names = [f"{prefix}{i}" for i in range(len(vals))]
    return vals, names


def extract_moments(trial, prefix="mom_"):
    skw = scipy.stats.skew(trial, axis=1, bias=False)
    krt = scipy.stats.kurtosis(trial, axis=1, bias=False)
    vals = np.concatenate([skw, krt])
    names = [f"{prefix}skew_{i}" for i in range(len(skw))] + [f"{prefix}kurt_{i}" for i in range(len(krt))]
    return vals, names


def extract_min(trial, prefix="min_"):
    vals = np.min(trial, axis=1)
    names = [f"{prefix}{i}" for i in range(len(vals))]
    return vals, names


def extract_max(trial, prefix="max_"):
    vals = np.max(trial, axis=1)
    names = [f"{prefix}{i}" for i in range(len(vals))]
    return vals, names


def extract_covariance(trial, prefix="cov_"):
    cov = np.cov(trial)
    idx = np.triu_indices_from(cov)
    vals = cov[idx]
    names = [f"{prefix}{i}_{j}" for i, j in zip(idx[0], idx[1])]
    return vals, names


def extract_eigenvalues(trial, prefix="eig_"):
    cov = np.cov(trial)
    vals = np.linalg.eigvals(cov).real
    names = [f"{prefix}{i}" for i in range(len(vals))]
    return vals, names


def extract_logcov(trial, prefix="logcov_"):
    cov = np.cov(trial)
    log_cov = scipy.linalg.logm(cov)
    idx = np.triu_indices_from(log_cov)
    vals = np.abs(log_cov[idx])
    names = [f"{prefix}{i}_{j}" for i, j in zip(idx[0], idx[1])]
    return vals, names


def extract_fft(trial, prefix="fft_", ntop=10):
    fft_vals = np.abs(scipy.fft.fft(trial, axis=1))
    fft_vals = fft_vals[:, :fft_vals.shape[1] // 2]
    idx = np.argsort(fft_vals, axis=1)[:, ::-1][:, :ntop]
    vals = idx.flatten()
    names = [f"{prefix}top_{k}_ch{i}" for i in range(idx.shape[0]) for k in range(ntop)]
    return vals, names


def extract_halves_diff(trial, prefix="h2h1_"):
    n = trial.shape[1] // 2
    h1 = trial[:, :n]
    h2 = trial[:, n:]
    vals = np.mean(h2, axis=1) - np.mean(h1, axis=1)
    names = [f"{prefix}{i}" for i in range(len(vals))]
    return vals, names


def extract_quarters_stats(trial, prefix="q_"):
    n = trial.shape[1]
    q = np.split(trial, [n // 4, n // 2, 3 * n // 4], axis=1)
    vals = []
    names = []
    for i, qi in enumerate(q):
        m = np.mean(qi, axis=1)
        vals.append(m)
        names += [f"{prefix}{i}_{j}" for j in range(len(m))]
    return np.concatenate(vals), names


def _format_band_tag(band_idx, band_labels=None):
    if band_labels is not None and band_idx < len(band_labels):
        low, high = band_labels[band_idx]
        return f"b{low}_{high}_"
    return f"b{band_idx}_"


def extract_features_from_trial(trial, extract_config, band_labels=None):
    vals_all = []
    names_all = []

    if trial.ndim == 2:
        for _, cfg in extract_config.items():
            fn = cfg["function"]
            params = cfg.get("params", {})
            vals, names = fn(trial, **params)
            vals_all.append(np.asarray(vals))
            names_all.extend(names)

    elif trial.ndim == 3:
        for band_idx in range(trial.shape[0]):
            band_trial = trial[band_idx]
            band_tag = _format_band_tag(band_idx, band_labels)

            for _, cfg in extract_config.items():
                fn = cfg["function"]
                params = cfg.get("params", {})
                vals, names = fn(band_trial, **params)
                vals_all.append(np.asarray(vals))
                names_all.extend([band_tag + name for name in names])

    else:
        raise ValueError(f"Unexpected trial shape: {trial.shape}")

    if not vals_all:
        raise ValueError("Empty extract_config")

    return np.concatenate(vals_all), names_all


def extract_features_to_dataframe(dataset, extract_config, show_progress=True, band_labels=None):
    all_features = []
    feature_names_ref = None

    iterator = dataset.items()
    if show_progress:
        iterator = tqdm(iterator, desc="Subjects")

    for subj_id, sessions in iterator:
        for sess_name, sess_data in sessions.items():
            X = sess_data["X"]
            y = sess_data["y"]

            if X.ndim == 3:
                trial_iterator = enumerate(X)
            elif X.ndim == 4:
                trial_iterator = ((i, X[:, i, :, :]) for i in range(X.shape[1]))
            else:
                raise ValueError(f"Unexpected X shape: {X.shape}")

            for trial_idx, trial in trial_iterator:
                vals, names = extract_features_from_trial(
                    trial,
                    extract_config,
                    band_labels=band_labels
                )

                if feature_names_ref is None:
                    feature_names_ref = names

                row = {
                    "subject": subj_id,
                    "session": sess_name,
                    "label": int(y[trial_idx]),
                }

                row.update({feature_names_ref[i]: vals[i] for i in range(len(vals))})
                all_features.append(row)

    return pd.DataFrame(all_features)