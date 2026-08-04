import numpy as np
import scipy.stats
import scipy.linalg
import scipy.fft
from tqdm import tqdm
import pandas as pd


def _get_channel_names(trial, channel_names=None):
    """
    Return electrode names or numeric indices as a fallback.
    """
    if channel_names is None:
        return [str(i) for i in range(trial.shape[0])]

    if len(channel_names) != trial.shape[0]:
        raise ValueError(
            f"{len(channel_names)} channel names were provided, "
            f"but the trial has {trial.shape[0]} channels."
        )

    return list(channel_names)


# New
def extract_logvar(trial, channel_names=None, prefix="logvar_"):
    channel_names = _get_channel_names(trial, channel_names)

    vals = np.log(np.var(trial, axis=1) + 1e-12)
    names = [f"{prefix}{channel}" for channel in channel_names]

    return vals, names


def extract_bandpower(
    trial,
    sfreq=128.0,
    bands=None,
    channel_names=None,
    prefix="bp_",
):
    channel_names = _get_channel_names(trial, channel_names)

    if bands is None:
        bands = [(8, 12), (13, 30)]

    n_samples = trial.shape[1]
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    psd = np.abs(np.fft.rfft(trial, axis=1)) ** 2

    vals = []
    names = []

    for low, high in bands:
        mask = (freqs >= low) & (freqs <= high)
        bp = psd[:, mask].sum(axis=1)

        vals.append(bp)
        names += [
            f"{prefix}{low}_{high}_{channel}"
            for channel in channel_names
        ]

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
        W = np.vstack([
            W[:n_components],
            W[-n_components:],
        ])
    else:
        W = W[:2 * n_components]

    return W


def apply_csp(trial, W):
    return W @ trial


def extract_csp_logvar(
    trial,
    W,
    channel_names=None,
    prefix="csp_logvar_",
):
    Z = apply_csp(trial, W)

    vals = np.log(np.var(Z, axis=1) + 1e-12)
    names = [f"{prefix}{i}" for i in range(len(vals))]

    return vals, names


# ===========================================================================
# ===========================================================================
# ===========================================================================

# Old
def extract_mean(trial, channel_names=None, prefix="mean_"):
    channel_names = _get_channel_names(trial, channel_names)

    vals = np.mean(trial, axis=1)
    names = [f"{prefix}{channel}" for channel in channel_names]

    return vals, names


def extract_std(trial, channel_names=None, prefix="std_"):
    channel_names = _get_channel_names(trial, channel_names)

    vals = np.std(trial, axis=1, ddof=1)
    names = [f"{prefix}{channel}" for channel in channel_names]

    return vals, names


def extract_moments(trial, channel_names=None, prefix="mom_"):
    channel_names = _get_channel_names(trial, channel_names)

    skw = scipy.stats.skew(trial, axis=1, bias=False)
    krt = scipy.stats.kurtosis(trial, axis=1, bias=False)

    vals = np.concatenate([skw, krt])

    names = [
        f"{prefix}skew_{channel}"
        for channel in channel_names
    ]

    names += [
        f"{prefix}kurt_{channel}"
        for channel in channel_names
    ]

    return vals, names


def extract_min(trial, channel_names=None, prefix="min_"):
    channel_names = _get_channel_names(trial, channel_names)

    vals = np.min(trial, axis=1)
    names = [f"{prefix}{channel}" for channel in channel_names]

    return vals, names


def extract_max(trial, channel_names=None, prefix="max_"):
    channel_names = _get_channel_names(trial, channel_names)

    vals = np.max(trial, axis=1)
    names = [f"{prefix}{channel}" for channel in channel_names]

    return vals, names


def extract_covariance(
    trial,
    channel_names=None,
    prefix="cov_",
):
    channel_names = _get_channel_names(trial, channel_names)

    cov = np.cov(trial)
    idx = np.triu_indices_from(cov)
    vals = cov[idx]

    names = [
        f"{prefix}{channel_names[i]}_{channel_names[j]}"
        for i, j in zip(idx[0], idx[1])
    ]

    return vals, names


def extract_eigenvalues(
    trial,
    channel_names=None,
    prefix="eig_",
):
    cov = np.cov(trial)
    vals = np.linalg.eigvals(cov).real

    # Eigenvalues do not correspond to individual electrodes.
    names = [f"{prefix}{i}" for i in range(len(vals))]

    return vals, names


def extract_logcov(
    trial,
    channel_names=None,
    prefix="logcov_",
):
    channel_names = _get_channel_names(trial, channel_names)

    cov = np.cov(trial)
    log_cov = scipy.linalg.logm(cov)
    idx = np.triu_indices_from(log_cov)
    vals = np.abs(log_cov[idx])

    names = [
        f"{prefix}{channel_names[i]}_{channel_names[j]}"
        for i, j in zip(idx[0], idx[1])
    ]

    return vals, names


def extract_fft(
    trial,
    channel_names=None,
    prefix="fft_",
    ntop=10,
):
    channel_names = _get_channel_names(trial, channel_names)

    fft_vals = np.abs(scipy.fft.fft(trial, axis=1))
    fft_vals = fft_vals[:, :fft_vals.shape[1] // 2]

    idx = np.argsort(
        fft_vals,
        axis=1,
    )[:, ::-1][:, :ntop]

    vals = idx.flatten()

    names = [
        f"{prefix}top_{k}_{channel}"
        for channel in channel_names
        for k in range(ntop)
    ]

    return vals, names


def extract_halves_diff(
    trial,
    channel_names=None,
    prefix="h2h1_",
):
    channel_names = _get_channel_names(trial, channel_names)

    n = trial.shape[1] // 2
    h1 = trial[:, :n]
    h2 = trial[:, n:]

    vals = np.mean(h2, axis=1) - np.mean(h1, axis=1)
    names = [f"{prefix}{channel}" for channel in channel_names]

    return vals, names


def extract_quarters_stats(
    trial,
    channel_names=None,
    prefix="q_",
):
    channel_names = _get_channel_names(trial, channel_names)

    n = trial.shape[1]

    q = np.split(
        trial,
        [n // 4, n // 2, 3 * n // 4],
        axis=1,
    )

    vals = []
    names = []

    for i, qi in enumerate(q):
        m = np.mean(qi, axis=1)
        vals.append(m)

        names += [
            f"{prefix}{i}_{channel}"
            for channel in channel_names
        ]

    return np.concatenate(vals), names


def _format_band_tag(band_idx, band_labels=None):
    if band_labels is not None and band_idx < len(band_labels):
        low, high = band_labels[band_idx]
        return f"b{low}_{high}_"

    return f"b{band_idx}_"


def extract_features_from_trial(
    trial,
    extract_config,
    channel_names=None,
    band_labels=None,
):
    vals_all = []
    names_all = []

    if trial.ndim == 2:
        for _, cfg in extract_config.items():
            fn = cfg["function"]
            params = cfg.get("params", {})

            vals, names = fn(
                trial,
                channel_names=channel_names,
                **params,
            )

            vals_all.append(np.asarray(vals))
            names_all.extend(names)

    elif trial.ndim == 3:
        for band_idx in range(trial.shape[0]):
            band_trial = trial[band_idx]

            band_tag = _format_band_tag(
                band_idx,
                band_labels,
            )

            for _, cfg in extract_config.items():
                fn = cfg["function"]
                params = cfg.get("params", {})

                vals, names = fn(
                    band_trial,
                    channel_names=channel_names,
                    **params,
                )

                vals_all.append(np.asarray(vals))

                names_all.extend([
                    band_tag + name
                    for name in names
                ])

    else:
        raise ValueError(
            f"Unexpected trial shape: {trial.shape}"
        )

    if not vals_all:
        raise ValueError("Empty extract_config")

    return np.concatenate(vals_all), names_all


def extract_features_to_dataframe(
    dataset,
    extract_config,
    show_progress=True,
    band_labels=None,
):
    all_features = []
    feature_names_ref = None

    iterator = dataset.items()

    if show_progress:
        iterator = tqdm(
            iterator,
            desc="Subjects",
        )

    for subj_id, sessions in iterator:
        for sess_name, sess_data in sessions.items():
            X = sess_data["X"]
            y = sess_data["y"]
            channel_names = sess_data["channel_names"]

            if X.ndim == 3:
                trial_iterator = enumerate(X)

            elif X.ndim == 4:
                trial_iterator = (
                    (i, X[:, i, :, :])
                    for i in range(X.shape[1])
                )

            else:
                raise ValueError(
                    f"Unexpected X shape: {X.shape}"
                )

            for trial_idx, trial in trial_iterator:
                vals, names = extract_features_from_trial(
                    trial,
                    extract_config,
                    channel_names=channel_names,
                    band_labels=band_labels,
                )

                if feature_names_ref is None:
                    feature_names_ref = names

                elif names != feature_names_ref:
                    raise ValueError(
                        "Feature names or channel order changed "
                        f"in {subj_id}/{sess_name}."
                    )

                row = {
                    "subject": subj_id,
                    "session": sess_name,
                    "label": int(y[trial_idx]),
                }

                row.update({
                    feature_names_ref[i]: vals[i]
                    for i in range(len(vals))
                })

                all_features.append(row)

    return pd.DataFrame(all_features)