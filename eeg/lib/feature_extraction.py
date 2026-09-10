# eeg/lib/feature_extraction.py

from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.stats
from tqdm import tqdm
import scipy.linalg
import scipy.signal
import scipy.spatial.distance
import pywt
import math


# ============================================================
# Feature registry
# ============================================================

FEATURE_FUNCTIONS = {}

DEFAULT_FEATURE_CONFIG = {
    # Statistical
    "mean": {},
    "std": {},
    "var": {},
    "logvar": {},
    "skew": {},
    "kurtosis": {},
    "min": {},
    "max": {},
    "rms": {},
    "ptp": {},

    # Temporal
    "line_length": {},
    "hjorth_activity": {},
    "hjorth_mobility": {},
    "hjorth_complexity": {},
    "zero_crossing": {},
    "ar": {},

    # Spectral
    "bandpower": {},
    "relative_bandpower": {},
    "psd_stats": {},
    "spectral_entropy": {},
    "differential_entropy": {},

    # Nonlinear
    "sample_entropy": {},
    "permutation_entropy": {},
    "higuchi_fd": {},
    "petrosian_fd": {},

    # Covariance
    "cov": {},
    "logcov": {},
    "eig": {},

    # Time-frequency
    "wavelet_energy": {},
    "wavelet_entropy": {},
}


def build_extract_config(feature_config=None):
    feature_config = DEFAULT_FEATURE_CONFIG if feature_config is None else feature_config
    extract_config = {}

    for name, params in feature_config.items():
        if name not in FEATURE_FUNCTIONS:
            raise ValueError(f"Unknown EEG feature '{name}'. Available: {sorted(FEATURE_FUNCTIONS)}")

        params = {} if params is None else params
        if not isinstance(params, dict):
            raise TypeError(f"Parameters for feature '{name}' must be a dictionary.")

        extract_config[name] = {
            "function": FEATURE_FUNCTIONS[name],
            "params": deepcopy(params),
        }

    if not extract_config:
        raise ValueError("At least one feature must be enabled.")

    return extract_config


# ============================================================
# Helpers
# ============================================================

def _get_channel_names(trial, channel_names=None):
    if channel_names is None:
        return [str(i) for i in range(trial.shape[0])]

    if len(channel_names) != trial.shape[0]:
        raise ValueError(
            f"{len(channel_names)} channel names were provided, "
            f"but the trial has {trial.shape[0]} channels."
        )

    return list(channel_names)


def _format_band_tag(band_idx, band_labels=None):
    if band_labels is not None and band_idx < len(band_labels):
        low, high = band_labels[band_idx]
        return f"b{low}_{high}_"

    return f"b{band_idx}_"


def _channel_feature(trial, vals, channel_names=None, prefix=""):
    channels = _get_channel_names(trial, channel_names)
    return np.asarray(vals), [f"{prefix}{channel}" for channel in channels]


# ============================================================
# Statistical features
# ============================================================

def extract_mean(trial, channel_names=None, prefix="mean_"):
    return _channel_feature(trial, np.mean(trial, axis=1), channel_names, prefix)


def extract_std(trial, channel_names=None, prefix="std_"):
    return _channel_feature(trial, np.std(trial, axis=1, ddof=1), channel_names, prefix)


def extract_variance(trial, channel_names=None, prefix="var_"):
    return _channel_feature(trial, np.var(trial, axis=1, ddof=1), channel_names, prefix)


def extract_logvar(trial, channel_names=None, prefix="logvar_"):
    vals = np.log(np.var(trial, axis=1, ddof=1) + 1e-12)
    return _channel_feature(trial, vals, channel_names, prefix)


def extract_skewness(trial, channel_names=None, prefix="skew_"):
    vals = scipy.stats.skew(trial, axis=1, bias=False)
    return _channel_feature(trial, vals, channel_names, prefix)


def extract_kurtosis(trial, channel_names=None, prefix="kurt_"):
    vals = scipy.stats.kurtosis(trial, axis=1, bias=False)
    return _channel_feature(trial, vals, channel_names, prefix)


def extract_min(trial, channel_names=None, prefix="min_"):
    return _channel_feature(trial, np.min(trial, axis=1), channel_names, prefix)


def extract_max(trial, channel_names=None, prefix="max_"):
    return _channel_feature(trial, np.max(trial, axis=1), channel_names, prefix)


def extract_rms(trial, channel_names=None, prefix="rms_"):
    vals = np.sqrt(np.mean(trial ** 2, axis=1))
    return _channel_feature(trial, vals, channel_names, prefix)


def extract_peak_to_peak(trial, channel_names=None, prefix="ptp_"):
    return _channel_feature(trial, np.ptp(trial, axis=1), channel_names, prefix)


FEATURE_FUNCTIONS.update({
    "mean": extract_mean,
    "std": extract_std,
    "var": extract_variance,
    "logvar": extract_logvar,
    "skew": extract_skewness,
    "kurtosis": extract_kurtosis,
    "min": extract_min,
    "max": extract_max,
    "rms": extract_rms,
    "ptp": extract_peak_to_peak,
})

# ============================================================
# Temporal features
# ============================================================

def extract_line_length(trial, channel_names=None, prefix="ll_"):
    vals = np.sum(np.abs(np.diff(trial, axis=1)), axis=1)
    return _channel_feature(trial, vals, channel_names, prefix)


def _hjorth(trial):
    dx = np.diff(trial, axis=1)
    ddx = np.diff(dx, axis=1)

    activity = np.var(trial, axis=1)
    var_dx = np.var(dx, axis=1)
    var_ddx = np.var(ddx, axis=1)

    mobility = np.sqrt(var_dx / (activity + 1e-12))
    complexity = np.sqrt(var_ddx / (var_dx + 1e-12)) / (mobility + 1e-12)

    return activity, mobility, complexity


def extract_hjorth_activity(trial, channel_names=None, prefix="hjorth_act_"):
    return _channel_feature(trial, _hjorth(trial)[0], channel_names, prefix)


def extract_hjorth_mobility(trial, channel_names=None, prefix="hjorth_mob_"):
    return _channel_feature(trial, _hjorth(trial)[1], channel_names, prefix)


def extract_hjorth_complexity(trial, channel_names=None, prefix="hjorth_comp_"):
    return _channel_feature(trial, _hjorth(trial)[2], channel_names, prefix)


def extract_zero_crossing(trial, channel_names=None, prefix="zc_"):
    vals = np.sum(np.diff(np.signbit(trial), axis=1), axis=1)
    return _channel_feature(trial, vals, channel_names, prefix)


def extract_ar(trial, channel_names=None, prefix="ar_", order=6):
    channels = _get_channel_names(trial, channel_names)
    vals, names = [], []

    for channel, x in zip(channels, trial):
        X = np.column_stack([x[order - lag - 1:-lag - 1] for lag in range(order)])
        y = x[order:]
        coef = np.linalg.lstsq(X, y, rcond=None)[0]

        vals.extend(coef)
        names.extend(f"{prefix}{i + 1}_{channel}" for i in range(order))

    return np.asarray(vals), names


FEATURE_FUNCTIONS.update({
    "line_length": extract_line_length,
    "hjorth_activity": extract_hjorth_activity,
    "hjorth_mobility": extract_hjorth_mobility,
    "hjorth_complexity": extract_hjorth_complexity,
    "zero_crossing": extract_zero_crossing,
    "ar": extract_ar,
})

# ============================================================
# Spectral features
# ============================================================

def _psd(trial, sfreq=128.0):
    freqs, psd = scipy.signal.welch(trial, fs=sfreq, axis=1, nperseg=min(256, trial.shape[1]))
    return freqs, psd


def extract_bandpower(trial, channel_names=None, prefix="bp_", sfreq=128.0, bands=((8, 12), (13, 30))):
    channels = _get_channel_names(trial, channel_names)
    freqs, psd = _psd(trial, sfreq)
    vals, names = [], []

    for low, high in bands:
        mask = (freqs >= low) & (freqs <= high)
        power = np.trapezoid(psd[:, mask], freqs[mask], axis=1)
        vals.extend(power)
        names.extend(f"{prefix}{low}_{high}_{ch}" for ch in channels)

    return np.asarray(vals), names


def extract_relative_bandpower(
    trial, channel_names=None, prefix="rbp_", sfreq=128.0,
    bands=((8, 12), (13, 30)), total_band=(1, 40),
):
    channels = _get_channel_names(trial, channel_names)
    freqs, psd = _psd(trial, sfreq)
    total_mask = (freqs >= total_band[0]) & (freqs <= total_band[1])
    total = np.trapezoid(psd[:, total_mask], freqs[total_mask], axis=1) + 1e-12
    vals, names = [], []

    for low, high in bands:
        mask = (freqs >= low) & (freqs <= high)
        power = np.trapezoid(psd[:, mask], freqs[mask], axis=1) / total
        vals.extend(power)
        names.extend(f"{prefix}{low}_{high}_{ch}" for ch in channels)

    return np.asarray(vals), names


def extract_psd_stats(trial, channel_names=None, prefix="psd_", sfreq=128.0, band=(1, 40)):
    channels = _get_channel_names(trial, channel_names)
    freqs, psd = _psd(trial, sfreq)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    f, p = freqs[mask], psd[:, mask]
    norm = p / (p.sum(axis=1, keepdims=True) + 1e-12)

    centroid = np.sum(norm * f, axis=1)
    spread = np.sqrt(np.sum(norm * (f - centroid[:, None]) ** 2, axis=1))
    peak = f[np.argmax(p, axis=1)]

    vals = np.concatenate([centroid, spread, peak])
    names = (
        [f"{prefix}centroid_{ch}" for ch in channels]
        + [f"{prefix}spread_{ch}" for ch in channels]
        + [f"{prefix}peak_{ch}" for ch in channels]
    )
    return vals, names


def extract_spectral_entropy(trial, channel_names=None, prefix="specent_", sfreq=128.0, band=(1, 40)):
    freqs, psd = _psd(trial, sfreq)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    p = psd[:, mask]
    p /= p.sum(axis=1, keepdims=True) + 1e-12
    vals = -np.sum(p * np.log(p + 1e-12), axis=1) / np.log(p.shape[1])
    return _channel_feature(trial, vals, channel_names, prefix)


def extract_differential_entropy(trial, channel_names=None, prefix="de_"):
    var = np.var(trial, axis=1, ddof=1)
    vals = 0.5 * np.log(2 * np.pi * np.e * var + 1e-12)
    return _channel_feature(trial, vals, channel_names, prefix)


FEATURE_FUNCTIONS.update({
    "bandpower": extract_bandpower,
    "relative_bandpower": extract_relative_bandpower,
    "psd_stats": extract_psd_stats,
    "spectral_entropy": extract_spectral_entropy,
    "differential_entropy": extract_differential_entropy,
})

# ============================================================
# Nonlinear features
# ============================================================

def extract_sample_entropy(trial, channel_names=None, prefix="sampen_", order=2, r=0.2):
    vals = []

    for x in trial:
        tol = r * np.std(x)
        counts = []

        for m in (order, order + 1):
            emb = np.lib.stride_tricks.sliding_window_view(x, m)
            dist = scipy.spatial.distance.pdist(emb, metric="chebyshev")
            counts.append(np.mean(dist <= tol))

        vals.append(-np.log((counts[1] + 1e-12) / (counts[0] + 1e-12)))

    return _channel_feature(trial, vals, channel_names, prefix)


def extract_permutation_entropy(trial, channel_names=None, prefix="perment_", order=3, delay=1):
    vals = []

    for x in trial:
        patterns = [
            tuple(np.argsort(x[i:i + order * delay:delay]))
            for i in range(len(x) - (order - 1) * delay)
        ]
        _, counts = np.unique(patterns, axis=0, return_counts=True)
        p = counts / counts.sum()
        vals.append(-np.sum(p * np.log(p)) / np.log(math.factorial(order)))

    return _channel_feature(trial, vals, channel_names, prefix)


def _higuchi_fd(x, kmax=10):
    n = len(x)
    lengths = []

    for k in range(1, kmax + 1):
        lm = []

        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 2:
                continue
            length = np.sum(np.abs(np.diff(x[idx])))
            length *= (n - 1) / ((len(idx) - 1) * k ** 2)
            lm.append(length)

        lengths.append(np.mean(lm))

    k = np.arange(1, kmax + 1)
    return np.polyfit(np.log(1.0 / k), np.log(np.asarray(lengths) + 1e-12), 1)[0]


def extract_higuchi_fd(trial, channel_names=None, prefix="hfd_", kmax=10):
    vals = [_higuchi_fd(x, kmax) for x in trial]
    return _channel_feature(trial, vals, channel_names, prefix)


def extract_petrosian_fd(trial, channel_names=None, prefix="pfd_"):
    vals = []

    for x in trial:
        diff = np.diff(x)
        changes = np.sum(diff[1:] * diff[:-1] < 0)
        n = len(x)
        fd = np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * changes + 1e-12)))
        vals.append(fd)

    return _channel_feature(trial, vals, channel_names, prefix)


FEATURE_FUNCTIONS.update({
    "sample_entropy": extract_sample_entropy,
    "permutation_entropy": extract_permutation_entropy,
    "higuchi_fd": extract_higuchi_fd,
    "petrosian_fd": extract_petrosian_fd,
})


# ============================================================
# Covariance features
# ============================================================

def _covariance(trial, eps=1e-6):
    cov = np.cov(trial)
    return cov + eps * (np.trace(cov) / cov.shape[0] + 1e-12) * np.eye(cov.shape[0])


def _matrix_feature(matrix, channel_names, prefix):
    idx = np.triu_indices_from(matrix)
    vals = matrix[idx]
    names = [f"{prefix}{channel_names[i]}_{channel_names[j]}" for i, j in zip(*idx)]
    return vals, names


def extract_covariance(trial, channel_names=None, prefix="cov_"):
    channels = _get_channel_names(trial, channel_names)
    return _matrix_feature(_covariance(trial), channels, prefix)


def extract_logcov(trial, channel_names=None, prefix="logcov_"):
    channels = _get_channel_names(trial, channel_names)
    cov = _covariance(trial)
    eigvals, eigvecs = np.linalg.eigh(cov)
    logcov = (eigvecs * np.log(np.clip(eigvals, 1e-12, None))) @ eigvecs.T
    return _matrix_feature(logcov, channels, prefix)


def extract_eigenvalues(trial, channel_names=None, prefix="eig_"):
    vals = np.linalg.eigvalsh(_covariance(trial))
    return np.asarray(vals), [f"{prefix}{i}" for i in range(len(vals))]


FEATURE_FUNCTIONS.update({
    "cov": extract_covariance,
    "logcov": extract_logcov,
    "eig": extract_eigenvalues,
})


# ============================================================
# Time-frequency features
# ============================================================

def _wavelet_coefficients(trial, wavelet="db4", level=4):
    return [pywt.wavedec(x, wavelet, level=level) for x in trial]


def extract_wavelet_energy(trial, channel_names=None, prefix="waveng_", wavelet="db4", level=4):
    channels = _get_channel_names(trial, channel_names)
    coeffs = _wavelet_coefficients(trial, wavelet, level)
    vals, names = [], []

    for channel, c in zip(channels, coeffs):
        energy = [np.sum(x ** 2) for x in c]
        vals.extend(energy)
        names.extend(f"{prefix}l{i}_{channel}" for i in range(len(energy)))

    return np.asarray(vals), names


def extract_wavelet_entropy(trial, channel_names=None, prefix="wavent_", wavelet="db4", level=4):
    vals = []

    for coeffs in _wavelet_coefficients(trial, wavelet, level):
        energy = np.asarray([np.sum(x ** 2) for x in coeffs])
        p = energy / (energy.sum() + 1e-12)
        vals.append(-np.sum(p * np.log(p + 1e-12)))

    return _channel_feature(trial, vals, channel_names, prefix)


FEATURE_FUNCTIONS.update({
    "wavelet_energy": extract_wavelet_energy,
    "wavelet_entropy": extract_wavelet_entropy,
})

# ============================================================
# Trial extraction
# ============================================================

def extract_features_from_trial(trial, extract_config, channel_names=None, band_labels=None):
    vals_all, names_all = [], []

    if trial.ndim == 2:
        trials = [(None, trial)]
    elif trial.ndim == 3:
        trials = [(i, trial[i]) for i in range(trial.shape[0])]
    else:
        raise ValueError(f"Unexpected trial shape: {trial.shape}")

    for band_idx, band_trial in trials:
        tag = "" if band_idx is None else _format_band_tag(band_idx, band_labels)

        for config in extract_config.values():
            vals, names = config["function"](
                band_trial,
                channel_names=channel_names,
                **config.get("params", {}),
            )
            vals_all.append(np.asarray(vals))
            names_all.extend(tag + name for name in names)

    if not vals_all:
        raise ValueError("Empty extract_config.")

    return np.concatenate(vals_all), names_all


# ============================================================
# Dataset extraction
# ============================================================

def extract_features_to_dataframe(
    dataset,
    extract_config,
    show_progress=True,
    band_labels=None,
    dataset_name=None,
    session_name=None,
):
    rows, feature_names_ref = [], None
    iterator = tqdm(dataset.items(), desc="Subjects") if show_progress else dataset.items()

    for subject_id, sessions in iterator:
        trial_indices = {}

        for session_id in sorted(sessions):
            data = sessions[session_id]
            X, y = data["X"], data["y"]
            channel_names = data["channel_names"]

            output_session = session_name if session_name is not None else session_id
            trial_indices.setdefault(output_session, 0)

            if X.ndim == 3:
                trials = enumerate(X)
            elif X.ndim == 4:
                trials = ((i, X[:, i, :, :]) for i in range(X.shape[1]))
            else:
                raise ValueError(f"Unexpected X shape: {X.shape}")

            for trial_idx, trial in trials:
                vals, names = extract_features_from_trial(
                    trial, extract_config, channel_names, band_labels
                )

                if feature_names_ref is None:
                    feature_names_ref = names
                elif names != feature_names_ref:
                    raise ValueError(
                        f"Feature names or channel order changed in "
                        f"{subject_id}/{session_id}."
                    )

                row = {
                    "dataset": dataset_name,
                    "subject": subject_id,
                    "session": output_session,
                    "trial_index": trial_indices[output_session],
                    "label": str(y[trial_idx]),
                }
                row.update(dict(zip(feature_names_ref, vals)))

                rows.append(row)
                trial_indices[output_session] += 1

    return pd.DataFrame(rows)


# ============================================================
# Validation
# ============================================================

def validate_feature_dataframe(dataframe, metadata_columns=None, identifier_columns=None):
    metadata_columns = metadata_columns or [
        "dataset", "subject", "session", "trial_index", "label"
    ]
    identifier_columns = identifier_columns or [
        "dataset", "subject", "session", "trial_index"
    ]

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if dataframe.empty:
        raise ValueError("The feature DataFrame is empty.")

    duplicated = dataframe.columns[dataframe.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Duplicated column names found: {duplicated}")

    missing = [c for c in metadata_columns if c not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")

    features = [c for c in dataframe.columns if c not in metadata_columns]
    if not features:
        raise ValueError("No feature columns were found.")

    missing_values = [c for c in metadata_columns if dataframe[c].isna().any()]
    if missing_values:
        raise ValueError(f"Missing metadata values found in: {missing_values}")

    trial_indices = pd.to_numeric(dataframe["trial_index"], errors="coerce")
    if trial_indices.isna().any():
        raise ValueError("'trial_index' contains non-numeric values.")
    if (trial_indices < 0).any():
        raise ValueError("'trial_index' contains negative values.")

    if dataframe.duplicated(subset=identifier_columns, keep=False).any():
        raise ValueError("Duplicated trial identifiers found.")

    non_numeric = [
        c for c in features
        if not pd.api.types.is_numeric_dtype(dataframe[c])
    ]
    if non_numeric:
        raise TypeError(f"Non-numeric feature columns found: {non_numeric[:10]}")

    values = dataframe[features].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("NaN or infinite feature values found.")

    return True