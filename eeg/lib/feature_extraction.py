# eeg/lib/feature_extraction.py

from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.fft
import scipy.linalg
import scipy.stats
from tqdm import tqdm


# ============================================================
# Feature registry
# ============================================================

FEATURE_FUNCTIONS = {}


DEFAULT_FEATURE_CONFIG = {
    "mean": {},
    "std": {},
    "mom": {},
    "min": {},
    "max": {},
    "cov": {},
    "eig": {},
    "h_diff": {},
    "q_stats": {},
    "logvar": {},
}


def build_extract_config(feature_config=None):
    """
    Convert a simple feature configuration into the format
    expected by the feature extraction functions.
    """

    if feature_config is None:
        feature_config = DEFAULT_FEATURE_CONFIG

    extract_config = {}

    for feature_name, feature_params in feature_config.items():

        if feature_name not in FEATURE_FUNCTIONS:
            raise ValueError(
                f"Unknown EEG feature '{feature_name}'. "
                f"Available features: {sorted(FEATURE_FUNCTIONS)}"
            )

        if feature_params is False:
            continue

        if feature_params is None:
            feature_params = {}

        if not isinstance(feature_params, dict):
            raise TypeError(
                f"Parameters for feature '{feature_name}' "
                "must be a dictionary."
            )

        extract_config[feature_name] = {
            "function": FEATURE_FUNCTIONS[feature_name],
            "params": deepcopy(feature_params),
        }

    if not extract_config:
        raise ValueError(
            "At least one feature must be enabled."
        )

    return extract_config


# ============================================================
# Helpers
# ============================================================

def _get_channel_names(trial, channel_names=None):
    """
    Return electrode names or numeric indices as fallback.
    """

    if channel_names is None:
        return [
            str(i)
            for i in range(trial.shape[0])
        ]

    if len(channel_names) != trial.shape[0]:
        raise ValueError(
            f"{len(channel_names)} channel names were provided, "
            f"but the trial has {trial.shape[0]} channels."
        )

    return list(channel_names)


def _format_band_tag(band_idx, band_labels=None):

    if (
        band_labels is not None
        and band_idx < len(band_labels)
    ):
        low, high = band_labels[band_idx]
        return f"b{low}_{high}_"

    return f"b{band_idx}_"


# ============================================================
# Feature functions
# ============================================================

def extract_logvar(
    trial,
    channel_names=None,
    prefix="logvar_",
):
    channel_names = _get_channel_names(
        trial,
        channel_names,
    )

    vals = np.log(
        np.var(trial, axis=1) + 1e-12
    )

    names = [
        f"{prefix}{channel}"
        for channel in channel_names
    ]

    return vals, names


def extract_bandpower(
    trial,
    sfreq=128.0,
    bands=None,
    channel_names=None,
    prefix="bp_",
):
    channel_names = _get_channel_names(
        trial,
        channel_names,
    )

    if bands is None:
        bands = [
            (8, 12),
            (13, 30),
        ]

    n_samples = trial.shape[1]

    freqs = np.fft.rfftfreq(
        n_samples,
        d=1.0 / sfreq,
    )

    psd = np.abs(
        np.fft.rfft(
            trial,
            axis=1,
        )
    ) ** 2

    vals = []
    names = []

    for low, high in bands:

        mask = (
            (freqs >= low)
            & (freqs <= high)
        )

        bp = psd[:, mask].sum(axis=1)

        vals.append(bp)

        names.extend([
            f"{prefix}{low}_{high}_{channel}"
            for channel in channel_names
        ])

    return np.concatenate(vals), names


def extract_mean(
    trial,
    channel_names=None,
    prefix="mean_",
):
    channel_names = _get_channel_names(
        trial,
        channel_names,
    )

    vals = np.mean(
        trial,
        axis=1,
    )

    names = [
        f"{prefix}{channel}"
        for channel in channel_names
    ]

    return vals, names


def extract_std(
    trial,
    channel_names=None,
    prefix="std_",
):
    channel_names = _get_channel_names(
        trial,
        channel_names,
    )

    vals = np.std(
        trial,
        axis=1,
        ddof=1,
    )

    names = [
        f"{prefix}{channel}"
        for channel in channel_names
    ]

    return vals, names


def extract_moments(
    trial,
    channel_names=None,
    prefix="mom_",
):
    channel_names = _get_channel_names(
        trial,
        channel_names,
    )

    skew = scipy.stats.skew(
        trial,
        axis=1,
        bias=False,
    )

    kurtosis = scipy.stats.kurtosis(
        trial,
        axis=1,
        bias=False,
    )

    vals = np.concatenate([
        skew,
        kurtosis,
    ])

    names = [
        f"{prefix}skew_{channel}"
        for channel in channel_names
    ]

    names.extend([
        f"{prefix}kurt_{channel}"
        for channel in channel_names
    ])

    return vals, names


def extract_min(
    trial,
    channel_names=None,
    prefix="min_",
):
    channel_names = _get_channel_names(
        trial,
        channel_names,
    )

    vals = np.min(
        trial,
        axis=1,
    )

    names = [
        f"{prefix}{channel}"
        for channel in channel_names
    ]

    return vals, names


def extract_max(
    trial,
    channel_names=None,
    prefix="max_",
):
    channel_names = _get_channel_names(
        trial,
        channel_names,
    )

    vals = np.max(
        trial,
        axis=1,
    )

    names = [
        f"{prefix}{channel}"
        for channel in channel_names
    ]

    return vals, names


def extract_covariance(
    trial,
    channel_names=None,
    prefix="cov_",
):
    channel_names = _get_channel_names(
        trial,
        channel_names,
    )

    cov = np.cov(trial)

    idx = np.triu_indices_from(
        cov
    )

    vals = cov[idx]

    names = [
        f"{prefix}{channel_names[i]}_{channel_names[j]}"
        for i, j in zip(
            idx[0],
            idx[1],
        )
    ]

    return vals, names


def extract_eigenvalues(
    trial,
    channel_names=None,
    prefix="eig_",
):
    cov = np.cov(trial)

    vals = np.linalg.eigvalsh(
        cov
    )

    names = [
        f"{prefix}{i}"
        for i in range(len(vals))
    ]

    return vals, names


def extract_logcov(
    trial,
    channel_names=None,
    prefix="logcov_",
):
    channel_names = _get_channel_names(
        trial,
        channel_names,
    )

    cov = np.cov(trial)

    log_cov = scipy.linalg.logm(
        cov
    )

    idx = np.triu_indices_from(
        log_cov
    )

    vals = np.real(
        log_cov[idx]
    )

    names = [
        f"{prefix}{channel_names[i]}_{channel_names[j]}"
        for i, j in zip(
            idx[0],
            idx[1],
        )
    ]

    return vals, names


def extract_fft(
    trial,
    channel_names=None,
    prefix="fft_",
    ntop=10,
):
    channel_names = _get_channel_names(
        trial,
        channel_names,
    )

    fft_vals = np.abs(
        scipy.fft.fft(
            trial,
            axis=1,
        )
    )

    fft_vals = fft_vals[
        :,
        :fft_vals.shape[1] // 2,
    ]

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
    channel_names = _get_channel_names(
        trial,
        channel_names,
    )

    n = trial.shape[1] // 2

    h1 = trial[:, :n]
    h2 = trial[:, n:]

    vals = (
        np.mean(h2, axis=1)
        - np.mean(h1, axis=1)
    )

    names = [
        f"{prefix}{channel}"
        for channel in channel_names
    ]

    return vals, names


def extract_quarters_stats(
    trial,
    channel_names=None,
    prefix="q_",
):
    channel_names = _get_channel_names(
        trial,
        channel_names,
    )

    n = trial.shape[1]

    quarters = np.split(
        trial,
        [
            n // 4,
            n // 2,
            3 * n // 4,
        ],
        axis=1,
    )

    vals = []
    names = []

    for i, quarter in enumerate(
        quarters
    ):

        mean = np.mean(
            quarter,
            axis=1,
        )

        vals.append(mean)

        names.extend([
            f"{prefix}{i}_{channel}"
            for channel in channel_names
        ])

    return np.concatenate(vals), names


# ============================================================
# Feature registry
# ============================================================

FEATURE_FUNCTIONS.update({
    "mean": extract_mean,
    "std": extract_std,
    "mom": extract_moments,
    "min": extract_min,
    "max": extract_max,
    "cov": extract_covariance,
    "eig": extract_eigenvalues,
    "logcov": extract_logcov,
    "fft": extract_fft,
    "h_diff": extract_halves_diff,
    "q_stats": extract_quarters_stats,
    "logvar": extract_logvar,
    "bandpower": extract_bandpower,
})


# ============================================================
# Trial extraction
# ============================================================

def extract_features_from_trial(
    trial,
    extract_config,
    channel_names=None,
    band_labels=None,
):
    vals_all = []
    names_all = []

    # Single signal representation.
    if trial.ndim == 2:

        for config in extract_config.values():

            function = config["function"]
            params = config.get(
                "params",
                {},
            )

            vals, names = function(
                trial,
                channel_names=channel_names,
                **params,
            )

            vals_all.append(
                np.asarray(vals)
            )

            names_all.extend(names)

    # Multiple filtered frequency bands.
    elif trial.ndim == 3:

        for band_idx in range(
            trial.shape[0]
        ):

            band_trial = trial[
                band_idx
            ]

            band_tag = _format_band_tag(
                band_idx,
                band_labels,
            )

            for config in extract_config.values():

                function = config[
                    "function"
                ]

                params = config.get(
                    "params",
                    {},
                )

                vals, names = function(
                    band_trial,
                    channel_names=channel_names,
                    **params,
                )

                vals_all.append(
                    np.asarray(vals)
                )

                names_all.extend([
                    band_tag + name
                    for name in names
                ])

    else:
        raise ValueError(
            f"Unexpected trial shape: "
            f"{trial.shape}"
        )

    if not vals_all:
        raise ValueError(
            "Empty extract_config."
        )

    return (
        np.concatenate(vals_all),
        names_all,
    )


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
    all_features = []

    feature_names_ref = None

    iterator = dataset.items()

    if show_progress:
        iterator = tqdm(
            iterator,
            desc="Subjects",
        )

    for subject_id, sessions in iterator:

        trial_indices = {}

        for session_id in sorted(
            sessions
        ):

            session_data = sessions[
                session_id
            ]

            X = session_data["X"]
            y = session_data["y"]

            channel_names = session_data[
                "channel_names"
            ]

            output_session = (
                session_name
                if session_name is not None
                else session_id
            )

            if output_session not in trial_indices:
                trial_indices[
                    output_session
                ] = 0

            if X.ndim == 3:

                trial_iterator = enumerate(
                    X
                )

            elif X.ndim == 4:

                trial_iterator = (
                    (
                        i,
                        X[:, i, :, :],
                    )
                    for i in range(
                        X.shape[1]
                    )
                )

            else:
                raise ValueError(
                    f"Unexpected X shape: "
                    f"{X.shape}"
                )

            for local_trial_idx, trial in trial_iterator:

                vals, names = (
                    extract_features_from_trial(
                        trial=trial,
                        extract_config=extract_config,
                        channel_names=channel_names,
                        band_labels=band_labels,
                    )
                )

                if feature_names_ref is None:
                    feature_names_ref = (
                        names
                    )

                elif names != feature_names_ref:
                    raise ValueError(
                        "Feature names or channel order "
                        f"changed in "
                        f"{subject_id}/{session_id}."
                    )

                row = {
                    "dataset": dataset_name,
                    "subject": subject_id,
                    "session": output_session,
                    "trial_index": (
                        trial_indices[
                            output_session
                        ]
                    ),
                    "label": str(
                        y[local_trial_idx]
                    ),
                }

                row.update({
                    feature_names_ref[i]: vals[i]
                    for i in range(
                        len(vals)
                    )
                })

                all_features.append(
                    row
                )

                trial_indices[
                    output_session
                ] += 1

    return pd.DataFrame(
        all_features
    )


# ============================================================
# Validation
# ============================================================

def validate_feature_dataframe(
    dataframe,
    metadata_columns=None,
    identifier_columns=None,
):
    """
    Validate an extracted EEG feature DataFrame.
    """

    if metadata_columns is None:
        metadata_columns = [
            "dataset",
            "subject",
            "session",
            "trial_index",
            "label",
        ]

    if identifier_columns is None:
        identifier_columns = [
            "dataset",
            "subject",
            "session",
            "trial_index",
        ]

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "Expected a pandas DataFrame."
        )

    if dataframe.empty:
        raise ValueError(
            "The feature DataFrame is empty."
        )

    # --------------------------------------------------------
    # Columns
    # --------------------------------------------------------

    duplicated_columns = (
        dataframe.columns[
            dataframe.columns.duplicated()
        ].tolist()
    )

    if duplicated_columns:
        raise ValueError(
            "Duplicated column names found: "
            f"{duplicated_columns}"
        )

    missing_metadata = [
        column
        for column in metadata_columns
        if column not in dataframe.columns
    ]

    if missing_metadata:
        raise ValueError(
            "Missing required metadata columns: "
            f"{missing_metadata}"
        )

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in metadata_columns
    ]

    if not feature_columns:
        raise ValueError(
            "No feature columns were found."
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    missing_values = [
        column
        for column in metadata_columns
        if dataframe[column].isna().any()
    ]

    if missing_values:
        raise ValueError(
            "Missing metadata values found in: "
            f"{missing_values}"
        )

    # --------------------------------------------------------
    # Trial identifiers
    # --------------------------------------------------------

    trial_indices = pd.to_numeric(
        dataframe["trial_index"],
        errors="coerce",
    )

    if trial_indices.isna().any():
        raise ValueError(
            "'trial_index' contains "
            "non-numeric values."
        )

    if (
        trial_indices < 0
    ).any():
        raise ValueError(
            "'trial_index' contains "
            "negative values."
        )

    duplicate_mask = dataframe.duplicated(
        subset=identifier_columns,
        keep=False,
    )

    if duplicate_mask.any():
        raise ValueError(
            "Duplicated trial identifiers found."
        )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    non_numeric_features = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(
            dataframe[column]
        )
    ]

    if non_numeric_features:
        raise TypeError(
            "Non-numeric feature columns found: "
            f"{non_numeric_features[:10]}"
        )

    feature_values = dataframe[
        feature_columns
    ].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    if not np.isfinite(
        feature_values
    ).all():
        raise ValueError(
            "NaN or infinite feature values found."
        )

    return True