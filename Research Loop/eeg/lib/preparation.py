# eeg/lib/preparation.py

from copy import deepcopy

import numpy as np
import pandas as pd

from eeg.lib.filtering import apply_filters_to_dataset
from eeg.lib.feature_extraction import (
    build_extract_config,
    extract_features_to_dataframe,
    validate_feature_dataframe,
)


# ============================================================
# Helpers
# ============================================================

def _split_batches(values, batch_size):
    values = list(values)

    if batch_size is None:
        return [values]

    if batch_size <= 0:
        raise ValueError("'batch_size' must be greater than zero.")

    return [
        values[i:i + batch_size]
        for i in range(0, len(values), batch_size)
    ]


def _get_dataset_info(dataset):
    for sessions in dataset.values():
        for data in sessions.values():
            return list(data["channel_names"]), float(data["sampling_rate"])

    raise ValueError("The dataset contains no subject/session data.")


# ============================================================
# Signal preprocessing
# ============================================================

def _exponential_standardize(X, factor_new=1e-3, eps=1e-4):
    """
    Exponential moving standardization along the time axis.

    Supports [..., time].
    """

    X = np.asarray(X, dtype=np.float32)
    flat = X.reshape(-1, X.shape[-1])

    mean = flat[:, 0].copy()
    var = np.zeros_like(mean)
    out = np.empty_like(flat)

    out[:, 0] = 0.0

    for t in range(1, flat.shape[1]):
        mean = factor_new * flat[:, t] + (1.0 - factor_new) * mean
        diff = flat[:, t] - mean
        var = factor_new * diff**2 + (1.0 - factor_new) * var
        out[:, t] = diff / np.maximum(np.sqrt(var), eps)

    return out.reshape(X.shape)


def _apply_signal_preprocessing(dataset, config=None):
    config = config or {}

    scale = float(config.get("scale", 1.0))
    exp_config = config.get("exponential_standardize", {})
    exp_enabled = exp_config.get("enabled", False)

    for sessions in dataset.values():
        for data in sessions.values():
            X = np.asarray(data["X"], dtype=np.float32)

            if scale != 1.0:
                X = X * scale

            if exp_enabled:
                X = _exponential_standardize(
                    X,
                    factor_new=exp_config.get("factor_new", 1e-3),
                    eps=exp_config.get("eps", 1e-4),
                )

            data["X"] = X

    return dataset


def _extract_signal(dataset, dataset_name, session_name=None):
    X_all = []
    dataset_all = []
    subject_all = []
    session_all = []
    trial_all = []
    label_all = []

    for subject, sessions in dataset.items():
        trial_indices = {}

        for session, data in sessions.items():
            X = np.asarray(data["X"])
            y = np.asarray(data["y"])

            output_session = session_name if session_name is not None else session
            trial_indices.setdefault(output_session, 0)

            if X.ndim == 4:
                X = np.transpose(X, (1, 0, 2, 3))
            elif X.ndim != 3:
                raise ValueError(f"Unexpected signal shape: {X.shape}")

            if len(X) != len(y):
                raise ValueError("Signal and label counts do not match.")

            n = len(X)
            start = trial_indices[output_session]

            X_all.append(X.astype(np.float32, copy=False))
            dataset_all.extend([dataset_name] * n)
            subject_all.extend([subject] * n)
            session_all.extend([output_session] * n)
            trial_all.extend(range(start, start + n))
            label_all.extend(y.astype(str))

            trial_indices[output_session] += n

    if not X_all:
        return None

    return {
        "X": np.concatenate(X_all, axis=0),
        "dataset": np.asarray(dataset_all),
        "subject": np.asarray(subject_all),
        "session": np.asarray(session_all),
        "trial_index": np.asarray(trial_all),
        "label": np.asarray(label_all),
    }


# ============================================================
# Batch preparation
# ============================================================

def _prepare_batch(
    loader,
    loader_kwargs,
    loader_config,
    filter_config,
    signal_preprocessing_config,
    representation,
    extract_config,
    dataset_name,
    band_labels=None,
    session_name=None,
    show_progress=False,
):
    dataset = loader(
        **loader_kwargs,
        config=loader_config,
    )

    if not dataset:
        return None, None, None

    # V -> µV or other dataset-specific scaling.
    scale = (signal_preprocessing_config or {}).get("scale", 1.0)

    if scale != 1.0:
        dataset = _apply_signal_preprocessing(
            dataset,
            {"scale": scale},
        )

    # Frequency filtering / resampling.
    dataset = apply_filters_to_dataset(
        dataset=dataset,
        config=filter_config,
    )

    # Signal normalization after filtering.
    exp_config = (
        signal_preprocessing_config or {}
    ).get("exponential_standardize", {})

    if exp_config.get("enabled", False):
        dataset = _apply_signal_preprocessing(
            dataset,
            {
                "exponential_standardize": exp_config,
            },
        )

    channels, sampling_rate = _get_dataset_info(dataset)

    if representation == "signal":
        return (
            _extract_signal(dataset, dataset_name, session_name),
            channels,
            sampling_rate,
        )

    dataframe = extract_features_to_dataframe(
        dataset=dataset,
        extract_config=extract_config,
        show_progress=show_progress,
        band_labels=band_labels,
        dataset_name=dataset_name,
        session_name=session_name,
    )

    if dataframe.empty:
        return None, channels, sampling_rate

    validate_feature_dataframe(dataframe)

    return dataframe, channels, sampling_rate


# ============================================================
# Main preparation
# ============================================================

def prepare_eeg_dataframe(
    loader,
    loader_kwargs,
    loader_config,
    filter_config,
    feature_config,
    dataset_name,
    representation="features",
    signal_preprocessing_config=None,
    subjects=None,
    subject_batch_size=None,
    band_labels=None,
    session_name=None,
    metadata=None,
    show_progress=False,
):
    """
    Common EEG preparation.

    representation:
        "features" -> [trials, features]
        "signal"   -> [trials, ..., channels, time]
    """

    if representation not in {"features", "signal"}:
        raise ValueError(f"Unknown representation: {representation}")

    loader_kwargs = deepcopy(loader_kwargs or {})
    loader_config = deepcopy(loader_config or {})
    filter_config = deepcopy(filter_config or {})
    signal_preprocessing_config = deepcopy(
        signal_preprocessing_config or {}
    )
    metadata = deepcopy(metadata or {})

    extract_config = (
        build_extract_config(feature_config)
        if representation == "features"
        else None
    )

    batches = (
        [None]
        if subjects is None
        else _split_batches(subjects, subject_batch_size)
    )

    prepared = []
    channels = None
    sampling_rate = None

    for subject_batch in batches:
        config = deepcopy(loader_config)

        if subject_batch is not None:
            config["subjects"] = subject_batch

        data, batch_channels, batch_fs = _prepare_batch(
            loader=loader,
            loader_kwargs=loader_kwargs,
            loader_config=config,
            filter_config=filter_config,
            signal_preprocessing_config=signal_preprocessing_config,
            representation=representation,
            extract_config=extract_config,
            dataset_name=dataset_name,
            band_labels=band_labels,
            session_name=session_name,
            show_progress=show_progress,
        )

        if data is None:
            continue

        if channels is None:
            channels = batch_channels
            sampling_rate = batch_fs
        elif batch_channels != channels or batch_fs != sampling_rate:
            raise ValueError(
                "Dataset configuration changed between batches."
            )

        prepared.append(data)

    if not prepared:
        raise RuntimeError(f"No data generated for '{dataset_name}'.")

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    if representation == "features":
        data = pd.concat(prepared, ignore_index=True)
        validate_feature_dataframe(data)

        metadata_columns = [
            "dataset", "subject", "session", "trial_index", "label",
        ]

        feature_columns = [
            column
            for column in data.columns
            if column not in metadata_columns
        ]

        input_shape = (len(feature_columns),)
        subjects_data = data["subject"]
        sessions_data = data["session"]
        n_trials = len(data)

    else:
        keys = [
            "X", "dataset", "subject", "session", "trial_index", "label",
        ]

        data = {
            key: np.concatenate(
                [batch[key] for batch in prepared],
                axis=0,
            )
            for key in keys
        }

        feature_columns = None
        input_shape = tuple(data["X"].shape[1:])
        subjects_data = data["subject"]
        sessions_data = data["session"]
        n_trials = len(data["X"])

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata.setdefault("n_subjects", len(np.unique(subjects_data)))
    metadata.setdefault("n_sessions", len(np.unique(sessions_data)))
    metadata.setdefault("n_trials", n_trials)

    info = {
        "dataset_name": dataset_name,
        "representation": representation,
        "channels": channels,
        "channel_names": channels,
        "sampling_rate": sampling_rate,
        "band_labels": band_labels,
        "input_shape": input_shape,
        "feature_columns": feature_columns,
        "label_column": "label",
        "domain_columns": ["dataset", "subject", "session"],
        "metadata_columns": ["trial_index"],
        "metadata": metadata,
    }

    return data, info