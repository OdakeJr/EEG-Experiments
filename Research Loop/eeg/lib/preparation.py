# eeg/lib/preparation.py

from copy import deepcopy

import pandas as pd

from eeg.lib.filtering import apply_filters_to_dataset
from eeg.lib.feature_extraction import (
    build_extract_config,
    extract_features_to_dataframe,
    validate_feature_dataframe,
)


def _split_batches(values, batch_size):
    """
    Split values into batches.

    If batch_size is None, process everything together.
    """
    values = list(values)

    if batch_size is None:
        return [values]

    if batch_size <= 0:
        raise ValueError(
            "'batch_size' must be greater than zero."
        )

    return [
        values[start:start + batch_size]
        for start in range(0, len(values), batch_size)
    ]


def _get_dataset_info(dataset):
    """
    Get channel names and sampling rate from a standardized
    EEG dataset.
    """
    for sessions in dataset.values():
        for session_data in sessions.values():
            return (
                list(session_data["channel_names"]),
                float(session_data["sampling_rate"]),
            )

    raise ValueError(
        "The dataset contains no subject/session data."
    )


def _prepare_batch(
    loader,
    loader_kwargs,
    loader_config,
    filter_config,
    extract_config,
    dataset_name,
    band_labels=None,
    session_name=None,
    show_progress=False,
):
    """
    Prepare one dataset batch.
    """

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    dataset = loader(
        **loader_kwargs,
        config=loader_config,
    )

    if not dataset:
        return None, None, None

    # --------------------------------------------------
    # Filter / resample
    # --------------------------------------------------

    dataset = apply_filters_to_dataset(
        dataset=dataset,
        config=filter_config,
    )

    channels, sampling_rate = _get_dataset_info(
        dataset
    )

    # --------------------------------------------------
    # Feature extraction
    # --------------------------------------------------

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


def prepare_eeg_dataframe(
    loader,
    loader_kwargs,
    loader_config,
    filter_config,
    feature_config,
    dataset_name,
    subjects=None,
    subject_batch_size=None,
    band_labels=None,
    session_name=None,
    metadata=None,
    show_progress=False,
):
    """
    Run the common EEG preparation pipeline.

    Processing
    ----------
    1. Load dataset.
    2. Optionally process subjects in batches.
    3. Filter and resample EEG signals.
    4. Extract features.
    5. Combine batches.
    6. Validate the final feature DataFrame.

    Returns
    -------
    dataframe : pandas.DataFrame
        Standardized feature-level dataset.

    info : dict
        Information describing the generated dataset.
    """

    loader_kwargs = deepcopy(
        loader_kwargs or {}
    )

    loader_config = deepcopy(
        loader_config or {}
    )

    filter_config = deepcopy(
        filter_config or {}
    )

    metadata = deepcopy(
        metadata or {}
    )

    # --------------------------------------------------
    # Feature configuration
    # --------------------------------------------------

    extract_config = build_extract_config(
        feature_config
    )

    # --------------------------------------------------
    # Determine batches
    # --------------------------------------------------

    if subjects is None:
        batches = [None]

    else:
        batches = _split_batches(
            subjects,
            subject_batch_size,
        )

    dataframes = []

    channels = None
    sampling_rate = None

    # --------------------------------------------------
    # Process batches
    # --------------------------------------------------

    for subject_batch in batches:

        current_loader_config = deepcopy(
            loader_config
        )

        if subject_batch is not None:
            current_loader_config[
                "subjects"
            ] = subject_batch

        (
            dataframe,
            batch_channels,
            batch_sampling_rate,
        ) = _prepare_batch(
            loader=loader,
            loader_kwargs=loader_kwargs,
            loader_config=current_loader_config,
            filter_config=filter_config,
            extract_config=extract_config,
            dataset_name=dataset_name,
            band_labels=band_labels,
            session_name=session_name,
            show_progress=show_progress,
        )

        if dataframe is None:
            continue

        # --------------------------------------------------
        # Ensure batches are compatible
        # --------------------------------------------------

        if channels is None:
            channels = batch_channels
            sampling_rate = batch_sampling_rate

        else:

            if batch_channels != channels:
                raise ValueError(
                    "Channel configuration changed "
                    "between batches."
                )

            if batch_sampling_rate != sampling_rate:
                raise ValueError(
                    "Sampling rate changed "
                    "between batches."
                )

        dataframes.append(dataframe)

    # --------------------------------------------------
    # Combine
    # --------------------------------------------------

    if not dataframes:
        raise RuntimeError(
            f"No feature data were generated for "
            f"dataset '{dataset_name}'."
        )

    dataframe = pd.concat(
        dataframes,
        ignore_index=True,
    )

    # --------------------------------------------------
    # Final validation
    # --------------------------------------------------

    validate_feature_dataframe(
        dataframe
    )

    metadata_columns = [
        "dataset",
        "subject",
        "session",
        "trial_index",
        "label",
    ]

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in metadata_columns
    ]

    # --------------------------------------------------
    # Dataset information
    # --------------------------------------------------

    metadata.setdefault(
        "n_subjects",
        dataframe["subject"].nunique(),
    )

    metadata.setdefault(
        "n_sessions",
        dataframe["session"].nunique(),
    )

    metadata.setdefault(
        "n_trials",
        len(dataframe),
    )

    info = {
        "dataset_name": dataset_name,
        "channels": channels,
        "sampling_rate": sampling_rate,
        "feature_columns": feature_columns,
        "label_column": "label",
        "domain_columns": [
            "dataset",
            "subject",
            "session",
        ],
        "metadata_columns": [
            "trial_index",
        ],
        "metadata": metadata,
    }

    return dataframe, info