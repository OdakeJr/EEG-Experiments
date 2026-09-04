# plugins/eeg/lib/preparation.py

from copy import deepcopy

import pandas as pd

from plugins.eeg.eeg_dataframe import EEGDataFrame
from plugins.eeg.lib.filtering import apply_filters_to_dataset
from plugins.eeg.lib.feature_extraction import (
    extract_features_to_dataframe,
    validate_feature_dataframe,
)


def _split_batches(values, batch_size):
    """
    Split a sequence into batches.

    If batch_size is None, the complete sequence is returned
    as a single batch.
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
    Extract common information from a loaded EEG dataset.

    Assumes the standardized intermediate structure:

        dataset[subject][session] = {
            "X": ...,
            "y": ...,
            "channel_names": ...,
            "sampling_rate": ...
        }
    """
    if not dataset:
        raise ValueError(
            "Cannot extract information from an empty dataset."
        )

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
    Prepare one batch of EEG data.
    """
    # ========================================================
    # Load
    # ========================================================

    dataset = loader(
        **loader_kwargs,
        config=loader_config,
    )

    if not dataset:
        return None, None, None

    # ========================================================
    # Filter / resample
    # ========================================================

    filtered_dataset = apply_filters_to_dataset(
        dataset=dataset,
        config=filter_config,
    )

    channels, sampling_rate = _get_dataset_info(
        filtered_dataset
    )

    # ========================================================
    # Feature extraction
    # ========================================================

    dataframe = extract_features_to_dataframe(
        dataset=filtered_dataset,
        extract_config=extract_config,
        show_progress=show_progress,
        band_labels=band_labels,
        dataset_name=dataset_name,
        session_name=session_name,
    )

    if dataframe.empty:
        return None, channels, sampling_rate

    # ========================================================
    # Validation
    # ========================================================

    validate_feature_dataframe(dataframe)

    return dataframe, channels, sampling_rate


def prepare_eeg_dataframe(
    loader,
    loader_kwargs,
    loader_config,
    filter_config,
    extract_config,
    dataset_name,
    subjects=None,
    subject_batch_size=None,
    band_labels=None,
    session_name=None,
    metadata=None,
    show_progress=False,
):
    """
    Run the common EEG dataset-preparation pipeline.

    Processing
    ----------
    1. Load raw/epoched dataset.
    2. Optionally process subjects in batches.
    3. Filter and resample EEG signals.
    4. Extract features.
    5. Validate each feature DataFrame.
    6. Concatenate all batches.
    7. Validate the complete DataFrame.
    8. Return a standardized EEGDataFrame.

    Parameters
    ----------
    loader : callable
        Dataset-specific loading function.

    loader_kwargs : dict
        Positional-independent arguments required by the loader,
        such as dataset paths.

    loader_config : dict
        Dataset-specific loader configuration.

    filter_config : dict
        Configuration passed to apply_filters_to_dataset.

    extract_config : dict
        Feature extraction configuration.

    dataset_name : str
        Standardized dataset identifier.

    subjects : sequence, optional
        Subjects to process. When provided, subject batching is
        enabled through the loader's ``subjects`` configuration.

    subject_batch_size : int, optional
        Number of subjects loaded simultaneously.

    band_labels : sequence, optional
        Frequency-band labels used to name extracted features.

    session_name : str, optional
        Override session names in the resulting DataFrame.

    metadata : dict, optional
        Additional information stored in the returned
        EEGDataFrame.

    show_progress : bool
        Whether feature extraction should display progress.

    Returns
    -------
    EEGDataFrame
        Standardized prepared EEG feature dataset.
    """
    loader_kwargs = deepcopy(loader_kwargs or {})
    base_loader_config = deepcopy(loader_config or {})
    filter_config = deepcopy(filter_config or {})
    metadata = deepcopy(metadata or {})

    dataframes = []

    channels = None
    sampling_rate = None

    # ========================================================
    # Determine batches
    # ========================================================

    if subjects is None:
        batches = [None]

    else:
        batches = _split_batches(
            values=subjects,
            batch_size=subject_batch_size,
        )

    # ========================================================
    # Process batches
    # ========================================================

    for subject_batch in batches:

        current_loader_config = deepcopy(
            base_loader_config
        )

        if subject_batch is not None:
            current_loader_config["subjects"] = (
                subject_batch
            )

        dataframe, batch_channels, batch_sampling_rate = (
            _prepare_batch(
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
        )

        if dataframe is None:
            continue

        if channels is None:
            channels = batch_channels
            sampling_rate = batch_sampling_rate

        else:
            if batch_channels != channels:
                raise ValueError(
                    "Channel configuration changed between "
                    "dataset batches."
                )

            if batch_sampling_rate != sampling_rate:
                raise ValueError(
                    "Sampling rate changed between "
                    "dataset batches."
                )

        dataframes.append(dataframe)

    # ========================================================
    # Combine batches
    # ========================================================

    if not dataframes:
        raise RuntimeError(
            f"No feature data were generated for "
            f"dataset '{dataset_name}'."
        )

    dataframe = pd.concat(
        dataframes,
        axis=0,
        ignore_index=True,
    )

    # ========================================================
    # Final validation
    # ========================================================

    validate_feature_dataframe(dataframe)

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

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in {
            "dataset",
            "subject",
            "session",
            "trial_index",
            "label",
        }
    ]

    # ========================================================
    # Standardized output
    # ========================================================

    return EEGDataFrame(
        data=dataframe,
        dataset_name=dataset_name,
        channels=channels,
        sampling_rate=sampling_rate,
        feature_columns=feature_columns,
        label_column="label",
        metadata=metadata,
    )