# pipeline/process_data.py

import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd

from eeg.datasets.bci2a import prepare_bci2a
from eeg.datasets.eegmmidb import prepare_eegmmidb
from eeg.datasets.weibo import prepare_weibo
from eeg.datasets.zhou import prepare_zhou

from models.dataset_view import DatasetView

from utils.storage import (
    exists,
    load_manifest,
    save_data,
    save_manifest,
)

from utils.status import (
    is_done,
    make_manifest,
    make_signature,
)


OUTPUT_ROOT = Path("outputs/preprocessing")


# ============================================================
# Dataset preparation registry
# ============================================================

DATASET_PREPARERS = {
    "bci2a": prepare_bci2a,
    "eegmmidb": prepare_eegmmidb,
    "weibo": prepare_weibo,
    "zhou": prepare_zhou,
}


# ============================================================
# Representation
# ============================================================

SUPPORTED_REPRESENTATIONS = {
    "features",
    "signal",
}


def _get_representation(params):
    """
    Return and validate the preprocessing output representation.

    features:
        X is a 2D feature matrix [trials, features].

    signal:
        X is a multidimensional EEG tensor, typically
        [trials, bands, channels, time].
    """

    representation = params.get(
        "representation",
        "features",
    )

    if representation not in SUPPORTED_REPRESENTATIONS:
        raise ValueError(
            f"Unknown representation '{representation}'. "
            f"Available representations: "
            f"{sorted(SUPPORTED_REPRESENTATIONS)}"
        )

    return representation


# ============================================================
# Helpers
# ============================================================

def _preprocessing_signature(params):
    return make_signature(params)


def _preprocessing_config_label(params):
    """
    Human-readable preprocessing label.

    Prefer the explicit params["name"] because this is what we use
    to name the preprocessing setup.
    """

    signature = _preprocessing_signature(params)

    return params.get(
        "name",
        f"{params['dataset']}_{signature[:8]}",
    )


def _add_preprocessing_trace(
    info,
    params,
    representation,
):
    """
    Add clean preprocessing trace to the saved output info.

    These fields are propagated through the pipeline and later used
    by analysis modules.
    """

    traced_info = deepcopy(info)

    traced_info["representation"] = representation

    traced_info["preprocessing_signature"] = (
        _preprocessing_signature(params)
    )

    traced_info["preprocessing_config_label"] = (
        _preprocessing_config_label(params)
    )

    traced_info["preprocessing_params"] = deepcopy(
        params
    )

    return traced_info


def _get_output_paths(params):
    """
    Determine where this preprocessing result will be stored.
    """

    dataset_name = params["dataset"]

    setup_name = params.get(
        "name",
        make_signature(params)[:8],
    )

    output_dir = (
        OUTPUT_ROOT
        / dataset_name
        / setup_name
    )

    return (
        output_dir / "data",
        output_dir / "manifest.json",
    )


# ============================================================
# Prepared data normalization
# ============================================================

def _prepare_feature_data(
    prepared_data,
    info,
):
    """
    Convert the current feature DataFrame representation into
    the dictionary format expected by storage.
    """

    if not isinstance(
        prepared_data,
        pd.DataFrame,
    ):
        raise TypeError(
            "Feature representation must be returned "
            "as a pandas DataFrame."
        )

    data = {
        column: prepared_data[
            column
        ].to_numpy()
        for column in prepared_data.columns
    }

    feature_columns = info.get(
        "feature_columns",
        [],
    )

    info["feature_columns"] = list(
        feature_columns
    )

    info["input_shape"] = (
        len(feature_columns),
    )

    return data, info


def _prepare_signal_data(
    prepared_data,
    info,
):
    """
    Validate the signal representation.

    Expected structure
    ------------------
    prepared_data = {
        "X": ndarray [trials, ..., channels, time],
        "dataset": ...,
        "subject": ...,
        "session": ...,
        "trial_index": ...,
        "label": ...,
    }

    All metadata arrays must share the same first dimension as X.
    """

    if not isinstance(
        prepared_data,
        dict,
    ):
        raise TypeError(
            "Signal representation must be returned "
            "as a dictionary."
        )

    if "X" not in prepared_data:
        raise ValueError(
            "Signal representation must contain an 'X' array."
        )

    X = np.asarray(
        prepared_data["X"]
    )

    if X.ndim < 3:
        raise ValueError(
            "Signal X must have at least 3 dimensions: "
            "[trials, channels, time], optionally with "
            "additional dimensions such as frequency bands."
        )

    n_trials = X.shape[0]

    metadata_columns = info.get(
        "metadata_columns",
        [],
    )

    for column in metadata_columns:

        if column not in prepared_data:
            raise ValueError(
                f"Signal representation is missing "
                f"metadata column '{column}'."
            )

        values = np.asarray(
            prepared_data[column]
        )

        if len(values) != n_trials:
            raise ValueError(
                f"Metadata column '{column}' has "
                f"{len(values)} samples, but X has "
                f"{n_trials} trials."
            )

    data = {
        key: np.asarray(value)
        for key, value in prepared_data.items()
    }

    info["feature_columns"] = None

    info["input_shape"] = tuple(
        int(value)
        for value in X.shape[1:]
    )

    return data, info


def _prepare_output_data(
    prepared_data,
    info,
    representation,
):
    """
    Normalize dataset-specific preparer outputs into one
    common storage contract.
    """

    info = deepcopy(info)

    if representation == "features":
        return _prepare_feature_data(
            prepared_data,
            info,
        )

    if representation == "signal":
        return _prepare_signal_data(
            prepared_data,
            info,
        )

    raise ValueError(
        f"Unsupported representation: "
        f"{representation}"
    )


# ============================================================
# DatasetView
# ============================================================

def _make_dataset_view(
    data_path,
    manifest_path,
    info,
):
    """
    Build the lightweight reference returned by preprocessing.
    """

    input_shape = info.get(
        "input_shape"
    )

    if input_shape is not None:
        input_shape = tuple(
            input_shape
        )

    band_labels = info.get(
        "band_labels"
    )

    if band_labels is not None:
        band_labels = [
            tuple(band)
            for band in band_labels
        ]

    return DatasetView(
        path=str(data_path),

        representation=info.get(
            "representation",
            "features",
        ),

        feature_columns=info.get(
            "feature_columns"
        ),

        label_column=info[
            "label_column"
        ],

        domain_columns=info[
            "domain_columns"
        ],

        metadata_columns=info[
            "metadata_columns"
        ],

        manifest_path=str(
            manifest_path
        ),

        input_shape=input_shape,

        channel_names=info.get(
            "channel_names"
        ),

        band_labels=band_labels,

        preprocessing_signature=info.get(
            "preprocessing_signature"
        ),

        preprocessing_config_label=info.get(
            "preprocessing_config_label"
        ),

        preprocessing_params=info.get(
            "preprocessing_params"
        ),
    )


# ============================================================
# Main preprocessing function
# ============================================================

def run_preprocessing(params):
    """
    Run one preprocessing configuration.

    Supported output representations
    --------------------------------
    features:
        [trials, features]

    signal:
        [trials, channels, time]
        or
        [trials, bands, channels, time]

    If the same configuration was already completed,
    reuse the stored result.

    Returns
    -------
    DatasetView
        Lightweight reference to the processed dataset.
    """

    representation = _get_representation(
        params
    )

    dataset_name = params["dataset"]

    if dataset_name not in DATASET_PREPARERS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available datasets: "
            f"{sorted(DATASET_PREPARERS)}"
        )

    preparer = DATASET_PREPARERS[
        dataset_name
    ]

    data_path, manifest_path = (
        _get_output_paths(params)
    )

    # --------------------------------------------------------
    # Reuse previous result
    # --------------------------------------------------------

    if (
        exists(data_path)
        and is_done(
            manifest_path,
            params,
        )
    ):
        manifest = load_manifest(
            manifest_path
        )

        info = _add_preprocessing_trace(
            manifest["output"],
            params,
            representation,
        )

        return _make_dataset_view(
            data_path=data_path,
            manifest_path=manifest_path,
            info=info,
        )

    # --------------------------------------------------------
    # Start execution
    # --------------------------------------------------------

    save_manifest(
        make_manifest(
            status="running",
            params=params,
        ),
        manifest_path,
    )

    start_time = time.time()

    try:

        # ----------------------------------------------------
        # Dataset-specific preparation
        # ----------------------------------------------------

        prepared_data, info = preparer(
            params
        )

        # ----------------------------------------------------
        # Normalize representation
        # ----------------------------------------------------

        data, info = _prepare_output_data(
            prepared_data=prepared_data,
            info=info,
            representation=representation,
        )

        info = _add_preprocessing_trace(
            info,
            params,
            representation,
        )

        # ----------------------------------------------------
        # Save processed dataset
        # ----------------------------------------------------

        save_data(
            data,
            data_path,
        )

        # ----------------------------------------------------
        # Save completed manifest
        # ----------------------------------------------------

        manifest = make_manifest(
            status="done",
            params=params,
            execution_time=(
                time.time()
                - start_time
            ),
        )

        # Keep enough information to reconstruct
        # the DatasetView without loading the dataset.
        manifest["output"] = info

        save_manifest(
            manifest,
            manifest_path,
        )

    except Exception as error:

        save_manifest(
            make_manifest(
                status="failed",
                params=params,
                execution_time=(
                    time.time()
                    - start_time
                ),
                error=str(error),
            ),
            manifest_path,
        )

        raise

    # --------------------------------------------------------
    # Return reference
    # --------------------------------------------------------

    return _make_dataset_view(
        data_path=data_path,
        manifest_path=manifest_path,
        info=info,
    )