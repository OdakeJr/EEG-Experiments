# pipeline/combine_data.py

import time
from pathlib import Path
from copy import deepcopy

import pandas as pd

from models.dataset_view import DatasetView

from utils.storage import (
    exists,
    load_data,
    load_manifest,
    save_data,
    save_manifest,
)

from utils.status import (
    is_done,
    make_manifest,
    make_signature,
)


OUTPUT_ROOT = Path("outputs/combined")


# ============================================================
# Helpers: preprocessing trace
# ============================================================

def _get_input_signatures(dataset_views):
    """
    Get the signatures of all upstream preprocessing artifacts.
    """

    signatures = []

    for view in dataset_views:

        if getattr(
            view,
            "preprocessing_signature",
            None,
        ) is not None:

            signatures.append(
                view.preprocessing_signature
            )

        else:

            manifest = load_manifest(
                view.manifest_path
            )

            signatures.append(
                manifest["signature"]
            )

    return signatures


def _get_input_preprocessing_trace(dataset_views):
    """
    Preserve preprocessing information from all input views.
    """

    traces = []

    for view in dataset_views:

        trace = {
            "path": view.path,

            "manifest_path": view.manifest_path,

            "preprocessing_signature": getattr(
                view,
                "preprocessing_signature",
                None,
            ),

            "preprocessing_config_label": getattr(
                view,
                "preprocessing_config_label",
                None,
            ),

            "preprocessing_params": getattr(
                view,
                "preprocessing_params",
                None,
            ),
        }

        traces.append(
            trace
        )

    return traces


def _combined_preprocessing_signature(
    params,
    input_preprocessing_trace,
):
    """
    Signature for the combined processed dataset.
    """

    return make_signature(
        {
            "type": "combined_preprocessing",
            "params": params,
            "inputs": input_preprocessing_trace,
        }
    )


def _combined_preprocessing_config_label(
    params,
    input_preprocessing_trace,
):
    """
    Human-readable label for the combined preprocessing setup.

    Prefer an explicit label/name.
    """

    if "preprocessing_config_label" in params:
        return params[
            "preprocessing_config_label"
        ]

    if "name" in params:
        return params[
            "name"
        ]

    labels = [
        trace.get(
            "preprocessing_config_label"
        )
        for trace in input_preprocessing_trace
        if trace.get(
            "preprocessing_config_label"
        )
    ]

    unique_labels = sorted(
        set(
            labels
        )
    )

    if len(
        unique_labels
    ) == 1:
        return unique_labels[
            0
        ]

    signature = make_signature(
        {
            "labels": unique_labels,
        }
    )

    return (
        "combined_"
        + signature[:8]
    )


def _add_combined_preprocessing_trace(
    info,
    params,
    input_preprocessing_trace,
):
    """
    Add preprocessing trace to the combined DatasetView info.
    """

    traced_info = deepcopy(
        info
    )

    traced_info[
        "preprocessing_signature"
    ] = _combined_preprocessing_signature(
        params,
        input_preprocessing_trace,
    )

    traced_info[
        "preprocessing_config_label"
    ] = _combined_preprocessing_config_label(
        params,
        input_preprocessing_trace,
    )

    traced_info[
        "preprocessing_params"
    ] = {
        "combined_params": deepcopy(
            params
        ),
        "inputs": deepcopy(
            input_preprocessing_trace
        ),
    }

    traced_info[
        "input_preprocessing_trace"
    ] = deepcopy(
        input_preprocessing_trace
    )

    return traced_info


# ============================================================
# Helpers
# ============================================================

def _get_output_paths(params):
    """
    Determine where the combined dataset will be stored.
    """

    name = params.get(
        "name",
        make_signature(params)[:8],
    )

    output_dir = (
        OUTPUT_ROOT
        / name
    )

    return (
        output_dir / "data",
        output_dir / "manifest.json",
    )


def _validate_views(dataset_views):
    """
    Make sure datasets can be concatenated.
    """

    if not dataset_views:
        raise ValueError(
            "No datasets were provided."
        )

    reference = dataset_views[0]

    for view in dataset_views[1:]:

        if (
            view.feature_columns
            != reference.feature_columns
        ):
            raise ValueError(
                "Datasets have different feature columns."
            )

        if (
            view.label_column
            != reference.label_column
        ):
            raise ValueError(
                "Datasets have different label columns."
            )

        if (
            view.domain_columns
            != reference.domain_columns
        ):
            raise ValueError(
                "Datasets have different domain columns."
            )


def _make_dataset_view(
    data_path,
    manifest_path,
    info,
):
    return DatasetView(
        path=str(data_path),

        feature_columns=info[
            "feature_columns"
        ],

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
# Main combination function
# ============================================================

def combine_datasets(
    dataset_views,
    params=None,
):
    """
    Combine multiple processed EEG datasets into one artifact.

    Returns
    -------
    DatasetView
        Reference to the combined dataset.
    """

    params = params or {}

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    _validate_views(
        dataset_views
    )

    # --------------------------------------------------------
    # Include upstream signatures/traces in status
    # --------------------------------------------------------

    input_signatures = _get_input_signatures(
        dataset_views
    )

    input_preprocessing_trace = (
        _get_input_preprocessing_trace(
            dataset_views
        )
    )

    effective_params = {
        "params": params,
        "inputs": input_signatures,
        "input_preprocessing_trace": input_preprocessing_trace,
    }

    data_path, manifest_path = (
        _get_output_paths(
            effective_params
        )
    )

    # --------------------------------------------------------
    # Reuse previous result
    # --------------------------------------------------------

    if (
        exists(data_path)
        and is_done(
            manifest_path,
            effective_params,
        )
    ):
        manifest = load_manifest(
            manifest_path
        )

        return _make_dataset_view(
            data_path,
            manifest_path,
            manifest["output"],
        )

    # --------------------------------------------------------
    # Start execution
    # --------------------------------------------------------

    save_manifest(
        make_manifest(
            status="running",
            params=effective_params,
        ),
        manifest_path,
    )

    start_time = time.time()

    try:

        # ----------------------------------------------------
        # Load datasets
        # ----------------------------------------------------

        dataframes = [
            load_data(
                view.path
            )
            for view in dataset_views
        ]

        # ----------------------------------------------------
        # Concatenate
        # ----------------------------------------------------

        dataframe = pd.concat(
            dataframes,
            axis=0,
            ignore_index=True,
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_data(
            dataframe,
            data_path,
        )

        reference = dataset_views[0]

        info = {
            "feature_columns": (
                reference.feature_columns
            ),

            "label_column": (
                reference.label_column
            ),

            "domain_columns": (
                reference.domain_columns
            ),

            "metadata_columns": (
                reference.metadata_columns
            ),

            "n_rows": len(
                dataframe
            ),

            "n_datasets": dataframe[
                "dataset"
            ].nunique(),
        }

        info = _add_combined_preprocessing_trace(
            info=info,
            params=params,
            input_preprocessing_trace=input_preprocessing_trace,
        )

        # ----------------------------------------------------
        # Manifest
        # ----------------------------------------------------

        manifest = make_manifest(
            status="done",
            params=effective_params,
            execution_time=(
                time.time()
                - start_time
            ),
        )

        manifest["output"] = info

        save_manifest(
            manifest,
            manifest_path,
        )

    except Exception as error:

        save_manifest(
            make_manifest(
                status="failed",
                params=effective_params,
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
        data_path,
        manifest_path,
        info,
    )