# pipeline/combine_data.py

import time
from pathlib import Path

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
# Helpers
# ============================================================

def _get_input_signatures(dataset_views):
    """
    Get the signatures of all preprocessing artifacts.
    """

    signatures = []

    for view in dataset_views:
        manifest = load_manifest(
            view.manifest_path
        )

        signatures.append(
            manifest["signature"]
        )

    return signatures


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
    # Include upstream signatures in status
    # --------------------------------------------------------

    effective_params = {
        "params": params,
        "inputs": _get_input_signatures(
            dataset_views
        ),
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
            load_data(view.path)
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