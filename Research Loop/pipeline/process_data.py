# pipeline/process_data.py

import time
from pathlib import Path

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
# Helpers
# ============================================================

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


def _make_dataset_view(
    data_path,
    manifest_path,
    info,
):
    """
    Build the lightweight reference returned by preprocessing.
    """

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
# Main preprocessing function
# ============================================================

def run_preprocessing(params):
    """
    Run one preprocessing configuration.

    If the same configuration was already completed,
    reuse the stored result.

    Returns
    -------
    DatasetView
        Lightweight reference to the processed dataset.
    """

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

        return _make_dataset_view(
            data_path=data_path,
            manifest_path=manifest_path,
            info=manifest["output"],
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

        dataframe, info = preparer(
            params
        )

        # ----------------------------------------------------
        # Save processed dataset
        # ----------------------------------------------------

        data = {
            column: dataframe[
                column
            ].to_numpy()
            for column
            in dataframe.columns
        }

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