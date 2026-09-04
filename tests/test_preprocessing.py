# tests/test_preprocessing.py

from pathlib import Path

import pipeline.process_data as process_data
from utils.storage import (
    exists,
    _data_path,
    load_data,
)


COMMON_CLASSES = [
    "left_hand_imagery",
    "right_hand_imagery",
    "both_feet_imagery",
]


# ============================================================
# Shared test
# ============================================================

def check_preprocessing(params):

    # --------------------------------------------------
    # First execution
    # --------------------------------------------------

    view = process_data.run_preprocessing(
        params
    )

    manifest_path = Path(
        view.manifest_path
    )

    actual_data_path = _data_path(
        view.path
    )

    # --------------------------------------------------
    # Check output
    # --------------------------------------------------

    assert exists(view.path)
    assert manifest_path.exists()

    assert view.label_column == "label"

    assert view.domain_columns == [
        "dataset",
        "subject",
        "session",
    ]

    assert len(
        view.feature_columns
    ) > 0

    # --------------------------------------------------
    # Check labels
    # --------------------------------------------------

    dataframe = load_data(
        view.path
    )

    assert set(
        dataframe["label"].unique()
    ).issubset(
        COMMON_CLASSES
    )

    # --------------------------------------------------
    # Check resume
    # --------------------------------------------------

    modified_time = (
        actual_data_path.stat().st_mtime_ns
    )

    second_view = (
        process_data.run_preprocessing(
            params
        )
    )

    assert second_view.path == view.path

    assert (
        actual_data_path.stat().st_mtime_ns
        == modified_time
    )


# ============================================================
# BCI IV 2a
# ============================================================

def test_bci2a_preprocessing():

    params = {
        "dataset": "bci2a",
        "name": "test_bci2a",

        "root_gdf": "datasets/bci2a/gdf",
        "root_mat": "datasets/bci2a/mat",

        "loader": {
            "subjects": [1],

            "channels": [
                "C3",
                "Cz",
                "C4",
            ],

            "classes": COMMON_CLASSES,
        },

        "filter": {
            "bandpass": {
                "bands": [
                    (8, 12),
                    (13, 30),
                ],
            },
        },

        "features": {
            "logvar": {},
            "cov": {},
        },

        "subject_batch_size": 1,
        "show_progress": False,
    }

    assert Path(
        params["root_gdf"]
    ).exists()

    assert Path(
        params["root_mat"]
    ).exists()

    process_data.OUTPUT_ROOT = Path(
        "tests/output/preprocessing"
    )

    check_preprocessing(
        params
    )


# ============================================================
# EEGMMIDB
# ============================================================

def test_eegmmidb_preprocessing():

    params = {
        "dataset": "eegmmidb",
        "name": "test_eegmmidb",

        "root_dir": "datasets/eegmmidb",

        "loader": {
            "subjects": [1],

            "runs": {
                4: {
                    "name": "run_04",
                    "label_map": {
                        "T1": "left_hand_imagery",
                        "T2": "right_hand_imagery",
                    },
                },

                6: {
                    "name": "run_06",
                    "label_map": {
                        "T1": "both_hands_imagery",
                        "T2": "both_feet_imagery",
                    },
                },
            },

            "channels": [
                "C3",
                "Cz",
                "C4",
            ],

            "classes": COMMON_CLASSES,
        },

        "filter": {
            "bandpass": {
                "bands": [
                    (8, 12),
                    (13, 30),
                ],
            },
        },

        "features": {
            "logvar": {},
            "cov": {},
        },

        "subject_batch_size": 1,
        "show_progress": False,
    }

    assert Path(
        params["root_dir"]
    ).exists()

    process_data.OUTPUT_ROOT = Path(
        "tests/output/preprocessing"
    )

    check_preprocessing(
        params
    )