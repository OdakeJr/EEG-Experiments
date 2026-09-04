from pathlib import Path

import pandas as pd

import pipeline.process_data as process_data
import pipeline.analysis.domain_variability_qualitative as qualitative

from utils.storage import (
    exists,
    load_manifest,
)


# --------------------------------------------------
# Test output locations
# --------------------------------------------------

TEST_ROOT = Path("tests/output")

process_data.OUTPUT_ROOT = (
    TEST_ROOT
    / "domain_variability_qualitative"
    / "preprocessing"
)

qualitative.OUTPUT_ROOT = (
    TEST_ROOT
    / "domain_variability_qualitative"
    / "analysis"
)


# --------------------------------------------------
# Shared configuration
# --------------------------------------------------

COMMON_CLASSES = [
    "left_hand_imagery",
    "right_hand_imagery",
    "both_feet_imagery",
]


CHANNELS = [
    "C3",
    "Cz",
    "C4",
]


FILTER_PARAMS = {
    "bandpass": {
        "bands": [
            (8, 12),
            (13, 30),
        ],
    },
}


FEATURE_PARAMS = {
    "logvar": {},
    "cov": {},
}


# ============================================================
# BCI Competition IV 2a
#
# Eight subjects are loaded because the cross-subject
# visualization contains four independent subject pairs.
#
# BCI2a is also used for the cross-session visualization.
# ============================================================

BCI2A_PARAMS = {
    "dataset": "bci2a",
    "name": "test_qualitative_bci2a",

    "root_gdf": "datasets/bci2a/gdf",
    "root_mat": "datasets/bci2a/mat",

    "loader": {
        "subjects": [
            1, 2, 3, 4,
            5, 6, 7, 8,
        ],

        "channels": CHANNELS,

        "classes": COMMON_CLASSES,
    },

    "filter": FILTER_PARAMS,

    "features": FEATURE_PARAMS,

    "subject_batch_size": 1,
    "show_progress": False,
}


# ============================================================
# PhysioNet EEGMMIDB
#
# Four subjects are used in the cross-dataset projection.
# ============================================================

EEGMMIDB_PARAMS = {
    "dataset": "eegmmidb",
    "name": "test_qualitative_eegmmidb",

    "root_dir": "datasets/eegmmidb",

    "loader": {
        "subjects": [
            1, 2, 3, 4,
        ],

        "runs": {
            4: {
                "name": "run_04",

                "label_map": {
                    "T1": (
                        "left_hand_imagery"
                    ),
                    "T2": (
                        "right_hand_imagery"
                    ),
                },
            },

            6: {
                "name": "run_06",

                "label_map": {
                    "T1": (
                        "both_hands_imagery"
                    ),
                    "T2": (
                        "both_feet_imagery"
                    ),
                },
            },
        },

        "channels": CHANNELS,

        "classes": COMMON_CLASSES,
    },

    "filter": FILTER_PARAMS,

    "features": FEATURE_PARAMS,

    "subject_batch_size": 1,
    "show_progress": False,
}


# ============================================================
# Weibo 2014
#
# Four subjects are used in the cross-dataset projection.
# ============================================================

WEIBO_PARAMS = {
    "dataset": "weibo",
    "name": "test_qualitative_weibo",

    "root": "datasets/weibo",

    "loader": {
        "subjects": [
            1, 2, 3, 4,
        ],

        "channels": CHANNELS,

        "classes": COMMON_CLASSES,
    },

    "filter": FILTER_PARAMS,

    "features": FEATURE_PARAMS,

    "subject_batch_size": 1,
    "show_progress": False,
}


# ============================================================
# Zhou 2016
#
# All four subjects are used in the cross-dataset projection.
# ============================================================

ZHOU_PARAMS = {
    "dataset": "zhou",
    "name": "test_qualitative_zhou",

    "root": "datasets/zhou",

    "loader": {
        "subjects": [
            1, 2, 3, 4,
        ],

        "channels": CHANNELS,

        "classes": COMMON_CLASSES,
    },

    "filter": FILTER_PARAMS,

    "features": FEATURE_PARAMS,

    "subject_batch_size": 1,
    "show_progress": False,
}


# ============================================================
# Qualitative analysis parameters
# ============================================================

QUALITATIVE_PARAMS = {

    "classes": COMMON_CLASSES,

    # --------------------------------------------------
    # Four independent subject-pair comparisons
    # --------------------------------------------------

    "cross_subject": {
        "dataset": "bci2a",

        "n_pairs": 4,

        "samples_per_subject_class": 60,
    },

    # --------------------------------------------------
    # Four subjects:
    # Session 1 versus Session 2
    # --------------------------------------------------

    "cross_session": {
        "dataset": "bci2a",

        "n_subjects": 4,

        "samples_per_session_class": 60,
    },

    # --------------------------------------------------
    # Four subjects from each of the four datasets
    # --------------------------------------------------

    "cross_dataset": {
        "subjects_per_dataset": 4,

        "samples_per_subject_class": 20,
    },

    "standardize": True,

    "seed": 42,

    "pca_params": {},

    "umap_params": {
        "n_neighbors": 15,
        "min_dist": 0.1,
    },
}


# ============================================================
# Test
# ============================================================

def test_domain_variability_qualitative():

    # --------------------------------------------------
    # Input data
    # --------------------------------------------------

    assert Path(
        BCI2A_PARAMS["root_gdf"]
    ).exists()

    assert Path(
        BCI2A_PARAMS["root_mat"]
    ).exists()

    assert Path(
        EEGMMIDB_PARAMS["root_dir"]
    ).exists()

    assert Path(
        WEIBO_PARAMS["root"]
    ).exists()

    assert Path(
        ZHOU_PARAMS["root"]
    ).exists()

    # ==================================================
    # 1. PREPROCESSING
    # ==================================================

    bci2a_view = (
        process_data.run_preprocessing(
            BCI2A_PARAMS
        )
    )

    eegmmidb_view = (
        process_data.run_preprocessing(
            EEGMMIDB_PARAMS
        )
    )

    weibo_view = (
        process_data.run_preprocessing(
            WEIBO_PARAMS
        )
    )

    zhou_view = (
        process_data.run_preprocessing(
            ZHOU_PARAMS
        )
    )

    # --------------------------------------------------
    # Check standardized artifacts
    # --------------------------------------------------

    for view in [
        bci2a_view,
        eegmmidb_view,
        weibo_view,
        zhou_view,
    ]:

        assert exists(
            view.path
        )

        assert len(
            view.feature_columns
        ) > 0

    # --------------------------------------------------
    # All datasets must have compatible features
    # --------------------------------------------------

    reference_features = (
        bci2a_view.feature_columns
    )

    assert (
        eegmmidb_view.feature_columns
        == reference_features
    )

    assert (
        weibo_view.feature_columns
        == reference_features
    )

    assert (
        zhou_view.feature_columns
        == reference_features
    )

    # ==================================================
    # 2. DATASET VIEW COLLECTION
    # ==================================================

    dataset_views = {
        "bci2a": bci2a_view,
        "eegmmidb": eegmmidb_view,
        "weibo": weibo_view,
        "zhou": zhou_view,
    }

    # ==================================================
    # 3. QUALITATIVE DOMAIN VARIABILITY
    # ==================================================

    artifact = (
        qualitative
        .run_domain_variability_qualitative(
            dataset_views,
            QUALITATIVE_PARAMS,
        )
    )

    # ==================================================
    # 4. ARTIFACT CHECKS
    # ==================================================

    figure_path = Path(
        artifact.figures[
            "qualitative_domain_variability"
        ]
    )

    selection_path = Path(
        artifact.tables[
            "qualitative_selection"
        ]
    )

    manifest_path = Path(
        artifact.manifest_path
    )

    assert (
        figure_path.exists()
    )

    assert (
        figure_path.stat().st_size
        > 0
    )

    assert (
        selection_path.exists()
    )

    assert (
        manifest_path.exists()
    )

    # ==================================================
    # 5. SELECTION TABLE
    # ==================================================

    selection = pd.read_csv(
        selection_path
    )

    assert len(
        selection
    ) > 0

    assert {
        "scenario",
        "dataset",
        "panel",
        "subject",
        "session",
        "n_samples",
    }.issubset(
        selection.columns
    )

    # --------------------------------------------------
    # Cross-subject
    #
    # Four panels x two subjects = eight entries.
    # --------------------------------------------------

    cross_subject = selection[
        selection["scenario"]
        == "cross_subject"
    ]

    assert (
        cross_subject["panel"]
        .nunique()
        == 4
    )

    assert (
        len(
            cross_subject
        )
        == 8
    )

    assert (
        cross_subject[
            "dataset"
        ]
        == "bci2a"
    ).all()

    # --------------------------------------------------
    # Cross-session
    #
    # Four subjects x two sessions = eight entries.
    # --------------------------------------------------

    cross_session = selection[
        selection["scenario"]
        == "cross_session"
    ]

    assert (
        cross_session["panel"]
        .nunique()
        == 4
    )

    assert (
        cross_session[
            "subject"
        ]
        .nunique()
        == 4
    )

    assert (
        len(
            cross_session
        )
        == 8
    )

    assert (
        cross_session[
            "dataset"
        ]
        == "bci2a"
    ).all()

    # --------------------------------------------------
    # Cross-dataset
    #
    # All four datasets must appear.
    # --------------------------------------------------

    cross_dataset = selection[
        selection["scenario"]
        == "cross_dataset"
    ]

    assert set(
        cross_dataset[
            "dataset"
        ]
    ) == {
        "bci2a",
        "eegmmidb",
        "weibo",
        "zhou",
    }

    # --------------------------------------------------
    # Four subjects from every dataset
    # --------------------------------------------------

    for dataset in [
        "bci2a",
        "eegmmidb",
        "weibo",
        "zhou",
    ]:

        dataset_selection = (
            cross_dataset[
                cross_dataset[
                    "dataset"
                ]
                == dataset
            ]
        )

        assert (
            dataset_selection[
                "subject"
            ]
            .nunique()
            == 4
        )

    # --------------------------------------------------
    # Every selected group must contain samples
    # --------------------------------------------------

    assert (
        selection[
            "n_samples"
        ] > 0
    ).all()

    # ==================================================
    # 6. MANIFEST
    # ==================================================

    manifest = load_manifest(
        artifact.manifest_path
    )

    assert (
        manifest["status"]
        == "done"
    )

    assert (
        manifest[
            "execution_time"
        ]
        >= 0
    )

    assert (
        manifest[
            "output"
        ][
            "n_cross_subject_panels"
        ]
        == 4
    )

    assert (
        manifest[
            "output"
        ][
            "n_cross_session_panels"
        ]
        == 4
    )

    assert (
        manifest[
            "output"
        ][
            "n_cross_dataset_datasets"
        ]
        == 4
    )

    # ==================================================
    # 7. RESUME
    # ==================================================

    old_mtime = (
        figure_path
        .stat()
        .st_mtime_ns
    )

    artifact_again = (
        qualitative
        .run_domain_variability_qualitative(
            dataset_views,
            QUALITATIVE_PARAMS,
        )
    )

    new_mtime = (
        figure_path
        .stat()
        .st_mtime_ns
    )

    assert (
        artifact_again.signature
        == artifact.signature
    )

    assert (
        new_mtime
        == old_mtime
    )
    
