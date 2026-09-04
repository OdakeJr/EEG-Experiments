# tests/test_domain_variability_quantitative.py

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd

import pipeline.process_data as process_data
import pipeline.combine_data as combine_data
import pipeline.scenarios as scenarios
import pipeline.evaluation.domain_results as domain_evaluation
import pipeline.analysis.domain_variability_quantitative as quantitative

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
    / "domain_variability_quantitative"
    / "preprocessing"
)

combine_data.OUTPUT_ROOT = (
    TEST_ROOT
    / "domain_variability_quantitative"
    / "combined"
)

scenarios.OUTPUT_ROOT = (
    TEST_ROOT
    / "domain_variability_quantitative"
    / "scenarios"
)

domain_evaluation.OUTPUT_ROOT = (
    TEST_ROOT
    / "domain_variability_quantitative"
    / "domain_results"
)

quantitative.OUTPUT_ROOT = (
    TEST_ROOT
    / "domain_variability_quantitative"
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
# Preprocessing
# ============================================================

BCI2A_PARAMS = {
    "dataset": "bci2a",
    "name": "test_quantitative_bci2a",

    "root_gdf": "datasets/bci2a/gdf",
    "root_mat": "datasets/bci2a/mat",

    "loader": {
        "subjects": [
            1,
            2,
            3,
        ],

        "channels": CHANNELS,

        "classes": COMMON_CLASSES,
    },

    "filter": FILTER_PARAMS,

    "features": FEATURE_PARAMS,

    "subject_batch_size": 1,
    "show_progress": False,
}


EEGMMIDB_PARAMS = {
    "dataset": "eegmmidb",
    "name": "test_quantitative_eegmmidb",

    "root_dir": "datasets/eegmmidb",

    "loader": {
        "subjects": [
            1,
            2,
            3,
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
# Scenario parameters
# ============================================================

CROSS_SESSION_PARAMS = {
    "source_counts": [
        1,
    ],

    "target_fractions": [
        0.5,
    ],

    "seed": 42,
}


CROSS_SUBJECT_PARAMS = {
    "source_counts": [
        1,
    ],

    "target_fractions": [
        0.5,
    ],

    "seed": 42,
}


CROSS_DATASET_PARAMS = {
    "source_dataset_counts": [
        1,
    ],

    "target_dataset_subject_counts": [
        0,
    ],

    "target_subject_fractions": [
        0.5,
    ],

    "seed": 42,
}


# ============================================================
# Domain evaluation parameters
# ============================================================

DOMAIN_EVALUATION_PARAMS = {
    "metrics": [
        {
            "name": "mmd",
            "params": {},
        },

        {
            "name": "energy",
            "params": {},
        },
    ],

    "representations": [
        "marginal",
        "class",
    ],

    "standardize": True,

    "min_samples_per_side": 2,
}


# ============================================================
# Quantitative RQ1 parameters
# ============================================================

QUANTITATIVE_PARAMS = {
    "scenarios": [
        "cross_subject",
        "cross_session",
        "cross_dataset",
    ],

    "metrics": [
        "mmd",
        "energy",
    ],

    "comparison": (
        "source_to_target_elementary"
    ),
}


# ============================================================
# Test
# ============================================================

def test_domain_variability_quantitative():

    # ==================================================
    # 1. INPUT DATA
    # ==================================================

    assert Path(
        BCI2A_PARAMS[
            "root_gdf"
        ]
    ).exists()

    assert Path(
        BCI2A_PARAMS[
            "root_mat"
        ]
    ).exists()

    assert Path(
        EEGMMIDB_PARAMS[
            "root_dir"
        ]
    ).exists()

    # ==================================================
    # 2. PREPROCESSING
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

    assert exists(
        bci2a_view.path
    )

    assert exists(
        eegmmidb_view.path
    )

    # --------------------------------------------------
    # Cross-dataset comparison requires compatible
    # feature representations.
    # --------------------------------------------------

    assert (
        bci2a_view.feature_columns
        == eegmmidb_view.feature_columns
    )

    # ==================================================
    # 3. COMBINE DATASETS
    # ==================================================

    combined_view = (
        combine_data.combine_datasets(
            [
                bci2a_view,
                eegmmidb_view,
            ],

            {
                "name": (
                    "test_quantitative_combined"
                ),
            },
        )
    )

    assert exists(
        combined_view.path
    )

    # ==================================================
    # 4. SCENARIOS
    # ==================================================

    cross_session_splits = (
        scenarios.run_scenario(
            bci2a_view,
            "cross_session",
            CROSS_SESSION_PARAMS,
        )
    )

    cross_subject_splits = (
        scenarios.run_scenario(
            bci2a_view,
            "cross_subject",
            CROSS_SUBJECT_PARAMS,
        )
    )

    cross_dataset_splits = (
        scenarios.run_scenario(
            combined_view,
            "cross_dataset",
            CROSS_DATASET_PARAMS,
        )
    )

    assert len(
        cross_session_splits
    ) > 0

    assert len(
        cross_subject_splits
    ) > 0

    assert len(
        cross_dataset_splits
    ) > 0

    # ==================================================
    # 5. REPRODUCE main.py SCENARIO STRUCTURE
    # ==================================================

    scenario_artifacts = {

        "cross_session": [
            {
                "group": "test_bci2a",
                "view": bci2a_view,
                "splits": (
                    cross_session_splits
                ),
            }
        ],

        "cross_subject": [
            {
                "group": "test_bci2a",
                "view": bci2a_view,
                "splits": (
                    cross_subject_splits
                ),
            }
        ],

        "cross_dataset": [
            {
                "group": "test_combined",
                "view": combined_view,
                "splits": (
                    cross_dataset_splits
                ),
            }
        ],
    }

    # ==================================================
    # 6. DOMAIN EVALUATION
    # ==================================================

    domain_results_artifact = (
        domain_evaluation
        .run_domain_evaluation(
            scenario_artifacts,
            DOMAIN_EVALUATION_PARAMS,
        )
    )

    assert exists(
        domain_results_artifact.path
    )

    assert (
        domain_results_artifact.n_rows
        > 0
    )

    domain_results = pd.read_csv(
        domain_results_artifact.path
    )

    # --------------------------------------------------
    # All RQ1 shift levels
    # --------------------------------------------------

    assert {
        "cross_session",
        "cross_subject",
        "cross_dataset",
    }.issubset(
        set(
            domain_results[
                "scenario"
            ]
        )
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    assert {
        "mmd",
        "energy",
    }.issubset(
        set(
            domain_results[
                "metric"
            ]
        )
    )

    # --------------------------------------------------
    # Marginal and class-specific measurements
    # --------------------------------------------------

    assert {
        "marginal",
        "class",
    }.issubset(
        set(
            domain_results[
                "representation"
            ]
        )
    )

    # ==================================================
    # 7. QUANTITATIVE DOMAIN VARIABILITY
    # ==================================================

    artifact = (
        quantitative
        .run_domain_variability_quantitative(
            domain_results_artifact,
            QUANTITATIVE_PARAMS,
        )
    )

    # ==================================================
    # 8. FIGURES
    # ==================================================

    discrepancy_figure = Path(
        artifact.figures[
            "domain_discrepancy_distribution"
        ]
    )

    marginal_conditional_figure = Path(
        artifact.figures[
            "marginal_conditional_discrepancy"
        ]
    )

    assert (
        discrepancy_figure.exists()
    )

    assert (
        discrepancy_figure.stat()
        .st_size
        > 0
    )

    assert (
        marginal_conditional_figure.exists()
    )

    assert (
        marginal_conditional_figure.stat()
        .st_size
        > 0
    )

    # ==================================================
    # 9. TABLES
    # ==================================================

    summary_path = Path(
        artifact.tables[
            "domain_variability_summary"
        ]
    )

    paired_path = Path(
        artifact.tables[
            "marginal_conditional_pairs"
        ]
    )

    assert (
        summary_path.exists()
    )

    assert (
        paired_path.exists()
    )

    summary = pd.read_csv(
        summary_path
    )

    paired = pd.read_csv(
        paired_path
    )

    # --------------------------------------------------
    # Summary:
    #
    # 3 shift levels x 2 discrepancy metrics
    # = 6 rows
    # --------------------------------------------------

    assert len(
        summary
    ) == 6

    assert set(
        summary[
            "shift"
        ]
    ) == {
        "cross_session",
        "cross_subject",
        "cross_dataset",
    }

    assert set(
        summary[
            "metric"
        ]
    ) == {
        "mmd",
        "energy",
    }

    required_columns = {
        "shift",
        "metric",
        "marginal",
        "conditional",
        "left",
        "right",
        "feet",
    }

    assert (
        required_columns
        .issubset(
            summary.columns
        )
    )

    # --------------------------------------------------
    # Every summary entry should have a value
    # --------------------------------------------------

    for column in [
        "marginal",
        "conditional",
        "left",
        "right",
        "feet",
    ]:

        assert (
            summary[
                column
            ]
            .notna()
            .all()
        )

    # ==================================================
    # 10. MARGINAL / CONDITIONAL PAIRS
    # ==================================================

    assert len(
        paired
    ) > 0

    assert {
        "cross_session",
        "cross_subject",
        "cross_dataset",
    }.issubset(
        set(
            paired[
                "scenario"
            ]
        )
    )

    assert {
        "mmd",
        "energy",
    }.issubset(
        set(
            paired[
                "metric"
            ]
        )
    )

    assert (
        paired[
            "marginal_value"
        ]
        .notna()
        .all()
    )

    assert (
        paired[
            "conditional_value"
        ]
        .notna()
        .all()
    )

    assert (
        paired[
            "n_classes"
        ] >= 1
    ).all()

    # ==================================================
    # 11. MANIFEST
    # ==================================================

    manifest = load_manifest(
        artifact.manifest_path
    )

    assert (
        manifest[
            "status"
        ]
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
            "n_domain_rows"
        ]
        > 0
    )

    assert (
        manifest[
            "output"
        ][
            "n_paired_rows"
        ]
        > 0
    )

    # ==================================================
    # 12. RESUME
    # ==================================================

    old_mtime = (
        discrepancy_figure
        .stat()
        .st_mtime_ns
    )

    artifact_again = (
        quantitative
        .run_domain_variability_quantitative(
            domain_results_artifact,
            QUANTITATIVE_PARAMS,
        )
    )

    new_mtime = (
        discrepancy_figure
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