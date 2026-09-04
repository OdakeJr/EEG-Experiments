from pathlib import Path

import pandas as pd

import pipeline.process_data as process_data
import pipeline.scenarios as scenarios
import pipeline.evaluation.domain_results as domain_evaluation

from utils.storage import (
    exists,
    load_manifest,
)


# --------------------------------------------------
# Test output locations
# --------------------------------------------------

TEST_ROOT = Path("tests/output")

process_data.OUTPUT_ROOT = (
    TEST_ROOT / "domain_evaluation" / "preprocessing"
)

scenarios.OUTPUT_ROOT = (
    TEST_ROOT / "domain_evaluation" / "scenarios"
)

domain_evaluation.OUTPUT_ROOT = (
    TEST_ROOT / "domain_evaluation" / "results"
)


# --------------------------------------------------
# Shared classes
# --------------------------------------------------

COMMON_CLASSES = [
    "left_hand_imagery",
    "right_hand_imagery",
    "both_feet_imagery",
]


# --------------------------------------------------
# Parameters
# --------------------------------------------------

PREPROCESSING_PARAMS = {
    "dataset": "bci2a",
    "name": "test_domain_evaluation",

    "root_gdf": "datasets/bci2a/gdf",
    "root_mat": "datasets/bci2a/mat",

    "loader": {
        "subjects": [1, 2, 3],

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


SCENARIO_PARAMS = {
    "source_counts": [1],
    "target_fractions": [0.5],
    "seed": 42,
}


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


# --------------------------------------------------
# Test
# --------------------------------------------------

def test_domain_evaluation_pipeline():

    # ----------------------------------------------
    # Preprocessing
    # ----------------------------------------------

    view = process_data.run_preprocessing(
        PREPROCESSING_PARAMS
    )

    # ----------------------------------------------
    # Scenario
    # ----------------------------------------------

    splits = scenarios.run_scenario(
        view,
        "cross_subject",
        SCENARIO_PARAMS,
    )

    assert len(splits) > 0

    split = splits[0]

    # ----------------------------------------------
    # Reproduce main.py artifact structure
    # ----------------------------------------------

    scenario_artifacts = {
        "cross_subject": [
            {
                "group": "test",
                "view": view,
                "splits": [
                    split
                ],
            }
        ]
    }

    # ----------------------------------------------
    # Domain evaluation
    # ----------------------------------------------

    result_artifact = (
        domain_evaluation.run_domain_evaluation(
            scenario_artifacts,
            DOMAIN_EVALUATION_PARAMS,
        )
    )

    # ----------------------------------------------
    # Artifact checks
    # ----------------------------------------------

    assert exists(
        result_artifact.path
    )

    assert exists(
        result_artifact.manifest_path
    )

    assert result_artifact.n_rows > 0

    # ----------------------------------------------
    # Manifest
    # ----------------------------------------------

    manifest = load_manifest(
        result_artifact.manifest_path
    )

    assert manifest["status"] == "done"

    assert (
        manifest["execution_time"]
        >= 0
    )

    # ----------------------------------------------
    # Canonical results table
    # ----------------------------------------------

    results = pd.read_csv(
        result_artifact.path
    )

    assert len(results) == (
        result_artifact.n_rows
    )

    required_columns = {
        "split_id",
        "scenario",
        "group",
        "seed",
        "target_fraction",
        "comparison",
        "left_group",
        "right_group",
        "left_domains",
        "right_domains",
        "n_left_domains",
        "n_right_domains",
        "metric",
        "metric_signature",
        "representation",
        "class_label",
        "value",
        "n_left_samples",
        "n_right_samples",
    }

    assert required_columns.issubset(
        results.columns
    )

    # ----------------------------------------------
    # Identity
    # ----------------------------------------------

    assert (
        results["split_id"]
        == split.id
    ).all()

    assert (
        results["scenario"]
        == "cross_subject"
    ).all()

    assert (
        results["group"]
        == "test"
    ).all()

    assert (
        results["target_fraction"]
        == 0.5
    ).all()

    # ----------------------------------------------
    # Expected comparison
    # ----------------------------------------------

    assert set(
        results["comparison"]
    ) == {
        "source_to_target_elementary"
    }

    assert set(
        results["left_group"]
    ) == {
        "source"
    }

    assert set(
        results["right_group"]
    ) == {
        "target_elementary_domain"
    }

    # ----------------------------------------------
    # Domain identity
    # ----------------------------------------------

    assert (
        results["n_left_domains"]
        == 1
    ).all()

    assert (
        results["n_right_domains"]
        == 1
    ).all()

    assert (
        results["left_domains"]
        == "bci_iv_2a|A02"
    ).all()

    assert (
        results["right_domains"]
        == "bci_iv_2a|A01"
    ).all()

    # ----------------------------------------------
    # Metrics
    # ----------------------------------------------

    assert set(
        results["metric"]
    ) == {
        "mmd",
        "energy",
    }

    assert (
        results["value"] >= 0
    ).all()

    # ----------------------------------------------
    # Representations
    # ----------------------------------------------

    assert set(
        results["representation"]
    ) == {
        "marginal",
        "class",
    }

    marginal = results[
        results["representation"]
        == "marginal"
    ]

    class_results = results[
        results["representation"]
        == "class"
    ]

    # One marginal row per metric
    assert len(marginal) == 2

    # Three classes per metric
    assert len(class_results) == 6

    assert set(
        class_results["class_label"]
    ) == set(
        COMMON_CLASSES
    )

    # ----------------------------------------------
    # Sample counts
    # ----------------------------------------------

    assert (
        results["n_left_samples"]
        > 0
    ).all()

    assert (
        results["n_right_samples"]
        > 0
    ).all()

    # Full cross-subject groups:
    # 144 trials per class x 3 classes
    assert (
        marginal["n_left_samples"]
        == 432
    ).all()

    assert (
        marginal["n_right_samples"]
        == 432
    ).all()

    # Class-specific:
    # 144 trials per class
    assert (
        class_results[
            "n_left_samples"
        ] == 144
    ).all()

    assert (
        class_results[
            "n_right_samples"
        ] == 144
    ).all()

    # ----------------------------------------------
    # Expected number of rows
    #
    # 2 metrics *
    # (1 marginal + 3 classes)
    # = 8 rows
    # ----------------------------------------------

    assert len(results) == 8

    # ----------------------------------------------
    # Resume
    # ----------------------------------------------

    results_path = Path(
        result_artifact.path
    )

    old_mtime = (
        results_path.stat().st_mtime
    )

    result_again = (
        domain_evaluation.run_domain_evaluation(
            scenario_artifacts,
            DOMAIN_EVALUATION_PARAMS,
        )
    )

    new_mtime = (
        results_path.stat().st_mtime
    )

    assert (
        result_again.signature
        == result_artifact.signature
    )

    assert new_mtime == old_mtime