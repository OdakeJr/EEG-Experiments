"""
from pathlib import Path

import pandas as pd

import pipeline.process_data as process_data
import pipeline.scenarios as scenarios
import pipeline.feature_selection as feature_selection
import pipeline.training as training
import pipeline.evaluation.model_results as model_evaluation

from utils.storage import (
    exists,
    load_manifest,
)


# --------------------------------------------------
# Test output locations
# --------------------------------------------------

TEST_ROOT = Path("tests/output")

process_data.OUTPUT_ROOT = (
    TEST_ROOT / "model_evaluation" / "preprocessing"
)

scenarios.OUTPUT_ROOT = (
    TEST_ROOT / "model_evaluation" / "scenarios"
)

feature_selection.OUTPUT_ROOT = (
    TEST_ROOT / "model_evaluation" / "feature_selection"
)

training.OUTPUT_ROOT = (
    TEST_ROOT / "model_evaluation" / "training"
)

model_evaluation.OUTPUT_ROOT = (
    TEST_ROOT / "model_evaluation" / "results"
)


# --------------------------------------------------
# Parameters
# --------------------------------------------------

PREPROCESSING_PARAMS = {
    "dataset": "bci2a",

    "root_gdf": "datasets/bci2a/gdf",
    "root_mat": "datasets/bci2a/mat",

    "subjects": [1, 2, 3],
    "sessions": ["train"],

    "classes": [
        "left_hand_imagery",
        "right_hand_imagery",
        "both_feet_imagery",
    ],

    "channels": [
        "C3",
        "Cz",
        "C4",
    ],

    "bands": [
        [8, 12],
        [13, 30],
    ],

    "features": {
        "logvar": {},
        "cov": {},
    },
}


SCENARIO_PARAMS = {
    "source_counts": [1],
    "target_fractions": [0.5],
    "seed": 42,
}


FS_PARAMS = {
    "method": "anova",
    "params": {
        "k": 3,
    },
}


TRAINING_PARAMS = {
    "learning": "sklearn_erm",
    "model": "logistic_regression",

    "model_params": {
        "max_iter": 200,
    },

    "training_params": {},
}


MODEL_EVALUATION_PARAMS = {
    "metrics": [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "auc",
    ],
    "include_computational": True,
}


# --------------------------------------------------
# Test
# --------------------------------------------------

def test_model_evaluation_pipeline():

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
    # Feature selection
    # ----------------------------------------------

    fs_artifact = (
        feature_selection.run_feature_selection(
            split,
            view,
            FS_PARAMS,
            group="test",
        )
    )

    # ----------------------------------------------
    # Training
    # ----------------------------------------------

    model_artifact = training.run_training(
        split,
        view,
        fs_artifact,
        TRAINING_PARAMS,
        group="test",
    )

    # ----------------------------------------------
    # Reproduce main.py artifact structure
    # ----------------------------------------------

    model_artifacts = {
        "cross_subject": [
            {
                "group": "test",
                "view": view,
                "split": split,
                "fs_artifact": fs_artifact,
                "artifacts": [
                    model_artifact
                ],
            }
        ]
    }

    # ----------------------------------------------
    # Model evaluation
    # ----------------------------------------------

    result_artifact = (
        model_evaluation.run_model_evaluation(
            model_artifacts,
            MODEL_EVALUATION_PARAMS,
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
    assert manifest["execution_time"] >= 0

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
        "n_source_domains",
        "target_fraction",
        "split_seed",
        "feature_selection_signature",
        "learning_method",
        "model_name",
        "model_signature",
        "evaluation_group",
        "partition",
        "n_samples",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "auc",
        "training_time",
        "inference_time",
        "inference_time_per_sample",
        "model_size_bytes",
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
        results["learning_method"]
        == "sklearn_erm"
    ).all()

    assert (
        results["model_name"]
        == "logistic_regression"
    ).all()

    # ----------------------------------------------
    # Expected evaluation partitions
    # ----------------------------------------------

    observed = set(
        zip(
            results["evaluation_group"],
            results["partition"],
        )
    )

    assert (
        "source",
        "train",
    ) in observed

    assert (
        "target_elementary_domain",
        "calibration",
    ) in observed

    assert (
        "target_elementary_domain",
        "test",
    ) in observed

    # ----------------------------------------------
    # Metric sanity
    # ----------------------------------------------

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]:
        assert results[metric].between(
            0.0,
            1.0,
        ).all()

    assert (
        results["n_samples"] > 0
    ).all()

    assert (
        results["training_time"] >= 0
    ).all()

    assert (
        results["inference_time"] >= 0
    ).all()

    assert (
        results["inference_time_per_sample"]
        >= 0
    ).all()

    assert (
        results["model_size_bytes"] > 0
    ).all()

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
        model_evaluation.run_model_evaluation(
            model_artifacts,
            MODEL_EVALUATION_PARAMS,
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
    
"""

from pathlib import Path

import numpy as np
import pandas as pd

import pipeline.process_data as process_data
import pipeline.scenarios as scenarios
import pipeline.feature_selection as feature_selection
import pipeline.training as training
import pipeline.evaluation.model_results as model_evaluation

from utils.storage import (
    exists,
    load_manifest,
    load_pickle,
)


# --------------------------------------------------
# Test output locations
# --------------------------------------------------

TEST_ROOT = Path("tests/output")

process_data.OUTPUT_ROOT = (
    TEST_ROOT / "model_evaluation" / "preprocessing"
)

scenarios.OUTPUT_ROOT = (
    TEST_ROOT / "model_evaluation" / "scenarios"
)

feature_selection.OUTPUT_ROOT = (
    TEST_ROOT / "model_evaluation" / "feature_selection"
)

training.OUTPUT_ROOT = (
    TEST_ROOT / "model_evaluation" / "training"
)

model_evaluation.OUTPUT_ROOT = (
    TEST_ROOT / "model_evaluation" / "results"
)


# --------------------------------------------------
# Parameters
# --------------------------------------------------

PREPROCESSING_PARAMS = {
    "dataset": "bci2a",
    "name": "test_model_evaluation",

    "root_gdf": "datasets/bci2a/gdf",
    "root_mat": "datasets/bci2a/mat",

    "loader": {
        "subjects": [1, 2, 3],

        "channels": [
            "C3",
            "Cz",
            "C4",
        ],

        "classes": [
            "left_hand_imagery",
            "right_hand_imagery",
            "both_feet_imagery",
        ],
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


FS_PARAMS = {
    "method": "anova",
    "params": {
        "k": 3,
    },
}


TRAINING_PARAMS = {
    "learning": "sklearn_erm",
    "model": "logistic_regression",

    "model_params": {
        "max_iter": 200,
    },

    "training_params": {},
}


MODEL_EVALUATION_PARAMS = {
    "metrics": [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "auc",
    ],
    "include_computational": True,
}


# --------------------------------------------------
# Test
# --------------------------------------------------

def test_model_evaluation_pipeline():

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
    # Materialize and inspect scenario
    # ----------------------------------------------

    data = split.materialize(view)

    print(
        "\nSource labels:",
        np.unique(
            data.source.y,
            return_counts=True,
        ),
    )

    print(
        "Source partitions:",
        np.unique(
            data.source.partitions,
            return_counts=True,
        ),
    )

    print(
        "Target labels:",
        np.unique(
            data.target_elementary_domain.y,
            return_counts=True,
        ),
    )

    print(
        "Target partitions:",
        np.unique(
            data.target_elementary_domain.partitions,
            return_counts=True,
        ),
    )

    # ----------------------------------------------
    # Feature selection
    # ----------------------------------------------

    fs_artifact = (
        feature_selection.run_feature_selection(
            split,
            view,
            FS_PARAMS,
            group="test",
        )
    )

    # ----------------------------------------------
    # Training
    # ----------------------------------------------

    model_artifact = training.run_training(
        split,
        view,
        fs_artifact,
        TRAINING_PARAMS,
        group="test",
    )

    # ----------------------------------------------
    # Inspect fitted model predictions
    # ----------------------------------------------

    transformer = load_pickle(
        fs_artifact.transformer_path
    )

    learner = load_pickle(
        model_artifact.model_path
    )

    target = (
        data.target_elementary_domain
    )

    test_mask = (
        target.partitions == "test"
    )

    X_test = transformer.transform(
        target.X[test_mask],
        target.elementary_domains[test_mask],
    )

    y_test = target.y[test_mask]

    domains_test = (
        target.elementary_domains[
            test_mask
        ]
    )

    super_domains_test = (
        None
        if target.super_domains is None
        else target.super_domains[
            test_mask
        ]
    )

    predictions = learner.predict(
        X_test,
        domains_test,
        super_domains_test,
    )

    print(
        "Test true labels:",
        np.unique(
            y_test,
            return_counts=True,
        ),
    )

    print(
        "Test predictions:",
        np.unique(
            predictions,
            return_counts=True,
        ),
    )

    # ----------------------------------------------
    # Reproduce main.py artifact structure
    # ----------------------------------------------

    model_artifacts = {
        "cross_subject": [
            {
                "group": "test",
                "view": view,
                "split": split,
                "fs_artifact": fs_artifact,
                "artifacts": [
                    model_artifact
                ],
            }
        ]
    }

    # ----------------------------------------------
    # Model evaluation
    # ----------------------------------------------

    result_artifact = (
        model_evaluation.run_model_evaluation(
            model_artifacts,
            MODEL_EVALUATION_PARAMS,
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
    assert manifest["execution_time"] >= 0

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
        "n_source_domains",
        "n_target_super_domains",
        "target_fraction",
        "split_seed",
        "source_domains",
        "target_super_domains",
        "target_domains",
        "feature_selection_signature",
        "learning_method",
        "model_name",
        "model_signature",
        "evaluation_group",
        "partition",
        "n_samples",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "auc",
        "training_time",
        "inference_time",
        "inference_time_per_sample",
        "model_size_bytes",
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
        results["learning_method"]
        == "sklearn_erm"
    ).all()

    assert (
        results["model_name"]
        == "logistic_regression"
    ).all()

    # ----------------------------------------------
    # Expected evaluation partitions
    # ----------------------------------------------

    observed = set(
        zip(
            results["evaluation_group"],
            results["partition"],
        )
    )

    assert (
        "source",
        "train",
    ) in observed

    assert (
        "target_elementary_domain",
        "calibration",
    ) in observed

    assert (
        "target_elementary_domain",
        "test",
    ) in observed

    # ----------------------------------------------
    # Metric sanity
    # ----------------------------------------------

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]:
        assert results[metric].between(
            0.0,
            1.0,
        ).all()

    assert (
        results["n_samples"] > 0
    ).all()

    assert (
        results["training_time"] >= 0
    ).all()

    assert (
        results["inference_time"] >= 0
    ).all()

    assert (
        results["inference_time_per_sample"]
        >= 0
    ).all()

    assert (
        results["model_size_bytes"] > 0
    ).all()

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
        model_evaluation.run_model_evaluation(
            model_artifacts,
            MODEL_EVALUATION_PARAMS,
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