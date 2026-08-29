from pathlib import Path

import numpy as np

import pipeline.process_data as process_data
import pipeline.scenarios as scenarios
import pipeline.feature_selection as feature_selection
import pipeline.training as training

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
    TEST_ROOT / "scenarios" / "preprocessing"
)

scenarios.OUTPUT_ROOT = (
    TEST_ROOT / "scenarios" / "splits"
)

feature_selection.OUTPUT_ROOT = (
    TEST_ROOT / "feature_selection"
)

training.OUTPUT_ROOT = (
    TEST_ROOT / "training"
)


# --------------------------------------------------
# Parameters
# --------------------------------------------------

PREPROCESSING_PARAMS = {
    "dataset": "bci2a",

    "root_gdf": "data/bci2a/gdf",
    "root_mat": "data/bci2a/mat",

    "subjects": [1, 2, 3, 4, 5],
    "sessions": ["train"],

    "classes": [
        "left_hand_imagery",
        "right_hand_imagery",
        "both_feet_imagery",
    ],

    "channels": ["C3", "Cz", "C4"],

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


TRAINING_CONFIGS = [
    {
        "learning": "sklearn_erm",
        "model": "logistic_regression",
        "model_params": {
            "max_iter": 200,
        },
        "training_params": {
            "seed": 42,
        },
    },
    {
        "learning": "neural_erm",
        "model": "mlp",
        "model_params": {
            "hidden_dims": [8],
        },
        "training_params": {
            "epochs": 2,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "seed": 42,
        },
    },
]


# --------------------------------------------------
# Test
# --------------------------------------------------

def test_training_pipeline():

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

    assert exists(
        fs_artifact.transformer_path
    )

    # ----------------------------------------------
    # Materialize data for prediction checks
    # ----------------------------------------------

    data = split.materialize(view)

    assert data.target_elementary_domain is not None

    transformer = load_pickle(
        fs_artifact.transformer_path
    )

    X_target = transformer.transform(
        data.target_elementary_domain.X,
        data.target_elementary_domain.elementary_domains,
    )

    # ----------------------------------------------
    # Train each learning/model configuration
    # ----------------------------------------------

    for params in TRAINING_CONFIGS:

        artifact = training.run_training(
            split,
            view,
            fs_artifact,
            params,
            group="test",
        )

        # ------------------------------------------
        # Artifact
        # ------------------------------------------

        assert artifact.split_id == split.id

        assert (
            artifact.feature_selection_signature
            == fs_artifact.signature
        )

        assert exists(
            artifact.model_path
        )

        assert exists(
            artifact.manifest_path
        )

        # ------------------------------------------
        # Manifest
        # ------------------------------------------

        manifest = load_manifest(
            artifact.manifest_path
        )

        assert manifest["status"] == "done"
        assert manifest["execution_time"] is not None
        assert manifest["execution_time"] >= 0

        # ------------------------------------------
        # Load fitted learner
        # ------------------------------------------

        learner = load_pickle(
            artifact.model_path
        )

        predictions = learner.predict(
            X_target,
            data.target_elementary_domain.elementary_domains,
            data.target_elementary_domain.super_domains,
        )

        assert len(predictions) == len(X_target)

        assert set(
            np.unique(predictions)
        ).issubset(
            set(
                np.unique(
                    data.target_elementary_domain.y
                )
            )
        )

        # ------------------------------------------
        # Probabilities
        # ------------------------------------------

        probabilities = learner.predict_proba(
            X_target,
            data.target_elementary_domain.elementary_domains,
            data.target_elementary_domain.super_domains,
        )

        assert probabilities.shape[0] == len(X_target)

        assert np.allclose(
            probabilities.sum(axis=1),
            1.0,
            atol=1e-5,
        )

        # ------------------------------------------
        # Resume
        # ------------------------------------------

        model_path = Path(
            artifact.model_path
        )

        old_mtime = (
            model_path.stat().st_mtime
        )

        artifact_again = training.run_training(
            split,
            view,
            fs_artifact,
            params,
            group="test",
        )

        new_mtime = (
            model_path.stat().st_mtime
        )

        assert (
            artifact_again.signature
            == artifact.signature
        )

        assert new_mtime == old_mtime