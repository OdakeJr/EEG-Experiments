from pathlib import Path

import pipeline.process_data as process_data
import pipeline.scenarios as scenarios
import pipeline.feature_selection as feature_selection

from utils.storage import (
    load_manifest,
    load_pickle,
)


# ============================================================
# Configuration
# ============================================================

COMMON_CLASSES = [
    "left_hand_imagery",
    "right_hand_imagery",
    "both_feet_imagery",
]


PREPROCESSING_PARAMS = {
    "dataset": "bci2a",
    "name": "scenario_test_bci2a",

    "root_gdf": "data/bci2a/gdf",
    "root_mat": "data/bci2a/mat",

    "loader": {
        "subjects": [1, 2, 3, 4, 5],

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

    "subject_batch_size": 2,
    "show_progress": False,
}


SCENARIO_PARAMS = {
    "source_counts": [1],
    "target_fractions": [0.5],
    "seed": 42,
}


FS_PARAMS = [
    {
        "method": "variance",
        "params": {
            "threshold": 0.0,
        },
    },

    {
        "method": "anova",
        "params": {
            "k": 3,
        },
    },

    {
        "method": "random",
        "params": {
            "k": 3,
            "seed": 42,
        },
    },
]


# ============================================================
# Test
# ============================================================

def test_feature_selection():

    # Reuse the previous scenario-test outputs.
    process_data.OUTPUT_ROOT = Path(
        "tests/output/scenarios/preprocessing"
    )

    scenarios.OUTPUT_ROOT = Path(
        "tests/output/scenarios/splits"
    )

    feature_selection.OUTPUT_ROOT = Path(
        "tests/output/feature_selection"
    )

    # --------------------------------------------------------
    # DatasetView
    # --------------------------------------------------------

    view = process_data.run_preprocessing(
        PREPROCESSING_PARAMS
    )

    # --------------------------------------------------------
    # One representative ScenarioSplit
    # --------------------------------------------------------

    splits = scenarios.run_scenario(
        view,
        "cross_subject",
        SCENARIO_PARAMS,
    )

    assert len(splits) > 0

    split = splits[0]

    data = split.materialize(
        view
    )

    assert data.source is not None

    n_input_features = (
        data.source.X.shape[1]
    )

    # --------------------------------------------------------
    # Run FS methods
    # --------------------------------------------------------

    for fs_params in FS_PARAMS:

        artifact = (
            feature_selection.run_feature_selection(
                split,
                view,
                fs_params,
            )
        )

        transformer_path = Path(
            artifact.transformer_path
        )

        manifest_path = Path(
            artifact.manifest_path
        )

        assert transformer_path.exists()
        assert manifest_path.exists()

        # ----------------------------------------------------
        # Manifest
        # ----------------------------------------------------

        manifest = load_manifest(
            manifest_path
        )

        assert (
            manifest["status"]
            == "done"
        )

        assert (
            artifact.split_id
            == split.id
        )

        assert (
            artifact.method
            == fs_params["method"]
        )

        # ----------------------------------------------------
        # Load and use fitted transformer
        # ----------------------------------------------------

        transformer = load_pickle(
            transformer_path
        )

        X_transformed = transformer.transform(
            data.source.X,
            data.source.elementary_domains,
        )

        assert (
            X_transformed.shape[0]
            == data.source.X.shape[0]
        )

        assert (
            X_transformed.shape[1]
            <= n_input_features
        )

        # ----------------------------------------------------
        # Resume
        # ----------------------------------------------------

        modified_time = (
            transformer_path
            .stat()
            .st_mtime_ns
        )

        second_artifact = (
            feature_selection.run_feature_selection(
                split,
                view,
                fs_params,
            )
        )

        assert (
            second_artifact.signature
            == artifact.signature
        )

        assert (
            Path(
                second_artifact.transformer_path
            ).stat().st_mtime_ns
            == modified_time
        )