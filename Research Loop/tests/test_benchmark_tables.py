from pathlib import Path

import pandas as pd
import pytest

import pipeline.process_data as process_data
import pipeline.combine_data as combine_data
import pipeline.scenarios as scenarios
import pipeline.feature_selection as feature_selection
import pipeline.training as training

import pipeline.evaluation.model_results as model_results
import pipeline.evaluation.domain_results as domain_results
import pipeline.analysis.benchmark_tables as benchmark_tables

from utils.storage import (
    exists,
)


# ============================================================
# Test output locations
# ============================================================

TEST_ROOT = Path(
    "tests/output/benchmark_tables"
)


# ============================================================
# Shared test configuration
# ============================================================

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

FILTER_CONFIG = {
    "bandpass": {
        "bands": [
            (8, 12),
            (13, 30),
        ],
    },
}

FEATURE_CONFIG = {
    "logvar": {},
    "cov": {},
}


# ============================================================
# Preprocessing params
# ============================================================

PREPROCESSING_PARAMS = [

    # --------------------------------------------------------
    # BCI Competition IV 2a
    # --------------------------------------------------------

    {
        "dataset": "bci2a",
        "name": "benchmark_table_bci2a",

        "root_gdf": "data/bci2a/gdf",
        "root_mat": "data/bci2a/mat",

        "loader": {
            "subjects": [
                1, 2, 3,
            ],

            "channels": CHANNELS,

            "classes": COMMON_CLASSES,
        },

        "filter": FILTER_CONFIG,

        "features": FEATURE_CONFIG,

        "subject_batch_size": 1,

        "show_progress": False,
    },

    # --------------------------------------------------------
    # Weibo2014
    # --------------------------------------------------------

    {
        "dataset": "weibo",
        "name": "benchmark_table_weibo",

        "root": "data/weibo",

        "loader": {
            "subjects": [
                1, 2, 3,
            ],

            "channels": CHANNELS,

            "classes": COMMON_CLASSES,
        },

        "filter": FILTER_CONFIG,

        "features": FEATURE_CONFIG,

        "subject_batch_size": 1,

        "show_progress": False,
    },

    # --------------------------------------------------------
    # Zhou2016
    # --------------------------------------------------------

    {
        "dataset": "zhou",
        "name": "benchmark_table_zhou",

        "root": "data/zhou",

        "loader": {
            "subjects": [
                1, 2, 3,
            ],

            "channels": CHANNELS,

            "classes": COMMON_CLASSES,
        },

        "filter": FILTER_CONFIG,

        "features": FEATURE_CONFIG,

        "subject_batch_size": 1,

        "show_progress": False,
    },
]


# ============================================================
# Scenario params
# ============================================================

INTRA_PARAMS = {
    "train_fraction": 0.5,
    "seed": 42,
}


SESSION_PARAMS = {
    "source_counts": [
        1,
        "all",
    ],

    "target_fractions": [
        0.5,
    ],

    "seed": 42,
}


SUBJECT_PARAMS = {
    "source_counts": [
        1,
        "all",
    ],

    "target_fractions": [
        0.5,
    ],

    "max_source_combinations": 2,

    "seed": 42,
}


DATASET_PARAMS = {
    "source_dataset_counts": [
        1,
        "all",
    ],

    # Current paper protocol:
    # source datasets -> held-out target subject.
    # No target-super-domain adaptation.
    "target_dataset_subject_counts": [
        0,
    ],

    "target_subject_fractions": [
        0.5,
    ],

    "max_source_combinations": 2,

    "seed": 42,
}


# ============================================================
# Feature selection and training params
# ============================================================

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
        "training_params": {},
    },
]


DOMAIN_EVALUATION_PARAMS = {
    "metrics": [
        {
            "name": "mmd",
            "params": {},
        },
    ],

    "representations": [
        "marginal",
    ],

    "standardize": True,
    "min_samples_per_side": 2,
}


# ============================================================
# Helpers
# ============================================================

def _check_raw_paths():

    required_paths = [
        "data/bci2a/gdf",
        "data/bci2a/mat",
        "data/weibo",
        "data/zhou",
    ]

    missing = [
        path
        for path in required_paths
        if not Path(path).exists()
    ]

    if missing:
        pytest.skip(
            "Missing raw EEG test data: "
            + ", ".join(
                missing
            )
        )


def _source_domain_count(
    split,
):

    return len(
        split.source_elementary_domains
    )


def _source_dataset_count(
    split,
):

    return len(
        {
            domain.split(
                "|",
                1,
            )[0]

            for domain
            in split.source_elementary_domains
        }
    )


def _take_representative_splits(
    splits,
    key_fn,
):
    """
    Keep a small subset while preserving the smallest and largest
    source protocols.

    This lets the test verify that benchmark_tables can receive
    both increasing-source and all-source cases, while the final
    benchmark table keeps the fixed/all-source protocol.
    """

    if not splits:
        return []

    ordered = sorted(
        splits,
        key=key_fn,
    )

    selected = [
        ordered[0],
        ordered[-1],
    ]

    unique = []
    seen = set()

    for split in selected:

        if split.id not in seen:

            unique.append(
                split
            )

            seen.add(
                split.id
            )

    return unique


def _assert_table_exists(
    path,
):

    assert exists(
        path
    )

    return pd.read_csv(
        path
    )


def _assert_table_columns(
    table,
):

    required_columns = {
        "Regime",
        "Method / Model",
        "Runs",
        "Source BA",
        "Target-Test BA",
        "Gap",
        "Macro-F1",
        "AUC",
        "Discrepancy",
    }

    assert required_columns.issubset(
        set(
            table.columns
        )
    )


def _add_scenario_item(
    scenario_items,
    split,
    view,
):

    scenario_items.append(
        {
            "split": split,
            "view": view,
        }
    )


# ============================================================
# Test
# ============================================================

def test_benchmark_tables_with_data():

    _check_raw_paths()

    # --------------------------------------------------------
    # Redirect outputs
    # --------------------------------------------------------

    process_data.OUTPUT_ROOT = (
        TEST_ROOT
        / "preprocessing"
    )

    combine_data.OUTPUT_ROOT = (
        TEST_ROOT
        / "combined"
    )

    scenarios.OUTPUT_ROOT = (
        TEST_ROOT
        / "scenarios"
    )

    feature_selection.OUTPUT_ROOT = (
        TEST_ROOT
        / "feature_selection"
    )

    training.OUTPUT_ROOT = (
        TEST_ROOT
        / "training"
    )

    model_results.OUTPUT_ROOT = (
        TEST_ROOT
        / "model_results"
    )

    domain_results.OUTPUT_ROOT = (
        TEST_ROOT
        / "domain_results"
    )

    benchmark_tables.OUTPUT_ROOT = (
        TEST_ROOT
        / "benchmark_tables"
    )

    # --------------------------------------------------------
    # Preprocess datasets
    # --------------------------------------------------------

    views = {}

    for params in PREPROCESSING_PARAMS:

        view = process_data.run_preprocessing(
            params
        )

        assert exists(
            view.path
        )

        views[
            params["dataset"]
        ] = view

    combined_view = combine_data.combine_datasets(
        list(
            views.values()
        ),
        {
            "name": "benchmark_table_combined",
        },
    )

    assert exists(
        combined_view.path
    )

    # --------------------------------------------------------
    # Generate representative scenario splits
    # --------------------------------------------------------

    scenario_items = []

    for dataset, view in views.items():

        # ----------------------------------------------------
        # Intra-subject
        # ----------------------------------------------------

        intra_splits = scenarios.run_scenario(
            view,
            "intra_subject",
            INTRA_PARAMS,
        )

        for split in intra_splits[
            :1
        ]:

            _add_scenario_item(
                scenario_items,
                split,
                view,
            )

        # ----------------------------------------------------
        # Cross-session
        # ----------------------------------------------------

        session_splits = scenarios.run_scenario(
            view,
            "cross_session",
            SESSION_PARAMS,
        )

        for split in _take_representative_splits(
            session_splits,
            _source_domain_count,
        ):

            _add_scenario_item(
                scenario_items,
                split,
                view,
            )

        # ----------------------------------------------------
        # Cross-subject
        # ----------------------------------------------------

        subject_splits = scenarios.run_scenario(
            view,
            "cross_subject",
            SUBJECT_PARAMS,
        )

        for split in _take_representative_splits(
            subject_splits,
            _source_domain_count,
        ):

            _add_scenario_item(
                scenario_items,
                split,
                view,
            )

    # --------------------------------------------------------
    # Cross-dataset
    # --------------------------------------------------------

    dataset_splits = scenarios.run_scenario(
        combined_view,
        "cross_dataset",
        DATASET_PARAMS,
    )

    for split in _take_representative_splits(
        dataset_splits,
        _source_dataset_count,
    ):

        _add_scenario_item(
            scenario_items,
            split,
            combined_view,
        )

    assert len(
        scenario_items
    ) > 0

    # --------------------------------------------------------
    # Feature selection + training
    # --------------------------------------------------------

    model_artifacts_by_scenario = {}
    scenario_artifacts = {}

    for item in scenario_items:

        split = item[
            "split"
        ]

        view = item[
            "view"
        ]

        fs_artifact = (
            feature_selection.run_feature_selection(
                split,
                view,
                FS_PARAMS,
                group="benchmark",
            )
        )

        assert exists(
            fs_artifact.transformer_path
        )

        split_model_artifacts = []

        for training_config in TRAINING_CONFIGS:

            model_artifact = training.run_training(
                split,
                view,
                fs_artifact,
                training_config,
                group="benchmark",
            )

            assert exists(
                model_artifact.model_path
            )

            split_model_artifacts.append(
                model_artifact
            )

        model_artifacts_by_scenario.setdefault(
            split.scenario,
            []
        ).append(
            {
                "group": "benchmark",
                "view": view,
                "split": split,
                "fs_artifact": fs_artifact,
                "artifacts": split_model_artifacts,
            }
        )

        scenario_artifacts.setdefault(
            split.scenario,
            []
        ).append(
            {
                "group": "benchmark",
                "view": view,
                "splits": [
                    split
                ],
            }
        )

    assert len(
        model_artifacts_by_scenario
    ) > 0

    assert len(
        scenario_artifacts
    ) > 0

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    model_results_artifact = (
        model_results.run_model_evaluation(
            model_artifacts_by_scenario
        )
    )

    assert exists(
        model_results_artifact.path
    )

    domain_results_artifact = (
        domain_results.run_domain_evaluation(
            scenario_artifacts,
            DOMAIN_EVALUATION_PARAMS,
        )
    )

    assert exists(
        domain_results_artifact.path
    )

    # --------------------------------------------------------
    # Benchmark tables
    # --------------------------------------------------------

    artifact = benchmark_tables.run_benchmark_tables(
        model_results_artifact,
        domain_results_artifact,
    )

    assert exists(
        artifact.manifest_path
    )

    assert set(
        artifact.tables.keys()
    ) == {
        "intra_subject",
        "cross_session",
        "cross_subject",
        "cross_dataset",
    }

    intra_table = _assert_table_exists(
        artifact.tables[
            "intra_subject"
        ]
    )

    cross_session_table = _assert_table_exists(
        artifact.tables[
            "cross_session"
        ]
    )

    cross_subject_table = _assert_table_exists(
        artifact.tables[
            "cross_subject"
        ]
    )

    cross_dataset_table = _assert_table_exists(
        artifact.tables[
            "cross_dataset"
        ]
    )

    # Some datasets may not produce cross-session splits.
    # Therefore only validate columns if the table is non-empty.
    for table in [
        intra_table,
        cross_session_table,
        cross_subject_table,
        cross_dataset_table,
    ]:

        if not table.empty:
            _assert_table_columns(
                table
            )

    assert not intra_table.empty
    assert not cross_subject_table.empty
    assert not cross_dataset_table.empty

    # --------------------------------------------------------
    # Protocol checks
    # --------------------------------------------------------

    assert (
        intra_table[
            "Discrepancy"
        ]
        .astype(str)
        .eq("—")
        .all()
    )

    model_table = pd.read_csv(
        model_results_artifact.path
    )

    cross_dataset_rows = model_table[
        model_table["scenario"]
        == "cross_dataset"
    ]

    assert not cross_dataset_rows.empty

    # Current cross-dataset benchmark:
    # no target-super-domain adaptation.
    assert (
        cross_dataset_rows[
            "n_target_super_domains"
        ]
        .fillna(
            0
        )
        .eq(
            0
        )
        .all()
    )

    # Make sure the test included both a smaller-source and
    # all-source cross-dataset protocol before table filtering.
    assert (
        cross_dataset_rows[
            "source_domains"
        ]
        .nunique()
        >= 2
    )

    assert not cross_dataset_table.empty