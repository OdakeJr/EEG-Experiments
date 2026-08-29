from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pipeline.process_data as process_data
import pipeline.combine_data as combine_data
import pipeline.scenarios as scenarios
import pipeline.feature_selection as feature_selection
import pipeline.training as training

import pipeline.evaluation.model_results as model_results
import pipeline.analysis.method_ranking as method_ranking

from utils.storage import (
    exists,
)


# ============================================================
# Test output locations
# ============================================================

TEST_ROOT = Path(
    "tests/output/method_ranking"
)


# ============================================================
# Test configuration
# ============================================================

BCI2A_SUBJECTS = [
    1, 2, 3,
]

WEIBO_SUBJECTS = [
    1, 2, 3,
]

ZHOU_SUBJECTS = [
    1, 2, 3,
]


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


PREPROCESSING_LABEL = "method_ranking_bp_logvar_cov"


# ============================================================
# Scenario params
# ============================================================

INTRA_PARAMS = {
    "train_fraction": 0.75,
    "seed": 42,
}


SESSION_PARAMS = {
    "source_counts": [
        1,
        "all",
    ],

    "target_fractions": [
        0.25,
    ],

    "seed": 42,
}


SUBJECT_PARAMS = {
    "source_counts": [
        1,
        "all",
    ],

    "target_fractions": [
        0.25,
    ],

    "max_source_combinations": 2,

    "seed": 42,
}


DATASET_PARAMS = {
    "source_dataset_counts": [
        1,
        "all",
    ],

    "target_dataset_subject_counts": [
        0,
    ],

    "target_subject_fractions": [
        0.25,
    ],

    "max_source_combinations": 2,

    "seed": 42,
}


# ============================================================
# Preprocessing params
# ============================================================

PREPROCESSING_PARAMS = [
    {
        "dataset": "bci2a",
        "name": PREPROCESSING_LABEL,

        "root_gdf": "data/bci2a/gdf",
        "root_mat": "data/bci2a/mat",

        "loader": {
            "subjects": BCI2A_SUBJECTS,
            "channels": CHANNELS,
            "classes": COMMON_CLASSES,
        },

        "filter": FILTER_CONFIG,

        "features": FEATURE_CONFIG,

        "subject_batch_size": 1,

        "show_progress": False,
    },
    {
        "dataset": "weibo",
        "name": PREPROCESSING_LABEL,

        "root": "data/weibo",

        "loader": {
            "subjects": WEIBO_SUBJECTS,
            "channels": CHANNELS,
            "classes": COMMON_CLASSES,
        },

        "filter": FILTER_CONFIG,

        "features": FEATURE_CONFIG,

        "subject_batch_size": 1,

        "show_progress": False,
    },
    {
        "dataset": "zhou",
        "name": PREPROCESSING_LABEL,

        "root": "data/zhou",

        "loader": {
            "subjects": ZHOU_SUBJECTS,
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
# Feature selection and training params
# ============================================================

FS_CONFIGS = [
    {
        "method": "anova",
        "config_label": "anova_k2",
        "params": {
            "k": 2,
        },
    },
]


TRAINING_CONFIGS = [
    {
        "learning": "sklearn_erm",
        "model": "logistic_regression",
        "model_params": {
            "max_iter": 300,
        },
        "training_params": {
            "seed": 42,
        },
    },
    {
        "learning": "sklearn_erm",
        "model": "random_forest",
        "model_params": {
            "n_estimators": 20,
            "random_state": 42,
        },
        "training_params": {
            "seed": 42,
        },
    },
]


METHOD_RANKING_PARAMS = {
    "feature_config_column": (
        "feature_selection_config_label"
    ),

    "preprocessing_config_column": (
        "preprocessing_config_label"
    ),

    "seed_column": "training_seed",

    "method_display": [
        {
            "learning_method": "sklearn_erm",
            "model_name": "logistic_regression",
            "regime": "Classical",
            "method": "Logistic Regression",
        },
        {
            "learning_method": "sklearn_erm",
            "model_name": "random_forest",
            "regime": "Classical",
            "method": "Random Forest",
        },
    ],
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
        if not Path(
            path
        ).exists()
    ]

    if missing:
        pytest.skip(
            "Missing raw EEG test data: "
            + ", ".join(
                missing
            )
        )


def _add_scenario_items(
    scenario_items,
    splits,
    view,
):

    for split in splits:

        scenario_items.append(
            {
                "split": split,
                "view": view,
            }
        )


def _build_model_artifacts(
    scenario_items,
):

    model_artifacts_by_scenario = {}

    for item in scenario_items:

        split = item[
            "split"
        ]

        view = item[
            "view"
        ]

        for fs_params in FS_CONFIGS:

            fs_artifact = feature_selection.run_feature_selection(
                split,
                view,
                fs_params,
                group="method_ranking",
            )

            assert exists(
                fs_artifact.transformer_path
            )

            assert (
                fs_artifact.feature_selection_config_label
                == "anova_k2"
            )

            split_model_artifacts = []

            for training_config in TRAINING_CONFIGS:

                model_artifact = training.run_training(
                    split,
                    view,
                    fs_artifact,
                    training_config,
                    group="method_ranking",
                )

                assert exists(
                    model_artifact.model_path
                )

                assert (
                    model_artifact.feature_selection_config_label
                    == "anova_k2"
                )

                assert (
                    model_artifact.preprocessing_config_label
                    is not None
                )

                split_model_artifacts.append(
                    model_artifact
                )

            model_artifacts_by_scenario.setdefault(
                split.scenario,
                []
            ).append(
                {
                    "group": "method_ranking",
                    "view": view,
                    "split": split,
                    "fs_artifact": fs_artifact,
                    "artifacts": split_model_artifacts,
                }
            )

    return model_artifacts_by_scenario


# ============================================================
# Assertions: table structure
# ============================================================

def _assert_ranking_columns(
    table,
):

    required_columns = {
        "Regime",
        "Method",
        "Runs",
        "Mean BA",
        "Median BA",
        "Std BA",
        "Mean Macro-F1",
        "Mean AUC",
        "Average Rank",
        "Best Rank (%)",
    }

    assert required_columns.issubset(
        set(
            table.columns
        )
    )


def _assert_methods_exist(
    table,
):

    observed_methods = set(
        table[
            "Method"
        ]
    )

    assert {
        "Logistic Regression",
        "Random Forest",
    }.issubset(
        observed_methods
    )


def _assert_numeric_columns(
    table,
):

    numeric_columns = [
        "Runs",
        "Mean BA",
        "Median BA",
        "Std BA",
        "Mean Macro-F1",
        "Average Rank",
        "Best Rank (%)",
    ]

    for column in numeric_columns:

        values = pd.to_numeric(
            table[
                column
            ],
            errors="coerce",
        )

        assert values.notna().all()


def _assert_metric_ranges(
    table,
):

    for column in [
        "Mean BA",
        "Median BA",
        "Mean Macro-F1",
    ]:

        values = pd.to_numeric(
            table[
                column
            ],
            errors="coerce",
        )

        assert (
            values
            >= 0
        ).all()

        assert (
            values
            <= 1
        ).all()

    best_rank = pd.to_numeric(
        table[
            "Best Rank (%)"
        ],
        errors="coerce",
    )

    assert (
        best_rank
        >= 0
    ).all()

    assert (
        best_rank
        <= 100
    ).all()

    average_rank = pd.to_numeric(
        table[
            "Average Rank"
        ],
        errors="coerce",
    )

    assert (
        average_rank
        >= 1
    ).all()


def _assert_runs_are_positive(
    table,
):

    assert (
        table[
            "Runs"
        ].astype(
            int
        )
        > 0
    ).all()


def _assert_rank_values_make_sense(
    table,
):

    average_rank = pd.to_numeric(
        table[
            "Average Rank"
        ],
        errors="coerce",
    )

    # We have exactly two methods in this test.
    assert (
        average_rank
        >= 1
    ).all()

    assert (
        average_rank
        <= 2
    ).all()


def _assert_best_rank_frequency_has_winner(
    table,
):

    best_rank_sum = pd.to_numeric(
        table[
            "Best Rank (%)"
        ],
        errors="coerce",
    ).sum()

    # With two methods and complete matched blocks,
    # best-rank percentages across methods should be positive.
    assert best_rank_sum > 0


def _assert_table_sorted_by_average_rank(
    table,
):

    ranks = pd.to_numeric(
        table[
            "Average Rank"
        ],
        errors="coerce",
    ).to_numpy()

    assert np.all(
        ranks[:-1]
        <= ranks[1:]
    )


# ============================================================
# Assertions: model-results trace
# ============================================================

def _assert_model_results_trace(
    model_results_artifact,
):

    dataframe = pd.read_csv(
        model_results_artifact.path
    )

    required_columns = {
        "feature_selection_config_label",
        "feature_selection_method",
        "feature_selection_params",
        "preprocessing_signature",
        "preprocessing_config_label",
    }

    assert required_columns.issubset(
        set(
            dataframe.columns
        )
    )

    assert set(
        dataframe[
            "feature_selection_config_label"
        ].dropna()
    ) == {
        "anova_k2",
    }

    assert (
        dataframe[
            "preprocessing_config_label"
        ]
        .notna()
        .all()
    )

    assert (
        dataframe[
            "preprocessing_signature"
        ]
        .notna()
        .all()
    )


# ============================================================
# Test
# ============================================================

def test_method_ranking_with_real_eeg_data():

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

    method_ranking.OUTPUT_ROOT = (
        TEST_ROOT
        / "method_ranking"
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

        assert (
            view.preprocessing_config_label
            == PREPROCESSING_LABEL
        )

        assert (
            view.preprocessing_signature
            is not None
        )

        views[
            params[
                "dataset"
            ]
        ] = view

    assert set(
        views.keys()
    ) == {
        "bci2a",
        "weibo",
        "zhou",
    }

    combined_view = combine_data.combine_datasets(
        list(
            views.values()
        ),
        {
            "name": "method_ranking_combined",
            "preprocessing_config_label": (
                PREPROCESSING_LABEL
            ),
        },
    )

    assert exists(
        combined_view.path
    )

    assert (
        combined_view.preprocessing_config_label
        == PREPROCESSING_LABEL
    )

    # --------------------------------------------------------
    # Generate scenario splits
    # --------------------------------------------------------

    scenario_items = []

    for _, view in views.items():

        intra_splits = scenarios.run_scenario(
            view,
            "intra_subject",
            INTRA_PARAMS,
        )

        assert len(
            intra_splits
        ) > 0

        _add_scenario_items(
            scenario_items,
            intra_splits,
            view,
        )

        session_splits = scenarios.run_scenario(
            view,
            "cross_session",
            SESSION_PARAMS,
        )

        _add_scenario_items(
            scenario_items,
            session_splits,
            view,
        )

        subject_splits = scenarios.run_scenario(
            view,
            "cross_subject",
            SUBJECT_PARAMS,
        )

        assert len(
            subject_splits
        ) > 0

        _add_scenario_items(
            scenario_items,
            subject_splits,
            view,
        )

    dataset_splits = scenarios.run_scenario(
        combined_view,
        "cross_dataset",
        DATASET_PARAMS,
    )

    assert len(
        dataset_splits
    ) > 0

    _add_scenario_items(
        scenario_items,
        dataset_splits,
        combined_view,
    )

    assert len(
        scenario_items
    ) > 0

    observed_scenarios = {
        item[
            "split"
        ].scenario
        for item in scenario_items
    }

    assert {
        "intra_subject",
        "cross_subject",
        "cross_dataset",
    }.issubset(
        observed_scenarios
    )

    # --------------------------------------------------------
    # Train and evaluate models
    # --------------------------------------------------------

    model_artifacts_by_scenario = _build_model_artifacts(
        scenario_items
    )

    assert len(
        model_artifacts_by_scenario
    ) > 0

    model_results_artifact = (
        model_results.run_model_evaluation(
            model_artifacts_by_scenario
        )
    )

    assert exists(
        model_results_artifact.path
    )

    _assert_model_results_trace(
        model_results_artifact
    )

    # --------------------------------------------------------
    # Run method ranking
    # --------------------------------------------------------

    artifact = method_ranking.run_method_ranking(
        model_results_artifact,
        METHOD_RANKING_PARAMS,
    )

    assert exists(
        artifact.manifest_path
    )

    expected_tables = {
        "intra_subject",
        "cross_session",
        "cross_subject",
        "cross_dataset",
    }

    assert expected_tables.issubset(
        set(
            artifact.tables.keys()
        )
    )

    # --------------------------------------------------------
    # Validate ranking tables
    # --------------------------------------------------------

    non_empty_tables = 0

    for scenario_name, table_path in artifact.tables.items():

        assert exists(
            table_path
        )

        table = pd.read_csv(
            table_path
        )

        _assert_ranking_columns(
            table
        )

        # Some datasets may not have sessions,
        # so cross_session may be empty depending on inputs.
        if table.empty:
            continue

        non_empty_tables += 1

        _assert_methods_exist(
            table
        )

        _assert_numeric_columns(
            table
        )

        _assert_metric_ranges(
            table
        )

        _assert_runs_are_positive(
            table
        )

        _assert_rank_values_make_sense(
            table
        )

        _assert_best_rank_frequency_has_winner(
            table
        )

        _assert_table_sorted_by_average_rank(
            table
        )

    assert non_empty_tables >= 3