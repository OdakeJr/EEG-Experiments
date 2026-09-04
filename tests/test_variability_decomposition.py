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
import pipeline.analysis.variability_decomposition as variability_decomposition

from utils.storage import (
    exists,
)


# ============================================================
# Test output locations
# ============================================================

TEST_ROOT = Path(
    "tests/output/variability_decomposition"
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


PREPROCESSING_LABEL = "common_bp_logvar_cov"


# ============================================================
# Variability protocol
# ============================================================

SOURCE_ONLY_TARGET_FRACTION = 0.0

VARIABILITY_TARGET_FRACTION = 0.25


# ============================================================
# Preprocessing params
# ============================================================

PREPROCESSING_PARAMS = [

    # --------------------------------------------------------
    # BCI Competition IV 2a
    # --------------------------------------------------------

    {
        "dataset": "bci2a",
        "name": PREPROCESSING_LABEL,

        "root_gdf": "datasets/bci2a/gdf",
        "root_mat": "datasets/bci2a/mat",

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

    # --------------------------------------------------------
    # Weibo2014
    # --------------------------------------------------------

    {
        "dataset": "weibo",
        "name": PREPROCESSING_LABEL,

        "root": "datasets/weibo",

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

    # --------------------------------------------------------
    # Zhou2016
    # --------------------------------------------------------

    {
        "dataset": "zhou",
        "name": PREPROCESSING_LABEL,

        "root": "datasets/zhou",

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
# Scenario params
# ============================================================

SESSION_PARAMS = {
    "source_counts": [
        1,
        "all",
    ],

    "target_fractions": [
        SOURCE_ONLY_TARGET_FRACTION,
        VARIABILITY_TARGET_FRACTION,
    ],

    "seed": 42,
}


SUBJECT_PARAMS = {
    "source_counts": [
        1,
        "all",
    ],

    "target_fractions": [
        SOURCE_ONLY_TARGET_FRACTION,
        VARIABILITY_TARGET_FRACTION,
    ],

    "max_source_combinations": 2,

    "seed": 42,
}


DATASET_PARAMS = {
    "source_dataset_counts": [
        1,
        "all",
    ],

    # Current cross-dataset protocol:
    # source datasets -> held-out target subject.
    # No target-super-domain adaptation.
    "target_dataset_subject_counts": [
        0,
    ],

    "target_subject_fractions": [
        SOURCE_ONLY_TARGET_FRACTION,
        VARIABILITY_TARGET_FRACTION,
    ],

    "max_source_combinations": 2,

    "seed": 42,
}


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
    {
        "method": "anova",
        "config_label": "anova_k3",
        "params": {
            "k": 3,
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
        "training_params": {},
    },
]


# ============================================================
# Variability-decomposition params
# ============================================================

VARIABILITY_DECOMPOSITION_PARAMS = {
    "metric_column": "Target-Test BA",

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
    ],

    "scenarios": [
        {
            "name": "cross_session",
            "scenario": "cross_session",
            "setting_column": "Dataset",
            "filters": {
                "n_target_super_domains": 0,
                "target_fraction": VARIABILITY_TARGET_FRACTION,
                "use_max_source_domains": True,
            },
        },
        {
            "name": "cross_subject",
            "scenario": "cross_subject",
            "setting_column": "Dataset",
            "filters": {
                "n_target_super_domains": 0,
                "target_fraction": VARIABILITY_TARGET_FRACTION,
                "use_max_source_domains": True,
            },
        },
        {
            "name": "cross_dataset",
            "scenario": "cross_dataset",
            "setting_column": "Held-out Dataset",
            "filters": {
                "n_target_super_domains": 0,
                "target_fraction": VARIABILITY_TARGET_FRACTION,
                "use_max_source_super_domains": True,
            },
        },
    ],
}


# ============================================================
# Helpers: paths
# ============================================================

def _check_raw_paths():

    required_paths = [
        "datasets/bci2a/gdf",
        "datasets/bci2a/mat",
        "datasets/weibo",
        "datasets/zhou",
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


# ============================================================
# Helpers: scenario item creation
# ============================================================

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


# ============================================================
# Helpers: artifact construction
# ============================================================

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
                group="variability",
            )

            assert exists(
                fs_artifact.transformer_path
            )

            assert fs_artifact.feature_selection_config_label in {
                "anova_k2",
                "anova_k3",
            }

            split_model_artifacts = []

            for training_config in TRAINING_CONFIGS:

                model_artifact = training.run_training(
                    split,
                    view,
                    fs_artifact,
                    training_config,
                    group="variability",
                )

                assert exists(
                    model_artifact.model_path
                )

                assert (
                    model_artifact.feature_selection_config_label
                    == fs_artifact.feature_selection_config_label
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
                    "group": "variability",
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

def _assert_points_columns(
    points,
):

    required_columns = {
        "split_id",
        "Scenario",
        "Dataset",
        "Held-out Dataset",
        "Target Domain",
        "Source Composition",
        "Target Fraction",
        "Preprocessing Config",
        "Feature Config",
        "Training Seed",
        "Regime",
        "Method",
        "learning_method",
        "model_name",
        "model_signature",
        "feature_selection_signature",
        "feature_selection_method",
        "feature_selection_params",
        "feature_selection_config_label",
        "preprocessing_signature",
        "preprocessing_config_label",
        "n_target_super_domains",
        "target_fraction",
        "Source Domain Count",
        "Source Super Domain Count",
        "source_domains",
        "target_domains",
        "Target-Test BA",
        "Macro-F1",
        "AUC",
    }

    assert required_columns.issubset(
        set(
            points.columns
        )
    )


def _assert_summary_columns(
    summary,
):

    required_columns = {
        "Scenario",
        "Regime",
        "Method",
        "Runs",
        "Mean BA",
        "BA Std",
        "BA SEM",
        "BA CI95 Low",
        "BA CI95 High",
        "Total BA Variance",
        "Isolated Variance Sum",
    }

    assert required_columns.issubset(
        set(
            summary.columns
        )
    )


def _assert_components_columns(
    components,
):

    required_columns = {
        "Scenario",
        "Regime",
        "Method",
        "Component",
        "Component Column",
        "Control Columns",
        "Isolated Variance",
        "Share (%)",
        "Estimable",
        "Matched Groups",
        "Matched Rows",
        "Unique Values",
    }

    assert required_columns.issubset(
        set(
            components.columns
        )
    )


# ============================================================
# Assertions: labels and filtering
# ============================================================

def _assert_labels(
    dataframe,
):

    assert set(
        dataframe[
            "Regime"
        ]
    ) == {
        "Classical",
    }

    assert set(
        dataframe[
            "Method"
        ]
    ) == {
        "Logistic Regression",
    }


def _assert_clean_feature_labels(
    points,
):

    observed = set(
        points[
            "Feature Config"
        ].dropna()
    )

    assert {
        "anova_k2",
        "anova_k3",
    }.issubset(
        observed
    )

    assert not any(
        len(
            str(value)
        ) == 32
        for value in observed
    )


def _assert_preprocessing_trace_exists(
    points,
):

    assert (
        points[
            "Preprocessing Config"
        ]
        .notna()
        .all()
    )

    assert "no_preprocessing_config" not in set(
        points[
            "Preprocessing Config"
        ]
    )

    assert (
        points[
            "preprocessing_config_label"
        ]
        .notna()
        .all()
    )

    assert (
        points[
            "preprocessing_signature"
        ]
        .notna()
        .all()
    )


def _assert_no_intra_subject(
    points,
):

    assert "intra_subject" not in set(
        points[
            "Scenario"
        ]
    )


def _assert_real_scenarios_exist(
    points,
):

    observed = set(
        points[
            "Scenario"
        ]
    )

    assert "cross_subject" in observed
    assert "cross_dataset" in observed


def _assert_only_fixed_target_fraction(
    points,
):

    observed = set(
        np.round(
            pd.to_numeric(
                points[
                    "Target Fraction"
                ],
                errors="coerce",
            ).dropna(),
            6,
        )
    )

    assert observed == {
        VARIABILITY_TARGET_FRACTION,
    }


def _assert_feature_variation_exists(
    points,
):

    assert (
        points[
            "Feature Config"
        ]
        .nunique()
        >= 2
    )


def _assert_all_source_protocol(
    points,
):

    for scenario, scenario_points in points.groupby(
        "Scenario"
    ):

        if scenario in [
            "cross_session",
            "cross_subject",
        ]:

            grouped = scenario_points.groupby(
                [
                    "Dataset",
                    "Target Fraction",
                    "Regime",
                    "Method",
                    "Preprocessing Config",
                    "Feature Config",
                    "Training Seed",
                ],
                dropna=False,
            )

            for _, group in grouped:

                max_count = group[
                    "Source Domain Count"
                ].max()

                assert (
                    group[
                        "Source Domain Count"
                    ]
                    .eq(
                        max_count
                    )
                    .all()
                )

        if scenario == "cross_dataset":

            grouped = scenario_points.groupby(
                [
                    "Held-out Dataset",
                    "Target Fraction",
                    "Regime",
                    "Method",
                    "Preprocessing Config",
                    "Feature Config",
                    "Training Seed",
                ],
                dropna=False,
            )

            for _, group in grouped:

                max_count = group[
                    "Source Super Domain Count"
                ].max()

                assert (
                    group[
                        "Source Super Domain Count"
                    ]
                    .eq(
                        max_count
                    )
                    .all()
                )


# ============================================================
# Assertions: numeric consistency
# ============================================================

def _assert_numeric_points(
    points,
):

    numeric_columns = [
        "Target Fraction",
        "target_fraction",
        "Source Domain Count",
        "Source Super Domain Count",
        "Target-Test BA",
        "Macro-F1",
        "AUC",
    ]

    for column in numeric_columns:

        values = pd.to_numeric(
            points[
                column
            ],
            errors="coerce",
        )

        assert values.notna().all()


def _assert_numeric_summary(
    summary,
):

    numeric_columns = [
        "Runs",
        "Mean BA",
        "BA Std",
        "BA SEM",
        "BA CI95 Low",
        "BA CI95 High",
        "Total BA Variance",
        "Isolated Variance Sum",
    ]

    for column in numeric_columns:

        values = pd.to_numeric(
            summary[
                column
            ],
            errors="coerce",
        )

        assert values.notna().all()


def _assert_summary_matches_points(
    points,
    summary,
):

    expected = (
        points
        .groupby(
            [
                "Scenario",
                "Regime",
                "Method",
            ],
            as_index=False,
        )[
            "Target-Test BA"
        ]
        .agg(
            Runs="count",
            MeanBA="mean",
            TotalVar=lambda x: x.var(
                ddof=1
            ),
        )
    )

    merged = summary.merge(
        expected,
        on=[
            "Scenario",
            "Regime",
            "Method",
        ],
        how="left",
    )

    assert not merged[
        "Runs_y"
    ].isna().any()

    assert (
        merged[
            "Runs_x"
        ]
        .astype(
            int
        )
        .eq(
            merged[
                "Runs_y"
            ].astype(
                int
            )
        )
        .all()
    )

    assert np.allclose(
        merged[
            "Mean BA"
        ],
        merged[
            "MeanBA"
        ],
    )

    assert np.allclose(
        merged[
            "Total BA Variance"
        ],
        merged[
            "TotalVar"
        ],
    )


# ============================================================
# Assertions: components
# ============================================================

def _assert_expected_components_exist(
    components,
):

    expected_components = {
        "Preprocessing config",
        "Dataset",
        "Target domain",
        "Feature config",
        "Seed",
    }

    observed_components = set(
        components[
            "Component"
        ]
    )

    assert expected_components.issubset(
        observed_components
    )

    assert "Scenario" not in observed_components
    assert "Source composition" not in observed_components
    assert "Target fraction" not in observed_components
    assert "Residual / repeated runs" not in observed_components


def _assert_component_variances_non_negative(
    components,
):

    values = pd.to_numeric(
        components[
            "Isolated Variance"
        ],
        errors="coerce",
    ).dropna()

    assert not values.empty

    assert (
        values
        >= 0
    ).all()


def _assert_component_shares_sum_to_100_when_estimable(
    components,
    summary,
):

    grouped = components.groupby(
        [
            "Scenario",
            "Regime",
            "Method",
        ]
    )

    for key, group in grouped:

        scenario, regime, method = key

        summary_row = summary[
            (
                summary[
                    "Scenario"
                ]
                == scenario
            )
            & (
                summary[
                    "Regime"
                ]
                == regime
            )
            & (
                summary[
                    "Method"
                ]
                == method
            )
        ]

        assert not summary_row.empty

        isolated_sum = float(
            summary_row[
                "Isolated Variance Sum"
            ].iloc[
                0
            ]
        )

        share_sum = pd.to_numeric(
            group[
                "Share (%)"
            ],
            errors="coerce",
        ).sum()

        if isolated_sum > 0:

            assert np.isclose(
                share_sum,
                100.0,
                atol=1e-6,
            )

        else:

            assert np.isclose(
                share_sum,
                0.0,
                atol=1e-6,
            )


def _assert_some_components_are_estimable(
    components,
):

    estimable_components = components[
        components[
            "Estimable"
        ].astype(
            bool
        )
    ]

    assert not estimable_components.empty


def _assert_feature_component_exists(
    components,
):

    feature_rows = components[
        components[
            "Component"
        ]
        == "Feature config"
    ]

    assert not feature_rows.empty

    assert (
        feature_rows[
            "Unique Values"
        ]
        .astype(
            int
        )
        .max()
        >= 2
    )


def _assert_preprocessing_component_exists_even_if_not_estimable(
    components,
):

    preprocessing_rows = components[
        components[
            "Component"
        ]
        == "Preprocessing config"
    ]

    assert not preprocessing_rows.empty

    assert (
        preprocessing_rows[
            "Unique Values"
        ]
        .astype(
            int
        )
        .min()
        >= 1
    )


def _assert_seed_component_exists_even_if_not_estimable(
    components,
):

    seed_rows = components[
        components[
            "Component"
        ]
        == "Seed"
    ]

    assert not seed_rows.empty

    assert (
        seed_rows[
            "Unique Values"
        ]
        .astype(
            int
        )
        .min()
        >= 1
    )


# ============================================================
# Assertions: figures
# ============================================================

def _assert_figures_exist(
    artifact,
    points,
):

    assert isinstance(
        artifact.figures,
        dict,
    )

    assert len(
        artifact.figures
    ) > 0

    observed_scenarios = set(
        points[
            "Scenario"
        ]
    )

    for scenario in observed_scenarios:

        expected_key = (
            f"{scenario}_method_stability"
        )

        assert expected_key in artifact.figures

        path = artifact.figures[
            expected_key
        ]

        assert exists(
            path
        )

        assert Path(
            path
        ).suffix == ".png"


# ============================================================
# Test
# ============================================================

def test_variability_decomposition_with_real_eeg_data():

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

    variability_decomposition.OUTPUT_ROOT = (
        TEST_ROOT
        / "variability_decomposition"
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
            params["dataset"]
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
            "name": "variability_combined",
            "preprocessing_config_label": PREPROCESSING_LABEL,
        },
    )

    assert exists(
        combined_view.path
    )

    assert (
        combined_view.preprocessing_config_label
        == PREPROCESSING_LABEL
    )

    assert (
        combined_view.preprocessing_signature
        is not None
    )

    # --------------------------------------------------------
    # Generate scenario splits
    # --------------------------------------------------------

    scenario_items = []

    for dataset, view in views.items():

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

    # --------------------------------------------------------
    # Run variability decomposition
    # --------------------------------------------------------

    artifact = (
        variability_decomposition
        .run_variability_decomposition(
            model_results_artifact,
            VARIABILITY_DECOMPOSITION_PARAMS,
        )
    )

    assert exists(
        artifact.manifest_path
    )

    assert "points" in artifact.tables
    assert "summary" in artifact.tables
    assert "components" in artifact.tables

    assert exists(
        artifact.tables[
            "points"
        ]
    )

    assert exists(
        artifact.tables[
            "summary"
        ]
    )

    assert exists(
        artifact.tables[
            "components"
        ]
    )

    points = pd.read_csv(
        artifact.tables[
            "points"
        ]
    )

    summary = pd.read_csv(
        artifact.tables[
            "summary"
        ]
    )

    components = pd.read_csv(
        artifact.tables[
            "components"
        ]
    )

    assert not points.empty
    assert not summary.empty
    assert not components.empty

    # --------------------------------------------------------
    # Table structure
    # --------------------------------------------------------

    _assert_points_columns(
        points
    )

    _assert_summary_columns(
        summary
    )

    _assert_components_columns(
        components
    )

    # --------------------------------------------------------
    # Labels and filtering
    # --------------------------------------------------------

    _assert_labels(
        points
    )

    _assert_labels(
        summary
    )

    _assert_labels(
        components
    )

    _assert_clean_feature_labels(
        points
    )

    _assert_preprocessing_trace_exists(
        points
    )

    _assert_no_intra_subject(
        points
    )

    _assert_real_scenarios_exist(
        points
    )

    _assert_only_fixed_target_fraction(
        points
    )

    _assert_feature_variation_exists(
        points
    )

    _assert_all_source_protocol(
        points
    )

    # --------------------------------------------------------
    # Numeric consistency
    # --------------------------------------------------------

    _assert_numeric_points(
        points
    )

    _assert_numeric_summary(
        summary
    )

    _assert_summary_matches_points(
        points,
        summary,
    )

    # --------------------------------------------------------
    # Variability components
    # --------------------------------------------------------

    _assert_expected_components_exist(
        components
    )

    _assert_component_variances_non_negative(
        components
    )

    _assert_component_shares_sum_to_100_when_estimable(
        components,
        summary,
    )

    _assert_some_components_are_estimable(
        components
    )

    _assert_feature_component_exists(
        components
    )

    _assert_preprocessing_component_exists_even_if_not_estimable(
        components
    )

    _assert_seed_component_exists_even_if_not_estimable(
        components
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    _assert_figures_exist(
        artifact,
        points,
    )