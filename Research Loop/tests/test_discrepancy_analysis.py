from pathlib import Path
import ast

import numpy as np
import pandas as pd
import pytest

import pipeline.process_data as process_data
import pipeline.combine_data as combine_data
import pipeline.scenarios as scenarios
import pipeline.feature_selection as feature_selection
import pipeline.training as training

import pipeline.evaluation.model_results as model_results
import pipeline.evaluation.domain_results as domain_results
import pipeline.analysis.discrepancy_analysis as discrepancy_analysis

from utils.storage import (
    exists,
)


# ============================================================
# Test output locations
# ============================================================

TEST_ROOT = Path(
    "tests/output/discrepancy_analysis"
)


# ============================================================
# Test configuration
# ============================================================

BCI2A_SUBJECTS = [
    1, 2, 3, 4,
]

WEIBO_SUBJECTS = [
    1, 2, 3, 4,
]

ZHOU_SUBJECTS = [
    1, 2, 3, 4,
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


# ============================================================
# Discrepancy-analysis protocol
# ============================================================

SOURCE_ONLY_TARGET_FRACTION = 0.0

DISCREPANCY_TARGET_FRACTION = 0.25


# ============================================================
# Preprocessing params
# ============================================================

PREPROCESSING_PARAMS = [

    # --------------------------------------------------------
    # BCI Competition IV 2a
    # --------------------------------------------------------

    {
        "dataset": "bci2a",
        "name": "discrepancy_bci2a",

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

    # --------------------------------------------------------
    # Weibo2014
    # --------------------------------------------------------

    {
        "dataset": "weibo",
        "name": "discrepancy_weibo",

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

    # --------------------------------------------------------
    # Zhou2016
    # --------------------------------------------------------

    {
        "dataset": "zhou",
        "name": "discrepancy_zhou",

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
# Scenario params
# ============================================================

SESSION_PARAMS = {
    "source_counts": [
        1,
        "all",
    ],

    "target_fractions": [
        SOURCE_ONLY_TARGET_FRACTION,
        DISCREPANCY_TARGET_FRACTION,
    ],

    "seed": 42,
}


SUBJECT_PARAMS = {
    "source_counts": [
        1,
        2,
        "all",
    ],

    "target_fractions": [
        SOURCE_ONLY_TARGET_FRACTION,
        DISCREPANCY_TARGET_FRACTION,
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
        DISCREPANCY_TARGET_FRACTION,
    ],

    "max_source_combinations": 2,

    "seed": 42,
}


# ============================================================
# Feature selection, training, and domain params
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
            "max_iter": 300,
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

    "comparisons": [
        "source_to_target_elementary",
    ],

    "standardize": True,

    "min_samples_per_side": 2,
}


# ============================================================
# Discrepancy-analysis params
# ============================================================

DISCREPANCY_ANALYSIS_PARAMS = {
    "discrepancy_metric": "mmd",
    "comparison": "source_to_target_elementary",
    "representation": "marginal",

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
            "output_prefix": "cross_session",
            "filters": {
                "n_target_super_domains": 0,
                "target_fraction": DISCREPANCY_TARGET_FRACTION,
                "use_max_source_domains": True,
            },
        },
        {
            "name": "cross_subject",
            "scenario": "cross_subject",
            "setting_column": "Dataset",
            "output_prefix": "cross_subject",
            "filters": {
                "n_target_super_domains": 0,
                "target_fraction": DISCREPANCY_TARGET_FRACTION,
                "use_max_source_domains": True,
            },
        },
        {
            "name": "cross_dataset",
            "scenario": "cross_dataset",
            "setting_column": "Held-out Dataset",
            "output_prefix": "cross_dataset",
            "filters": {
                "n_target_super_domains": 0,
                "target_fraction": DISCREPANCY_TARGET_FRACTION,
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


# ============================================================
# Helpers: domain parsing
# ============================================================

def _parse_domain_list(
    value,
):

    if value is None:
        return []

    if isinstance(
        value,
        float,
    ) and pd.isna(
        value
    ):
        return []

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            str(item)
            for item in value
        ]

    text = str(
        value
    ).strip()

    if text in [
        "",
        "[]",
        "None",
        "nan",
    ]:
        return []

    try:

        parsed = ast.literal_eval(
            text
        )

        if isinstance(
            parsed,
            (
                list,
                tuple,
                set,
            ),
        ):

            return [
                str(item)
                for item in parsed
            ]

    except Exception:

        pass

    # Domain IDs use "|", e.g. bci_iv_2a|A01.
    # Domain lists are separated by ";" in result CSVs.
    for separator in [
        ";",
        ",",
    ]:

        if separator in text:

            return [
                item.strip()
                for item in text.split(
                    separator
                )
                if item.strip()
            ]

    return [
        text
    ]


def _dataset_from_domains(
    value,
):

    domains = _parse_domain_list(
        value
    )

    datasets = sorted(
        {
            domain.split(
                "|",
                1,
            )[0]
            for domain in domains
        }
    )

    return "+".join(
        datasets
    )


def _dataset_count_from_domains(
    value,
):

    domains = _parse_domain_list(
        value
    )

    datasets = {
        domain.split(
            "|",
            1,
        )[0]
        for domain in domains
    }

    return len(
        datasets
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

        fs_artifact = feature_selection.run_feature_selection(
            split,
            view,
            FS_PARAMS,
            group="discrepancy",
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
                group="discrepancy",
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
                "group": "discrepancy",
                "view": view,
                "split": split,
                "fs_artifact": fs_artifact,
                "artifacts": split_model_artifacts,
            }
        )

    return model_artifacts_by_scenario


def _build_domain_scenario_artifacts(
    scenario_items,
):

    scenario_artifacts = {}

    for item in scenario_items:

        split = item[
            "split"
        ]

        view = item[
            "view"
        ]

        scenario_artifacts.setdefault(
            split.scenario,
            []
        ).append(
            {
                "group": "discrepancy",
                "view": view,
                "splits": [
                    split,
                ],
            }
        )

    return scenario_artifacts


# ============================================================
# Helpers: expected raw points
# ============================================================

def _target_test_rows(
    model_table,
    scenario,
):

    rows = model_table[
        (
            model_table["scenario"]
            == scenario
        )
        & (
            model_table["evaluation_group"]
            == "target_elementary_domain"
        )
        & (
            model_table["partition"]
            == "test"
        )
    ].copy()

    rows[
        "Dataset"
    ] = rows[
        "target_domains"
    ].apply(
        _dataset_from_domains
    )

    rows[
        "Held-out Dataset"
    ] = rows[
        "Dataset"
    ]

    rows[
        "source_super_domain_count"
    ] = rows[
        "source_domains"
    ].apply(
        _dataset_count_from_domains
    )

    return rows


def _source_train_rows(
    model_table,
):

    rows = model_table[
        (
            model_table["evaluation_group"]
            == "source"
        )
        & (
            model_table["partition"]
            == "train"
        )
    ].copy()

    return rows


def _model_gap_rows(
    model_table,
    scenario,
):

    target = _target_test_rows(
        model_table,
        scenario,
    )

    source = _source_train_rows(
        model_table
    )

    key_columns = [
        "split_id",
        "scenario",
        "group",
        "n_source_domains",
        "n_target_super_domains",
        "target_fraction",
        "source_domains",
        "target_domains",
        "feature_selection_signature",
        "learning_method",
        "model_name",
        "model_signature",
    ]

    if "training_seed" in model_table.columns:
        key_columns.append(
            "training_seed"
        )

    source = source[
        key_columns
        + [
            "balanced_accuracy",
        ]
    ].rename(
        columns={
            "balanced_accuracy": "source_ba",
        }
    )

    target = target.rename(
        columns={
            "balanced_accuracy": "target_ba",
        }
    )

    merged = target.merge(
        source,
        on=key_columns,
        how="inner",
    )

    merged[
        "Gap"
    ] = (
        merged[
            "source_ba"
        ]
        - merged[
            "target_ba"
        ]
    )

    return merged


def _discrepancy_rows(
    domain_table,
):

    rows = domain_table[
        (
            domain_table[
                "comparison"
            ]
            == DISCREPANCY_ANALYSIS_PARAMS[
                "comparison"
            ]
        )
        & (
            domain_table[
                "metric"
            ]
            == DISCREPANCY_ANALYSIS_PARAMS[
                "discrepancy_metric"
            ]
        )
        & (
            domain_table[
                "representation"
            ]
            == DISCREPANCY_ANALYSIS_PARAMS[
                "representation"
            ]
        )
    ].copy()

    if rows.empty:

        return pd.DataFrame(
            columns=[
                "split_id",
                "Discrepancy",
            ]
        )

    rows = (
        rows
        .groupby(
            [
                "split_id",
            ],
            as_index=False,
        )[
            "value"
        ]
        .mean()
        .rename(
            columns={
                "value": "Discrepancy",
            }
        )
    )

    return rows


def _filter_expected_points(
    rows,
    scenario_config,
):

    if rows.empty:
        return rows

    setting_column = scenario_config[
        "setting_column"
    ]

    filters = scenario_config[
        "filters"
    ]

    result = rows.copy()

    for column, value in filters.items():

        if column in [
            "use_max_source_domains",
            "use_max_source_super_domains",
        ]:
            continue

        if column not in result.columns:
            continue

        if isinstance(
            value,
            (int, float),
        ):

            result = result[
                np.isclose(
                    pd.to_numeric(
                        result[column],
                        errors="coerce",
                    ),
                    value,
                )
            ]

        else:

            result = result[
                result[column]
                == value
            ]

    if filters.get(
        "use_max_source_domains",
        False,
    ):

        max_values = (
            result
            .groupby(
                [
                    setting_column,
                    "target_fraction",
                ]
            )[
                "n_source_domains"
            ]
            .transform(
                "max"
            )
        )

        result = result[
            result[
                "n_source_domains"
            ]
            == max_values
        ]

    if filters.get(
        "use_max_source_super_domains",
        False,
    ):

        max_values = (
            result
            .groupby(
                [
                    setting_column,
                    "target_fraction",
                ]
            )[
                "source_super_domain_count"
            ]
            .transform(
                "max"
            )
        )

        result = result[
            result[
                "source_super_domain_count"
            ]
            == max_values
        ]

    return result


def _expected_points(
    model_table,
    domain_table,
    scenario_config,
):

    scenario = scenario_config[
        "scenario"
    ]

    setting_column = scenario_config[
        "setting_column"
    ]

    model_rows = _model_gap_rows(
        model_table,
        scenario,
    )

    discrepancy_rows = _discrepancy_rows(
        domain_table
    )

    rows = model_rows.merge(
        discrepancy_rows,
        on="split_id",
        how="inner",
    )

    if rows.empty:

        return pd.DataFrame(
            columns=[
                "Scenario",
                "Setting",
                "Expected Runs",
            ]
        )

    rows[
        "Setting"
    ] = rows[
        setting_column
    ]

    rows = _filter_expected_points(
        rows,
        scenario_config,
    )

    rows[
        "Scenario"
    ] = scenario

    rows[
        "Target Fraction"
    ] = pd.to_numeric(
        rows[
            "target_fraction"
        ],
        errors="coerce",
    )

    rows[
        "Source Domain Count"
    ] = pd.to_numeric(
        rows[
            "n_source_domains"
        ],
        errors="coerce",
    )

    rows[
        "Source Super Domain Count"
    ] = pd.to_numeric(
        rows[
            "source_super_domain_count"
        ],
        errors="coerce",
    )

    return rows


def _expected_summary_counts(
    model_table,
    domain_table,
):

    all_counts = []

    for scenario_config in DISCREPANCY_ANALYSIS_PARAMS[
        "scenarios"
    ]:

        rows = _expected_points(
            model_table,
            domain_table,
            scenario_config,
        )

        if rows.empty:
            continue

        counts = (
            rows
            .groupby(
                [
                    "Scenario",
                    "Setting",
                    "learning_method",
                    "model_name",
                ],
                as_index=False,
            )
            .size()
            .rename(
                columns={
                    "size": "Expected Runs",
                }
            )
        )

        all_counts.append(
            counts
        )

    if not all_counts:

        return pd.DataFrame(
            columns=[
                "Scenario",
                "Setting",
                "learning_method",
                "model_name",
                "Expected Runs",
            ]
        )

    return pd.concat(
        all_counts,
        ignore_index=True,
    )


# ============================================================
# Helpers: assertions
# ============================================================

def _assert_points_columns(
    points,
):

    required_columns = {
        "split_id",
        "Scenario",
        "Setting",
        "Regime",
        "Method",
        "learning_method",
        "model_name",
        "model_signature",
        "feature_selection_signature",
        "Source Domain Count",
        "Source Super Domain Count",
        "Target Fraction",
        "source_domains",
        "target_domains",
        "Discrepancy",
        "Source BA",
        "Target-Test BA",
        "Gap",
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
        "Setting",
        "Regime",
        "Method",
        "Runs",
        "Pearson r",
        "Spearman rho",
        "Linear Slope",
        "Linear Intercept",
        "Discrepancy Mean",
        "Discrepancy Std",
        "Gap Mean",
        "Gap Std",
    }

    assert required_columns.issubset(
        set(
            summary.columns
        )
    )


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


def _assert_numeric_points(
    points,
):

    numeric_columns = [
        "Source Domain Count",
        "Source Super Domain Count",
        "Target Fraction",
        "Discrepancy",
        "Source BA",
        "Target-Test BA",
        "Gap",
        "Macro-F1",
        "AUC",
    ]

    for column in numeric_columns:

        converted = pd.to_numeric(
            points[
                column
            ],
            errors="coerce",
        )

        assert len(
            converted
        ) == len(
            points
        )


def _assert_numeric_summary(
    summary,
):

    numeric_columns = [
        "Runs",
        "Pearson r",
        "Spearman rho",
        "Linear Slope",
        "Linear Intercept",
        "Discrepancy Mean",
        "Discrepancy Std",
        "Gap Mean",
        "Gap Std",
    ]

    for column in numeric_columns:

        converted = pd.to_numeric(
            summary[
                column
            ],
            errors="coerce",
        )

        assert len(
            converted
        ) == len(
            summary
        )


def _assert_no_intra_subject(
    points,
    summary,
):

    assert "intra_subject" not in set(
        points[
            "Scenario"
        ]
    )

    assert "intra_subject" not in set(
        summary[
            "Scenario"
        ]
    )


def _assert_only_requested_target_fraction(
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
        DISCREPANCY_TARGET_FRACTION,
    }


def _assert_gap_is_correct(
    points,
):

    source_ba = pd.to_numeric(
        points[
            "Source BA"
        ],
        errors="coerce",
    )

    target_ba = pd.to_numeric(
        points[
            "Target-Test BA"
        ],
        errors="coerce",
    )

    gap = pd.to_numeric(
        points[
            "Gap"
        ],
        errors="coerce",
    )

    assert np.allclose(
        gap,
        source_ba - target_ba,
        equal_nan=True,
    )


def _assert_summary_matches_points(
    points,
    summary,
):

    counts = (
        points
        .groupby(
            [
                "Scenario",
                "Setting",
                "Regime",
                "Method",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "Expected Runs",
            }
        )
    )

    merged = summary.merge(
        counts,
        on=[
            "Scenario",
            "Setting",
            "Regime",
            "Method",
        ],
        how="left",
    )

    assert not merged[
        "Expected Runs"
    ].isna().any()

    assert (
        merged[
            "Runs"
        ]
        .astype(
            int
        )
        .eq(
            merged[
                "Expected Runs"
            ].astype(
                int
            )
        )
        .all()
    )


def _assert_summary_matches_raw_counts(
    points,
    summary,
    model_table,
    domain_table,
):

    expected = _expected_summary_counts(
        model_table,
        domain_table,
    )

    expected = expected.rename(
        columns={
            "learning_method": "learning_method",
            "model_name": "model_name",
        }
    )

    # Map raw method names to display labels used by the analysis.
    expected[
        "Regime"
    ] = "Classical"

    expected[
        "Method"
    ] = "Logistic Regression"

    merged = summary.merge(
        expected[
            [
                "Scenario",
                "Setting",
                "Regime",
                "Method",
                "Expected Runs",
            ]
        ],
        on=[
            "Scenario",
            "Setting",
            "Regime",
            "Method",
        ],
        how="left",
    )

    assert not merged[
        "Expected Runs"
    ].isna().any()

    assert (
        merged[
            "Runs"
        ]
        .astype(
            int
        )
        .eq(
            merged[
                "Expected Runs"
            ].astype(
                int
            )
        )
        .all()
    )


def _assert_correlation_values_are_valid(
    summary,
):

    for column in [
        "Pearson r",
        "Spearman rho",
    ]:

        values = pd.to_numeric(
            summary[
                column
            ],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        assert (
            values
            .between(
                -1.0,
                1.0,
            )
            .all()
        )


def _assert_figures_exist(
    artifact,
):

    assert isinstance(
        artifact.figures,
        dict,
    )

    assert len(
        artifact.figures
    ) > 0

    for path in artifact.figures.values():

        assert exists(
            path
        )

        assert Path(
            path
        ).suffix == ".png"


# ============================================================
# Test
# ============================================================

def test_discrepancy_analysis_gap_vs_domain_distance():

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

    discrepancy_analysis.OUTPUT_ROOT = (
        TEST_ROOT
        / "discrepancy_analysis"
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
            "name": "discrepancy_combined",
        },
    )

    assert exists(
        combined_view.path
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
    # Evaluate domain discrepancies
    # --------------------------------------------------------

    domain_scenario_artifacts = _build_domain_scenario_artifacts(
        scenario_items
    )

    domain_results_artifact = domain_results.run_domain_evaluation(
        domain_scenario_artifacts,
        DOMAIN_EVALUATION_PARAMS,
    )

    assert exists(
        domain_results_artifact.path
    )

    # --------------------------------------------------------
    # Run discrepancy analysis
    # --------------------------------------------------------

    artifact = discrepancy_analysis.run_discrepancy_analysis(
        model_results_artifact,
        domain_results_artifact,
        DISCREPANCY_ANALYSIS_PARAMS,
    )

    assert exists(
        artifact.manifest_path
    )

    assert "points" in artifact.tables
    assert "summary" in artifact.tables

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

    model_table = pd.read_csv(
        model_results_artifact.path
    )

    domain_table = pd.read_csv(
        domain_results_artifact.path
    )

    assert not points.empty
    assert not summary.empty
    assert not model_table.empty
    assert not domain_table.empty

    # --------------------------------------------------------
    # Points checks
    # --------------------------------------------------------

    _assert_points_columns(
        points
    )

    _assert_labels(
        points
    )

    _assert_numeric_points(
        points
    )

    _assert_no_intra_subject(
        points,
        summary,
    )

    _assert_only_requested_target_fraction(
        points
    )

    _assert_gap_is_correct(
        points
    )

    # --------------------------------------------------------
    # Summary checks
    # --------------------------------------------------------

    _assert_summary_columns(
        summary
    )

    _assert_labels(
        summary
    )

    _assert_numeric_summary(
        summary
    )

    _assert_summary_matches_points(
        points,
        summary,
    )

    _assert_summary_matches_raw_counts(
        points,
        summary,
        model_table,
        domain_table,
    )

    _assert_correlation_values_are_valid(
        summary
    )

    # --------------------------------------------------------
    # Figure checks
    # --------------------------------------------------------

    _assert_figures_exist(
        artifact
    )