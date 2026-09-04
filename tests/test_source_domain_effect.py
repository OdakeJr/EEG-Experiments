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
import pipeline.analysis.source_domain_effect as source_domain_effect

from utils.storage import (
    exists,
)


# ============================================================
# Test output locations
# ============================================================

TEST_ROOT = Path(
    "tests/output/source_domain_effect"
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
# Source-domain effect protocol
# ============================================================

SOURCE_ONLY_TARGET_FRACTION = 0.0

SOURCE_EFFECT_TARGET_FRACTION = 0.25


# ============================================================
# Preprocessing params
# ============================================================

PREPROCESSING_PARAMS = [

    # --------------------------------------------------------
    # BCI Competition IV 2a
    # --------------------------------------------------------

    {
        "dataset": "bci2a",
        "name": "source_effect_bci2a",

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
        "name": "source_effect_weibo",

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
        "name": "source_effect_zhou",

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
        SOURCE_EFFECT_TARGET_FRACTION,
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
        SOURCE_EFFECT_TARGET_FRACTION,
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
        SOURCE_EFFECT_TARGET_FRACTION,
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
            "max_iter": 300,
        },
        "training_params": {},
    },
]


# ============================================================
# Source-domain effect params
# ============================================================

SOURCE_DOMAIN_EFFECT_PARAMS = {
    "plot_metric": "Target-Test BA",

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
            "source_axis_column": "n_source_domains",
            "source_axis_label": "Number of source sessions",
            "output_prefix": "cross_session",
            "filters": {
                "n_target_super_domains": 0,
                "target_fraction": SOURCE_EFFECT_TARGET_FRACTION,
            },
        },
        {
            "name": "cross_subject",
            "scenario": "cross_subject",
            "setting_column": "Dataset",
            "source_axis_column": "n_source_domains",
            "source_axis_label": "Number of source subjects",
            "output_prefix": "cross_subject",
            "filters": {
                "n_target_super_domains": 0,
                "target_fraction": SOURCE_EFFECT_TARGET_FRACTION,
            },
        },
        {
            "name": "cross_dataset",
            "scenario": "cross_dataset",
            "setting_column": "Held-out Dataset",
            "source_axis_column": "source_super_domain_count",
            "source_axis_label": "Number of source datasets",
            "output_prefix": "cross_dataset",
            "filters": {
                "n_target_super_domains": 0,
                "target_fraction": SOURCE_EFFECT_TARGET_FRACTION,
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

    # Domain IDs use "|", for example:
    # bci_iv_2a|A01
    #
    # Domain lists are separated by ";" in model_results.csv.
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
            group="source_effect",
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
                group="source_effect",
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
                "group": "source_effect",
                "view": view,
                "split": split,
                "fs_artifact": fs_artifact,
                "artifacts": split_model_artifacts,
            }
        )

    return model_artifacts_by_scenario


# ============================================================
# Helpers: result filtering
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
        & np.isclose(
            pd.to_numeric(
                model_table["target_fraction"],
                errors="coerce",
            ),
            SOURCE_EFFECT_TARGET_FRACTION,
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

    rows[
        "source_dataset_count"
    ] = rows[
        "source_super_domain_count"
    ]

    return rows


def _expected_raw_summary_counts(
    model_table,
    scenario,
    setting_column,
    source_axis_column,
):

    rows = _target_test_rows(
        model_table,
        scenario,
    )

    if rows.empty:

        return pd.DataFrame(
            columns=[
                "Scenario",
                "Setting",
                "Source Domain Count",
                "Expected Runs",
            ]
        )

    rows[
        "Scenario"
    ] = scenario

    rows[
        "Setting"
    ] = rows[
        setting_column
    ]

    rows[
        "Source Domain Count"
    ] = pd.to_numeric(
        rows[
            source_axis_column
        ],
        errors="coerce",
    )

    counts = (
        rows
        .groupby(
            [
                "Scenario",
                "Setting",
                "Source Domain Count",
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

    return counts


def _assert_summary_matches_raw_counts(
    summary,
    model_table,
):

    for scenario_config in SOURCE_DOMAIN_EFFECT_PARAMS[
        "scenarios"
    ]:

        scenario = scenario_config[
            "scenario"
        ]

        setting_column = scenario_config[
            "setting_column"
        ]

        source_axis_column = scenario_config[
            "source_axis_column"
        ]

        raw_counts = _expected_raw_summary_counts(
            model_table=model_table,
            scenario=scenario,
            setting_column=setting_column,
            source_axis_column=source_axis_column,
        )

        scenario_summary = summary[
            summary["Scenario"]
            == scenario
        ].copy()

        if raw_counts.empty:

            assert scenario_summary.empty
            continue

        assert not scenario_summary.empty

        merged = scenario_summary.merge(
            raw_counts,
            on=[
                "Scenario",
                "Setting",
                "Source Domain Count",
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


# ============================================================
# Helpers: summary assertions
# ============================================================

def _assert_summary_columns(
    summary,
):

    required_columns = {
        "Scenario",
        "Setting",
        "Regime",
        "Method",
        "Source Domain Count",
        "Source Domain Axis",
        "Runs",

        "Source BA Mean",
        "Source BA Std",
        "Source BA SEM",
        "Source BA CI95 Low",
        "Source BA CI95 High",

        "Target-Test BA Mean",
        "Target-Test BA Std",
        "Target-Test BA SEM",
        "Target-Test BA CI95 Low",
        "Target-Test BA CI95 High",

        "Gap Mean",
        "Gap Std",
        "Gap SEM",
        "Gap CI95 Low",
        "Gap CI95 High",

        "Macro-F1 Mean",
        "Macro-F1 Std",
        "Macro-F1 SEM",
        "Macro-F1 CI95 Low",
        "Macro-F1 CI95 High",

        "AUC Mean",
        "AUC Std",
        "AUC SEM",
        "AUC CI95 Low",
        "AUC CI95 High",
    }

    assert required_columns.issubset(
        set(
            summary.columns
        )
    )


def _assert_summary_labels(
    summary,
):

    assert set(
        summary[
            "Regime"
        ]
    ) == {
        "Classical",
    }

    assert set(
        summary[
            "Method"
        ]
    ) == {
        "Logistic Regression",
    }


def _assert_summary_is_numeric(
    summary,
):

    numeric_columns = [
        "Source Domain Count",
        "Runs",

        "Source BA Mean",
        "Source BA Std",
        "Source BA SEM",
        "Source BA CI95 Low",
        "Source BA CI95 High",

        "Target-Test BA Mean",
        "Target-Test BA Std",
        "Target-Test BA SEM",
        "Target-Test BA CI95 Low",
        "Target-Test BA CI95 High",

        "Gap Mean",
        "Gap Std",
        "Gap SEM",
        "Gap CI95 Low",
        "Gap CI95 High",

        "Macro-F1 Mean",
        "Macro-F1 Std",
        "Macro-F1 SEM",
        "Macro-F1 CI95 Low",
        "Macro-F1 CI95 High",

        "AUC Mean",
        "AUC Std",
        "AUC SEM",
        "AUC CI95 Low",
        "AUC CI95 High",
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
    summary,
):

    assert "intra_subject" not in set(
        summary[
            "Scenario"
        ]
    )


def _assert_source_axis_values_exist(
    summary,
):

    cross_subject = summary[
        summary[
            "Scenario"
        ]
        == "cross_subject"
    ]

    assert not cross_subject.empty

    assert (
        cross_subject[
            "Source Domain Count"
        ]
        .nunique()
        >= 2
    )

    cross_dataset = summary[
        summary[
            "Scenario"
        ]
        == "cross_dataset"
    ]

    assert not cross_dataset.empty

    assert (
        cross_dataset[
            "Source Domain Count"
        ]
        .nunique()
        >= 2
    )


def _assert_ci_columns_are_consistent(
    summary,
):

    metric_prefixes = [
        "Source BA",
        "Target-Test BA",
        "Gap",
        "Macro-F1",
        "AUC",
    ]

    for prefix in metric_prefixes:

        mean_col = (
            f"{prefix} Mean"
        )

        sem_col = (
            f"{prefix} SEM"
        )

        low_col = (
            f"{prefix} CI95 Low"
        )

        high_col = (
            f"{prefix} CI95 High"
        )

        rows = summary[
            summary[
                sem_col
            ].notna()
        ]

        if rows.empty:
            continue

        assert (
            rows[
                low_col
            ]
            .le(
                rows[
                    mean_col
                ]
            )
            .all()
        )

        assert (
            rows[
                high_col
            ]
            .ge(
                rows[
                    mean_col
                ]
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

def test_source_domain_effect_with_natural_source_growth():

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

    source_domain_effect.OUTPUT_ROOT = (
        TEST_ROOT
        / "source_domain_effect"
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
            "name": "source_effect_combined",
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
    # Run source-domain-effect analysis
    # --------------------------------------------------------

    artifact = source_domain_effect.run_source_domain_effect(
        model_results_artifact,
        SOURCE_DOMAIN_EFFECT_PARAMS,
    )

    assert exists(
        artifact.manifest_path
    )

    assert "summary" in artifact.tables

    assert exists(
        artifact.tables[
            "summary"
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

    assert not summary.empty
    assert not model_table.empty

    # --------------------------------------------------------
    # Summary checks
    # --------------------------------------------------------

    _assert_summary_columns(
        summary
    )

    _assert_summary_labels(
        summary
    )

    _assert_summary_is_numeric(
        summary
    )

    _assert_no_intra_subject(
        summary
    )

    _assert_source_axis_values_exist(
        summary
    )

    _assert_ci_columns_are_consistent(
        summary
    )

    _assert_summary_matches_raw_counts(
        summary=summary,
        model_table=model_table,
    )

    # --------------------------------------------------------
    # Figure checks
    # --------------------------------------------------------

    _assert_figures_exist(
        artifact
    )