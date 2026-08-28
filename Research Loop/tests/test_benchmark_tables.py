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
# Full-subject test configuration
# ============================================================

BCI2A_SUBJECTS = list(
    range(
        1,
        10,
    )
)

WEIBO_SUBJECTS = list(
    range(
        1,
        11,
    )
)

ZHOU_SUBJECTS = list(
    range(
        1,
        5,
    )
)


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
# Benchmark protocol choices
# ============================================================

INTRA_TRAIN_FRACTION = 0.8

SOURCE_ONLY_TARGET_FRACTION = 0.0

BENCHMARK_TARGET_FRACTION = 0.25


# ============================================================
# Preprocessing params
# ============================================================

PREPROCESSING_PARAMS = [

    # --------------------------------------------------------
    # BCI Competition IV 2a
    # --------------------------------------------------------

    {
        "dataset": "bci2a",
        "name": "benchmark_table_bci2a_full",

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
        "name": "benchmark_table_weibo_full",

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
        "name": "benchmark_table_zhou_full",

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

INTRA_PARAMS = {
    "train_fraction": INTRA_TRAIN_FRACTION,
    "seed": 42,
}


SESSION_PARAMS = {
    # Include both a smaller-source protocol and the all-source
    # protocol. The benchmark table must keep only all-source.
    "source_counts": [
        1,
        "all",
    ],

    # Include more than one target fraction. The benchmark table
    # must keep only BENCHMARK_TARGET_FRACTION.
    "target_fractions": [
        SOURCE_ONLY_TARGET_FRACTION,
        BENCHMARK_TARGET_FRACTION,
    ],

    "seed": 42,
}


SUBJECT_PARAMS = {
    # Include smaller-source and all-source protocols.
    # The benchmark table must keep only all subjects except
    # the held-out subject.
    "source_counts": [
        1,
        "all",
    ],

    # Include more than one target fraction. The benchmark table
    # must keep only BENCHMARK_TARGET_FRACTION.
    "target_fractions": [
        SOURCE_ONLY_TARGET_FRACTION,
        BENCHMARK_TARGET_FRACTION,
    ],

    # This limits only the smaller-source combinations.
    # The all-source protocol is still generated.
    "max_source_combinations": 2,

    "seed": 42,
}


DATASET_PARAMS = {
    # Include one-source-dataset and all-source-dataset protocols.
    # The benchmark table must keep only all source datasets except
    # the held-out target dataset.
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

    # Include more than one target fraction. The benchmark table
    # must keep only BENCHMARK_TARGET_FRACTION.
    "target_subject_fractions": [
        SOURCE_ONLY_TARGET_FRACTION,
        BENCHMARK_TARGET_FRACTION,
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
# Benchmark table params
# ============================================================

BENCHMARK_TABLE_PARAMS = {
    "discrepancy_metric": "mmd",

    "method_display": [
        {
            "learning_method": "sklearn_erm",
            "model_name": "logistic_regression",
            "regime": "Classical",
            "method": "Logistic Regression",
        },
    ],

    "tables": [
        {
            "name": "intra_subject",
            "scenario": "intra_subject",
            "setting_column": "Dataset",
            "output_name": "intra_subject_table.csv",
            "include_discrepancy": False,
            "filters": {
                "target_fraction": INTRA_TRAIN_FRACTION,
            },
        },
        {
            "name": "cross_session",
            "scenario": "cross_session",
            "setting_column": "Dataset",
            "output_name": "cross_session_table.csv",
            "include_discrepancy": True,
            "filters": {
                "n_target_super_domains": 0,
                "target_fraction": BENCHMARK_TARGET_FRACTION,
                "use_max_source_domains": True,
            },
        },
        {
            "name": "cross_subject",
            "scenario": "cross_subject",
            "setting_column": "Dataset",
            "output_name": "cross_subject_table.csv",
            "include_discrepancy": True,
            "filters": {
                "n_target_super_domains": 0,
                "target_fraction": BENCHMARK_TARGET_FRACTION,
                "use_max_source_domains": True,
            },
        },
        {
            "name": "cross_dataset",
            "scenario": "cross_dataset",
            "setting_column": "Held-out Dataset",
            "output_name": "cross_dataset_table.csv",
            "include_discrepancy": True,
            "filters": {
                "n_target_super_domains": 0,
                "target_fraction": BENCHMARK_TARGET_FRACTION,
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

    # Domain IDs use "|", for example:
    # bci_iv_2a|A01
    #
    # Therefore we must not split on "|".
    # Domain lists in model_results.csv are separated by ";".
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


def _all_domains_from_column(
    values,
):

    domains = set()

    for value in values:

        domains.update(
            _parse_domain_list(
                value
            )
        )

    return domains


# ============================================================
# Helpers: table assertions
# ============================================================

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
    setting_column,
):

    required_columns = {
        setting_column,
        "Regime",
        "Method",
        "Runs",

        "Source BA Mean",
        "Source BA Std",

        "Target-Test BA Mean",
        "Target-Test BA Std",

        "Gap Mean",
        "Gap Std",

        "Macro-F1 Mean",
        "Macro-F1 Std",

        "AUC Mean",
        "AUC Std",

        "Discrepancy Mean",
        "Discrepancy Std",
    }

    assert required_columns.issubset(
        set(
            table.columns
        )
    )


def _assert_display_labels(
    table,
):

    if table.empty:
        return

    assert set(
        table["Regime"]
    ) == {
        "Classical",
    }

    assert set(
        table["Method"]
    ) == {
        "Logistic Regression",
    }


def _assert_numeric_summary_columns(
    table,
):

    if table.empty:
        return

    numeric_columns = [
        "Runs",

        "Source BA Mean",
        "Source BA Std",

        "Target-Test BA Mean",
        "Target-Test BA Std",

        "Gap Mean",
        "Gap Std",

        "Macro-F1 Mean",
        "Macro-F1 Std",

        "AUC Mean",
        "AUC Std",

        "Discrepancy Mean",
        "Discrepancy Std",
    ]

    for column in numeric_columns:

        converted = pd.to_numeric(
            table[column],
            errors="coerce",
        )

        assert len(
            converted
        ) == len(
            table
        )


def _assert_no_formatted_mean_std_strings(
    table,
):

    for column in table.columns:

        if table[column].dtype == object:

            text = table[column].astype(
                str
            )

            assert not text.str.contains(
                "±",
                regex=False,
            ).any()


# ============================================================
# Helpers: model-result filtering
# ============================================================

def _target_test_rows(
    model_table,
    scenario,
    target_fraction,
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
            target_fraction,
        )
    ].copy()

    rows["Dataset"] = (
        rows["target_domains"]
        .apply(
            _dataset_from_domains
        )
    )

    rows["Held-out Dataset"] = (
        rows["Dataset"]
    )

    rows["source_dataset_count"] = (
        rows["source_domains"]
        .apply(
            _dataset_count_from_domains
        )
    )

    return rows


def _keep_max_source_domains(
    rows,
    setting_column,
):

    if rows.empty:
        return rows

    max_values = (
        rows
        .groupby(
            setting_column
        )[
            "n_source_domains"
        ]
        .transform(
            "max"
        )
    )

    return rows[
        rows["n_source_domains"]
        == max_values
    ].copy()


def _keep_max_source_datasets(
    rows,
    setting_column,
):

    if rows.empty:
        return rows

    max_values = (
        rows
        .groupby(
            setting_column
        )[
            "source_dataset_count"
        ]
        .transform(
            "max"
        )
    )

    return rows[
        rows["source_dataset_count"]
        == max_values
    ].copy()


def _assert_runs_match_raw_rows(
    table,
    raw_rows,
    setting_column,
):

    counts = (
        raw_rows
        .groupby(
            setting_column
        )
        .size()
    )

    assert set(
        table[
            setting_column
        ]
    ) == set(
        counts.index
    )

    for _, row in table.iterrows():

        setting = row[
            setting_column
        ]

        assert int(
            row["Runs"]
        ) == int(
            counts.loc[
                setting
            ]
        )


# ============================================================
# Helpers: protocol assertions
# ============================================================

def _assert_intra_subject_runs_across_subjects(
    model_table,
    intra_table,
):

    rows = _target_test_rows(
        model_table=model_table,
        scenario="intra_subject",
        target_fraction=INTRA_TRAIN_FRACTION,
    )

    assert not rows.empty

    _assert_runs_match_raw_rows(
        table=intra_table,
        raw_rows=rows,
        setting_column="Dataset",
    )

    # Intra-subject has no source-target domain discrepancy.
    assert (
        intra_table[
            "Discrepancy Mean"
        ]
        .isna()
        .all()
    )

    assert (
        intra_table[
            "Discrepancy Std"
        ]
        .isna()
        .all()
    )


def _assert_cross_session_uses_max_source_sessions(
    model_table,
    cross_session_table,
):

    rows = _target_test_rows(
        model_table=model_table,
        scenario="cross_session",
        target_fraction=BENCHMARK_TARGET_FRACTION,
    )

    if rows.empty:

        assert cross_session_table.empty
        return

    full_source_rows = _keep_max_source_domains(
        rows,
        setting_column="Dataset",
    )

    assert not full_source_rows.empty

    _assert_runs_match_raw_rows(
        table=cross_session_table,
        raw_rows=full_source_rows,
        setting_column="Dataset",
    )


def _assert_cross_subject_is_all_except_one(
    model_table,
    cross_subject_table,
):

    rows = _target_test_rows(
        model_table=model_table,
        scenario="cross_subject",
        target_fraction=BENCHMARK_TARGET_FRACTION,
    )

    assert not rows.empty

    # The raw results should contain more than one source-size
    # protocol, otherwise the table filtering is not being tested.
    assert (
        rows[
            "n_source_domains"
        ]
        .nunique()
        >= 2
    )

    full_source_rows = _keep_max_source_domains(
        rows,
        setting_column="Dataset",
    )

    assert not full_source_rows.empty

    # Count how many target subjects/domains were loaded for each
    # dataset based on the held-out target domains that appear in
    # the cross-subject experiment.
    loaded_target_domains = (
        rows
        .groupby(
            "Dataset"
        )[
            "target_domains"
        ]
        .apply(
            _all_domains_from_column
        )
    )

    for dataset, group in full_source_rows.groupby(
        "Dataset"
    ):

        n_loaded_subjects = len(
            loaded_target_domains.loc[
                dataset
            ]
        )

        expected_source_domains = (
            n_loaded_subjects
            - 1
        )

        assert expected_source_domains > 0

        assert (
            group[
                "n_source_domains"
            ]
            .eq(
                expected_source_domains
            )
            .all()
        )

    _assert_runs_match_raw_rows(
        table=cross_subject_table,
        raw_rows=full_source_rows,
        setting_column="Dataset",
    )


def _assert_cross_dataset_is_all_except_one_dataset(
    model_table,
    cross_dataset_table,
):

    rows = _target_test_rows(
        model_table=model_table,
        scenario="cross_dataset",
        target_fraction=BENCHMARK_TARGET_FRACTION,
    )

    assert not rows.empty

    assert (
        rows[
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

    # The raw results should contain one-source-dataset and
    # all-source-dataset protocols.
    assert (
        rows[
            "source_dataset_count"
        ]
        .nunique()
        >= 2
    )

    full_source_rows = _keep_max_source_datasets(
        rows,
        setting_column="Held-out Dataset",
    )

    assert not full_source_rows.empty

    loaded_datasets = set(
        rows[
            "Held-out Dataset"
        ]
    )

    n_loaded_datasets = len(
        loaded_datasets
    )

    expected_source_datasets = (
        n_loaded_datasets
        - 1
    )

    assert expected_source_datasets > 0

    assert (
        full_source_rows[
            "source_dataset_count"
        ]
        .eq(
            expected_source_datasets
        )
        .all()
    )

    _assert_runs_match_raw_rows(
        table=cross_dataset_table,
        raw_rows=full_source_rows,
        setting_column="Held-out Dataset",
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
# Test
# ============================================================

def test_benchmark_tables_with_full_source_protocols():

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
    # Preprocess full-subject datasets
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
            "name": "benchmark_table_combined_full",
        },
    )

    assert exists(
        combined_view.path
    )

    # --------------------------------------------------------
    # Generate all scenario splits
    # --------------------------------------------------------

    scenario_items = []

    for dataset, view in views.items():

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
        BENCHMARK_TABLE_PARAMS,
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

    # --------------------------------------------------------
    # Schema checks
    # --------------------------------------------------------

    table_specs = [
        (
            intra_table,
            "Dataset",
        ),
        (
            cross_session_table,
            "Dataset",
        ),
        (
            cross_subject_table,
            "Dataset",
        ),
        (
            cross_dataset_table,
            "Held-out Dataset",
        ),
    ]

    for table, setting_column in table_specs:

        if not table.empty:

            _assert_table_columns(
                table,
                setting_column,
            )

            _assert_display_labels(
                table
            )

            _assert_numeric_summary_columns(
                table
            )

            _assert_no_formatted_mean_std_strings(
                table
            )

    assert not intra_table.empty
    assert not cross_subject_table.empty
    assert not cross_dataset_table.empty

    # --------------------------------------------------------
    # Raw-result protocol checks
    # --------------------------------------------------------

    model_table = pd.read_csv(
        model_results_artifact.path
    )

    assert not model_table.empty

    _assert_intra_subject_runs_across_subjects(
        model_table=model_table,
        intra_table=intra_table,
    )

    _assert_cross_session_uses_max_source_sessions(
        model_table=model_table,
        cross_session_table=cross_session_table,
    )

    _assert_cross_subject_is_all_except_one(
        model_table=model_table,
        cross_subject_table=cross_subject_table,
    )

    _assert_cross_dataset_is_all_except_one_dataset(
        model_table=model_table,
        cross_dataset_table=cross_dataset_table,
    )

    # --------------------------------------------------------
    # Final sanity checks
    # --------------------------------------------------------

    assert (
        cross_subject_table[
            "Runs"
        ]
        .min()
        > 1
    )

    assert (
        cross_dataset_table[
            "Runs"
        ]
        .min()
        > 1
    )

    assert (
        cross_subject_table[
            "Target-Test BA Std"
        ]
        .notna()
        .any()
    )

    assert (
        cross_dataset_table[
            "Target-Test BA Std"
        ]
        .notna()
        .any()
    )