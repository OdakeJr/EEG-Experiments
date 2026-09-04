from pathlib import Path
import ast
import re

import numpy as np
import pandas as pd

from models.analysis_artifact import AnalysisArtifact

from utils.storage import (
    exists,
    save_manifest,
)

from utils.status import (
    is_done,
    make_manifest,
    make_signature,
)


OUTPUT_ROOT = Path(
    "outputs/analysis/method_ranking"
)


DEFAULT_SCENARIOS = [
    {
        "name": "intra_subject",
        "scenario": "intra_subject",
        "filters": {},
    },
    {
        "name": "cross_session",
        "scenario": "cross_session",
        "filters": {},
    },
    {
        "name": "cross_subject",
        "scenario": "cross_subject",
        "filters": {},
    },
    {
        "name": "cross_dataset",
        "scenario": "cross_dataset",
        "filters": {},
    },
]


DEFAULT_MATCHED_BLOCK_COLUMNS = [
    "split_id",
    "Dataset",
    "Target Domain",
    "Source Composition",
    "Target Fraction",
    "Preprocessing Config",
    "Feature Config",
    "Training Seed",
]


# ============================================================
# Public function
# ============================================================

def run_method_ranking(
    model_results_artifact,
    params=None,
):
    """
    Build scenario-wise method ranking tables.

    This analysis uses target-test results only:

        evaluation_group == target_elementary_domain
        partition == test

    It summarizes:
    - mean/median/std Target-Test BA
    - mean Macro-F1
    - mean AUC
    - average rank across matched blocks
    - best-rank frequency across matched blocks

    No p-values are computed here.
    """

    params = _with_default_params(
        params
    )

    model_path = _artifact_path(
        model_results_artifact
    )

    effective_params = {
        "analysis": "method_ranking",
        "params": params,
        "model_results_path": str(
            model_path
        ),
    }

    signature = make_signature(
        effective_params
    )

    output_dir = (
        OUTPUT_ROOT
        / signature[:12]
    )

    manifest_path = (
        output_dir
        / "manifest.json"
    )

    table_paths = {
        config["name"]: (
            output_dir
            / f"{config['name']}_method_ranking.csv"
        )
        for config in params["scenarios"]
    }

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    if (
        all(
            exists(path)
            for path in table_paths.values()
        )
        and is_done(
            manifest_path,
            effective_params,
        )
    ):
        return AnalysisArtifact(
            name="method_ranking",
            output_dir=str(
                output_dir
            ),
            tables={
                key: str(path)
                for key, path in table_paths.items()
            },
            figures={},
            manifest_path=str(
                manifest_path
            ),
            signature=signature,
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load and prepare points
    # --------------------------------------------------------

    model_results = pd.read_csv(
        model_path
    )

    display_lookup = _build_display_lookup(
        params[
            "method_display"
        ]
    )

    points = _prepare_points(
        results=model_results,
        display_lookup=display_lookup,
        feature_config_column=params[
            "feature_config_column"
        ],
        preprocessing_config_column=params[
            "preprocessing_config_column"
        ],
        seed_column=params[
            "seed_column"
        ],
    )

    # --------------------------------------------------------
    # Build one table per scenario
    # --------------------------------------------------------

    tables = {}

    for config in params[
        "scenarios"
    ]:

        scenario_name = config[
            "name"
        ]

        scenario = config[
            "scenario"
        ]

        scenario_points = points[
            points[
                "Scenario"
            ]
            == scenario
        ].copy()

        scenario_points = _apply_filters(
            scenario_points,
            config.get(
                "filters",
                {},
            ),
        )

        table = _build_scenario_ranking_table(
            points=scenario_points,
            metric_column=params[
                "metric_column"
            ],
            matched_block_columns=params[
                "matched_block_columns"
            ],
        )

        table_path = table_paths[
            scenario_name
        ]

        table.to_csv(
            table_path,
            index=False,
        )

        tables[
            scenario_name
        ] = str(
            table_path
        )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = make_manifest(
        status="done",
        params=effective_params,
    )

    manifest[
        "output"
    ] = {
        "output_dir": str(
            output_dir
        ),
        "tables": tables,
    }

    save_manifest(
        manifest,
        manifest_path,
    )

    return AnalysisArtifact(
        name="method_ranking",
        output_dir=str(
            output_dir
        ),
        tables=tables,
        figures={},
        manifest_path=str(
            manifest_path
        ),
        signature=signature,
    )


# ============================================================
# Params
# ============================================================

def _with_default_params(
    params,
):
    defaults = {
        "scenarios": DEFAULT_SCENARIOS,

        "metric_column": "Target-Test BA",

        "matched_block_columns": DEFAULT_MATCHED_BLOCK_COLUMNS,

        "feature_config_column": "feature_selection_config_label",

        "preprocessing_config_column": "preprocessing_config_label",

        "seed_column": "training_seed",

        "method_display": [],
    }

    if params is None:
        return defaults

    merged = defaults.copy()

    merged.update(
        params
    )

    return merged


def _artifact_path(
    artifact,
):
    if isinstance(
        artifact,
        (str, Path),
    ):
        return Path(
            artifact
        )

    if hasattr(
        artifact,
        "path",
    ):
        return Path(
            artifact.path
        )

    raise TypeError(
        "Expected path-like object or artifact with .path."
    )


def _build_display_lookup(
    method_display,
):
    lookup = {}

    for item in method_display:

        key = (
            str(
                item[
                    "learning_method"
                ]
            ),
            str(
                item[
                    "model_name"
                ]
            ),
        )

        lookup[
            key
        ] = {
            "regime": item.get(
                "regime",
                item[
                    "learning_method"
                ],
            ),
            "method": item.get(
                "method",
                item[
                    "model_name"
                ],
            ),
        }

    return lookup


# ============================================================
# Point preparation
# ============================================================

def _prepare_points(
    results,
    display_lookup,
    feature_config_column,
    preprocessing_config_column,
    seed_column,
):
    required_columns = [
        "split_id",
        "scenario",
        "n_source_domains",
        "n_target_super_domains",
        "target_fraction",
        "source_domains",
        "target_domains",
        "feature_selection_signature",
        "learning_method",
        "model_name",
        "model_signature",
        "evaluation_group",
        "partition",
        "balanced_accuracy",
        "macro_f1",
        "auc",
    ]

    _require_columns(
        results,
        required_columns,
    )

    points = results[
        (
            results[
                "evaluation_group"
            ]
            == "target_elementary_domain"
        )
        & (
            results[
                "partition"
            ]
            == "test"
        )
    ].copy()

    points[
        "Scenario"
    ] = points[
        "scenario"
    ].astype(
        str
    )

    points[
        "Dataset"
    ] = points[
        "target_domains"
    ].apply(
        _dataset_from_domains
    )

    points[
        "Target Domain"
    ] = points[
        "target_domains"
    ].apply(
        _canonical_domain_string
    )

    points[
        "Source Composition"
    ] = points[
        "source_domains"
    ].apply(
        _canonical_domain_string
    )

    points[
        "Target Fraction"
    ] = pd.to_numeric(
        points[
            "target_fraction"
        ],
        errors="coerce",
    )

    points[
        "Feature Config"
    ] = _build_feature_config(
        points,
        feature_config_column,
    )

    points[
        "Preprocessing Config"
    ] = _build_preprocessing_config(
        points,
        preprocessing_config_column,
    )

    points[
        "Training Seed"
    ] = _build_seed_config(
        points,
        seed_column,
    )

    points[
        "Target-Test BA"
    ] = pd.to_numeric(
        points[
            "balanced_accuracy"
        ],
        errors="coerce",
    )

    points[
        "Macro-F1"
    ] = pd.to_numeric(
        points[
            "macro_f1"
        ],
        errors="coerce",
    )

    points[
        "AUC"
    ] = pd.to_numeric(
        points[
            "auc"
        ],
        errors="coerce",
    )

    points[
        "Source Domain Count"
    ] = pd.to_numeric(
        points[
            "n_source_domains"
        ],
        errors="coerce",
    )

    points[
        "Source Super Domain Count"
    ] = points[
        "source_domains"
    ].apply(
        _dataset_count_from_domains
    )

    labels = points.apply(
        lambda row: _display_labels(
            row,
            display_lookup,
        ),
        axis=1,
        result_type="expand",
    )

    points[
        "Regime"
    ] = labels[
        "Regime"
    ]

    points[
        "Method"
    ] = labels[
        "Method"
    ]

    keep_columns = [
        "split_id",
        "Scenario",
        "Dataset",
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
        "n_target_super_domains",
        "target_fraction",
        "Source Domain Count",
        "Source Super Domain Count",
        "Target-Test BA",
        "Macro-F1",
        "AUC",
    ]

    optional_columns = [
        "feature_selection_config_label",
        "preprocessing_config_label",
        "preprocessing_signature",
    ]

    keep_columns.extend(
        [
            column
            for column in optional_columns
            if column in points.columns
        ]
    )

    return points[
        keep_columns
    ].copy()


def _build_feature_config(
    dataframe,
    feature_config_column,
):
    if feature_config_column in dataframe.columns:

        return (
            dataframe[
                feature_config_column
            ]
            .fillna(
                "no_feature_config"
            )
            .astype(
                str
            )
        )

    if "feature_selection_config_label" in dataframe.columns:

        return (
            dataframe[
                "feature_selection_config_label"
            ]
            .fillna(
                "no_feature_config"
            )
            .astype(
                str
            )
        )

    return (
        dataframe[
            "feature_selection_signature"
        ]
        .fillna(
            "no_feature_config"
        )
        .astype(
            str
        )
    )


def _build_preprocessing_config(
    dataframe,
    preprocessing_config_column,
):
    if preprocessing_config_column in dataframe.columns:

        return (
            dataframe[
                preprocessing_config_column
            ]
            .fillna(
                "no_preprocessing_config"
            )
            .astype(
                str
            )
        )

    if "preprocessing_config_label" in dataframe.columns:

        return (
            dataframe[
                "preprocessing_config_label"
            ]
            .fillna(
                "no_preprocessing_config"
            )
            .astype(
                str
            )
        )

    if "preprocessing_signature" in dataframe.columns:

        return (
            dataframe[
                "preprocessing_signature"
            ]
            .fillna(
                "no_preprocessing_config"
            )
            .astype(
                str
            )
        )

    return pd.Series(
        [
            "no_preprocessing_config"
        ]
        * len(
            dataframe
        ),
        index=dataframe.index,
    )


def _build_seed_config(
    dataframe,
    seed_column,
):
    if seed_column in dataframe.columns:

        return (
            dataframe[
                seed_column
            ]
            .fillna(
                "no_seed"
            )
            .astype(
                str
            )
        )

    return pd.Series(
        [
            "no_seed"
        ]
        * len(
            dataframe
        ),
        index=dataframe.index,
    )


def _display_labels(
    row,
    display_lookup,
):
    learning_method = str(
        row[
            "learning_method"
        ]
    )

    model_name = str(
        row[
            "model_name"
        ]
    )

    key = (
        learning_method,
        model_name,
    )

    if key in display_lookup:

        return {
            "Regime": display_lookup[
                key
            ][
                "regime"
            ],
            "Method": display_lookup[
                key
            ][
                "method"
            ],
        }

    return {
        "Regime": learning_method,
        "Method": model_name,
    }


# ============================================================
# Filtering
# ============================================================

def _apply_filters(
    dataframe,
    filters,
):
    result = dataframe.copy()

    for column, value in filters.items():

        if column not in result.columns:
            continue

        result = _filter_column(
            result,
            column,
            value,
        )

    return result


def _filter_column(
    dataframe,
    column,
    value,
):
    if value is None:
        return dataframe

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return dataframe[
            dataframe[
                column
            ].isin(
                list(
                    value
                )
            )
        ]

    if isinstance(
        value,
        (int, float),
    ):
        return dataframe[
            np.isclose(
                pd.to_numeric(
                    dataframe[
                        column
                    ],
                    errors="coerce",
                ),
                value,
                equal_nan=False,
            )
        ]

    return dataframe[
        dataframe[
            column
        ]
        == value
    ]


# ============================================================
# Ranking table
# ============================================================

def _build_scenario_ranking_table(
    points,
    metric_column,
    matched_block_columns,
):
    output_columns = [
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
    ]

    if points.empty:

        return pd.DataFrame(
            columns=output_columns
        )

    performance = _summarize_performance(
        points
    )

    ranks = _summarize_ranks(
        points=points,
        metric_column=metric_column,
        matched_block_columns=matched_block_columns,
    )

    table = performance.merge(
        ranks,
        on=[
            "Regime",
            "Method",
        ],
        how="left",
    )

    table = table[
        output_columns
    ]

    table = table.sort_values(
        [
            "Average Rank",
            "Mean BA",
            "Regime",
            "Method",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    return table


def _summarize_performance(
    points,
):
    grouped = points.groupby(
        [
            "Regime",
            "Method",
        ],
        dropna=False,
    )

    table = grouped.agg(
        Runs=(
            "Target-Test BA",
            "count",
        ),
        Mean_BA=(
            "Target-Test BA",
            "mean",
        ),
        Median_BA=(
            "Target-Test BA",
            "median",
        ),
        Std_BA=(
            "Target-Test BA",
            lambda x: x.std(
                ddof=1
            ),
        ),
        Mean_Macro_F1=(
            "Macro-F1",
            "mean",
        ),
        Mean_AUC=(
            "AUC",
            "mean",
        ),
    ).reset_index()

    table = table.rename(
        columns={
            "Mean_BA": "Mean BA",
            "Median_BA": "Median BA",
            "Std_BA": "Std BA",
            "Mean_Macro_F1": "Mean Macro-F1",
            "Mean_AUC": "Mean AUC",
        }
    )

    return table


def _summarize_ranks(
    points,
    metric_column,
    matched_block_columns,
):
    available_block_columns = [
        column
        for column in matched_block_columns
        if column in points.columns
    ]

    # Collapse possible duplicated rows per method inside
    # the same matched condition.
    block_method_values = (
        points
        .dropna(
            subset=[
                metric_column
            ]
        )
        .groupby(
            available_block_columns
            + [
                "Regime",
                "Method",
            ],
            as_index=False,
            dropna=False,
        )[
            metric_column
        ]
        .mean()
    )

    if block_method_values.empty:

        return pd.DataFrame(
            columns=[
                "Regime",
                "Method",
                "Average Rank",
                "Best Rank (%)",
            ]
        )

    rank_rows = []

    for _, block in block_method_values.groupby(
        available_block_columns,
        dropna=False,
    ):

        if block.empty:
            continue

        block = block.copy()

        block[
            "Rank"
        ] = block[
            metric_column
        ].rank(
            method="average",
            ascending=False,
        )

        max_value = block[
            metric_column
        ].max()

        block[
            "Is Best"
        ] = np.isclose(
            block[
                metric_column
            ],
            max_value,
            equal_nan=False,
        )

        rank_rows.append(
            block[
                [
                    "Regime",
                    "Method",
                    "Rank",
                    "Is Best",
                ]
            ]
        )

    if not rank_rows:

        return pd.DataFrame(
            columns=[
                "Regime",
                "Method",
                "Average Rank",
                "Best Rank (%)",
            ]
        )

    ranks = pd.concat(
        rank_rows,
        ignore_index=True,
    )

    summary = (
        ranks
        .groupby(
            [
                "Regime",
                "Method",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            Average_Rank=(
                "Rank",
                "mean",
            ),
            Best_Rank_Percent=(
                "Is Best",
                lambda x: 100.0
                * float(
                    np.mean(
                        x
                    )
                ),
            ),
        )
    )

    summary = summary.rename(
        columns={
            "Average_Rank": "Average Rank",
            "Best_Rank_Percent": "Best Rank (%)",
        }
    )

    return summary


# ============================================================
# Domain parsing
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
            str(
                item
            )
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
            (list, tuple, set),
        ):
            return [
                str(
                    item
                )
                for item in parsed
            ]

    except Exception:

        pass

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


def _canonical_domain_string(
    value,
):
    domains = _parse_domain_list(
        value
    )

    return ";".join(
        domains
    )


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
# Validation
# ============================================================

def _require_columns(
    dataframe,
    columns,
):
    missing = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                missing
            )
        )