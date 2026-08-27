from pathlib import Path
import ast

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
    "outputs/analysis/benchmark_tables"
)


DEFAULT_TABLE_CONFIGS = [
    {
        "name": "intra_subject",
        "scenario": "intra_subject",
        "setting_column": "Dataset",
        "output_name": "intra_subject_table.csv",
        "include_discrepancy": False,
        "filters": {},
        "method_labels": {},
    },
    {
        "name": "cross_session",
        "scenario": "cross_session",
        "setting_column": "Dataset",
        "output_name": "cross_session_table.csv",
        "include_discrepancy": True,
        "filters": {
            "n_target_super_domains": 0,
            "use_max_source_domains": True,
        },
        "method_labels": {},
    },
    {
        "name": "cross_subject",
        "scenario": "cross_subject",
        "setting_column": "Dataset",
        "output_name": "cross_subject_table.csv",
        "include_discrepancy": True,
        "filters": {
            "n_target_super_domains": 0,
            "use_max_source_domains": True,
        },
        "method_labels": {},
    },
    {
        "name": "cross_dataset",
        "scenario": "cross_dataset",
        "setting_column": "Held-out Dataset",
        "output_name": "cross_dataset_table.csv",
        "include_discrepancy": True,
        "filters": {
            "n_target_super_domains": 0,
            "use_max_source_super_domains": True,
        },
        "method_labels": {},
    },
]


# ============================================================
# Public function
# ============================================================

def run_benchmark_tables(
    model_results_artifact,
    domain_results_artifact=None,
    params=None,
):
    """
    Create benchmark summary tables from model/domain result tables.

    This is analysis-only. It does not infer learning regimes.
    If a table should display regime/group labels, provide them
    in params["tables"][i]["method_labels"].
    """

    params = _default_params(
        params
    )

    model_path = _artifact_path(
        model_results_artifact
    )

    domain_path = (
        None
        if domain_results_artifact is None
        else _artifact_path(
            domain_results_artifact
        )
    )

    effective_params = {
        "analysis": "benchmark_tables",
        "params": params,
        "model_results_path": str(model_path),
        "domain_results_path": (
            None
            if domain_path is None
            else str(domain_path)
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
        table_config["name"]: (
            output_dir
            / table_config["output_name"]
        )
        for table_config in params["tables"]
    }

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
            name="benchmark_tables",
            output_dir=str(output_dir),
            tables={
                name: str(path)
                for name, path in table_paths.items()
            },
            figures={},
            manifest_path=str(manifest_path),
            signature=signature,
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_results = pd.read_csv(
        model_path
    )

    domain_results = (
        None
        if domain_path is None
        else pd.read_csv(
            domain_path
        )
    )

    model_summary = _prepare_model_summary(
        model_results
    )

    discrepancy = _prepare_discrepancy(
        domain_results,
        params["discrepancy_metric"],
    )

    written_tables = {}

    for table_config in params["tables"]:

        table = _build_table(
            model_summary,
            discrepancy,
            table_config,
            params["precision"],
        )

        path = table_paths[
            table_config["name"]
        ]

        table.to_csv(
            path,
            index=False,
        )

        written_tables[
            table_config["name"]
        ] = str(path)

    manifest = make_manifest(
        status="done",
        params=effective_params,
    )

    manifest["output"] = {
        "output_dir": str(output_dir),
        "tables": written_tables,
    }

    save_manifest(
        manifest,
        manifest_path,
    )

    return AnalysisArtifact(
        name="benchmark_tables",
        output_dir=str(output_dir),
        tables=written_tables,
        figures={},
        manifest_path=str(manifest_path),
        signature=signature,
    )


# ============================================================
# Parameters
# ============================================================

def _default_params(
    params,
):
    default = {
        "tables": DEFAULT_TABLE_CONFIGS,
        "discrepancy_metric": "mmd",
        "precision": 3,
    }

    if params is None:
        return default

    updated = default.copy()
    updated.update(
        params
    )

    return updated


def _artifact_path(
    artifact,
):
    return Path(
        getattr(
            artifact,
            "path",
            artifact,
        )
    )


# ============================================================
# Model result preparation
# ============================================================

def _prepare_model_summary(
    results,
):
    required_columns = [
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

    target_test = results[
        (
            results["evaluation_group"]
            == "target_elementary_domain"
        )
        & (
            results["partition"]
            == "test"
        )
    ].copy()

    target_test = target_test.rename(
        columns={
            "balanced_accuracy": "target_ba",
            "macro_f1": "target_macro_f1",
            "auc": "target_auc",
        }
    )

    source_train = results[
        (
            results["evaluation_group"]
            == "source"
        )
        & (
            results["partition"]
            == "train"
        )
    ].copy()

    intra_train = results[
        (
            results["scenario"]
            == "intra_subject"
        )
        & (
            results["evaluation_group"]
            == "target_elementary_domain"
        )
        & (
            results["partition"]
            == "train"
        )
    ].copy()

    train_rows = pd.concat(
        [
            source_train,
            intra_train,
        ],
        ignore_index=True,
    )

    train_rows = train_rows[
        key_columns
        + [
            "balanced_accuracy",
        ]
    ].rename(
        columns={
            "balanced_accuracy": "source_ba",
        }
    )

    summary = target_test.merge(
        train_rows,
        on=key_columns,
        how="left",
    )

    summary["gap"] = (
        summary["source_ba"]
        - summary["target_ba"]
    )

    summary["Dataset"] = (
        summary["target_domains"]
        .apply(
            _dataset_from_domains
        )
    )

    summary["Held-out Dataset"] = (
        summary["target_domains"]
        .apply(
            _dataset_from_domains
        )
    )

    summary["source_super_domain_count"] = (
        summary["source_domains"]
        .apply(
            _dataset_count_from_domains
        )
    )

    summary["Method / Model"] = (
        summary["learning_method"].astype(str)
        + " / "
        + summary["model_name"].astype(str)
    )

    return summary


# ============================================================
# Domain discrepancy preparation
# ============================================================

def _prepare_discrepancy(
    results,
    metric,
):
    if results is None:
        return pd.DataFrame(
            columns=[
                "split_id",
                "discrepancy",
            ]
        )

    required_columns = [
        "split_id",
        "comparison",
        "representation",
        "metric",
        "value",
    ]

    _require_columns(
        results,
        required_columns,
    )

    selected = results[
        (
            results["comparison"]
            == "source_to_target_elementary"
        )
        & (
            results["representation"]
            == "marginal"
        )
        & (
            results["metric"]
            == metric
        )
    ].copy()

    return (
        selected
        .groupby(
            "split_id",
            as_index=False,
        )["value"]
        .mean()
        .rename(
            columns={
                "value": "discrepancy",
            }
        )
    )


# ============================================================
# Table construction
# ============================================================

def _build_table(
    model_summary,
    discrepancy,
    table_config,
    precision,
):
    scenario = table_config[
        "scenario"
    ]

    setting_column = table_config[
        "setting_column"
    ]

    df = model_summary[
        model_summary["scenario"]
        == scenario
    ].copy()

    df = _apply_filters(
        df,
        table_config.get(
            "filters",
            {},
        ),
        setting_column,
    )

    if table_config.get(
        "include_discrepancy",
        True,
    ):
        df = df.merge(
            discrepancy,
            on="split_id",
            how="left",
        )
    else:
        df["discrepancy"] = np.nan

    df["Regime"] = df.apply(
        lambda row: _method_label(
            row,
            table_config.get(
                "method_labels",
                {},
            ),
        ),
        axis=1,
    )

    group_columns = [
        setting_column,
        "Regime",
        "Method / Model",
    ]

    rows = []

    for group_values, group in df.groupby(
        group_columns,
        dropna=False,
    ):

        if not isinstance(
            group_values,
            tuple,
        ):
            group_values = (
                group_values,
            )

        row = dict(
            zip(
                group_columns,
                group_values,
            )
        )

        row["Runs"] = int(
            group["target_ba"]
            .notna()
            .sum()
        )

        row["Source BA"] = _mean_std(
            group["source_ba"],
            precision,
        )

        row["Target-Test BA"] = _mean_std(
            group["target_ba"],
            precision,
        )

        row["Gap"] = _mean_std(
            group["gap"],
            precision,
        )

        row["Macro-F1"] = _mean_std(
            group["target_macro_f1"],
            precision,
        )

        row["AUC"] = _mean_std(
            group["target_auc"],
            precision,
        )

        if table_config.get(
            "include_discrepancy",
            True,
        ):
            row["Discrepancy"] = _mean_std(
                group["discrepancy"],
                precision,
            )
        else:
            row["Discrepancy"] = "—"

        rows.append(
            row
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            group_columns
        )
        .reset_index(
            drop=True
        )
    )


def _apply_filters(
    df,
    filters,
    setting_column,
):
    result = df.copy()

    for column, value in filters.items():

        if column in [
            "use_max_source_domains",
            "use_max_source_super_domains",
        ]:
            continue

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
                setting_column
            )[
                "n_source_domains"
            ]
            .transform(
                "max"
            )
        )

        result = result[
            result["n_source_domains"]
            == max_values
        ]

    if filters.get(
        "use_max_source_super_domains",
        False,
    ):

        max_values = (
            result
            .groupby(
                setting_column
            )[
                "source_super_domain_count"
            ]
            .transform(
                "max"
            )
        )

        result = result[
            result["source_super_domain_count"]
            == max_values
        ]

    return result


def _method_label(
    row,
    labels,
):
    key = (
        str(
            row["learning_method"]
        ),
        str(
            row["model_name"]
        ),
    )

    return labels.get(
        key,
        str(
            row["learning_method"]
        ),
    )


# ============================================================
# Formatting and parsing
# ============================================================

def _mean_std(
    values,
    precision,
):
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return "—"

    if len(values) == 1:
        return f"{values.iloc[0]:.{precision}f}"

    return (
        f"{values.mean():.{precision}f}"
        " ± "
        f"{values.std(ddof=1):.{precision}f}"
    )


def _parse_domain_list(
    value,
):
    text = str(
        value
    )

    try:
        parsed = ast.literal_eval(
            text
        )

        if isinstance(
            parsed,
            list,
        ):
            return [
                str(item)
                for item in parsed
            ]

    except Exception:
        pass

    return [
        item.strip()
        for item in text.split(",")
        if item.strip()
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

    return len(
        {
            domain.split(
                "|",
                1,
            )[0]
            for domain in domains
        }
    )


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