# pipeline/analysis/benchmark_tables.py

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from models.analysis_artifact import AnalysisArtifact
from utils.storage import exists, save_manifest
from utils.status import is_done, make_manifest, make_signature


OUTPUT_ROOT = Path("outputs/analysis/benchmark_tables")


DEFAULT_TABLE_CONFIGS = [
    {
        "name": "intra_subject",
        "scenario": "intra_subject",
        "setting_column": "Dataset",
        "output_name": "intra_subject_table.csv",
        "include_discrepancy": False,
        "group_by_configs": True,
        "filters": {},
    },
    {
        "name": "cross_session",
        "scenario": "cross_session",
        "setting_column": "Dataset",
        "output_name": "cross_session_table.csv",
        "include_discrepancy": True,
        "group_by_configs": True,
        "filters": {
            "n_target_super_domains": 0,
            "use_max_source_domains": True,
        },
    },
    {
        "name": "cross_subject",
        "scenario": "cross_subject",
        "setting_column": "Dataset",
        "output_name": "cross_subject_table.csv",
        "include_discrepancy": True,
        "group_by_configs": True,
        "filters": {
            "n_target_super_domains": 0,
            "use_max_source_domains": True,
        },
    },
    {
        "name": "cross_dataset",
        "scenario": "cross_dataset",
        "setting_column": "Held-out Dataset",
        "output_name": "cross_dataset_table.csv",
        "include_discrepancy": True,
        "group_by_configs": True,
        "filters": {
            "n_target_super_domains": 0,
            "use_max_source_super_domains": True,
        },
    },
]


# ============================================================
# Public
# ============================================================

def run_benchmark_tables(model_results_artifact, domain_results_artifact=None, params=None):
    params = _with_default_params(params)

    model_path = _artifact_path(model_results_artifact)
    domain_path = (
        None if domain_results_artifact is None
        else _artifact_path(domain_results_artifact)
    )

    effective_params = {
        "analysis": "benchmark_tables",
        "params": params,
        "model_results_path": str(model_path),
        "domain_results_path": None if domain_path is None else str(domain_path),
    }

    signature = make_signature(effective_params)
    output_dir = OUTPUT_ROOT / signature[:12]
    manifest_path = output_dir / "manifest.json"

    table_paths = {
        config["name"]: output_dir / config["output_name"]
        for config in params["tables"]
    }

    if (
        all(exists(path) for path in table_paths.values())
        and is_done(manifest_path, effective_params)
    ):
        return AnalysisArtifact(
            name="benchmark_tables",
            output_dir=str(output_dir),
            tables={name: str(path) for name, path in table_paths.items()},
            figures={},
            manifest_path=str(manifest_path),
            signature=signature,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    model_results = pd.read_csv(model_path)
    domain_results = None if domain_path is None else pd.read_csv(domain_path)

    display_lookup = _build_display_lookup(params["method_display"])
    model_summary = _prepare_model_summary(model_results, display_lookup)
    discrepancy = _prepare_discrepancy(
        domain_results,
        params["discrepancy_metric"],
    )

    written_tables = {}

    for config in params["tables"]:
        table = _build_table(model_summary, discrepancy, config)
        path = table_paths[config["name"]]
        table.to_csv(path, index=False)
        written_tables[config["name"]] = str(path)

    manifest = make_manifest(status="done", params=effective_params)
    manifest["output"] = {
        "output_dir": str(output_dir),
        "tables": written_tables,
    }
    save_manifest(manifest, manifest_path)

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

def _with_default_params(params):
    defaults = {
        "tables": DEFAULT_TABLE_CONFIGS,
        "discrepancy_metric": "mmd",
        "method_display": [],
    }

    if params is None:
        return defaults

    merged = defaults.copy()
    merged.update(params)
    return merged


def _artifact_path(artifact):
    if isinstance(artifact, (str, Path)):
        return Path(artifact)

    if hasattr(artifact, "path"):
        return Path(artifact.path)

    raise TypeError("Expected path-like object or artifact with .path.")


def _build_display_lookup(method_display):
    lookup = {}

    for item in method_display:
        key = (str(item["learning_method"]), str(item["model_name"]))
        lookup[key] = {
            "regime": item.get("regime", item["learning_method"]),
            "method": item.get("method", item["model_name"]),
        }

    return lookup


# ============================================================
# Model result preparation
# ============================================================

def _prepare_model_summary(results, display_lookup):
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

    _require_columns(results, required_columns)

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

    # Preserve preprocessing / FS configurations if available.
    for column in [
        "preprocessing_config_label",
        "feature_selection_config_label",
        "preprocessing_signature",
    ]:
        if column in results.columns:
            key_columns.append(column)

    if "training_seed" in results.columns:
        key_columns.append("training_seed")

    # --------------------------------------------------------
    # Source/train
    # --------------------------------------------------------

    source_train = results[
        (results["evaluation_group"] == "source")
        & (results["partition"] == "train")
    ].copy()

    intra_train = results[
        (results["scenario"] == "intra_subject")
        & (results["evaluation_group"] == "target_elementary_domain")
        & (results["partition"] == "train")
    ].copy()

    train_rows = pd.concat([source_train, intra_train], ignore_index=True)

    train_rows = _collapse_partition_metrics(
        train_rows,
        key_columns,
        {
            "balanced_accuracy": "source_ba",
            "macro_f1": "source_macro_f1",
            "auc": "source_auc",
        },
    )

    # --------------------------------------------------------
    # Target/calibration
    # --------------------------------------------------------

    target_calibration = results[
        (results["evaluation_group"] == "target_elementary_domain")
        & (results["partition"] == "calibration")
    ].copy()

    calibration_rows = _collapse_partition_metrics(
        target_calibration,
        key_columns,
        {
            "balanced_accuracy": "target_calibration_ba",
            "macro_f1": "target_calibration_macro_f1",
            "auc": "target_calibration_auc",
        },
    )

    # --------------------------------------------------------
    # Target/test
    # --------------------------------------------------------

    target_test = results[
        (results["evaluation_group"] == "target_elementary_domain")
        & (results["partition"] == "test")
    ].copy()

    test_rows = _collapse_partition_metrics(
        target_test,
        key_columns,
        {
            "balanced_accuracy": "target_test_ba",
            "macro_f1": "target_test_macro_f1",
            "auc": "target_test_auc",
        },
    )

    # --------------------------------------------------------
    # Experiment-level summary
    # --------------------------------------------------------

    base_parts = [
        df[key_columns]
        for df in [train_rows, calibration_rows, test_rows]
        if not df.empty
    ]

    if not base_parts:
        return pd.DataFrame()

    summary = (
        pd.concat(base_parts, ignore_index=True)
        .drop_duplicates(subset=key_columns)
        .reset_index(drop=True)
    )

    summary = summary.merge(train_rows, on=key_columns, how="left")
    summary = summary.merge(calibration_rows, on=key_columns, how="left")
    summary = summary.merge(test_rows, on=key_columns, how="left")

    summary["calibration_gap"] = (
        summary["source_ba"] - summary["target_calibration_ba"]
    )
    summary["test_gap"] = summary["source_ba"] - summary["target_test_ba"]

    # --------------------------------------------------------
    # Dataset labels
    # --------------------------------------------------------

    summary["Dataset"] = summary["target_domains"].apply(_dataset_from_domains)
    summary["Held-out Dataset"] = summary["target_domains"].apply(
        _dataset_from_domains
    )
    summary["source_super_domain_count"] = summary["source_domains"].apply(
        _dataset_count_from_domains
    )

    # --------------------------------------------------------
    # Configuration labels
    # --------------------------------------------------------

    if "preprocessing_config_label" in summary.columns:
        summary["Preprocessing"] = summary["preprocessing_config_label"]

    if "feature_selection_config_label" in summary.columns:
        summary["Feature Selection"] = summary["feature_selection_config_label"]

    # --------------------------------------------------------
    # Display labels
    # --------------------------------------------------------

    labels = summary.apply(
        lambda row: _display_labels(row, display_lookup),
        axis=1,
        result_type="expand",
    )

    summary["Regime"] = labels["Regime"]
    summary["Method"] = labels["Method"]

    return summary


def _collapse_partition_metrics(dataframe, key_columns, metric_mapping):
    output_columns = key_columns + list(metric_mapping.values())

    if dataframe.empty:
        return pd.DataFrame(columns=output_columns)

    selected = dataframe[
        key_columns + list(metric_mapping.keys())
    ].copy()

    selected = (
        selected.groupby(key_columns, dropna=False, as_index=False)[
            list(metric_mapping.keys())
        ]
        .mean()
    )

    return selected.rename(columns=metric_mapping)


def _display_labels(row, display_lookup):
    key = (
        str(row["learning_method"]),
        str(row["model_name"]),
    )

    if key in display_lookup:
        return {
            "Regime": display_lookup[key]["regime"],
            "Method": display_lookup[key]["method"],
        }

    return {
        "Regime": key[0],
        "Method": key[1],
    }


# ============================================================
# Domain discrepancy
# ============================================================

def _prepare_discrepancy(results, metric):
    if results is None:
        return pd.DataFrame(columns=["split_id", "discrepancy"])

    required_columns = [
        "split_id",
        "comparison",
        "representation",
        "metric",
        "value",
    ]
    _require_columns(results, required_columns)

    selected = results[
        (results["comparison"] == "source_to_target_elementary")
        & (results["representation"] == "marginal")
        & (results["metric"] == metric)
    ].copy()

    if selected.empty:
        return pd.DataFrame(columns=["split_id", "discrepancy"])

    return (
        selected.groupby("split_id", as_index=False)["value"]
        .mean()
        .rename(columns={"value": "discrepancy"})
    )


# ============================================================
# Table construction
# ============================================================

def _build_table(model_summary, discrepancy, table_config):
    scenario = table_config["scenario"]
    setting_column = table_config["setting_column"]

    if model_summary.empty:
        return _empty_table(setting_column)

    df = model_summary[
        model_summary["scenario"] == scenario
    ].copy()

    df = _apply_filters(
        df,
        table_config.get("filters", {}),
        setting_column,
    )

    if df.empty:
        return _empty_table(setting_column)

    if table_config.get("include_discrepancy", True):
        df = df.merge(discrepancy, on="split_id", how="left")
    else:
        df["discrepancy"] = np.nan

    group_columns = [setting_column]

    if table_config.get("group_by_configs", True):
        for column in ["Preprocessing", "Feature Selection"]:
            if column in df.columns:
                group_columns.append(column)

    group_columns += ["Regime", "Method"]

    rows = []

    for group_values, group in df.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        row = dict(zip(group_columns, group_values))

        row["Runs"] = int(len(group))
        row["Calibration Runs"] = int(group["target_calibration_ba"].notna().sum())
        row["Test Runs"] = int(group["target_test_ba"].notna().sum())

        _add_metric_columns(row, "Source BA", group["source_ba"])
        _add_metric_columns(row, "Source Macro-F1", group["source_macro_f1"])
        _add_metric_columns(row, "Source AUC", group["source_auc"])

        _add_metric_columns(
            row,
            "Target-Calibration BA",
            group["target_calibration_ba"],
        )
        _add_metric_columns(
            row,
            "Target-Calibration Macro-F1",
            group["target_calibration_macro_f1"],
        )
        _add_metric_columns(
            row,
            "Target-Calibration AUC",
            group["target_calibration_auc"],
        )
        _add_metric_columns(row, "Calibration Gap", group["calibration_gap"])

        _add_metric_columns(row, "Target-Test BA", group["target_test_ba"])
        _add_metric_columns(row, "Gap", group["test_gap"])
        _add_metric_columns(row, "Macro-F1", group["target_test_macro_f1"])
        _add_metric_columns(row, "AUC", group["target_test_auc"])

        _add_metric_columns(row, "Discrepancy", group["discrepancy"])

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(group_columns)
        .reset_index(drop=True)
    )


def _add_metric_columns(row, name, values):
    values = pd.to_numeric(values, errors="coerce").dropna()

    mean_column = f"{name} Mean"
    std_column = f"{name} Std"

    if len(values) == 0:
        row[mean_column] = np.nan
        row[std_column] = np.nan
        return

    row[mean_column] = float(values.mean())
    row[std_column] = (
        np.nan if len(values) == 1
        else float(values.std(ddof=1))
    )


# ============================================================
# Filters
# ============================================================

def _apply_filters(df, filters, setting_column):
    result = df.copy()

    for column, value in filters.items():
        if column in [
            "use_max_source_domains",
            "use_max_source_super_domains",
        ]:
            continue

        if column in result.columns:
            result = _filter_column(result, column, value)

    if filters.get("use_max_source_domains", False):
        if result.empty:
            return result

        max_values = (
            result.groupby(setting_column)["n_source_domains"]
            .transform("max")
        )

        result = result[
            result["n_source_domains"] == max_values
        ]

    if filters.get("use_max_source_super_domains", False):
        if result.empty:
            return result

        max_values = (
            result.groupby(setting_column)["source_super_domain_count"]
            .transform("max")
        )

        result = result[
            result["source_super_domain_count"] == max_values
        ]

    return result


def _filter_column(dataframe, column, value):
    if value is None:
        return dataframe

    if isinstance(value, (list, tuple, set)):
        return dataframe[
            dataframe[column].isin(list(value))
        ]

    if (
        pd.api.types.is_numeric_dtype(dataframe[column])
        and isinstance(value, (int, float))
    ):
        return dataframe[
            np.isclose(
                pd.to_numeric(dataframe[column], errors="coerce"),
                value,
                equal_nan=False,
            )
        ]

    return dataframe[
        dataframe[column] == value
    ]


# ============================================================
# Empty table
# ============================================================

def _empty_table(setting_column):
    return pd.DataFrame(
        columns=[
            setting_column,
            "Preprocessing",
            "Feature Selection",
            "Regime",
            "Method",
            "Runs",
            "Calibration Runs",
            "Test Runs",
            "Source BA Mean",
            "Source BA Std",
            "Source Macro-F1 Mean",
            "Source Macro-F1 Std",
            "Source AUC Mean",
            "Source AUC Std",
            "Target-Calibration BA Mean",
            "Target-Calibration BA Std",
            "Target-Calibration Macro-F1 Mean",
            "Target-Calibration Macro-F1 Std",
            "Target-Calibration AUC Mean",
            "Target-Calibration AUC Std",
            "Calibration Gap Mean",
            "Calibration Gap Std",
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
    )


# ============================================================
# Domain parsing
# ============================================================

def _parse_domain_list(value):
    if value is None:
        return []

    if isinstance(value, float) and pd.isna(value):
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]

    text = str(value).strip()

    if text in ["", "[]", "None", "nan"]:
        return []

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, (list, tuple, set)):
            return [str(item) for item in parsed]

    except Exception:
        pass

    for separator in [";", ","]:
        if separator in text:
            return [
                item.strip()
                for item in text.split(separator)
                if item.strip()
            ]

    return [text]


def _dataset_from_domains(value):
    domains = _parse_domain_list(value)

    datasets = sorted({
        domain.split("|", 1)[0]
        for domain in domains
    })

    return "+".join(datasets)


def _dataset_count_from_domains(value):
    domains = _parse_domain_list(value)

    return len({
        domain.split("|", 1)[0]
        for domain in domains
    })


# ============================================================
# Validation
# ============================================================

def _require_columns(dataframe, columns):
    missing = [
        column for column in columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )