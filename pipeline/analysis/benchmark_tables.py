# pipeline/analysis/benchmark_tables.py

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

from models.analysis_artifact import AnalysisArtifact
from utils.status import is_done, make_manifest, make_signature
from utils.storage import exists, save_manifest


OUTPUT_ROOT = Path("outputs/analysis/benchmark_tables")

DEFAULT_TABLE_CONFIGS = [
    {
        "name": "intra_subject", "scenario": "intra_subject",
        "setting_column": "Dataset", "output_name": "intra_subject_table.csv",
        "include_discrepancy": False, "group_by_configs": True, "filters": {},
    },
    {
        "name": "cross_session", "scenario": "cross_session",
        "setting_column": "Dataset", "output_name": "cross_session_table.csv",
        "include_discrepancy": True, "group_by_configs": True,
        "filters": {"n_target_super_domains": 0, "use_max_source_domains": True},
    },
    {
        "name": "cross_subject", "scenario": "cross_subject",
        "setting_column": "Dataset", "output_name": "cross_subject_table.csv",
        "include_discrepancy": True, "group_by_configs": True,
        "filters": {"n_target_super_domains": 0, "use_max_source_domains": True},
    },
    {
        "name": "cross_dataset", "scenario": "cross_dataset",
        "setting_column": "Held-out Dataset", "output_name": "cross_dataset_table.csv",
        "include_discrepancy": True, "group_by_configs": True,
        "filters": {"n_target_super_domains": 0, "use_max_source_super_domains": True},
    },
]


def run_benchmark_tables(model_results_artifact, domain_results_artifact=None, params=None):
    params = _with_default_params(params)
    model_path = _artifact_path(model_results_artifact)
    domain_path = None if domain_results_artifact is None else _artifact_path(domain_results_artifact)

    effective_params = {
        "analysis": "benchmark_tables",
        "params": params,
        "model_results_path": str(model_path),
        "domain_results_path": None if domain_path is None else str(domain_path),
    }
    signature = make_signature(effective_params)
    output_dir = _output_dir(params, signature)
    manifest_path = output_dir / "manifest.json"
    table_paths = {
        config["name"]: output_dir / config["output_name"]
        for config in params["tables"]
    }

    if all(exists(path) for path in table_paths.values()) and is_done(manifest_path, effective_params):
        return AnalysisArtifact(
            name="benchmark_tables", output_dir=str(output_dir),
            tables={name: str(path) for name, path in table_paths.items()},
            figures={}, manifest_path=str(manifest_path), signature=signature,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    model_results = pd.read_csv(model_path)
    domain_results = None if domain_path is None else pd.read_csv(domain_path)

    display_lookup = _build_display_lookup(params["method_display"])
    model_summary = _prepare_model_summary(model_results, display_lookup)
    discrepancy = _prepare_discrepancy(domain_results, params["discrepancy_metric"])

    written_tables = {}
    for config in params["tables"]:
        table = _build_table(model_summary, discrepancy, config)
        path = table_paths[config["name"]]
        table.to_csv(path, index=False)
        written_tables[config["name"]] = str(path)

    manifest = make_manifest(status="done", params=effective_params)
    manifest["output"] = {"output_dir": str(output_dir), "tables": written_tables}
    save_manifest(manifest, manifest_path)

    return AnalysisArtifact(
        name="benchmark_tables", output_dir=str(output_dir),
        tables=written_tables, figures={},
        manifest_path=str(manifest_path), signature=signature,
    )


def _with_default_params(params):
    defaults = {
        "tables": DEFAULT_TABLE_CONFIGS,
        "discrepancy_metric": "mmd",
        "method_display": [],
    }
    return defaults if params is None else {**defaults, **params}


def _artifact_path(artifact):
    if isinstance(artifact, (str, Path)):
        return Path(artifact)
    if hasattr(artifact, "path"):
        return Path(artifact.path)
    raise TypeError("Expected path-like object or artifact with .path.")


def _slug(value):
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()


def _output_dir(params, signature):
    names = [config["name"] for config in params["tables"]]
    default_names = [config["name"] for config in DEFAULT_TABLE_CONFIGS]
    tables_label = "all_scenarios" if names == default_names else "-".join(map(_slug, names))
    metric = _slug(params["discrepancy_metric"])
    return OUTPUT_ROOT / f"{tables_label}__{metric}__{signature[:12]}"


def _build_display_lookup(method_display):
    return {
        (str(item["learning_method"]), str(item["model_name"])): {
            "regime": item.get("regime", item["learning_method"]),
            "method": item.get("method", item["model_name"]),
        }
        for item in method_display
    }


def _prepare_model_summary(results, display_lookup):
    required = [
        "split_id", "scenario", "group", "n_source_domains",
        "n_target_super_domains", "target_fraction", "source_domains",
        "target_domains", "feature_selection_signature", "learning_method",
        "model_name", "model_signature", "evaluation_group", "partition",
        "balanced_accuracy", "macro_f1", "auc",
    ]
    _require_columns(results, required)

    keys = [
        "split_id", "scenario", "group", "n_source_domains",
        "n_target_super_domains", "target_fraction", "source_domains",
        "target_domains", "feature_selection_signature", "learning_method",
        "model_name", "model_signature",
    ]

    for column in [
        "preprocessing_config_label",
        "feature_selection_config_label",
        "preprocessing_signature",
    ]:
        if column in results.columns:
            keys.append(column)

    if "training_seed" in results.columns:
        keys.append("training_seed")

    source_train = results[
        (results["evaluation_group"] == "source") &
        (results["partition"] == "train")
    ].copy()

    intra_train = results[
        (results["scenario"] == "intra_subject") &
        (results["evaluation_group"] == "target_elementary_domain") &
        (results["partition"] == "train")
    ].copy()

    train = _collapse_partition_metrics(
        pd.concat([source_train, intra_train], ignore_index=True), keys,
        {
            "balanced_accuracy": "source_ba",
            "macro_f1": "source_macro_f1",
            "auc": "source_auc",
        },
    )

    calibration = _collapse_partition_metrics(
        results[
            (results["evaluation_group"] == "target_elementary_domain") &
            (results["partition"] == "calibration")
        ].copy(),
        keys,
        {
            "balanced_accuracy": "target_calibration_ba",
            "macro_f1": "target_calibration_macro_f1",
            "auc": "target_calibration_auc",
        },
    )

    test = _collapse_partition_metrics(
        results[
            (results["evaluation_group"] == "target_elementary_domain") &
            (results["partition"] == "test")
        ].copy(),
        keys,
        {
            "balanced_accuracy": "target_test_ba",
            "macro_f1": "target_test_macro_f1",
            "auc": "target_test_auc",
        },
    )

    base = [df[keys] for df in [train, calibration, test] if not df.empty]
    if not base:
        return pd.DataFrame()

    summary = pd.concat(base, ignore_index=True).drop_duplicates(subset=keys).reset_index(drop=True)
    for dataframe in [train, calibration, test]:
        summary = summary.merge(dataframe, on=keys, how="left")

    summary["calibration_gap"] = summary["source_ba"] - summary["target_calibration_ba"]
    summary["test_gap"] = summary["source_ba"] - summary["target_test_ba"]
    summary["Dataset"] = summary["target_domains"].apply(_dataset_from_domains)
    summary["Held-out Dataset"] = summary["target_domains"].apply(_dataset_from_domains)
    summary["source_super_domain_count"] = summary["source_domains"].apply(_dataset_count_from_domains)

    if "preprocessing_config_label" in summary:
        summary["Preprocessing"] = summary["preprocessing_config_label"]
    if "feature_selection_config_label" in summary:
        summary["Feature Selection"] = summary["feature_selection_config_label"]

    labels = summary.apply(
        lambda row: _display_labels(row, display_lookup),
        axis=1, result_type="expand",
    )
    summary["Regime"], summary["Method"] = labels["Regime"], labels["Method"]
    return summary


def _collapse_partition_metrics(dataframe, keys, mapping):
    columns = keys + list(mapping.values())
    if dataframe.empty:
        return pd.DataFrame(columns=columns)

    metrics = list(mapping)
    return (
        dataframe[keys + metrics]
        .groupby(keys, dropna=False, as_index=False)[metrics]
        .mean()
        .rename(columns=mapping)
    )


def _display_labels(row, lookup):
    key = (str(row["learning_method"]), str(row["model_name"]))
    item = lookup.get(key)
    return {
        "Regime": key[0] if item is None else item["regime"],
        "Method": key[1] if item is None else item["method"],
    }


def _prepare_discrepancy(results, metric):
    if results is None:
        return pd.DataFrame(columns=["split_id", "discrepancy"])

    _require_columns(results, ["split_id", "comparison", "representation", "metric", "value"])

    selected = results[
        (results["comparison"] == "source_to_target_elementary") &
        (results["representation"] == "marginal") &
        (results["metric"] == metric)
    ]

    if selected.empty:
        return pd.DataFrame(columns=["split_id", "discrepancy"])

    return (
        selected.groupby("split_id", as_index=False)["value"]
        .mean()
        .rename(columns={"value": "discrepancy"})
    )


def _build_table(model_summary, discrepancy, config):
    scenario, setting = config["scenario"], config["setting_column"]

    if model_summary.empty:
        return _empty_table(setting)

    df = model_summary[model_summary["scenario"] == scenario].copy()
    df = _apply_filters(df, config.get("filters", {}), setting)

    if df.empty:
        return _empty_table(setting)

    if config.get("include_discrepancy", True):
        df = df.merge(discrepancy, on="split_id", how="left")
    else:
        df["discrepancy"] = np.nan

    group_columns = [setting]
    if config.get("group_by_configs", True):
        group_columns += [
            column for column in ["Preprocessing", "Feature Selection"]
            if column in df.columns
        ]
    group_columns += ["Regime", "Method"]

    rows = []
    for values, group in df.groupby(group_columns, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        row = dict(zip(group_columns, values))

        row.update({
            "Runs": len(group),
            "Calibration Runs": int(group["target_calibration_ba"].notna().sum()),
            "Test Runs": int(group["target_test_ba"].notna().sum()),
        })

        metrics = {
            "Source BA": "source_ba",
            "Source Macro-F1": "source_macro_f1",
            "Source AUC": "source_auc",
            "Target-Calibration BA": "target_calibration_ba",
            "Target-Calibration Macro-F1": "target_calibration_macro_f1",
            "Target-Calibration AUC": "target_calibration_auc",
            "Calibration Gap": "calibration_gap",
            "Target-Test BA": "target_test_ba",
            "Gap": "test_gap",
            "Macro-F1": "target_test_macro_f1",
            "AUC": "target_test_auc",
            "Discrepancy": "discrepancy",
        }

        for name, column in metrics.items():
            _add_metric_columns(row, name, group[column])

        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def _add_metric_columns(row, name, values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    row[f"{name} Mean"] = np.nan if values.empty else float(values.mean())
    row[f"{name} Std"] = np.nan if len(values) <= 1 else float(values.std(ddof=1))


def _apply_filters(df, filters, setting_column):
    result = df.copy()
    special = {"use_max_source_domains", "use_max_source_super_domains"}

    for column, value in filters.items():
        if column not in special and column in result.columns:
            result = _filter_column(result, column, value)

    if filters.get("use_max_source_domains") and not result.empty:
        max_values = result.groupby(setting_column)["n_source_domains"].transform("max")
        result = result[result["n_source_domains"] == max_values]

    if filters.get("use_max_source_super_domains") and not result.empty:
        max_values = result.groupby(setting_column)["source_super_domain_count"].transform("max")
        result = result[result["source_super_domain_count"] == max_values]

    return result


def _filter_column(dataframe, column, value):
    if value is None:
        return dataframe
    if isinstance(value, (list, tuple, set)):
        return dataframe[dataframe[column].isin(value)]
    if pd.api.types.is_numeric_dtype(dataframe[column]) and isinstance(value, (int, float)):
        return dataframe[np.isclose(pd.to_numeric(dataframe[column], errors="coerce"), value)]
    return dataframe[dataframe[column] == value]


def _empty_table(setting_column):
    metrics = [
        "Source BA", "Source Macro-F1", "Source AUC",
        "Target-Calibration BA", "Target-Calibration Macro-F1",
        "Target-Calibration AUC", "Calibration Gap",
        "Target-Test BA", "Gap", "Macro-F1", "AUC", "Discrepancy",
    ]
    columns = [
        setting_column, "Preprocessing", "Feature Selection", "Regime", "Method",
        "Runs", "Calibration Runs", "Test Runs",
    ]
    columns += [f"{metric} {stat}" for metric in metrics for stat in ["Mean", "Std"]]
    return pd.DataFrame(columns=columns)


def _parse_domain_list(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]

    text = str(value).strip()
    if text in {"", "[]", "None", "nan"}:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(item) for item in parsed]
    except Exception:
        pass

    for separator in [";", ","]:
        if separator in text:
            return [item.strip() for item in text.split(separator) if item.strip()]

    return [text]


def _dataset_from_domains(value):
    return "+".join(sorted({
        domain.split("|", 1)[0]
        for domain in _parse_domain_list(value)
    }))


def _dataset_count_from_domains(value):
    return len({
        domain.split("|", 1)[0]
        for domain in _parse_domain_list(value)
    })


def _require_columns(dataframe, columns):
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))