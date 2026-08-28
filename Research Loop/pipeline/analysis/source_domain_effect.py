from pathlib import Path
import ast
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    "outputs/analysis/source_domain_effect"
)


DEFAULT_SCENARIO_CONFIGS = [
    {
        "name": "cross_session",
        "scenario": "cross_session",
        "setting_column": "Dataset",
        "source_axis_column": "n_source_domains",
        "source_axis_label": "Number of source sessions",
        "output_prefix": "cross_session",
        "filters": {
            "n_target_super_domains": 0,
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
        },
    },
]


# ============================================================
# Public function
# ============================================================

def run_source_domain_effect(
    model_results_artifact,
    params=None,
):
    """
    Analyze the natural effect of increasing the number of
    source domains.

    This is analysis-only. It assumes the model-results table
    already contains runs with different numbers of source
    sessions, subjects, or datasets.

    The output contains:
    - one summary CSV
    - one line plot per scenario x dataset/held-out dataset

    Confidence intervals are computed across repeated runs with
    the same scenario, dataset, method, and number of source
    domains.
    """

    params = _with_default_params(
        params
    )

    model_path = _artifact_path(
        model_results_artifact
    )

    effective_params = {
        "analysis": "source_domain_effect",
        "params": params,
        "model_results_path": str(model_path),
    }

    signature = make_signature(
        effective_params
    )

    output_dir = (
        OUTPUT_ROOT
        / signature[:12]
    )

    figures_dir = (
        output_dir
        / "figures"
    )

    summary_path = (
        output_dir
        / "source_domain_effect_summary.csv"
    )

    manifest_path = (
        output_dir
        / "manifest.json"
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    if (
        exists(summary_path)
        and is_done(
            manifest_path,
            effective_params,
        )
    ):

        manifest = pd.read_json(
            manifest_path,
            typ="series",
        )

        figures = (
            manifest.get(
                "output",
                {},
            )
            .get(
                "figures",
                {},
            )
            if isinstance(
                manifest.get(
                    "output",
                    {},
                ),
                dict,
            )
            else {}
        )

        return AnalysisArtifact(
            name="source_domain_effect",
            output_dir=str(output_dir),
            tables={
                "summary": str(summary_path),
            },
            figures=figures,
            manifest_path=str(manifest_path),
            signature=signature,
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load and prepare rows
    # --------------------------------------------------------

    model_results = pd.read_csv(
        model_path
    )

    display_lookup = _build_display_lookup(
        params["method_display"]
    )

    prepared = _prepare_model_summary(
        model_results,
        display_lookup,
    )

    # --------------------------------------------------------
    # Build summary table
    # --------------------------------------------------------

    scenario_summaries = []

    for scenario_config in params[
        "scenarios"
    ]:

        scenario_summary = _build_scenario_summary(
            prepared=prepared,
            scenario_config=scenario_config,
            ci_z=params["ci_z"],
        )

        if not scenario_summary.empty:

            scenario_summaries.append(
                scenario_summary
            )

    if scenario_summaries:

        summary = pd.concat(
            scenario_summaries,
            ignore_index=True,
        )

    else:

        summary = _empty_summary_table()

    summary.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Build plots
    # --------------------------------------------------------

    figures = _build_plots(
        summary=summary,
        scenario_configs=params["scenarios"],
        figures_dir=figures_dir,
        metric_prefix=params["plot_metric"],
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = make_manifest(
        status="done",
        params=effective_params,
    )

    manifest["output"] = {
        "output_dir": str(output_dir),
        "summary": str(summary_path),
        "figures": figures,
    }

    save_manifest(
        manifest,
        manifest_path,
    )

    return AnalysisArtifact(
        name="source_domain_effect",
        output_dir=str(output_dir),
        tables={
            "summary": str(summary_path),
        },
        figures=figures,
        manifest_path=str(manifest_path),
        signature=signature,
    )


# ============================================================
# Params
# ============================================================

def _with_default_params(
    params,
):
    defaults = {
        "scenarios": DEFAULT_SCENARIO_CONFIGS,

        # Main plotted metric.
        # Must match one of the prefixes produced below:
        # Source BA, Target-Test BA, Gap, Macro-F1, AUC
        "plot_metric": "Target-Test BA",

        # 95% normal-approximation confidence interval.
        "ci_z": 1.96,

        # Display-only labels.
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
                item["learning_method"]
            ),
            str(
                item["model_name"]
            ),
        )

        lookup[
            key
        ] = {
            "regime": item.get(
                "regime",
                item["learning_method"],
            ),
            "method": item.get(
                "method",
                item["model_name"],
            ),
        }

    return lookup


# ============================================================
# Model preparation
# ============================================================

def _prepare_model_summary(
    results,
    display_lookup,
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

    if "training_seed" in results.columns:
        key_columns.append(
            "training_seed"
        )

    # --------------------------------------------------------
    # Target-test performance
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Source/train performance
    # --------------------------------------------------------

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

    source_train = source_train[
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
        source_train,
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

    labels = summary.apply(
        lambda row: _display_labels(
            row,
            display_lookup,
        ),
        axis=1,
        result_type="expand",
    )

    summary["Regime"] = labels[
        "Regime"
    ]

    summary["Method"] = labels[
        "Method"
    ]

    return summary


def _display_labels(
    row,
    display_lookup,
):
    learning_method = str(
        row["learning_method"]
    )

    model_name = str(
        row["model_name"]
    )

    key = (
        learning_method,
        model_name,
    )

    if key in display_lookup:

        return {
            "Regime": display_lookup[
                key
            ]["regime"],
            "Method": display_lookup[
                key
            ]["method"],
        }

    return {
        "Regime": learning_method,
        "Method": model_name,
    }


# ============================================================
# Summary construction
# ============================================================

def _build_scenario_summary(
    prepared,
    scenario_config,
    ci_z,
):
    scenario = scenario_config[
        "scenario"
    ]

    setting_column = scenario_config[
        "setting_column"
    ]

    source_axis_column = scenario_config[
        "source_axis_column"
    ]

    source_axis_label = scenario_config[
        "source_axis_label"
    ]

    df = prepared[
        prepared["scenario"]
        == scenario
    ].copy()

    df = _apply_filters(
        df=df,
        filters=scenario_config.get(
            "filters",
            {},
        ),
    )

    if df.empty:
        return _empty_summary_table()

    df["Scenario"] = scenario
    df["Setting"] = df[
        setting_column
    ]

    df["Source Domain Count"] = pd.to_numeric(
        df[
            source_axis_column
        ],
        errors="coerce",
    )

    df["Source Domain Axis"] = (
        source_axis_label
    )

    group_columns = [
        "Scenario",
        "Setting",
        "Regime",
        "Method",
        "Source Domain Count",
        "Source Domain Axis",
    ]

    rows = []

    for group_values, group in df.groupby(
        group_columns,
        dropna=False,
    ):

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

        _add_stats(
            row=row,
            prefix="Source BA",
            values=group["source_ba"],
            ci_z=ci_z,
        )

        _add_stats(
            row=row,
            prefix="Target-Test BA",
            values=group["target_ba"],
            ci_z=ci_z,
        )

        _add_stats(
            row=row,
            prefix="Gap",
            values=group["gap"],
            ci_z=ci_z,
        )

        _add_stats(
            row=row,
            prefix="Macro-F1",
            values=group["target_macro_f1"],
            ci_z=ci_z,
        )

        _add_stats(
            row=row,
            prefix="AUC",
            values=group["target_auc"],
            ci_z=ci_z,
        )

        rows.append(
            row
        )

    if not rows:
        return _empty_summary_table()

    summary = pd.DataFrame(
        rows
    )

    return (
        summary
        .sort_values(
            [
                "Scenario",
                "Setting",
                "Regime",
                "Method",
                "Source Domain Count",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def _add_stats(
    row,
    prefix,
    values,
    ci_z,
):
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    n = len(
        values
    )

    row[
        f"{prefix} Mean"
    ] = np.nan

    row[
        f"{prefix} Std"
    ] = np.nan

    row[
        f"{prefix} SEM"
    ] = np.nan

    row[
        f"{prefix} CI95 Low"
    ] = np.nan

    row[
        f"{prefix} CI95 High"
    ] = np.nan

    if n == 0:
        return

    mean = float(
        values.mean()
    )

    row[
        f"{prefix} Mean"
    ] = mean

    if n == 1:
        return

    std = float(
        values.std(
            ddof=1
        )
    )

    sem = float(
        std
        / np.sqrt(
            n
        )
    )

    margin = (
        ci_z
        * sem
    )

    row[
        f"{prefix} Std"
    ] = std

    row[
        f"{prefix} SEM"
    ] = sem

    row[
        f"{prefix} CI95 Low"
    ] = mean - margin

    row[
        f"{prefix} CI95 High"
    ] = mean + margin


def _empty_summary_table():
    metric_prefixes = [
        "Source BA",
        "Target-Test BA",
        "Gap",
        "Macro-F1",
        "AUC",
    ]

    columns = [
        "Scenario",
        "Setting",
        "Regime",
        "Method",
        "Source Domain Count",
        "Source Domain Axis",
        "Runs",
    ]

    for prefix in metric_prefixes:

        columns.extend(
            [
                f"{prefix} Mean",
                f"{prefix} Std",
                f"{prefix} SEM",
                f"{prefix} CI95 Low",
                f"{prefix} CI95 High",
            ]
        )

    return pd.DataFrame(
        columns=columns
    )


# ============================================================
# Plotting
# ============================================================

def _build_plots(
    summary,
    scenario_configs,
    figures_dir,
    metric_prefix,
):
    figures = {}

    if summary.empty:
        return figures

    mean_column = (
        f"{metric_prefix} Mean"
    )

    low_column = (
        f"{metric_prefix} CI95 Low"
    )

    high_column = (
        f"{metric_prefix} CI95 High"
    )

    required_columns = [
        mean_column,
        low_column,
        high_column,
    ]

    _require_columns(
        summary,
        required_columns,
    )

    for scenario_config in scenario_configs:

        scenario = scenario_config[
            "scenario"
        ]

        scenario_summary = summary[
            summary["Scenario"]
            == scenario
        ].copy()

        if scenario_summary.empty:
            continue

        for setting, setting_summary in scenario_summary.groupby(
            "Setting"
        ):

            if setting_summary.empty:
                continue

            figure_key = _safe_name(
                f"{scenario}_{setting}_{metric_prefix}"
            )

            figure_path = (
                figures_dir
                / f"{figure_key}.png"
            )

            _plot_single_setting(
                data=setting_summary,
                scenario=scenario,
                setting=setting,
                metric_prefix=metric_prefix,
                mean_column=mean_column,
                low_column=low_column,
                high_column=high_column,
                path=figure_path,
            )

            figures[
                figure_key
            ] = str(
                figure_path
            )

    return figures


def _plot_single_setting(
    data,
    scenario,
    setting,
    metric_prefix,
    mean_column,
    low_column,
    high_column,
    path,
):
    fig, ax = plt.subplots(
        figsize=(
            7,
            4.5,
        )
    )

    for label, group in data.groupby(
        [
            "Regime",
            "Method",
        ]
    ):

        if isinstance(
            label,
            tuple,
        ):

            regime, method = label
            series_label = (
                f"{regime} / {method}"
            )

        else:

            series_label = str(
                label
            )

        group = group.sort_values(
            "Source Domain Count"
        )

        x = pd.to_numeric(
            group[
                "Source Domain Count"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        y = pd.to_numeric(
            group[
                mean_column
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        low = pd.to_numeric(
            group[
                low_column
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        high = pd.to_numeric(
            group[
                high_column
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        valid = (
            np.isfinite(
                x
            )
            & np.isfinite(
                y
            )
        )

        if not valid.any():
            continue

        x = x[
            valid
        ]

        y = y[
            valid
        ]

        low = low[
            valid
        ]

        high = high[
            valid
        ]

        ax.plot(
            x,
            y,
            marker="o",
            label=series_label,
        )

        ci_valid = (
            np.isfinite(
                low
            )
            & np.isfinite(
                high
            )
        )

        if ci_valid.any():

            ax.fill_between(
                x[
                    ci_valid
                ],
                low[
                    ci_valid
                ],
                high[
                    ci_valid
                ],
                alpha=0.2,
            )

    source_axis_label = str(
        data[
            "Source Domain Axis"
        ].iloc[
            0
        ]
    )

    ax.set_title(
        f"{_pretty_scenario(scenario)} — {setting}"
    )

    ax.set_xlabel(
        source_axis_label
    )

    ax.set_ylabel(
        metric_prefix
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=200,
    )

    plt.close(
        fig
    )


def _pretty_scenario(
    scenario,
):
    return str(
        scenario
    ).replace(
        "_",
        " ",
    ).title()


def _safe_name(
    text,
):
    text = str(
        text
    )

    text = text.replace(
        "/",
        "_"
    )

    text = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        text,
    )

    text = text.strip(
        "_"
    )

    return text.lower()


# ============================================================
# Filtering
# ============================================================

def _apply_filters(
    df,
    filters,
):
    result = df.copy()

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
            dataframe[column].isin(
                list(
                    value
                )
            )
        ]

    if (
        pd.api.types.is_numeric_dtype(
            dataframe[column]
        )
        and isinstance(
            value,
            (int, float)
        )
    ):

        return dataframe[
            np.isclose(
                pd.to_numeric(
                    dataframe[column],
                    errors="coerce",
                ),
                value,
                equal_nan=False,
            )
        ]

    return dataframe[
        dataframe[column]
        == value
    ]


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
            (list, tuple, set),
        ):

            return [
                str(item)
                for item in parsed
            ]

    except Exception:

        pass

    # Domain IDs internally use "|", for example:
    # bci_iv_2a|A01
    #
    # Therefore we must not split on "|".
    # Canonical serialized domain lists use ";".
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