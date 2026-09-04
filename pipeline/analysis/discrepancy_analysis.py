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
    "outputs/analysis/discrepancy_analysis"
)


DEFAULT_SCENARIO_CONFIGS = [
    {
        "name": "cross_session",
        "scenario": "cross_session",
        "setting_column": "Dataset",
        "output_prefix": "cross_session",
        "filters": {
            "n_target_super_domains": 0,
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
            "use_max_source_super_domains": True,
        },
    },
]


# ============================================================
# Public function
# ============================================================

def run_discrepancy_analysis(
    model_results_artifact,
    domain_results_artifact,
    params=None,
):
    """
    Analyze whether larger source-target discrepancy is associated
    with a larger source-target generalization gap.

    Main relationship:

        x = source-target discrepancy
        y = gap = Source BA - Target-Test BA

    Output:
    - discrepancy_analysis_points.csv
    - discrepancy_analysis_summary.csv
    - one scatter plot per scenario x dataset/held-out dataset
    """

    params = _with_default_params(
        params
    )

    model_path = _artifact_path(
        model_results_artifact
    )

    domain_path = _artifact_path(
        domain_results_artifact
    )

    effective_params = {
        "analysis": "discrepancy_analysis",
        "params": params,
        "model_results_path": str(model_path),
        "domain_results_path": str(domain_path),
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

    points_path = (
        output_dir
        / "discrepancy_analysis_points.csv"
    )

    summary_path = (
        output_dir
        / "discrepancy_analysis_summary.csv"
    )

    manifest_path = (
        output_dir
        / "manifest.json"
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    if (
        exists(points_path)
        and exists(summary_path)
        and is_done(
            manifest_path,
            effective_params,
        )
    ):

        figures = {
            path.stem: str(path)
            for path in figures_dir.glob(
                "*.png"
            )
        }

        return AnalysisArtifact(
            name="discrepancy_analysis",
            output_dir=str(output_dir),
            tables={
                "points": str(points_path),
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

    domain_results = pd.read_csv(
        domain_path
    )

    display_lookup = _build_display_lookup(
        params["method_display"]
    )

    model_summary = _prepare_model_summary(
        model_results,
        display_lookup,
    )

    discrepancy = _prepare_discrepancy(
        domain_results,
        params,
    )

    points = _merge_model_and_discrepancy(
        model_summary,
        discrepancy,
    )

    # --------------------------------------------------------
    # Apply scenario configs
    # --------------------------------------------------------

    scenario_points = []

    for scenario_config in params[
        "scenarios"
    ]:

        scenario_df = _build_scenario_points(
            points,
            scenario_config,
        )

        if not scenario_df.empty:

            scenario_points.append(
                scenario_df
            )

    if scenario_points:

        points = pd.concat(
            scenario_points,
            ignore_index=True,
        )

    else:

        points = _empty_points_table()

    points.to_csv(
        points_path,
        index=False,
    )

    # --------------------------------------------------------
    # Correlation summary
    # --------------------------------------------------------

    summary = _build_correlation_summary(
        points,
        min_points=params[
            "min_points_for_fit"
        ],
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Scatter plots
    # --------------------------------------------------------

    figures = _build_plots(
        points=points,
        summary=summary,
        scenario_configs=params["scenarios"],
        figures_dir=figures_dir,
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
        "points": str(points_path),
        "summary": str(summary_path),
        "figures": figures,
    }

    save_manifest(
        manifest,
        manifest_path,
    )

    return AnalysisArtifact(
        name="discrepancy_analysis",
        output_dir=str(output_dir),
        tables={
            "points": str(points_path),
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

        "discrepancy_metric": "mmd",
        "comparison": "source_to_target_elementary",
        "representation": "marginal",

        "min_points_for_fit": 2,

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

    keep_columns = [
        "split_id",
        "scenario",
        "group",
        "Dataset",
        "Held-out Dataset",
        "Regime",
        "Method",
        "learning_method",
        "model_name",
        "model_signature",
        "feature_selection_signature",
        "n_source_domains",
        "source_super_domain_count",
        "n_target_super_domains",
        "target_fraction",
        "source_domains",
        "target_domains",
        "source_ba",
        "target_ba",
        "gap",
        "target_macro_f1",
        "target_auc",
    ]

    if "training_seed" in summary.columns:
        keep_columns.append(
            "training_seed"
        )

    return summary[
        keep_columns
    ].copy()


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
# Discrepancy preparation
# ============================================================

def _prepare_discrepancy(
    domain_results,
    params,
):
    required_columns = [
        "split_id",
        "scenario",
        "group",
        "target_fraction",
        "comparison",
        "metric",
        "representation",
        "value",
    ]

    _require_columns(
        domain_results,
        required_columns,
    )

    df = domain_results.copy()

    df = df[
        df["comparison"]
        == params["comparison"]
    ]

    df = df[
        df["metric"]
        == params["discrepancy_metric"]
    ]

    df = df[
        df["representation"]
        == params["representation"]
    ]

    if df.empty:

        return pd.DataFrame(
            columns=[
                "split_id",
                "discrepancy",
            ]
        )

    discrepancy = (
        df
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
                "value": "discrepancy",
            }
        )
    )

    return discrepancy


def _merge_model_and_discrepancy(
    model_summary,
    discrepancy,
):
    merged = model_summary.merge(
        discrepancy,
        on="split_id",
        how="inner",
    )

    merged["discrepancy"] = pd.to_numeric(
        merged["discrepancy"],
        errors="coerce",
    )

    merged["gap"] = pd.to_numeric(
        merged["gap"],
        errors="coerce",
    )

    merged = merged[
        merged["discrepancy"].notna()
        & merged["gap"].notna()
    ].copy()

    return merged


# ============================================================
# Scenario point construction
# ============================================================

def _build_scenario_points(
    points,
    scenario_config,
):
    scenario = scenario_config[
        "scenario"
    ]

    setting_column = scenario_config[
        "setting_column"
    ]

    df = points[
        points["scenario"]
        == scenario
    ].copy()

    df = _apply_filters(
        df=df,
        filters=scenario_config.get(
            "filters",
            {},
        ),
        setting_column=setting_column,
    )

    if df.empty:
        return _empty_points_table()

    df["Scenario"] = scenario

    df["Setting"] = df[
        setting_column
    ]

    df["Source Domain Count"] = pd.to_numeric(
        df["n_source_domains"],
        errors="coerce",
    )

    df["Source Super Domain Count"] = pd.to_numeric(
        df["source_super_domain_count"],
        errors="coerce",
    )

    df["Target Fraction"] = pd.to_numeric(
        df["target_fraction"],
        errors="coerce",
    )

    df = df.rename(
        columns={
            "source_ba": "Source BA",
            "target_ba": "Target-Test BA",
            "gap": "Gap",
            "discrepancy": "Discrepancy",
            "target_macro_f1": "Macro-F1",
            "target_auc": "AUC",
        }
    )

    columns = _empty_points_table().columns

    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    return df[
        available_columns
    ].copy()


def _empty_points_table():
    columns = [
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
    ]

    return pd.DataFrame(
        columns=columns
    )


# ============================================================
# Correlation summary
# ============================================================

def _build_correlation_summary(
    points,
    min_points,
):
    columns = [
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
    ]

    if points.empty:

        return pd.DataFrame(
            columns=columns
        )

    rows = []

    group_columns = [
        "Scenario",
        "Setting",
        "Regime",
        "Method",
    ]

    for group_values, group in points.groupby(
        group_columns,
        dropna=False,
    ):

        row = dict(
            zip(
                group_columns,
                group_values,
            )
        )

        x = pd.to_numeric(
            group["Discrepancy"],
            errors="coerce",
        )

        y = pd.to_numeric(
            group["Gap"],
            errors="coerce",
        )

        valid = (
            x.notna()
            & y.notna()
        )

        x = x[
            valid
        ].to_numpy(
            dtype=float
        )

        y = y[
            valid
        ].to_numpy(
            dtype=float
        )

        row["Runs"] = int(
            len(
                x
            )
        )

        row["Pearson r"] = np.nan
        row["Spearman rho"] = np.nan
        row["Linear Slope"] = np.nan
        row["Linear Intercept"] = np.nan
        row["Discrepancy Mean"] = np.nan
        row["Discrepancy Std"] = np.nan
        row["Gap Mean"] = np.nan
        row["Gap Std"] = np.nan

        if len(x) > 0:

            row["Discrepancy Mean"] = float(
                np.mean(
                    x
                )
            )

            row["Gap Mean"] = float(
                np.mean(
                    y
                )
            )

        if len(x) > 1:

            row["Discrepancy Std"] = float(
                np.std(
                    x,
                    ddof=1,
                )
            )

            row["Gap Std"] = float(
                np.std(
                    y,
                    ddof=1,
                )
            )

        if len(x) >= min_points:

            row["Pearson r"] = _safe_corr(
                x,
                y,
            )

            row["Spearman rho"] = _safe_corr(
                _rank_values(
                    x
                ),
                _rank_values(
                    y
                ),
            )

            slope, intercept = _safe_linear_fit(
                x,
                y,
            )

            row["Linear Slope"] = slope
            row["Linear Intercept"] = intercept

        rows.append(
            row
        )

    return (
        pd.DataFrame(
            rows,
            columns=columns,
        )
        .sort_values(
            [
                "Scenario",
                "Setting",
                "Regime",
                "Method",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def _safe_corr(
    x,
    y,
):
    if len(x) < 2:
        return np.nan

    if np.isclose(
        np.std(
            x
        ),
        0.0,
    ):
        return np.nan

    if np.isclose(
        np.std(
            y
        ),
        0.0,
    ):
        return np.nan

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


def _rank_values(
    values,
):
    return (
        pd.Series(
            values
        )
        .rank(
            method="average"
        )
        .to_numpy(
            dtype=float
        )
    )


def _safe_linear_fit(
    x,
    y,
):
    if len(x) < 2:
        return (
            np.nan,
            np.nan,
        )

    if np.isclose(
        np.std(
            x
        ),
        0.0,
    ):
        return (
            np.nan,
            np.nan,
        )

    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )

    return (
        float(
            slope
        ),
        float(
            intercept
        ),
    )


# ============================================================
# Plotting
# ============================================================

def _build_plots(
    points,
    summary,
    scenario_configs,
    figures_dir,
):
    figures = {}

    if points.empty:
        return figures

    for scenario_config in scenario_configs:

        scenario = scenario_config[
            "scenario"
        ]

        scenario_points = points[
            points["Scenario"]
            == scenario
        ].copy()

        if scenario_points.empty:
            continue

        for setting, setting_points in scenario_points.groupby(
            "Setting"
        ):

            if setting_points.empty:
                continue

            figure_key = _safe_name(
                f"{scenario}_{setting}_discrepancy_vs_gap"
            )

            figure_path = (
                figures_dir
                / f"{figure_key}.png"
            )

            _plot_single_setting(
                data=setting_points,
                summary=summary,
                scenario=scenario,
                setting=setting,
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
    summary,
    scenario,
    setting,
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

            regime = None
            method = None
            series_label = str(
                label
            )

        x = pd.to_numeric(
            group["Discrepancy"],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        y = pd.to_numeric(
            group["Gap"],
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

        ax.scatter(
            x,
            y,
            label=series_label,
            alpha=0.75,
        )

        if (
            regime is not None
            and method is not None
        ):

            fit_row = summary[
                (
                    summary["Scenario"]
                    == scenario
                )
                & (
                    summary["Setting"]
                    == setting
                )
                & (
                    summary["Regime"]
                    == regime
                )
                & (
                    summary["Method"]
                    == method
                )
            ]

            if not fit_row.empty:

                slope = fit_row[
                    "Linear Slope"
                ].iloc[
                    0
                ]

                intercept = fit_row[
                    "Linear Intercept"
                ].iloc[
                    0
                ]

                if (
                    pd.notna(
                        slope
                    )
                    and pd.notna(
                        intercept
                    )
                ):

                    x_fit = np.linspace(
                        np.min(
                            x
                        ),
                        np.max(
                            x
                        ),
                        100,
                    )

                    y_fit = (
                        slope
                        * x_fit
                        + intercept
                    )

                    ax.plot(
                        x_fit,
                        y_fit,
                        linestyle="--",
                        alpha=0.8,
                    )

    ax.axhline(
        0.0,
        linestyle=":",
        linewidth=1,
        alpha=0.6,
    )

    ax.set_title(
        f"{_pretty_scenario(scenario)} — {setting}"
    )

    ax.set_xlabel(
        "Source-target discrepancy"
    )

    ax.set_ylabel(
        "Generalization gap"
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
    setting_column,
):
    result = df.copy()

    # --------------------------------------------------------
    # Direct filters
    # --------------------------------------------------------

    for column, value in filters.items():

        if column in [
            "use_max_source_domains",
            "use_max_source_super_domains",
        ]:
            continue

        if column not in result.columns:
            continue

        result = _filter_column(
            result,
            column,
            value,
        )

    # --------------------------------------------------------
    # Fix source-domain protocol for session/subject cases
    # --------------------------------------------------------

    if filters.get(
        "use_max_source_domains",
        False,
    ):

        if result.empty:
            return result

        max_values = (
            result
            .groupby(
                [
                    setting_column,
                    "target_fraction",
                    "Regime",
                    "Method",
                ]
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

    # --------------------------------------------------------
    # Fix source-dataset protocol for cross-dataset
    # --------------------------------------------------------

    if filters.get(
        "use_max_source_super_domains",
        False,
    ):

        if result.empty:
            return result

        max_values = (
            result
            .groupby(
                [
                    setting_column,
                    "target_fraction",
                    "Regime",
                    "Method",
                ]
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