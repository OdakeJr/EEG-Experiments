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
    "outputs/analysis/variability_decomposition"
)


DEFAULT_SCENARIO_CONFIGS = [
    {
        "name": "cross_session",
        "scenario": "cross_session",
        "setting_column": "Dataset",
        "filters": {
            "n_target_super_domains": 0,
            "use_max_source_domains": True,
        },
    },
    {
        "name": "cross_subject",
        "scenario": "cross_subject",
        "setting_column": "Dataset",
        "filters": {
            "n_target_super_domains": 0,
            "use_max_source_domains": True,
        },
    },
    {
        "name": "cross_dataset",
        "scenario": "cross_dataset",
        "setting_column": "Held-out Dataset",
        "filters": {
            "n_target_super_domains": 0,
            "use_max_source_super_domains": True,
        },
    },
]


DEFAULT_FACTORS = [
    {
        "name": "Preprocessing config",
        "column": "Preprocessing Config",
        "controls": [
            "Dataset",
            "Target Domain",
            "Feature Config",
            "Training Seed",
        ],
    },
    {
        "name": "Dataset",
        "column": "Dataset",
        "controls": [
            "Preprocessing Config",
            "Feature Config",
            "Training Seed",
        ],
    },
    {
        "name": "Target domain",
        "column": "Target Domain",
        "controls": [
            "Dataset",
            "Preprocessing Config",
            "Feature Config",
            "Training Seed",
        ],
    },
    {
        "name": "Feature config",
        "column": "Feature Config",
        "controls": [
            "Dataset",
            "Target Domain",
            "Preprocessing Config",
            "Training Seed",
        ],
    },
    {
        "name": "Seed",
        "column": "Training Seed",
        "controls": [
            "Dataset",
            "Target Domain",
            "Preprocessing Config",
            "Feature Config",
        ],
    },
]


# ============================================================
# Public function
# ============================================================

def run_variability_decomposition(
    model_results_artifact,
    params=None,
):
    """
    Estimate scenario-wise isolated-factor variability.

    This analysis uses a benchmark-like protocol:
    - one scenario at a time
    - all-source setting
    - optional fixed target fraction through filters

    For each scenario and method, it asks:

        When the other relevant factors are fixed,
        how much can this factor move Target-Test BA?

    Factors:
    - preprocessing config
    - dataset
    - target domain
    - feature config
    - seed

    This is not strict ANOVA. It is a conditional
    factor-wise stability analysis.
    """

    params = _with_default_params(
        params
    )

    model_path = _artifact_path(
        model_results_artifact
    )

    effective_params = {
        "analysis": "variability_decomposition",
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

    points_path = (
        output_dir
        / "variability_points.csv"
    )

    summary_path = (
        output_dir
        / "variability_summary.csv"
    )

    components_path = (
        output_dir
        / "variability_components.csv"
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
        and exists(components_path)
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
            name="variability_decomposition",
            output_dir=str(output_dir),
            tables={
                "points": str(points_path),
                "summary": str(summary_path),
                "components": str(components_path),
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

    points = _apply_scenario_configs(
        points=points,
        scenario_configs=params["scenarios"],
    )

    points.to_csv(
        points_path,
        index=False,
    )

    # --------------------------------------------------------
    # Variability tables
    # --------------------------------------------------------

    summary, components = _build_variability_tables(
        points=points,
        factors=params["factors"],
        y_column=params["metric_column"],
        ci_z=params["ci_z"],
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    components.to_csv(
        components_path,
        index=False,
    )

    # --------------------------------------------------------
    # Scenario-wise figures
    # --------------------------------------------------------

    figures = _build_plots(
        points=points,
        summary=summary,
        components=components,
        factors=params["factors"],
        y_column=params["metric_column"],
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
        "components": str(components_path),
        "figures": figures,
    }

    save_manifest(
        manifest,
        manifest_path,
    )

    return AnalysisArtifact(
        name="variability_decomposition",
        output_dir=str(output_dir),
        tables={
            "points": str(points_path),
            "summary": str(summary_path),
            "components": str(components_path),
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

        "metric_column": "Target-Test BA",

        "factors": DEFAULT_FACTORS,

        # Clean labels now preferred.
        # Fallbacks are handled automatically.
        "feature_config_column": "feature_selection_config_label",

        "preprocessing_config_column": "preprocessing_config_label",

        "seed_column": "training_seed",

        "ci_z": 1.96,

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

    points = results[
        (
            results["evaluation_group"]
            == "target_elementary_domain"
        )
        & (
            results["partition"]
            == "test"
        )
    ].copy()

    points["Scenario"] = points[
        "scenario"
    ].astype(
        str
    )

    points["Dataset"] = points[
        "target_domains"
    ].apply(
        _dataset_from_domains
    )

    points["Held-out Dataset"] = points[
        "Dataset"
    ]

    points["Target Domain"] = points[
        "target_domains"
    ].apply(
        _canonical_domain_string
    )

    points["Source Composition"] = points[
        "source_domains"
    ].apply(
        _canonical_domain_string
    )

    points["Target Fraction"] = pd.to_numeric(
        points["target_fraction"],
        errors="coerce",
    )

    points["Feature Config"] = _build_feature_config(
        points,
        feature_config_column,
    )

    points["Preprocessing Config"] = _build_preprocessing_config(
        points,
        preprocessing_config_column,
    )

    points["Training Seed"] = _build_seed_config(
        points,
        seed_column,
    )

    points["Target-Test BA"] = pd.to_numeric(
        points["balanced_accuracy"],
        errors="coerce",
    )

    points["Macro-F1"] = pd.to_numeric(
        points["macro_f1"],
        errors="coerce",
    )

    points["AUC"] = pd.to_numeric(
        points["auc"],
        errors="coerce",
    )

    points["Source Domain Count"] = pd.to_numeric(
        points["n_source_domains"],
        errors="coerce",
    )

    points["Source Super Domain Count"] = points[
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

    points["Regime"] = labels[
        "Regime"
    ]

    points["Method"] = labels[
        "Method"
    ]

    keep_columns = [
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
        "n_target_super_domains",
        "target_fraction",
        "Source Domain Count",
        "Source Super Domain Count",
        "source_domains",
        "target_domains",
        "Target-Test BA",
        "Macro-F1",
        "AUC",
    ]

    optional_columns = [
        "feature_selection_method",
        "feature_selection_params",
        "feature_selection_config_label",
        "preprocessing_signature",
        "preprocessing_config_label",
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
# Scenario filtering
# ============================================================

def _apply_scenario_configs(
    points,
    scenario_configs,
):
    scenario_frames = []

    for config in scenario_configs:

        scenario = config[
            "scenario"
        ]

        setting_column = config.get(
            "setting_column",
            "Dataset",
        )

        df = points[
            points["Scenario"]
            == scenario
        ].copy()

        df = _apply_filters(
            df=df,
            filters=config.get(
                "filters",
                {},
            ),
            setting_column=setting_column,
        )

        if not df.empty:

            scenario_frames.append(
                df
            )

    if not scenario_frames:

        return points.iloc[
            0:0
        ].copy()

    return pd.concat(
        scenario_frames,
        ignore_index=True,
    )


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
    # Benchmark-like all-source protocol
    # --------------------------------------------------------

    if filters.get(
        "use_max_source_domains",
        False,
    ):

        if result.empty:
            return result

        group_columns = [
            setting_column,
            "Target Fraction",
            "Regime",
            "Method",
            "Preprocessing Config",
            "Feature Config",
            "Training Seed",
        ]

        group_columns = [
            column
            for column in group_columns
            if column in result.columns
        ]

        max_values = (
            result
            .groupby(
                group_columns,
                dropna=False,
            )[
                "Source Domain Count"
            ]
            .transform(
                "max"
            )
        )

        result = result[
            result[
                "Source Domain Count"
            ]
            == max_values
        ]

    if filters.get(
        "use_max_source_super_domains",
        False,
    ):

        if result.empty:
            return result

        group_columns = [
            setting_column,
            "Target Fraction",
            "Regime",
            "Method",
            "Preprocessing Config",
            "Feature Config",
            "Training Seed",
        ]

        group_columns = [
            column
            for column in group_columns
            if column in result.columns
        ]

        max_values = (
            result
            .groupby(
                group_columns,
                dropna=False,
            )[
                "Source Super Domain Count"
            ]
            .transform(
                "max"
            )
        )

        result = result[
            result[
                "Source Super Domain Count"
            ]
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

    if isinstance(
        value,
        (int, float),
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
# Variability tables
# ============================================================

def _build_variability_tables(
    points,
    factors,
    y_column,
    ci_z,
):
    summary_columns = [
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
    ]

    component_columns = [
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
    ]

    if points.empty:

        return (
            pd.DataFrame(
                columns=summary_columns
            ),
            pd.DataFrame(
                columns=component_columns
            ),
        )

    summary_rows = []
    component_rows = []

    group_columns = [
        "Scenario",
        "Regime",
        "Method",
    ]

    for group_values, method_data in points.groupby(
        group_columns,
        dropna=False,
    ):

        scenario, regime, method = group_values

        method_data = method_data.copy()

        y = pd.to_numeric(
            method_data[
                y_column
            ],
            errors="coerce",
        ).dropna()

        summary_row = {
            "Scenario": scenario,
            "Regime": regime,
            "Method": method,
            "Runs": int(
                len(
                    y
                )
            ),
            "Mean BA": np.nan,
            "BA Std": np.nan,
            "BA SEM": np.nan,
            "BA CI95 Low": np.nan,
            "BA CI95 High": np.nan,
            "Total BA Variance": np.nan,
            "Isolated Variance Sum": np.nan,
        }

        if len(y) > 0:

            mean = float(
                y.mean()
            )

            summary_row[
                "Mean BA"
            ] = mean

        if len(y) > 1:

            std = float(
                y.std(
                    ddof=1
                )
            )

            sem = float(
                std
                / np.sqrt(
                    len(
                        y
                    )
                )
            )

            margin = (
                ci_z
                * sem
            )

            summary_row[
                "BA Std"
            ] = std

            summary_row[
                "BA SEM"
            ] = sem

            summary_row[
                "BA CI95 Low"
            ] = mean - margin

            summary_row[
                "BA CI95 High"
            ] = mean + margin

            summary_row[
                "Total BA Variance"
            ] = float(
                y.var(
                    ddof=1
                )
            )

        method_components = []

        for factor in factors:

            component = _estimate_isolated_factor_variance(
                data=method_data,
                factor=factor,
                y_column=y_column,
            )

            component.update(
                {
                    "Scenario": scenario,
                    "Regime": regime,
                    "Method": method,
                }
            )

            method_components.append(
                component
            )

        variance_values = [
            (
                component[
                    "Isolated Variance"
                ]
                if pd.notna(
                    component[
                        "Isolated Variance"
                    ]
                )
                else 0.0
            )
            for component in method_components
        ]

        variance_sum = float(
            np.sum(
                variance_values
            )
        )

        summary_row[
            "Isolated Variance Sum"
        ] = variance_sum

        for component, value in zip(
            method_components,
            variance_values,
        ):

            if variance_sum > 0:

                component[
                    "Share (%)"
                ] = float(
                    value
                    / variance_sum
                    * 100.0
                )

            else:

                component[
                    "Share (%)"
                ] = 0.0

            component_rows.append(
                {
                    column: component.get(
                        column,
                        np.nan,
                    )
                    for column in component_columns
                }
            )

        summary_rows.append(
            summary_row
        )

    summary = (
        pd.DataFrame(
            summary_rows,
            columns=summary_columns,
        )
        .sort_values(
            [
                "Scenario",
                "Mean BA",
                "Regime",
                "Method",
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    components = (
        pd.DataFrame(
            component_rows,
            columns=component_columns,
        )
        .sort_values(
            [
                "Scenario",
                "Regime",
                "Method",
                "Component",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        summary,
        components,
    )


def _estimate_isolated_factor_variance(
    data,
    factor,
    y_column,
):
    factor_name = factor[
        "name"
    ]

    factor_column = factor[
        "column"
    ]

    control_columns = factor.get(
        "controls",
        [],
    )

    control_columns = [
        column
        for column in control_columns
        if column in data.columns
    ]

    if factor_column not in data.columns:

        return {
            "Component": factor_name,
            "Component Column": factor_column,
            "Control Columns": ";".join(
                control_columns
            ),
            "Isolated Variance": np.nan,
            "Share (%)": np.nan,
            "Estimable": False,
            "Matched Groups": 0,
            "Matched Rows": 0,
            "Unique Values": 0,
        }

    valid_data = data[
        data[
            y_column
        ].notna()
    ].copy()

    unique_values = int(
        valid_data[
            factor_column
        ].nunique(
            dropna=False
        )
    )

    if valid_data.empty:

        return {
            "Component": factor_name,
            "Component Column": factor_column,
            "Control Columns": ";".join(
                control_columns
            ),
            "Isolated Variance": np.nan,
            "Share (%)": np.nan,
            "Estimable": False,
            "Matched Groups": 0,
            "Matched Rows": 0,
            "Unique Values": unique_values,
        }

    numerator = 0.0
    denominator = 0
    matched_groups = 0
    matched_rows = 0

    if control_columns:

        grouped = valid_data.groupby(
            control_columns,
            dropna=False,
        )

    else:

        grouped = [
            (
                "__all__",
                valid_data,
            )
        ]

    for _, group in grouped:

        if (
            group[
                factor_column
            ].nunique(
                dropna=False
            )
            < 2
        ):
            continue

        level_means = (
            group
            .groupby(
                factor_column,
                dropna=False,
            )[
                y_column
            ]
            .mean()
            .dropna()
        )

        if len(
            level_means
        ) < 2:
            continue

        values = level_means.to_numpy(
            dtype=float
        )

        numerator += float(
            np.sum(
                (
                    values
                    - np.mean(
                        values
                    )
                )
                ** 2
            )
        )

        denominator += int(
            len(
                values
            )
            - 1
        )

        matched_groups += 1
        matched_rows += int(
            len(
                group
            )
        )

    if denominator <= 0:

        isolated_variance = np.nan
        estimable = False

    else:

        isolated_variance = float(
            numerator
            / denominator
        )

        estimable = True

    return {
        "Component": factor_name,
        "Component Column": factor_column,
        "Control Columns": ";".join(
            control_columns
        ),
        "Isolated Variance": isolated_variance,
        "Share (%)": np.nan,
        "Estimable": estimable,
        "Matched Groups": int(
            matched_groups
        ),
        "Matched Rows": int(
            matched_rows
        ),
        "Unique Values": unique_values,
    }


# ============================================================
# Plotting
# ============================================================

def _build_plots(
    points,
    summary,
    components,
    factors,
    y_column,
    figures_dir,
):
    figures = {}

    if points.empty or summary.empty:
        return figures

    for scenario in sorted(
        points[
            "Scenario"
        ].dropna().unique()
    ):

        scenario_points = points[
            points[
                "Scenario"
            ]
            == scenario
        ].copy()

        scenario_summary = summary[
            summary[
                "Scenario"
            ]
            == scenario
        ].copy()

        scenario_components = components[
            components[
                "Scenario"
            ]
            == scenario
        ].copy()

        if (
            scenario_points.empty
            or scenario_summary.empty
            or scenario_components.empty
        ):
            continue

        figure_key = _safe_name(
            f"{scenario}_method_stability"
        )

        figure_path = (
            figures_dir
            / f"{figure_key}.png"
        )

        _plot_scenario_stability(
            scenario=scenario,
            points=scenario_points,
            summary=scenario_summary,
            components=scenario_components,
            factors=factors,
            y_column=y_column,
            path=figure_path,
        )

        figures[
            figure_key
        ] = str(
            figure_path
        )

    return figures


def _plot_scenario_stability(
    scenario,
    points,
    summary,
    components,
    factors,
    y_column,
    path,
):
    summary = summary.sort_values(
        [
            "Mean BA",
            "Regime",
            "Method",
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    method_keys = list(
        zip(
            summary["Regime"],
            summary["Method"],
        )
    )

    method_labels = [
        f"{regime} / {method}"
        for regime, method in method_keys
    ]

    y_positions = np.arange(
        len(
            method_keys
        )
    )

    fig = plt.figure(
        figsize=(
            15,
            max(
                5,
                0.65
                * len(
                    method_keys
                )
                + 2,
            ),
        )
    )

    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[
            1.35,
            3.65,
        ],
        wspace=0.06,
    )

    ax_perf = fig.add_subplot(
        gs[
            0,
            0
        ]
    )

    ax_var = fig.add_subplot(
        gs[
            0,
            1
        ],
        sharey=ax_perf,
    )

    violin_data = []

    for regime, method in method_keys:

        values = points[
            (
                points["Regime"]
                == regime
            )
            & (
                points["Method"]
                == method
            )
        ][
            y_column
        ]

        values = pd.to_numeric(
            values,
            errors="coerce",
        ).dropna()

        violin_data.append(
            values.to_numpy(
                dtype=float
            )
        )

    valid_positions = [
        y_positions[
            i
        ]
        for i, values in enumerate(
            violin_data
        )
        if len(
            values
        ) > 1
    ]

    valid_violin_data = [
        values
        for values in violin_data
        if len(
            values
        ) > 1
    ]

    if valid_violin_data:

        parts = ax_perf.violinplot(
            valid_violin_data,
            positions=valid_positions,
            vert=False,
            widths=0.75,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        for body in parts[
            "bodies"
        ]:

            body.set_alpha(
                0.35
            )

    for i, row in summary.iterrows():

        mean = row[
            "Mean BA"
        ]

        ci_low = row[
            "BA CI95 Low"
        ]

        ci_high = row[
            "BA CI95 High"
        ]

        if pd.isna(
            mean
        ):
            continue

        if (
            pd.notna(
                ci_low
            )
            and pd.notna(
                ci_high
            )
        ):

            xerr = np.array(
                [
                    [
                        mean
                        - ci_low
                    ],
                    [
                        ci_high
                        - mean
                    ],
                ]
            )

            ax_perf.errorbar(
                mean,
                i,
                xerr=xerr,
                fmt="o",
                capsize=3,
                zorder=4,
            )

        else:

            ax_perf.scatter(
                mean,
                i,
                s=45,
                zorder=4,
            )

        ax_perf.text(
            mean,
            i + 0.22,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax_perf.set_yticks(
        y_positions
    )

    ax_perf.set_yticklabels(
        method_labels,
        fontsize=9,
    )

    ax_perf.set_xlabel(
        "Target-test BA",
        fontsize=10,
    )

    ax_perf.grid(
        axis="x",
        linestyle="--",
        linewidth=0.7,
        alpha=0.3,
    )

    ax_perf.spines[
        "top"
    ].set_visible(
        False
    )

    ax_perf.spines[
        "right"
    ].set_visible(
        False
    )

    component_order = [
        factor[
            "name"
        ]
        for factor in factors
    ]

    left = np.zeros(
        len(
            method_keys
        )
    )

    for component in component_order:

        values = []

        for regime, method in method_keys:

            row = components[
                (
                    components["Regime"]
                    == regime
                )
                & (
                    components["Method"]
                    == method
                )
                & (
                    components["Component"]
                    == component
                )
            ]

            if row.empty:

                values.append(
                    0.0
                )

            else:

                values.append(
                    float(
                        row[
                            "Share (%)"
                        ].iloc[
                            0
                        ]
                    )
                )

        values = np.asarray(
            values,
            dtype=float,
        )

        ax_var.barh(
            y_positions,
            values,
            left=left,
            height=0.66,
            label=component,
        )

        for i, width in enumerate(
            values
        ):

            if width >= 8:

                ax_var.text(
                    left[
                        i
                    ]
                    + width
                    / 2,
                    i,
                    f"{width:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        left += values

    ax_var.set_xlim(
        0,
        100,
    )

    ax_var.set_xlabel(
        "Relative isolated variability contribution (%)",
        fontsize=10,
    )

    ax_var.tick_params(
        axis="y",
        left=False,
        labelleft=False,
    )

    ax_var.grid(
        axis="x",
        linestyle="--",
        linewidth=0.7,
        alpha=0.25,
    )

    ax_var.spines[
        "top"
    ].set_visible(
        False
    )

    ax_var.spines[
        "right"
    ].set_visible(
        False
    )

    ax_var.spines[
        "left"
    ].set_visible(
        False
    )

    for i, row in summary.iterrows():

        total_var = row[
            "Total BA Variance"
        ]

        if pd.notna(
            total_var
        ):

            label = (
                f"Var. {total_var:.4f}"
            )

        else:

            label = "Var. n/a"

        ax_var.text(
            101.5,
            i,
            label,
            va="center",
            fontsize=8.5,
            clip_on=False,
        )

    fig.suptitle(
        f"{_pretty_scenario(scenario)} — Method Stability",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )

    fig.text(
        0.5,
        0.935,
        (
            "Target-test BA distribution and isolated variability "
            "from preprocessing, dataset, target domain, feature config, and seed"
        ),
        ha="center",
        fontsize=10,
    )

    ax_var.legend(
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            -0.22,
        ),
        ncol=5,
        frameon=True,
        fontsize=8.5,
    )

    plt.subplots_adjust(
        left=0.12,
        right=0.92,
        top=0.88,
        bottom=0.20,
    )

    fig.savefig(
        path,
        dpi=220,
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