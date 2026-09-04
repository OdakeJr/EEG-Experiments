# pipeline/analysis/domain_variability_quantitative.py

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.analysis_artifact import (
    AnalysisArtifact,
)

from utils.storage import (
    exists,
    load_manifest,
    save_data,
    save_manifest,
)

from utils.status import (
    is_done,
    make_manifest,
    make_signature,
)

OUTPUT_ROOT = Path(
    "outputs/analysis/domain_variability_quantitative"
)


# ============================================================
# Helpers
# ============================================================

def _filter_domain_results(
    results,
    comparison,
    metrics,
    scenarios,
):
    """
    Keep only the domain comparisons relevant to RQ1.
    """

    filtered = results[
        results["comparison"].eq(
            comparison
        )
        & results["metric"].isin(
            metrics
        )
        & results["scenario"].isin(
            scenarios
        )
    ].copy()

    return filtered


def _build_conditional_table(
    results,
):
    """
    Compute one macro class-conditional discrepancy
    per split / scenario / metric.

    The canonical domain-results table stores one row
    per class. Here these class-specific values are
    averaged to produce D_cond.
    """

    class_results = results[
        results["representation"]
        == "class"
    ].copy()

    if class_results.empty:
        return pd.DataFrame()

    group_columns = [
        "split_id",
        "scenario",
        "group",
        "seed",
        "target_fraction",
        "comparison",
        "left_group",
        "right_group",
        "left_domains",
        "right_domains",
        "n_left_domains",
        "n_right_domains",
        "metric",
        "metric_signature",
    ]

    conditional = (
        class_results
        .groupby(
            group_columns,
            dropna=False,
            as_index=False,
        )
        .agg(
            conditional_value=(
                "value",
                "mean",
            ),
            n_classes=(
                "class_label",
                "nunique",
            ),
        )
    )

    return conditional


def _build_marginal_conditional_pairs(
    results,
):
    """
    Pair marginal discrepancy with the macro-average
    class-conditional discrepancy for each split.
    """

    marginal = results[
        results["representation"]
        == "marginal"
    ].copy()

    marginal = marginal.rename(
        columns={
            "value": "marginal_value",
        }
    )

    conditional = (
        _build_conditional_table(
            results
        )
    )

    if conditional.empty:
        return pd.DataFrame()

    merge_columns = [
        "split_id",
        "scenario",
        "group",
        "seed",
        "target_fraction",
        "comparison",
        "left_group",
        "right_group",
        "left_domains",
        "right_domains",
        "n_left_domains",
        "n_right_domains",
        "metric",
        "metric_signature",
    ]

    paired = marginal.merge(
        conditional,
        on=merge_columns,
        how="inner",
    )

    return paired


def _median_iqr(
    values,
):
    """
    Return median and interquartile range.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    median = np.median(
        values
    )

    q1 = np.percentile(
        values,
        25,
    )

    q3 = np.percentile(
        values,
        75,
    )

    return (
        float(median),
        float(q1),
        float(q3),
    )


def _format_summary(
    values,
):
    """
    Format values as:

        median [Q1, Q3]
    """

    median, q1, q3 = (
        _median_iqr(
            values
        )
    )

    if np.isnan(median):
        return ""

    return (
        f"{median:.4f} "
        f"[{q1:.4f}, {q3:.4f}]"
    )


def _build_summary_table(
    results,
    paired,
    scenarios,
    metrics,
):
    """
    Build the numerical domain-variability summary:

        Shift
        Metric
        Marginal
        Conditional
        Left
        Right
        Feet
    """

    rows = []

    class_map = {
        "Left": "left_hand_imagery",
        "Right": "right_hand_imagery",
        "Feet": "both_feet_imagery",
    }

    for scenario in scenarios:

        for metric in metrics:

            subset = results[
                results["scenario"].eq(
                    scenario
                )
                & results["metric"].eq(
                    metric
                )
            ]

            marginal = subset[
                subset["representation"]
                == "marginal"
            ]["value"]

            conditional_subset = paired[
                paired["scenario"].eq(
                    scenario
                )
                & paired["metric"].eq(
                    metric
                )
            ]

            conditional = (
                conditional_subset[
                    "conditional_value"
                ]
            )

            row = {
                "shift": scenario,
                "metric": metric,

                "marginal": (
                    _format_summary(
                        marginal
                    )
                ),

                "conditional": (
                    _format_summary(
                        conditional
                    )
                ),
            }

            for output_name, class_label in (
                class_map.items()
            ):

                class_values = subset[
                    (
                        subset[
                            "representation"
                        ]
                        == "class"
                    )
                    & (
                        subset[
                            "class_label"
                        ]
                        == class_label
                    )
                ]["value"]

                row[
                    output_name.lower()
                ] = _format_summary(
                    class_values
                )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Figure 1
# Quantitative discrepancy distributions
# ============================================================

def _plot_discrepancy_distribution(
    results,
    scenarios,
    metrics,
    output_path,
):
    """
    Plot marginal discrepancy distributions across
    cross-subject, cross-session, and cross-dataset
    configurations.

    One panel is generated per discrepancy metric.
    """

    marginal = results[
        results["representation"]
        == "marginal"
    ]

    n_metrics = len(
        metrics
    )

    fig, axes = plt.subplots(
        1,
        n_metrics,
        figsize=(
            5 * n_metrics,
            4.5,
        ),
        squeeze=False,
    )

    axes = axes[0]

    for index, metric in enumerate(
        metrics
    ):

        ax = axes[index]

        metric_data = marginal[
            marginal["metric"]
            == metric
        ]

        distributions = []

        labels = []

        for scenario in scenarios:

            values = metric_data[
                metric_data["scenario"]
                == scenario
            ]["value"].dropna().values

            if len(values) == 0:
                continue

            distributions.append(
                values
            )

            labels.append(
                scenario
                .replace("_", " ")
                .title()
            )

        if len(distributions) == 0:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
            continue

        ax.boxplot(
            distributions,
            labels=labels,
            showfliers=True,
        )

        # ------------------------------------------
        # Add individual observations
        # ------------------------------------------

        rng = np.random.default_rng(
            42
        )

        for position, values in enumerate(
            distributions,
            start=1,
        ):

            jitter = rng.normal(
                0,
                0.04,
                size=len(values),
            )

            ax.scatter(
                position + jitter,
                values,
                s=12,
                alpha=0.45,
            )

        ax.set_title(
            metric.upper()
        )

        ax.set_ylabel(
            "Domain discrepancy"
        )

        ax.tick_params(
            axis="x",
            rotation=20,
        )

    fig.suptitle(
        "Distribution of EEG Domain Discrepancy"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Figure 2
# Marginal vs class-conditional discrepancy
# ============================================================

def _plot_marginal_conditional(
    paired,
    scenarios,
    metrics,
    output_path,
):
    """
    Compare marginal discrepancy with the macro-averaged
    class-conditional discrepancy.
    """

    n_metrics = len(
        metrics
    )

    fig, axes = plt.subplots(
        1,
        n_metrics,
        figsize=(
            5 * n_metrics,
            4.5,
        ),
        squeeze=False,
    )

    axes = axes[0]

    for metric_index, metric in enumerate(
        metrics
    ):

        ax = axes[
            metric_index
        ]

        metric_data = paired[
            paired["metric"]
            == metric
        ]

        positions = []

        distributions = []

        labels = []

        current_position = 1

        for scenario in scenarios:

            scenario_data = metric_data[
                metric_data["scenario"]
                == scenario
            ]

            if scenario_data.empty:
                continue

            marginal_values = (
                scenario_data[
                    "marginal_value"
                ]
                .dropna()
                .values
            )

            conditional_values = (
                scenario_data[
                    "conditional_value"
                ]
                .dropna()
                .values
            )

            distributions.extend([
                marginal_values,
                conditional_values,
            ])

            positions.extend([
                current_position,
                current_position + 1,
            ])

            labels.extend([
                (
                    scenario
                    .replace("_", " ")
                    .title()
                    + "\nMarg."
                ),
                (
                    scenario
                    .replace("_", " ")
                    .title()
                    + "\nCond."
                ),
            ])

            current_position += 3

        if len(distributions) == 0:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
            )

            ax.set_axis_off()
            continue

        ax.boxplot(
            distributions,
            positions=positions,
            widths=0.7,
            showfliers=True,
        )

        ax.set_xticks(
            positions
        )

        ax.set_xticklabels(
            labels,
            rotation=20,
        )

        ax.set_title(
            metric.upper()
        )

        ax.set_ylabel(
            "Domain discrepancy"
        )

    fig.suptitle(
        "Marginal vs Class-Conditional Domain Shift"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Main pipeline
#
# This analysis consumes the canonical domain-results table
# generated by the domain-evaluation stage.
#
# It produces:
#
#   1. Distribution of marginal discrepancy
#   2. Marginal vs conditional discrepancy figure
#   3. Numerical domain-variability summary
#   4. Paired marginal/conditional table for later analyses
#
# No model predictions or feature-level data are required.
# ============================================================

def run_domain_variability_quantitative(
    domain_results_artifact,
    params=None,
):

    params = params or {}

    # --------------------------------------------------
    # Analysis configuration
    # --------------------------------------------------

    scenarios = params.get(
        "scenarios",
        [
            "cross_subject",
            "cross_session",
            "cross_dataset",
        ],
    )

    metrics = params.get(
        "metrics",
        [
            "mmd",
            "energy",
        ],
    )

    comparison = params.get(
        "comparison",
        "source_to_target_elementary",
    )

    # --------------------------------------------------
    # Analysis identity
    # --------------------------------------------------

    effective_params = {
        "domain_results_signature": (
            domain_results_artifact.signature
        ),
        "params": params,
    }

    signature = make_signature(
        effective_params
    )

    # --------------------------------------------------
    # Output paths
    # --------------------------------------------------

    output_dir = (
        OUTPUT_ROOT
        / signature[:12]
    )

    discrepancy_figure_path = (
        output_dir
        / "domain_discrepancy_distribution.png"
    )

    marginal_conditional_figure_path = (
        output_dir
        / "marginal_conditional_discrepancy.png"
    )

    summary_table_path = (
        output_dir
        / "domain_variability_summary.csv"
    )

    paired_table_path = (
        output_dir
        / "marginal_conditional_pairs.csv"
    )

    manifest_path = (
        output_dir
        / "manifest.json"
    )

    # --------------------------------------------------
    # Resume
    # --------------------------------------------------

    outputs_exist = (
        exists(
            discrepancy_figure_path
        )
        and exists(
            marginal_conditional_figure_path
        )
        and exists(
            summary_table_path
        )
        and exists(
            paired_table_path
        )
    )

    if (
        outputs_exist
        and is_done(
            manifest_path,
            effective_params,
        )
    ):

        return AnalysisArtifact(
            name=(
                "domain_variability_quantitative"
            ),

            output_dir=str(
                output_dir
            ),

            tables={
                "domain_variability_summary": str(
                    summary_table_path
                ),

                "marginal_conditional_pairs": str(
                    paired_table_path
                ),
            },

            figures={
                "domain_discrepancy_distribution": str(
                    discrepancy_figure_path
                ),

                "marginal_conditional_discrepancy": str(
                    marginal_conditional_figure_path
                ),
            },

            manifest_path=str(
                manifest_path
            ),

            signature=signature,
        )

    # --------------------------------------------------
    # Start analysis
    # --------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    start = time.time()

    save_manifest(
        make_manifest(
            "running",
            effective_params,
        ),
        manifest_path,
    )

    try:

        # --------------------------------------------------
        # Load canonical domain-results table
        # --------------------------------------------------

        results = pd.read_csv(
            domain_results_artifact.path
        )

        # --------------------------------------------------
        # Select RQ1 comparisons
        # --------------------------------------------------

        results = (
            _filter_domain_results(
                results=results,
                comparison=comparison,
                metrics=metrics,
                scenarios=scenarios,
            )
        )

        if results.empty:
            raise ValueError(
                "No domain results matched the "
                "domain-variability configuration."
            )

        # --------------------------------------------------
        # Marginal / conditional paired table
        # --------------------------------------------------

        paired = (
            _build_marginal_conditional_pairs(
                results
            )
        )

        save_data(
            paired,
            paired_table_path,
        )

        # --------------------------------------------------
        # Numerical summary table
        # --------------------------------------------------

        summary = (
            _build_summary_table(
                results=results,
                paired=paired,
                scenarios=scenarios,
                metrics=metrics,
            )
        )

        save_data(
            summary,
            summary_table_path,
        )

        # --------------------------------------------------
        # Figure:
        # discrepancy distributions
        # --------------------------------------------------

        _plot_discrepancy_distribution(
            results=results,
            scenarios=scenarios,
            metrics=metrics,
            output_path=(
                discrepancy_figure_path
            ),
        )

        # --------------------------------------------------
        # Figure:
        # marginal vs conditional
        # --------------------------------------------------

        _plot_marginal_conditional(
            paired=paired,
            scenarios=scenarios,
            metrics=metrics,
            output_path=(
                marginal_conditional_figure_path
            ),
        )

        # --------------------------------------------------
        # Manifest
        # --------------------------------------------------

        execution_time = (
            time.time()
            - start
        )

        manifest = make_manifest(
            "done",
            effective_params,
            execution_time=execution_time,
        )

        manifest["output"] = {
            "domain_discrepancy_distribution": str(
                discrepancy_figure_path
            ),

            "marginal_conditional_discrepancy": str(
                marginal_conditional_figure_path
            ),

            "domain_variability_summary": str(
                summary_table_path
            ),

            "marginal_conditional_pairs": str(
                paired_table_path
            ),

            "n_domain_rows": len(
                results
            ),

            "n_paired_rows": len(
                paired
            ),
        }

        save_manifest(
            manifest,
            manifest_path,
        )

    except Exception as error:

        execution_time = (
            time.time()
            - start
        )

        save_manifest(
            make_manifest(
                "failed",
                effective_params,
                execution_time=execution_time,
                error=str(error),
            ),
            manifest_path,
        )

        raise

    # --------------------------------------------------
    # Analysis artifact
    # --------------------------------------------------

    return AnalysisArtifact(
        name=(
            "domain_variability_quantitative"
        ),

        output_dir=str(
            output_dir
        ),

        tables={
            "domain_variability_summary": str(
                summary_table_path
            ),

            "marginal_conditional_pairs": str(
                paired_table_path
            ),
        },

        figures={
            "domain_discrepancy_distribution": str(
                discrepancy_figure_path
            ),

            "marginal_conditional_discrepancy": str(
                marginal_conditional_figure_path
            ),
        },

        manifest_path=str(
            manifest_path
        ),

        signature=signature,
    )