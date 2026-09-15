# analysis/paper1/paper_1_analysis.py

import re
import time
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from models.analysis_artifact import AnalysisArtifact
from utils.status import is_done, make_manifest, make_signature
from utils.storage import exists, load_manifest, save_manifest


# ============================================================
# Paths and global configuration
# ============================================================

OUTPUT_ROOT = Path("outputs/analysis/paper1")

PAPER_OUTPUT_DIR = None
FIGURES_DIR = None
TABLES_DIR = None

DEFAULT_PAPER_ANALYSIS_PARAMS = {
    "name": "paper1",
    "collection": "paper1",
    "required_scenarios": [
        "intra_subject",
        "cross_session",
        "cross_subject",
    ],
}

SCENARIO_DISPLAY = {
    "intra_subject": "Intra-subject",
    "cross_session": "Cross-session",
    "cross_subject": "Cross-subject",
}

BAND_DISPLAY = {
    "broad_1_38": "1–38 Hz",
    "delta_1_4": "Delta",
    "theta_4_8": "Theta",
    "alpha_8_12": "Alpha",
    "beta_12_30": "Beta",
    "gamma_30_38": "Gamma",
}

REPRESENTATION_DISPLAY = {
    "statistical": "Statistical",
    "temporal": "Temporal",
    "spectral": "Spectral",
    "nonlinear": "Nonlinear",
    "wavelet": "Wavelet",
    "cov": "Covariance",
    "logcov": "Log-Covariance",
    "eig": "Eigenvalues",
    "csp_4": "CSP-4",
    "csp_6": "CSP-6",
    "csp_8": "CSP-8",
    "rcsp_4": "RCSP-4",
    "rcsp_6": "RCSP-6",
    "rcsp_8": "RCSP-8",
    "riemann": "Riemannian",
    "all_fused": "Fused",
}

MODEL_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "svm": "SVM",
    "random_forest": "Random Forest",
    "eegnet": "EEGNet",
}

DEEP_REPRESENTATION_DISPLAY = {"eegnet": "EEGNet"}
METRICS = ["Balanced Accuracy", "Macro-F1", "AUC"]


# ============================================================
# 1. Collect experiment results
# ============================================================

def _with_default_params(params):
    return {**DEFAULT_PAPER_ANALYSIS_PARAMS, **(params or {})}


def _slug(value):
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()


def _artifact_path(artifact):
    if isinstance(artifact, (str, Path)):
        return Path(artifact)
    if hasattr(artifact, "path"):
        return Path(artifact.path)
    raise TypeError("Expected path-like object or artifact with .path.")


def _artifact_signature(artifact):
    signature = getattr(artifact, "signature", None)
    if signature is not None:
        return signature

    manifest_path = getattr(artifact, "manifest_path", None)
    if manifest_path and exists(manifest_path):
        return load_manifest(manifest_path)["signature"]

    path = _artifact_path(artifact)
    return make_signature({"path": str(path), "mtime": path.stat().st_mtime_ns})


def _artifact_manifest_path(artifact):
    path = getattr(artifact, "manifest_path", None)
    return None if path is None else str(path)


def _artifact_scenarios(artifact):
    values = pd.read_csv(_artifact_path(artifact), usecols=["scenario"])["scenario"]
    return sorted(map(str, values.dropna().unique()))


def _collection_manifest_path(params):
    return OUTPUT_ROOT / "collections" / _slug(params["collection"]) / "manifest.json"


def _register_model_results(model_results_artifact, params):
    manifest_path = _collection_manifest_path(params)

    if exists(manifest_path):
        inputs = dict(load_manifest(manifest_path).get("inputs", {}))
    else:
        inputs = {}

    entry = {
        "path": str(_artifact_path(model_results_artifact)),
        "manifest_path": _artifact_manifest_path(model_results_artifact),
        "signature": _artifact_signature(model_results_artifact),
    }

    for scenario in _artifact_scenarios(model_results_artifact):
        inputs[scenario] = entry

    inputs = {
        scenario: item
        for scenario, item in inputs.items()
        if exists(item["path"])
    }

    required = params["required_scenarios"]
    missing = [scenario for scenario in required if scenario not in inputs]

    effective_params = {
        "analysis": "paper1_collection",
        "collection": params["collection"],
        "required_scenarios": required,
        "inputs": inputs,
    }

    manifest = make_manifest(
        "ready" if not missing else "collecting",
        effective_params,
    )
    manifest["inputs"] = inputs
    manifest["missing_scenarios"] = missing
    save_manifest(manifest, manifest_path)

    return inputs, missing


def _load_registered_results(inputs, required_scenarios):
    frames = []

    for scenario in required_scenarios:
        frame = pd.read_csv(inputs[scenario]["path"])
        frame = frame[frame["scenario"] == scenario].copy()

        if frame.empty:
            raise ValueError(
                f"Registered result for '{scenario}' contains no matching rows."
            )

        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def _set_output_dirs(output_dir):
    global PAPER_OUTPUT_DIR, FIGURES_DIR, TABLES_DIR

    PAPER_OUTPUT_DIR = output_dir
    FIGURES_DIR = output_dir / "figures"
    TABLES_DIR = output_dir / "tables"

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Normalize result schema
# ============================================================

def _first_target_domain(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value.split(";")[0] if value else None


def _domain_parts(value):
    domain = _first_target_domain(value)
    return [] if domain is None else domain.split("|")


def _dataset_from_row(row):
    parts = _domain_parts(row["target_domains"])
    return parts[0] if parts else None


def _subject_from_row(row):
    parts = _domain_parts(row["target_domains"])
    return parts[1] if len(parts) >= 2 else None


def _session_from_row(row):
    parts = _domain_parts(row["target_domains"])
    return parts[2] if len(parts) >= 3 else None


def _band_from_row(row):
    label = row.get("preprocessing_config_label")
    if pd.isna(label):
        return None

    label = str(label)
    marker = f"_{row['scenario']}_"

    if marker in label:
        band = label.split(marker, 1)[1]
    else:
        band = next((name for name in BAND_DISPLAY if name in label), label)

    return BAND_DISPLAY.get(band, band)


def _model_from_row(row):
    learning, model = str(row["learning_method"]), str(row["model_name"])

    mlp_names = {
        "neural_erm__mlp_small": "MLP Small",
        "neural_erm__mlp_medium": "MLP Medium",
        "neural_erm__mlp_large": "MLP Large",
    }

    if learning in mlp_names:
        return mlp_names[learning]

    return MODEL_DISPLAY.get(model, model.replace("_", " ").title())


def normalize_results(results):
    required = [
        "scenario", "target_domains", "preprocessing_config_label",
        "feature_extraction_config_label", "signal_transform_config_label",
        "model_input_representation", "learning_method", "model_name",
        "evaluation_group", "partition", "accuracy", "balanced_accuracy",
        "macro_f1", "auc",
    ]

    missing = [column for column in required if column not in results.columns]
    if missing:
        raise ValueError(f"Missing required result columns: {missing}")

    df = results.copy()

    df["Dataset"] = df.apply(_dataset_from_row, axis=1)
    df["Scenario"] = df["scenario"].map(SCENARIO_DISPLAY).fillna(df["scenario"])
    df["Subject"] = df.apply(_subject_from_row, axis=1)
    df["Session"] = df.apply(_session_from_row, axis=1)
    df["Band"] = df.apply(_band_from_row, axis=1)
    df["Representation"] = df.apply(_representation_from_row, axis=1)
    df["Model"] = df.apply(_model_from_row, axis=1)

    df["Accuracy"] = df["accuracy"]
    df["Balanced Accuracy"] = df["balanced_accuracy"]
    df["Macro-F1"] = df["macro_f1"]
    df["AUC"] = df["auc"]
    df["Evaluation Group"] = df["evaluation_group"]
    df["Partition"] = df["partition"]

    canonical = [
        "Dataset", "Scenario", "Subject", "Session", "Band", "Representation",
        "Model", "Evaluation Group", "Partition", "Accuracy",
        "Balanced Accuracy", "Macro-F1", "AUC",
    ]
    remaining = [column for column in df.columns if column not in canonical]

    return df[canonical + remaining]


# ============================================================
# 3. Representation mapping
# ============================================================

def _representation_from_row(row):
    feature = row.get("feature_extraction_config_label")

    if pd.notna(feature) and str(feature).strip():
        feature = str(feature)
        return REPRESENTATION_DISPLAY.get(
            feature,
            feature.replace("_", " ").title(),
        )

    if row.get("model_input_representation") == "signal":
        model = str(row["model_name"])
        return DEEP_REPRESENTATION_DISPLAY.get(
            model,
            MODEL_DISPLAY.get(model, model.replace("_", " ").title()),
        )

    signal = row.get("signal_transform_config_label")
    if pd.notna(signal) and str(signal).strip():
        return str(signal).replace("_", " ").title()

    raise ValueError(
        "Could not determine representation for result row: "
        f"model={row.get('model_name')}, feature={feature}."
    )


# ============================================================
# 4. Representation generalization
# ============================================================

def _target_test_results(results):
    df = results[
        (results["Evaluation Group"] == "target_elementary_domain")
        & (results["Partition"] == "test")
    ].copy()

    if df.empty:
        raise ValueError("No target test results were found.")

    return df


def _representation_order(dataframe):
    preferred = list(dict.fromkeys(
        list(REPRESENTATION_DISPLAY.values())
        + list(DEEP_REPRESENTATION_DISPLAY.values())
    ))
    available = dataframe["Representation"].dropna().unique().tolist()

    return (
        [x for x in preferred if x in available]
        + sorted(x for x in available if x not in preferred)
    )


def build_representation_subject_results(results):
    df = _target_test_results(results)

    return (
        df.groupby(
            ["Dataset", "Scenario", "Subject", "Representation"],
            dropna=False,
            as_index=False,
        )[METRICS]
        .mean()
    )


def build_representation_summary(subject_results):
    grouped = (
        subject_results
        .groupby(["Dataset", "Representation", "Scenario"], dropna=False)
        .agg(
            ba_mean=("Balanced Accuracy", "mean"),
            ba_std=("Balanced Accuracy", "std"),
            f1_mean=("Macro-F1", "mean"),
            f1_std=("Macro-F1", "std"),
            auc_mean=("AUC", "mean"),
            auc_std=("AUC", "std"),
            subjects=("Subject", "nunique"),
        )
        .reset_index()
    )

    rows = []

    for (dataset, representation), group in grouped.groupby(
        ["Dataset", "Representation"], dropna=False
    ):
        row = {"Dataset": dataset, "Representation": representation}

        for scenario in SCENARIO_DISPLAY.values():
            current = group[group["Scenario"] == scenario]

            if current.empty:
                for metric in ["BA", "F1", "AUC"]:
                    row[f"{scenario} {metric} Mean"] = np.nan
                    row[f"{scenario} {metric} Std"] = np.nan
                continue

            values = current.iloc[0]

            row[f"{scenario} BA Mean"] = values["ba_mean"]
            row[f"{scenario} BA Std"] = values["ba_std"]
            row[f"{scenario} F1 Mean"] = values["f1_mean"]
            row[f"{scenario} F1 Std"] = values["f1_std"]
            row[f"{scenario} AUC Mean"] = values["auc_mean"]
            row[f"{scenario} AUC Std"] = values["auc_std"]

        intra = row.get("Intra-subject BA Mean")
        cs = row.get("Cross-session BA Mean")
        csub = row.get("Cross-subject BA Mean")

        row["Gap CS"] = (
            intra - cs if pd.notna(intra) and pd.notna(cs) else np.nan
        )
        row["Gap CSub"] = (
            intra - csub if pd.notna(intra) and pd.notna(csub) else np.nan
        )

        rows.append(row)

    return pd.DataFrame(rows)


def plot_representation_generalization(subject_results):
    datasets = sorted(subject_results["Dataset"].dropna().unique())
    scenarios = [
        x for x in SCENARIO_DISPLAY.values()
        if x in subject_results["Scenario"].unique()
    ]
    representations = _representation_order(subject_results)

    fig, axes = plt.subplots(
        len(datasets),
        len(scenarios),
        figsize=(5.2 * len(scenarios), 3.8 * len(datasets)),
        sharey=True,
        squeeze=False,
    )

    rng = np.random.default_rng(0)

    for i, dataset in enumerate(datasets):
        for j, scenario in enumerate(scenarios):
            ax = axes[i, j]

            subset = subject_results[
                (subject_results["Dataset"] == dataset)
                & (subject_results["Scenario"] == scenario)
            ]

            active = [
                rep for rep in representations
                if rep in subset["Representation"].values
            ]

            values = [
                subset.loc[
                    subset["Representation"] == rep,
                    "Balanced Accuracy",
                ].dropna().to_numpy()
                for rep in active
            ]

            positions = np.arange(1, len(active) + 1)

            if values:
                violin_values = [
                    x if len(x) > 1 else np.repeat(x, 2)
                    for x in values
                ]

                ax.violinplot(
                    violin_values,
                    positions=positions,
                    showmeans=False,
                    showmedians=True,
                    showextrema=False,
                )

                for position, value in zip(positions, values):
                    ax.scatter(
                        position + rng.normal(0, 0.035, len(value)),
                        value,
                        s=16,
                        alpha=0.65,
                    )

            ax.set_xticks(positions)
            ax.set_xticklabels(active, rotation=45, ha="right")
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.25)

            if i == 0:
                ax.set_title(scenario)
            if j == 0:
                ax.set_ylabel(f"{dataset}\nBalanced Accuracy")

    fig.tight_layout()

    for ext in ["pdf", "png"]:
        fig.savefig(
            FIGURES_DIR / f"representation_generalization.{ext}",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)


# ============================================================
# 5. Best configurations
# ============================================================

def _configuration_columns(dataframe):
    columns = ["Dataset", "Scenario", "Representation", "Model", "Band"]
    optional = [
        "preprocessing_config_label",
        "feature_selection_config_label",
        "signal_transform_config_label",
    ]
    return columns + [
        column for column in optional
        if column in dataframe.columns
    ]


def build_configuration_subject_results(results):
    df = _target_test_results(results)
    configuration = _configuration_columns(df)

    return (
        df.groupby(
            configuration + ["Subject"],
            dropna=False,
            as_index=False,
        )[METRICS]
        .mean()
    )


def select_best_configurations(results):
    subject_results = build_configuration_subject_results(results)
    configuration = _configuration_columns(subject_results)

    summary = (
        subject_results
        .groupby(configuration, dropna=False)
        .agg(
            ba_mean=("Balanced Accuracy", "mean"),
            ba_std=("Balanced Accuracy", "std"),
            f1_mean=("Macro-F1", "mean"),
            f1_std=("Macro-F1", "std"),
            auc_mean=("AUC", "mean"),
            auc_std=("AUC", "std"),
            subjects=("Subject", "nunique"),
        )
        .reset_index()
    )

    best = (
        summary
        .sort_values(
            ["Dataset", "Scenario", "ba_mean", "f1_mean", "auc_mean"],
            ascending=[True, True, False, False, False],
            na_position="last",
        )
        .groupby(["Dataset", "Scenario"], as_index=False, group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )

    display = best[
        [
            "Dataset", "Scenario", "Representation", "Model", "Band",
            "ba_mean", "ba_std", "f1_mean", "f1_std",
            "auc_mean", "auc_std", "subjects",
        ]
    ].copy()

    display = display.rename(columns={
        "Band": "Preprocessing",
        "ba_mean": "BA Mean",
        "ba_std": "BA Std",
        "f1_mean": "F1 Mean",
        "f1_std": "F1 Std",
        "auc_mean": "AUC Mean",
        "auc_std": "AUC Std",
        "subjects": "Subjects",
    })

    selected_subjects = subject_results.merge(
        best[configuration],
        on=configuration,
        how="inner",
    )

    return display, selected_subjects


# ============================================================
# 6. Subject-level performance of best configuration
# ============================================================

def _subject_sort_key(value):
    text = str(value)

    try:
        return 0, float(text)
    except ValueError:
        return 1, text


def plot_best_configuration_subjects(subject_results):
    datasets = sorted(subject_results["Dataset"].dropna().unique())
    scenarios = [
        x for x in SCENARIO_DISPLAY.values()
        if x in subject_results["Scenario"].unique()
    ]

    fig, axes = plt.subplots(
        len(datasets),
        len(scenarios),
        figsize=(5.2 * len(scenarios), 3.6 * len(datasets)),
        sharey=True,
        squeeze=False,
    )

    for i, dataset in enumerate(datasets):
        for j, scenario in enumerate(scenarios):
            ax = axes[i, j]

            subset = subject_results[
                (subject_results["Dataset"] == dataset)
                & (subject_results["Scenario"] == scenario)
            ].copy()

            subset = subset.sort_values(
                "Subject",
                key=lambda x: x.map(_subject_sort_key),
            )

            if not subset.empty:
                x = np.arange(len(subset))
                y = subset["Balanced Accuracy"].to_numpy()
                config = subset.iloc[0]

                ax.plot(x, y, marker="o")
                ax.axhline(np.nanmean(y), linestyle="--", linewidth=1.2)

                ax.set_xticks(x)
                ax.set_xticklabels(
                    subset["Subject"].astype(str),
                    rotation=45,
                    ha="right",
                )

                ax.text(
                    0.02,
                    0.04,
                    f"{config['Representation']} | {config['Model']} | {config['Band']}",
                    transform=ax.transAxes,
                    fontsize=8,
                    va="bottom",
                )

            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.25)
            ax.set_xlabel("Subject")

            if i == 0:
                ax.set_title(scenario)
            if j == 0:
                ax.set_ylabel(f"{dataset}\nBalanced Accuracy")

    fig.tight_layout()

    for ext in ["pdf", "png"]:
        fig.savefig(
            FIGURES_DIR / f"best_configuration_subjects.{ext}",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)


# ============================================================
# 7. Subject variability and representation ranking
# ============================================================

def build_subject_rankings(representation_subjects):
    rankings = representation_subjects.copy()

    rankings["Rank"] = (
        rankings
        .groupby(
            ["Dataset", "Scenario", "Subject"]
        )["Balanced Accuracy"]
        .rank(method="min", ascending=False)
    )

    return rankings


def plot_subject_rankings(rankings):
    datasets = sorted(rankings["Dataset"].dropna().unique())
    scenarios = [
        x for x in SCENARIO_DISPLAY.values()
        if x in rankings["Scenario"].unique()
    ]
    representations = _representation_order(rankings)

    fig, axes = plt.subplots(
        len(datasets),
        len(scenarios),
        figsize=(5.2 * len(scenarios), 4.5 * len(datasets)),
        squeeze=False,
    )

    image = None

    for i, dataset in enumerate(datasets):
        for j, scenario in enumerate(scenarios):
            ax = axes[i, j]

            subset = rankings[
                (rankings["Dataset"] == dataset)
                & (rankings["Scenario"] == scenario)
            ]

            subjects = sorted(
                subset["Subject"].dropna().unique(),
                key=_subject_sort_key,
            )

            active = [
                rep for rep in representations
                if rep in subset["Representation"].values
            ]

            matrix = (
                subset
                .pivot_table(
                    index="Representation",
                    columns="Subject",
                    values="Rank",
                    aggfunc="mean",
                )
                .reindex(index=active, columns=subjects)
            )

            image = ax.imshow(
                matrix.to_numpy(dtype=float),
                aspect="auto",
                interpolation="nearest",
            )

            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    value = matrix.iloc[row, col]

                    if pd.notna(value):
                        ax.text(
                            col,
                            row,
                            f"{value:.0f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                        )

            ax.set_xticks(np.arange(len(subjects)))
            ax.set_xticklabels(subjects, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(active)))
            ax.set_yticklabels(active)
            ax.set_xlabel("Subject")

            if i == 0:
                ax.set_title(scenario)
            if j == 0:
                ax.set_ylabel(dataset)

    if image is not None:
        fig.colorbar(
            image,
            ax=axes,
            label="Representation rank",
            fraction=0.02,
            pad=0.02,
        )

    fig.subplots_adjust(
        left=0.12,
        right=0.90,
        bottom=0.10,
        top=0.93,
        hspace=0.35,
        wspace=0.30,
    )

    for ext in ["pdf", "png"]:
        fig.savefig(
            FIGURES_DIR / f"subject_ranking.{ext}",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)


# ============================================================
# 8. Subject ranking summary
# ============================================================

def build_subject_ranking_summary(rankings):
    summary = (
        rankings
        .groupby(
            ["Dataset", "Scenario", "Representation"],
            dropna=False,
        )
        .agg(
            mean_rank=("Rank", "mean"),
            rank_std=("Rank", "std"),
            subjects=("Subject", "nunique"),
            best_count=("Rank", lambda x: (x == 1).sum()),
            top3_count=("Rank", lambda x: (x <= 3).sum()),
        )
        .reset_index()
    )

    summary["Best (%)"] = 100 * summary["best_count"] / summary["subjects"]
    summary["Top-3 (%)"] = 100 * summary["top3_count"] / summary["subjects"]

    summary = summary.rename(columns={
        "mean_rank": "Mean Rank",
        "rank_std": "Rank Variability",
        "subjects": "Subjects",
    })

    return summary[
        [
            "Dataset", "Scenario", "Representation", "Mean Rank",
            "Rank Variability", "Best (%)", "Top-3 (%)", "Subjects",
        ]
    ]


# ============================================================
# 9. Configuration robustness
# ============================================================

def build_configuration_robustness(results):
    subject_results = build_configuration_subject_results(results)

    baseline = (
        subject_results
        .groupby(
            ["Dataset", "Scenario", "Representation"],
            dropna=False,
        )["Balanced Accuracy"]
        .mean()
        .rename("Representation Mean")
        .reset_index()
    )

    subject_results = subject_results.merge(
        baseline,
        on=["Dataset", "Scenario", "Representation"],
        how="left",
    )

    subject_results["Delta BA"] = (
        subject_results["Balanced Accuracy"]
        - subject_results["Representation Mean"]
    )

    model = (
        subject_results
        .groupby(
            ["Dataset", "Scenario", "Representation", "Model"],
            dropna=False,
        )["Delta BA"]
        .mean()
        .reset_index()
    )

    band = (
        subject_results
        .groupby(
            ["Dataset", "Scenario", "Representation", "Band"],
            dropna=False,
        )["Delta BA"]
        .mean()
        .reset_index()
    )

    return model, band


def _plot_delta_heatmap(ax, dataframe, column, representations, title):
    values = sorted(dataframe[column].dropna().unique())

    matrix = (
        dataframe
        .pivot_table(
            index="Representation",
            columns=column,
            values="Delta BA",
            aggfunc="mean",
        )
        .reindex(index=representations, columns=values)
    )

    image = ax.imshow(
        matrix.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        vmin=-0.15,
        vmax=0.15,
    )

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.iloc[row, col]

            if pd.notna(value):
                ax.text(
                    col,
                    row,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(values, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(representations)))
    ax.set_yticklabels(representations)
    ax.set_title(title)

    return image


def plot_configuration_robustness(model_robustness, band_robustness):
    datasets = sorted(
        set(model_robustness["Dataset"].dropna())
        | set(band_robustness["Dataset"].dropna())
    )

    scenarios = [
        x for x in SCENARIO_DISPLAY.values()
        if x in model_robustness["Scenario"].values
        or x in band_robustness["Scenario"].values
    ]

    representations = _representation_order(pd.concat(
        [
            model_robustness[["Representation"]],
            band_robustness[["Representation"]],
        ],
        ignore_index=True,
    ))

    fig, axes = plt.subplots(
        len(datasets) * 2,
        len(scenarios),
        figsize=(5.4 * len(scenarios), 8.4 * len(datasets)),
        squeeze=False,
    )

    image = None

    for dataset_index, dataset in enumerate(datasets):
        model_row = dataset_index * 2
        band_row = model_row + 1

        for scenario_index, scenario in enumerate(scenarios):
            model_ax = axes[model_row, scenario_index]
            band_ax = axes[band_row, scenario_index]

            model_subset = model_robustness[
                (model_robustness["Dataset"] == dataset)
                & (model_robustness["Scenario"] == scenario)
            ]

            band_subset = band_robustness[
                (band_robustness["Dataset"] == dataset)
                & (band_robustness["Scenario"] == scenario)
            ]

            model_reps = [
                rep for rep in representations
                if rep in model_subset["Representation"].values
            ]

            band_reps = [
                rep for rep in representations
                if rep in band_subset["Representation"].values
            ]

            if not model_subset.empty:
                image = _plot_delta_heatmap(
                    model_ax,
                    model_subset,
                    "Model",
                    model_reps,
                    f"{scenario} — Model",
                )

            if not band_subset.empty:
                image = _plot_delta_heatmap(
                    band_ax,
                    band_subset,
                    "Band",
                    band_reps,
                    f"{scenario} — Band",
                )

            if scenario_index == 0:
                model_ax.set_ylabel(f"{dataset}\nRepresentation")
                band_ax.set_ylabel(f"{dataset}\nRepresentation")

    if image is not None:
        fig.colorbar(
            image,
            ax=axes,
            label="Δ Balanced Accuracy",
            fraction=0.015,
            pad=0.02,
        )

    fig.subplots_adjust(
        left=0.12,
        right=0.90,
        bottom=0.08,
        top=0.95,
        hspace=0.45,
        wspace=0.30,
    )

    for ext in ["pdf", "png"]:
        fig.savefig(
            FIGURES_DIR / f"configuration_robustness.{ext}",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)


# ============================================================
# 10. Statistical analysis
# ============================================================

def _holm_correction(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0

    for rank, index in enumerate(order):
        value = min((len(p_values) - rank) * p_values[index], 1.0)
        running_max = max(running_max, value)
        adjusted[index] = running_max

    return adjusted


def _rank_biserial(x, y):
    differences = np.asarray(x) - np.asarray(y)
    differences = differences[differences != 0]

    if len(differences) == 0:
        return 0.0

    ranks = stats.rankdata(np.abs(differences))
    positive = ranks[differences > 0].sum()
    negative = ranks[differences < 0].sum()

    return (positive - negative) / (positive + negative)


def run_friedman_tests(representation_subjects):
    rows = []

    for (dataset, scenario), group in representation_subjects.groupby(
        ["Dataset", "Scenario"]
    ):
        pivot = group.pivot_table(
            index="Subject",
            columns="Representation",
            values="Balanced Accuracy",
            aggfunc="mean",
        ).dropna()

        if pivot.shape[0] < 2 or pivot.shape[1] < 3:
            continue

        statistic, p_value = stats.friedmanchisquare(
            *[pivot[column].to_numpy() for column in pivot.columns]
        )

        rows.append({
            "Dataset": dataset,
            "Scenario": scenario,
            "Subjects": len(pivot),
            "Representations": pivot.shape[1],
            "Statistic": statistic,
            "P-value": p_value,
        })

    return pd.DataFrame(rows)


def run_pairwise_wilcoxon(representation_subjects):
    rows = []

    for (dataset, scenario), group in representation_subjects.groupby(
        ["Dataset", "Scenario"]
    ):
        comparisons = []

        for rep_a, rep_b in combinations(_representation_order(group), 2):
            pair = (
                group[group["Representation"].isin([rep_a, rep_b])]
                .pivot_table(
                    index="Subject",
                    columns="Representation",
                    values="Balanced Accuracy",
                    aggfunc="mean",
                )
            )

            if rep_a not in pair or rep_b not in pair:
                continue

            pair = pair[[rep_a, rep_b]].dropna()

            if len(pair) < 2:
                continue

            x = pair[rep_a].to_numpy()
            y = pair[rep_b].to_numpy()

            try:
                statistic, p_value = stats.wilcoxon(
                    x,
                    y,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
            except ValueError:
                statistic, p_value = 0.0, 1.0

            comparisons.append({
                "Dataset": dataset,
                "Scenario": scenario,
                "Representation A": rep_a,
                "Representation B": rep_b,
                "Subjects": len(pair),
                "Mean A": np.mean(x),
                "Mean B": np.mean(y),
                "Mean Difference": np.mean(x - y),
                "Statistic": statistic,
                "P-value": p_value,
                "Rank-Biserial": _rank_biserial(x, y),
            })

        if comparisons:
            adjusted = _holm_correction(
                [item["P-value"] for item in comparisons]
            )

            for comparison, p_adjusted in zip(comparisons, adjusted):
                comparison["P-value Holm"] = p_adjusted
                comparison["Significant"] = p_adjusted < 0.05
                rows.append(comparison)

    return pd.DataFrame(rows)


def run_statistical_analysis(representation_subjects):
    return (
        run_friedman_tests(representation_subjects),
        run_pairwise_wilcoxon(representation_subjects),
    )


# ============================================================
# 11. Save figures
# ============================================================
# Figures are saved directly by the plotting functions.


# ============================================================
# 12. Save numerical tables
# ============================================================
# CSV files are saved directly by the pipeline runner.


# ============================================================
# 13. Generate LaTeX tables
# ============================================================

def _latex_escape(value):
    if pd.isna(value):
        return "-"

    text = str(value)

    for old, new in {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }.items():
        text = text.replace(old, new)

    return text


def _mean_std(mean, std):
    if pd.isna(mean):
        return "-"
    return f"{mean:.3f}" if pd.isna(std) else f"{mean:.3f} $\\pm$ {std:.3f}"


def generate_representation_summary_latex(summary):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Overall performance of the evaluated EEG representations.}",
        r"\label{tab:representation_summary}",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Representation} &",
        r"\multicolumn{2}{c}{\textbf{Intra-subject}} &",
        r"\multicolumn{2}{c}{\textbf{Cross-session}} &",
        r"\multicolumn{2}{c}{\textbf{Cross-subject}} &",
        r"\textbf{Gap CS} & \textbf{Gap CSub} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"& & \textbf{BA} & \textbf{F1} & \textbf{BA} & \textbf{F1} &",
        r"\textbf{BA} & \textbf{F1} & & \\",
        r"\midrule",
    ]

    for dataset, group in summary.groupby("Dataset", sort=True):
        first = True

        for _, row in group.iterrows():
            values = [
                _latex_escape(dataset) if first else "",
                _latex_escape(row["Representation"]),
                _mean_std(
                    row.get("Intra-subject BA Mean"),
                    row.get("Intra-subject BA Std"),
                ),
                _mean_std(
                    row.get("Intra-subject F1 Mean"),
                    row.get("Intra-subject F1 Std"),
                ),
                _mean_std(
                    row.get("Cross-session BA Mean"),
                    row.get("Cross-session BA Std"),
                ),
                _mean_std(
                    row.get("Cross-session F1 Mean"),
                    row.get("Cross-session F1 Std"),
                ),
                _mean_std(
                    row.get("Cross-subject BA Mean"),
                    row.get("Cross-subject BA Std"),
                ),
                _mean_std(
                    row.get("Cross-subject F1 Mean"),
                    row.get("Cross-subject F1 Std"),
                ),
                "-" if pd.isna(row.get("Gap CS")) else f"{row['Gap CS']:.3f}",
                "-" if pd.isna(row.get("Gap CSub")) else f"{row['Gap CSub']:.3f}",
            ]

            first = False
            lines.append(" & ".join(values) + r" \\")

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]

    (TABLES_DIR / "representation_summary.tex").write_text("\n".join(lines))


def generate_best_configurations_latex(best):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Best observed configuration for each dataset and generalization regime.}",
        r"\label{tab:best_configurations}",
        r"\begin{tabular}{lllllcc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Regime} & \textbf{Representation} &",
        r"\textbf{Classifier} & \textbf{Preprocessing} & \textbf{BA} & \textbf{F1} \\",
        r"\midrule",
    ]

    for dataset, group in best.groupby("Dataset", sort=True):
        first = True

        for _, row in group.iterrows():
            values = [
                _latex_escape(dataset) if first else "",
                _latex_escape(row["Scenario"]),
                _latex_escape(row["Representation"]),
                _latex_escape(row["Model"]),
                _latex_escape(row["Preprocessing"]),
                _mean_std(row["BA Mean"], row["BA Std"]),
                _mean_std(row["F1 Mean"], row["F1 Std"]),
            ]

            first = False
            lines.append(" & ".join(values) + r" \\")

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]

    (TABLES_DIR / "best_configurations.tex").write_text("\n".join(lines))


def generate_subject_ranking_latex(summary):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Summary of subject-specific representation rankings.}",
        r"\label{tab:subject_ranking_summary}",
        r"\begin{tabular}{lllcccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Regime} & \textbf{Representation} &",
        r"\textbf{Mean Rank} & \textbf{Rank Variability} &",
        r"\textbf{Best (\%)} & \textbf{Top-3 (\%)} \\",
        r"\midrule",
    ]

    for dataset, group in summary.groupby("Dataset", sort=True):
        first = True

        for _, row in group.iterrows():
            values = [
                _latex_escape(dataset) if first else "",
                _latex_escape(row["Scenario"]),
                _latex_escape(row["Representation"]),
                f"{row['Mean Rank']:.2f}",
                f"{row['Rank Variability']:.2f}",
                f"{row['Best (%)']:.1f}",
                f"{row['Top-3 (%)']:.1f}",
            ]

            first = False
            lines.append(" & ".join(values) + r" \\")

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]

    (TABLES_DIR / "subject_ranking_summary.tex").write_text("\n".join(lines))


def generate_statistics_latex(friedman_results):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Friedman tests for differences among EEG representations.}",
        r"\label{tab:friedman_tests}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Regime} & "
        r"\textbf{$\chi^2_F$} & \textbf{$p$} & \textbf{Subjects} \\",
        r"\midrule",
    ]

    for _, row in friedman_results.iterrows():
        p = row["P-value"]
        p_text = r"$<0.001$" if p < 0.001 else f"{p:.3f}"

        lines.append(" & ".join([
            _latex_escape(row["Dataset"]),
            _latex_escape(row["Scenario"]),
            f"{row['Statistic']:.2f}",
            p_text,
            str(int(row["Subjects"])),
        ]) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    (TABLES_DIR / "friedman_tests.tex").write_text("\n".join(lines))


def generate_latex_tables(
    representation_summary,
    best_configurations,
    ranking_summary,
):
    generate_representation_summary_latex(representation_summary)
    generate_best_configurations_latex(best_configurations)
    generate_subject_ranking_latex(ranking_summary)


# ============================================================
# 14. Generate LaTeX figure snippets
# ============================================================

def generate_figure_snippets():
    figures = [
        (
            "representation_generalization",
            "Distribution of Balanced Accuracy for the evaluated EEG "
            "representations under the considered generalization regimes.",
            "fig:representation_generalization",
        ),
        (
            "best_configuration_subjects",
            "Subject-level Balanced Accuracy obtained by the best observed "
            "configuration for each dataset and evaluation regime.",
            "fig:best_configuration_subjects",
        ),
        (
            "subject_ranking",
            "Subject-specific ranking of EEG representations across datasets "
            "and generalization regimes.",
            "fig:subject_ranking",
        ),
        (
            "configuration_robustness",
            "Sensitivity of EEG representations to classifier and preprocessing choices.",
            "fig:configuration_robustness",
        ),
    ]

    blocks = []

    for filename, caption, label in figures:
        blocks += [
            r"\begin{figure*}[t]",
            r"    \centering",
            rf"    \includegraphics[width=\textwidth]{{{filename}.pdf}}",
            rf"    \caption{{{caption}}}",
            rf"    \label{{{label}}}",
            r"\end{figure*}",
            "",
        ]

    (PAPER_OUTPUT_DIR / "figure_snippets.tex").write_text("\n".join(blocks))


# ============================================================
# 15. Analysis validation and coverage
# ============================================================

def build_analysis_coverage(results):
    df = _target_test_results(results)

    return (
        df.groupby(
            ["Dataset", "Scenario", "Representation", "Model", "Band"],
            dropna=False,
        )
        .agg(
            Subjects=("Subject", "nunique"),
            Runs=("Balanced Accuracy", "size"),
        )
        .reset_index()
    )


def build_dataset_summary(results):
    df = _target_test_results(results)

    return (
        df.groupby(["Dataset", "Scenario"], dropna=False)
        .agg(
            Subjects=("Subject", "nunique"),
            Representations=("Representation", "nunique"),
            Models=("Model", "nunique"),
            Bands=("Band", "nunique"),
            Runs=("Balanced Accuracy", "size"),
        )
        .reset_index()
    )


def validate_analysis_results(results):
    required = [
        "Dataset", "Scenario", "Subject", "Band",
        "Representation", "Model", "Balanced Accuracy", "Macro-F1",
    ]

    missing = [
        column for column in required
        if column not in results.columns
    ]

    if missing:
        raise ValueError(f"Missing canonical columns: {missing}")

    missing_ba = results["Balanced Accuracy"].isna().sum()

    if missing_ba:
        print(
            f"[Paper 1] Warning: {missing_ba} rows have missing Balanced Accuracy."
        )

    available = set(results["Scenario"].dropna().unique())

    for scenario in SCENARIO_DISPLAY.values():
        if scenario not in available:
            print(
                f"[Paper 1] Warning: scenario '{scenario}' is not currently available."
            )

    for dataset in sorted(results["Dataset"].dropna().unique()):
        subset = results[results["Dataset"] == dataset]

        print(
            f"[Paper 1] {dataset}: "
            f"{subset['Subject'].nunique()} subjects, "
            f"{subset['Representation'].nunique()} representations, "
            f"{subset['Model'].nunique()} models, "
            f"{subset['Band'].nunique()} bands."
        )


def save_analysis_coverage(results):
    coverage = build_analysis_coverage(results)
    summary = build_dataset_summary(results)

    coverage.to_csv(
        TABLES_DIR / "analysis_coverage.csv",
        index=False,
    )

    summary.to_csv(
        TABLES_DIR / "dataset_summary.csv",
        index=False,
    )

    return coverage, summary


# ============================================================
# 16. Pipeline runner
# ============================================================

def _output_paths():
    tables = {
        name: TABLES_DIR / name
        for name in [
            "analysis_coverage.csv",
            "dataset_summary.csv",
            "representation_subject_results.csv",
            "representation_summary.csv",
            "best_configurations.csv",
            "best_configuration_subjects.csv",
            "subject_rankings.csv",
            "subject_ranking_summary.csv",
            "model_robustness.csv",
            "band_robustness.csv",
            "friedman_tests.csv",
            "pairwise_wilcoxon.csv",
            "representation_summary.tex",
            "best_configurations.tex",
            "subject_ranking_summary.tex",
            "friedman_tests.tex",
        ]
    }

    figures = {
        f"{name}_{ext}": FIGURES_DIR / f"{name}.{ext}"
        for name in [
            "representation_generalization",
            "best_configuration_subjects",
            "subject_ranking",
            "configuration_robustness",
        ]
        for ext in ["pdf", "png"]
    }

    return tables, figures


def run_paper1_analysis(model_results_artifact, params=None):
    params = _with_default_params(params)

    inputs, missing = _register_model_results(
        model_results_artifact,
        params,
    )

    if missing:
        print(
            "[Paper 1] Waiting for scenarios: "
            + ", ".join(missing)
        )
        return None

    effective_params = {
        "analysis": "paper1",
        "params": params,
        "inputs": {
            scenario: {
                "path": inputs[scenario]["path"],
                "signature": inputs[scenario]["signature"],
            }
            for scenario in params["required_scenarios"]
        },
    }

    signature = make_signature(effective_params)

    output_dir = (
        OUTPUT_ROOT
        / "runs"
        / _slug(params["collection"])
        / signature[:12]
    )

    manifest_path = output_dir / "manifest.json"

    _set_output_dirs(output_dir)

    table_paths, figure_paths = _output_paths()
    snippet_path = PAPER_OUTPUT_DIR / "figure_snippets.tex"

    expected = [
        *table_paths.values(),
        *figure_paths.values(),
        snippet_path,
    ]

    if (
        all(exists(path) for path in expected)
        and is_done(manifest_path, effective_params)
    ):
        return AnalysisArtifact(
            name=params["name"],
            output_dir=str(output_dir),
            tables={
                name: str(path)
                for name, path in table_paths.items()
            },
            figures={
                name: str(path)
                for name, path in figure_paths.items()
            },
            manifest_path=str(manifest_path),
            signature=signature,
        )

    start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    save_manifest(
        make_manifest("running", effective_params),
        manifest_path,
    )

    try:
        results = normalize_results(
            _load_registered_results(
                inputs,
                params["required_scenarios"],
            )
        )

        validate_analysis_results(results)
        save_analysis_coverage(results)

        representation_subjects = (
            build_representation_subject_results(results)
        )

        representation_summary = (
            build_representation_summary(
                representation_subjects
            )
        )

        representation_subjects.to_csv(
            TABLES_DIR / "representation_subject_results.csv",
            index=False,
        )

        representation_summary.to_csv(
            TABLES_DIR / "representation_summary.csv",
            index=False,
        )

        plot_representation_generalization(
            representation_subjects
        )

        best_configurations, best_configuration_subjects = (
            select_best_configurations(results)
        )

        best_configurations.to_csv(
            TABLES_DIR / "best_configurations.csv",
            index=False,
        )

        best_configuration_subjects.to_csv(
            TABLES_DIR / "best_configuration_subjects.csv",
            index=False,
        )

        plot_best_configuration_subjects(
            best_configuration_subjects
        )

        subject_rankings = build_subject_rankings(
            representation_subjects
        )

        ranking_summary = build_subject_ranking_summary(
            subject_rankings
        )

        subject_rankings.to_csv(
            TABLES_DIR / "subject_rankings.csv",
            index=False,
        )

        ranking_summary.to_csv(
            TABLES_DIR / "subject_ranking_summary.csv",
            index=False,
        )

        plot_subject_rankings(subject_rankings)

        model_robustness, band_robustness = (
            build_configuration_robustness(results)
        )

        model_robustness.to_csv(
            TABLES_DIR / "model_robustness.csv",
            index=False,
        )

        band_robustness.to_csv(
            TABLES_DIR / "band_robustness.csv",
            index=False,
        )

        plot_configuration_robustness(
            model_robustness,
            band_robustness,
        )

        friedman_results, pairwise_results = (
            run_statistical_analysis(
                representation_subjects
            )
        )

        friedman_results.to_csv(
            TABLES_DIR / "friedman_tests.csv",
            index=False,
        )

        pairwise_results.to_csv(
            TABLES_DIR / "pairwise_wilcoxon.csv",
            index=False,
        )

        generate_latex_tables(
            representation_summary,
            best_configurations,
            ranking_summary,
        )

        generate_statistics_latex(
            friedman_results
        )

        generate_figure_snippets()

        manifest = make_manifest(
            "done",
            effective_params,
            execution_time=time.time() - start,
        )

        manifest["output"] = {
            "output_dir": str(output_dir),
            "tables": {
                name: str(path)
                for name, path in table_paths.items()
            },
            "figures": {
                name: str(path)
                for name, path in figure_paths.items()
            },
            "figure_snippets": str(snippet_path),
        }

        save_manifest(manifest, manifest_path)

    except Exception as error:
        save_manifest(
            make_manifest(
                "failed",
                effective_params,
                execution_time=time.time() - start,
                error=str(error),
            ),
            manifest_path,
        )
        raise

    return AnalysisArtifact(
        name=params["name"],
        output_dir=str(output_dir),
        tables={
            name: str(path)
            for name, path in table_paths.items()
        },
        figures={
            name: str(path)
            for name, path in figure_paths.items()
        },
        manifest_path=str(manifest_path),
        signature=signature,
    )