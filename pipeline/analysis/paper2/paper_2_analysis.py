
# pipeline/analysis/paper2/paper_2_analysis.py

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from models.analysis_artifact import AnalysisArtifact
from utils.status import is_done, make_manifest, make_signature
from utils.storage import exists, load_manifest, save_manifest


OUTPUT_ROOT = Path("outputs/analysis/paper2")

DATASETS = {
    "bci2a": "BCI IV 2a",
    "eegmmidb": "EEGMMIDB",
    "weibo": "Weibo2014",
    "zhou": "Zhou2016",
}

METHODS = {
    "neural_erm": "ERM",
    "irm": "IRM",
    "vrex": "VREx",
    "groupdro": "GroupDRO",
    "coral": "CORAL",
    "mmd": "MMD",
    "dg_dann": "DG-DANN",
    "mldg": "MLDG",
}

DEFAULTS = {
    "name": "paper2",
    "collection": "paper2",
    "required_datasets": list(DATASETS),
    "n_boot": 2000,
    "seed": 42,
}

FIGURES = [
    "generalization_improvement",
    "generalization_gap",
    "subject_improvement",
    "improvement_vs_difficulty",
]

TABLES = [
    "generalization_performance.csv",
    "generalization_performance.tex",
    "performance_summary.csv",
    "subject_results.csv",
    "subject_improvement.csv",
    "improvement_summary.csv",
    "paired_wilcoxon.csv",
    "gap_summary.csv",
    "analysis_coverage.csv",
]


# ============================================================
# 1. Result collection
# ============================================================

def _dataset(value):
    return str(value).split(";")[0].split("|")[0]


def _subject(value):
    domain = str(value).split(";")[0].split("|")
    return domain[1] if len(domain) > 1 else None


def _method(value):
    name = str(value).split("__")[0]
    return METHODS.get(name, name.replace("_", " ").title())


def _methods(df):
    available = set(df["Method"].dropna())
    preferred = list(METHODS.values())
    return [x for x in preferred if x in available] + sorted(available - set(preferred))


def _artifact_path(artifact):
    return Path(artifact if isinstance(artifact, (str, Path)) else artifact.path)


def _register(artifact, params):
    path = OUTPUT_ROOT / "collections" / params["collection"] / "manifest.json"
    inputs = load_manifest(path).get("inputs", {}) if exists(path) else {}
    inputs = {k: v for k, v in inputs.items() if exists(v["path"])}

    file = _artifact_path(artifact)
    frame = pd.read_csv(file, usecols=["scenario", "target_fraction", "target_domains"])
    frame = frame[(frame["scenario"] == "cross_subject") & (frame["target_fraction"] == 0)]
    datasets = sorted(set(frame["target_domains"].map(_dataset)))

    if not datasets:
        raise ValueError("The input contains no zero-calibration cross-subject results.")

    signature = getattr(artifact, "signature", None) or make_signature({
        "path": str(file), "mtime": file.stat().st_mtime_ns,
    })

    for dataset in datasets:
        inputs[dataset] = {"path": str(file), "signature": signature}

    missing = [d for d in params["required_datasets"] if d not in inputs]
    state = make_manifest("collecting" if missing else "ready", {
        "analysis": "paper2_collection",
        "collection": params["collection"],
        "inputs": inputs,
    })
    state.update(inputs=inputs, missing_datasets=missing)
    save_manifest(state, path)
    return inputs, missing


def _load_results(inputs, datasets):
    frames = []

    for dataset in datasets:
        frame = pd.read_csv(inputs[dataset]["path"])
        frame = frame[(frame["scenario"] == "cross_subject") & (frame["target_fraction"] == 0)]
        frame = frame[frame["target_domains"].map(_dataset) == dataset]

        if frame.empty:
            raise ValueError(f"No matching results for {dataset}.")
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


# ============================================================
# 2. Data preparation
# ============================================================

def normalize_results(results):
    required = [
        "scenario", "target_domains", "learning_method", "model_name",
        "split_id", "model_signature", "evaluation_group", "partition",
        "balanced_accuracy", "macro_f1", "training_time",
    ]
    missing = set(required) - set(results)

    if missing:
        raise ValueError(f"Missing model-results columns: {sorted(missing)}")

    df = results.copy()
    df["Dataset"] = df["target_domains"].map(_dataset).replace(DATASETS)
    df["Subject"] = df["target_domains"].map(_subject)
    df["Method"] = df["learning_method"].map(_method)
    df["Balanced Accuracy"] = pd.to_numeric(df["balanced_accuracy"], errors="coerce")
    df["Macro-F1"] = pd.to_numeric(df["macro_f1"], errors="coerce")
    df["Training Time (s)"] = pd.to_numeric(df["training_time"], errors="coerce")

    if "auc" in df:
        df["AUC"] = pd.to_numeric(df["auc"], errors="coerce")

    return df


def build_subject_results(results):
    keys = ["Dataset", "Subject", "Method", "split_id", "model_signature"]

    train = results[
        (results["evaluation_group"] == "source") &
        (results["partition"] == "train")
    ]
    test = results[
        (results["evaluation_group"] == "target_elementary_domain") &
        (results["partition"] == "test")
    ]

    train = train.groupby(keys, as_index=False)["Balanced Accuracy"].mean()
    train = train.rename(columns={"Balanced Accuracy": "Train BA"})

    test = test.groupby(keys, as_index=False).agg(**{
        "Test BA": ("Balanced Accuracy", "mean"),
        "Macro-F1": ("Macro-F1", "mean"),
        "Training Time (s)": ("Training Time (s)", "mean"),
    })

    df = train.merge(test, on=keys, validate="one_to_one")
    if df.empty:
        raise ValueError("No paired source-train / target-test results found.")

    df = df.groupby(["Dataset", "Subject", "Method"], as_index=False).agg({
        "Train BA": "mean",
        "Test BA": "mean",
        "Macro-F1": "mean",
        "Training Time (s)": "mean",
    })

    df["Generalization Gap"] = df["Train BA"] - df["Test BA"]

    for dataset, group in df.groupby("Dataset"):
        if "ERM" not in group["Method"].values:
            raise ValueError(f"Missing ERM baseline for {dataset}.")

    return df


def build_performance_summary(subjects):
    return subjects.groupby(["Method", "Dataset"], as_index=False).agg(
        BA_mean=("Test BA", "mean"),
        BA_std=("Test BA", "std"),
        F1_mean=("Macro-F1", "mean"),
        F1_std=("Macro-F1", "std"),
        time_mean=("Training Time (s)", "mean"),
        time_std=("Training Time (s)", "std"),
        subjects=("Subject", "nunique"),
    )


def _fmt(mean, std, digits=3):
    if pd.isna(mean):
        return "--"
    return (
        f"{mean:.{digits}f} ± {std:.{digits}f}"
        if pd.notna(std) else f"{mean:.{digits}f}"
    )


# ============================================================
# 3. Generalization performance table
# ============================================================

def build_generalization_performance_table(subjects, methods=None, datasets=None):
    methods = list(methods) if methods is not None else _methods(subjects)
    datasets = list(datasets) if datasets is not None else subjects["Dataset"].unique().tolist()

    summary = build_performance_summary(subjects).set_index(["Method", "Dataset"])
    times = subjects.groupby("Method")["Training Time (s)"].agg(["mean", "std"])
    table = pd.DataFrame(index=pd.Index(methods, name="Method"))

    for dataset in datasets:
        table[dataset] = [
            _fmt(summary.loc[(m, dataset), "BA_mean"], summary.loc[(m, dataset), "BA_std"])
            if (m, dataset) in summary.index else "--"
            for m in methods
        ]

    table["Time (s)"] = [
        _fmt(times.loc[m, "mean"], times.loc[m, "std"], 1)
        if m in times.index else "--"
        for m in methods
    ]

    return table


# ============================================================
# 4. Paired improvements and statistics
# ============================================================

def _paired(subjects, baseline="ERM"):
    pivot = subjects.pivot_table(
        index=["Dataset", "Subject"], columns="Method",
        values="Test BA", aggfunc="mean"
    )
    if baseline not in pivot:
        raise ValueError(f"Missing {baseline} baseline.")

    rows = []
    for method in [m for m in pivot if m != baseline]:
        pairs = pivot[[baseline, method]].dropna()

        for (dataset, subject), row in pairs.iterrows():
            rows.append({
                "Dataset": dataset,
                "Subject": subject,
                "Method": method,
                "ERM BA": row[baseline],
                "Delta BA": row[method] - row[baseline],
            })

    return pd.DataFrame(rows)


def _holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.empty(len(pvalues))
    top = 0.0

    for rank, idx in enumerate(np.argsort(pvalues)):
        top = max(top, min(1.0, (len(pvalues) - rank) * pvalues[idx]))
        adjusted[idx] = top

    return adjusted


def build_improvement_summary(paired, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    for (dataset, method), group in paired.groupby(["Dataset", "Method"]):
        delta = group["Delta BA"].to_numpy()
        bounds = [np.nan, np.nan]

        if len(delta) > 1:
            boot = rng.choice(
                delta, size=(n_boot, len(delta)), replace=True
            ).mean(axis=1)
            bounds = np.quantile(boot, [0.025, 0.975])

        try:
            pvalue = (
                1.0 if np.allclose(delta, 0) or len(delta) < 2
                else wilcoxon(delta).pvalue
            )
        except ValueError:
            pvalue = np.nan

        rows.append({
            "Dataset": dataset,
            "Method": method,
            "Subjects": len(delta),
            "Mean Delta BA": delta.mean(),
            "CI Lower": bounds[0],
            "CI Upper": bounds[1],
            "Win Rate": np.mean(delta > 0),
            "Worst Delta BA": delta.min(),
            "P-value": pvalue,
        })

    out = pd.DataFrame(rows)

    if not out.empty:
        out["P-value Holm"] = np.nan

        for _, idx in out.groupby("Dataset").groups.items():
            valid = out.loc[idx, "P-value"].dropna()
            if len(valid):
                out.loc[valid.index, "P-value Holm"] = _holm(valid.values)

    return out


# ============================================================
# 5. Figure: Generalization improvement
# ============================================================

def plot_generalization_improvement(summary, datasets, methods):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)

    values = summary[["CI Lower", "CI Upper", "Mean Delta BA"]].to_numpy()
    limit = max(0.03, np.nanmax(np.abs(values)) * 1.15)

    for ax, dataset in zip(axes.flat, datasets):
        data = summary[summary["Dataset"] == dataset].set_index("Method")

        for i, method in enumerate(methods):
            if method not in data.index:
                continue

            row = data.loc[method]
            mean = row["Mean Delta BA"]

            if pd.notna(row["CI Lower"]):
                ax.errorbar(
                    mean, i,
                    xerr=[[mean - row["CI Lower"]], [row["CI Upper"] - mean]],
                    fmt="o", capsize=3
                )
            else:
                ax.plot(mean, i, "o")

        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set(
            yticks=np.arange(len(methods)),
            yticklabels=methods,
            xlim=(-limit, limit),
            title=dataset,
            xlabel="ΔBA vs. ERM",
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    return fig


# ============================================================
# 6. Figure: Generalization gap
# ============================================================

def plot_generalization_gap(subjects, datasets, methods):
    gap = subjects.groupby(["Dataset", "Method"], as_index=False).agg(
        Train_BA=("Train BA", "mean"),
        Test_BA=("Test BA", "mean"),
        Gap=("Generalization Gap", "mean"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=True, squeeze=False)
    x, width = np.arange(len(methods)), 0.36

    for ax, dataset in zip(axes.flat, datasets):
        subset = gap[gap["Dataset"] == dataset].set_index("Method").reindex(methods)
        train = subset["Train_BA"].to_numpy()
        test = subset["Test_BA"].to_numpy()

        ax.bar(x - width / 2, train, width, label="Source Train BA")
        ax.bar(x + width / 2, test, width, label="Target Test BA")

        for i, value in enumerate(subset["Gap"]):
            if pd.notna(value):
                ax.text(
                    i, max(train[i], test[i]) + 0.02,
                    f"{value:+.2f}", ha="center", fontsize=7
                )

        ax.set(
            title=dataset,
            xticks=x,
            xticklabels=methods,
            ylim=(0, 1.13),
            ylabel="Balanced Accuracy",
        )
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    return fig, gap


# ============================================================
# 7. Figure: Subject-level improvement
# ============================================================

def plot_subject_improvement(paired, datasets, methods):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True, squeeze=False)
    limit = max(0.05, paired["Delta BA"].abs().max() * 1.1)

    for ax, dataset in zip(axes.flat, datasets):
        subset = paired[paired["Dataset"] == dataset]
        base = (
            subset[["Subject", "ERM BA"]]
            .drop_duplicates()
            .sort_values("ERM BA")
        )

        for method in methods:
            values = subset[subset["Method"] == method].set_index("Subject")
            ax.plot(
                np.arange(len(base)),
                values["Delta BA"].reindex(base["Subject"]),
                marker="o", linewidth=1.3, markersize=3, label=method
            )

        step = max(1, int(np.ceil(len(base) / 12)))
        ticks = np.arange(0, len(base), step)

        ax.set(
            xticks=ticks,
            xticklabels=base.iloc[ticks]["Subject"],
            title=dataset,
            ylabel="ΔBA vs. ERM",
            ylim=(-limit, limit),
            xlabel="Held-out subject (ordered by ERM BA)",
        )
        ax.tick_params(axis="x", rotation=45)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.grid(alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center",
        ncol=min(5, len(methods))
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    return fig


# ============================================================
# 8. Figure: Improvement vs. subject difficulty
# ============================================================

def plot_improvement_vs_difficulty(paired, datasets, methods):
    fig, axes = plt.subplots(
        len(datasets), len(methods),
        figsize=(2.5 * len(methods), 2.5 * len(datasets)),
        sharex=True, sharey=True, squeeze=False
    )
    limit = max(0.05, paired["Delta BA"].abs().max() * 1.15)

    for i, dataset in enumerate(datasets):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            group = paired[
                (paired["Dataset"] == dataset) &
                (paired["Method"] == method)
            ]

            ax.scatter(group["ERM BA"], group["Delta BA"], s=22, alpha=0.7)
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax.set(xlim=(0, 1), ylim=(-limit, limit))
            ax.grid(alpha=0.2)

            if i == 0:
                ax.set_title(method, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{dataset}\nΔBA")
            if i == len(datasets) - 1:
                ax.set_xlabel("ERM BA")

    fig.tight_layout()
    return fig


# ============================================================
# 9. LaTeX generation
# ============================================================

def _latex_table(summary, subjects, methods, datasets, path):
    means = summary.set_index(["Method", "Dataset"])
    times = subjects.groupby("Method")["Training Time (s)"].agg(["mean", "std"])

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Cross-subject test Balanced Accuracy (mean $\pm$ SD) and pooled training-stage time across experiments.}",
        r"\label{tab:generalization_performance}",
        r"\begin{tabular}{l" + "c" * (len(datasets) + 1) + "}",
        r"\toprule",
        " & ".join(
            [r"\textbf{Method}"] +
            [rf"\textbf{{{d}}}" for d in datasets] +
            [r"\textbf{Time (s)}"]
        ) + r" \\",
        r"\midrule",
    ]

    for method in methods:
        values = []

        for dataset in datasets:
            if (method, dataset) not in means.index:
                values.append("--")
                continue

            row = means.loc[(method, dataset)]
            cell = _fmt(row["BA_mean"], row["BA_std"]).replace("±", r"$\pm$")
            best = means.xs(dataset, level="Dataset")["BA_mean"].max()

            if np.isfinite(row["BA_mean"]) and np.isclose(row["BA_mean"], best):
                cell = r"\textbf{" + cell + "}"

            values.append(cell)

        t = times.loc[method] if method in times.index else None
        values.append(
            _fmt(t["mean"], t["std"], 1).replace("±", r"$\pm$")
            if t is not None else "--"
        )

        lines.append(
            " & ".join([method.replace("_", r"\_")] + values) + r" \\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def _figure_snippets(path):
    captions = {
        "generalization_improvement":
            "Mean paired change in test Balanced Accuracy relative to ERM, with 95\\% bootstrap confidence intervals.",
        "generalization_gap":
            "Source training and unseen-subject test Balanced Accuracy. Labels give their mean difference.",
        "subject_improvement":
            "Subject-wise Balanced Accuracy changes relative to ERM, ordered by ERM performance.",
        "improvement_vs_difficulty":
            "Subject-wise improvement as a function of ERM performance, by method and dataset.",
    }

    blocks = []
    for name in FIGURES:
        blocks.extend([
            r"\begin{figure*}[t]",
            r"\centering",
            rf"\includegraphics[width=\textwidth]{{figures/{name}.pdf}}",
            rf"\caption{{{captions[name]}}}",
            rf"\label{{fig:{name}}}",
            r"\end{figure*}",
            "",
        ])

    path.write_text("\n".join(blocks), encoding="utf-8")


# ============================================================
# 10. Analysis pipeline
# ============================================================

def run_paper2_analysis(model_results, params=None):
    params = {**DEFAULTS, **(params or {})}
    inputs, missing = _register(model_results, params)

    if missing:
        print("[Paper 2] Waiting for datasets: " + ", ".join(missing))
        return None

    effective = {
        "analysis": "paper2",
        "version": 1,
        "params": params,
        "inputs": {k: inputs[k] for k in params["required_datasets"]},
    }

    signature = make_signature(effective)
    output = OUTPUT_ROOT / "runs" / params["collection"] / signature[:12]
    manifest = output / "manifest.json"

    paths = {name: output / "tables" / name for name in TABLES}
    paths.update({
        f"{name}.{ext}": output / "figures" / f"{name}.{ext}"
        for name in FIGURES for ext in ("pdf", "png")
    })
    paths["figure_snippets.tex"] = output / "figure_snippets.tex"

    def artifact():
        return AnalysisArtifact(
            name=params["name"],
            output_dir=str(output),
            tables={k: str(v) for k, v in paths.items() if k in TABLES},
            figures={
                k: str(v) for k, v in paths.items()
                if k.endswith((".pdf", ".png"))
            },
            manifest_path=str(manifest),
            signature=signature,
        )

    if all(exists(p) for p in paths.values()) and is_done(manifest, effective):
        return artifact()

    start = time.time()
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    save_manifest(make_manifest("running", effective), manifest)

    try:
        raw = normalize_results(
            _load_results(inputs, params["required_datasets"])
        )
        subjects = build_subject_results(raw)
        methods = _methods(subjects)
        datasets = [DATASETS[d] for d in params["required_datasets"]]
        dg_methods = [m for m in methods if m != "ERM"]

        if not dg_methods:
            raise ValueError("At least one DG method is needed for comparison with ERM.")

        performance = build_performance_summary(subjects)
        table = build_generalization_performance_table(
            subjects, methods, datasets
        )
        paired = _paired(subjects)
        improvement = build_improvement_summary(
            paired, params["n_boot"], params["seed"]
        )

        gap = subjects.groupby(["Dataset", "Method"], as_index=False).agg(
            Train_BA=("Train BA", "mean"),
            Test_BA=("Test BA", "mean"),
            Gap=("Generalization Gap", "mean"),
        )
        coverage = subjects.groupby(["Dataset", "Method"], as_index=False).agg(
            Subjects=("Subject", "nunique"),
            Runs=("Test BA", "size"),
        )

        tables = {
            "generalization_performance.csv": table.reset_index(),
            "performance_summary.csv": performance,
            "subject_results.csv": subjects,
            "subject_improvement.csv": paired,
            "improvement_summary.csv": improvement,
            "paired_wilcoxon.csv": improvement[
                ["Dataset", "Method", "Subjects", "P-value", "P-value Holm"]
            ],
            "gap_summary.csv": gap,
            "analysis_coverage.csv": coverage,
        }

        for name, frame in tables.items():
            frame.to_csv(paths[name], index=False)

        _latex_table(
            performance, subjects, methods, datasets,
            paths["generalization_performance.tex"]
        )

        plots = {
            "generalization_improvement": plot_generalization_improvement(
                improvement, datasets, dg_methods
            ),
            "generalization_gap": plot_generalization_gap(
                subjects, datasets, methods
            )[0],
            "subject_improvement": plot_subject_improvement(
                paired, datasets, dg_methods
            ),
            "improvement_vs_difficulty": plot_improvement_vs_difficulty(
                paired, datasets, dg_methods
            ),
        }

        for name, fig in plots.items():
            for ext in ("pdf", "png"):
                fig.savefig(
                    paths[f"{name}.{ext}"], dpi=300,
                    bbox_inches="tight"
                )
            plt.close(fig)

        _figure_snippets(paths["figure_snippets.tex"])

        done = make_manifest(
            "done", effective, execution_time=time.time() - start
        )
        done["output"] = {
            "output_dir": str(output),
            "tables": artifact().tables,
            "figures": artifact().figures,
        }
        save_manifest(done, manifest)

    except Exception as error:
        save_manifest(
            make_manifest(
                "failed", effective,
                execution_time=time.time() - start,
                error=str(error),
            ),
            manifest,
        )
        raise

    return artifact()
