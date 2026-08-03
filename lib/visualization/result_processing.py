import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics_table(results_df):
    df = results_df.copy()

    exclude_cols = {
        "y_true", "y_pred", "y_prob",
        "model", "fs_method", "split", "split_type"
    }

    meta_cols = [c for c in df.columns if c not in exclude_cols]

    group_cols = meta_cols + ["model", "fs_method", "split"]

    metrics_list = []

    for keys, group in df.groupby(group_cols):
        y_true = group["y_true"].values
        y_pred = group["y_pred"].values

        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        row["accuracy"] = accuracy_score(y_true, y_pred)
        row["f1_macro"] = f1_score(y_true, y_pred, average="macro")
        row["n_samples"] = len(group)  # <-- added

        metrics_list.append(row)

    metrics_df = pd.DataFrame(metrics_list)

    summary_df = (
        metrics_df
        .groupby(["model", "fs_method", "split"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            n_samples_total=("n_samples", "sum")  # <-- added
        )
        .reset_index()
    )

    return metrics_df, summary_df