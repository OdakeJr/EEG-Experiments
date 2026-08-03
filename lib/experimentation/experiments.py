import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import time
import os


# Helper
def load_dataset(path):
    df = pd.read_csv(path)
    return df

def save_results(result_df, time_df, path, prefix):
    os.makedirs(path, exist_ok=True)

    result_df.to_csv(os.path.join(path, f"{prefix}_results.csv"), index=False)
    time_df.to_csv(os.path.join(path, f"{prefix}_time.csv"), index=False)

def load_results(path, prefix):
    result_df = pd.read_csv(os.path.join(path, f"{prefix}_results.csv"))
    time_df = pd.read_csv(os.path.join(path, f"{prefix}_time.csv"))
    return result_df, time_df

# Base Pipeline
def run_pipeline(
    X_train, y_train, domains_train,
    X_test,  y_test,  domains_test,
    fs_dict,
    model_dict,
    meta=None,
    split_pbar=None
):
    results = []
    timing_results = []

    if meta is None:
        meta = {}

    # ==========================
    # SCALE
    # ==========================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ==========================
    # TOTAL INNER STEPS
    # ==========================
    total_inner = len(fs_dict) * len(model_dict)

    inner_desc = f"FS/Model [{meta.get('split_type', 'split')}]"
    with tqdm(total=total_inner, desc=inner_desc, leave=False) as inner_pbar:

        # ==========================
        # FEATURE SELECTION LOOP
        # ==========================
        for fs_name, fs_cfg in fs_dict.items():

            fs_fn = fs_cfg["function"]
            fs_params = fs_cfg.get("params", {})

            inner_pbar.set_postfix({
                "fs": fs_name,
                "model": "-"
            })

            # ---- FS timing ----
            t0 = time.time()

            X_train_fs, X_test_fs, fs_info = fs_fn(
                X_train, y_train, domains_train,
                X_test,  domains_test,
                **fs_params
            )

            fs_time = time.time() - t0

            # ==========================
            # MODEL LOOP
            # ==========================
            for model_name, model_cfg in model_dict.items():

                model_fn = model_cfg["function"]
                model_params = model_cfg.get("params", {})

                inner_pbar.set_postfix({
                    "fs": fs_name,
                    "model": model_name
                })

                # ---- TRAIN timing ----
                t0 = time.time()

                model = model_fn(
                    X_train_fs, y_train, domains_train,
                    **model_params
                )

                train_time = time.time() - t0

                # ==========================
                # PREDICTIONS
                # ==========================
                y_train_pred = model["predict"](X_train_fs)
                y_test_pred  = model["predict"](X_test_fs)

                y_train_prob = model["predict_proba"](X_train_fs)
                y_test_prob  = model["predict_proba"](X_test_fs)

                # ==========================
                # SAVE TIMING
                # ==========================
                timing_results.append({
                    **meta,
                    "fs_method": fs_name,
                    "model": model_name,
                    "fs_time": fs_time,
                    "train_time": train_time,
                    "n_train": len(y_train),
                    "n_test": len(y_test)
                })

                # ==========================
                # SAVE TRAIN
                # ==========================
                for i in range(len(y_train)):
                    results.append({
                        **meta,
                        "split": "train",
                        "fs_method": fs_name,
                        "model": model_name,
                        "y_true": int(y_train[i]),
                        "y_pred": int(y_train_pred[i]),
                        "y_prob": (
                            y_train_prob[i].tolist()
                            if isinstance(y_train_prob[i], np.ndarray)
                            else float(y_train_prob[i])
                        )
                    })

                # ==========================
                # SAVE TEST
                # ==========================
                for i in range(len(y_test)):
                    results.append({
                        **meta,
                        "split": "test",
                        "fs_method": fs_name,
                        "model": model_name,
                        "y_true": int(y_test[i]),
                        "y_pred": int(y_test_pred[i]),
                        "y_prob": (
                            y_test_prob[i].tolist()
                            if isinstance(y_test_prob[i], np.ndarray)
                            else float(y_test_prob[i])
                        )
                    })

                inner_pbar.update(1)

    if split_pbar is not None:
        split_pbar.update(1)

    return pd.DataFrame(results), pd.DataFrame(timing_results)


# ==========================================================
# Experiment Runner
# ==========================================================
def run_experiment(df, split_fn, fs_dict, model_dict):
    all_results = []
    all_timing = []

    splits = list(split_fn(df))

    with tqdm(total=len(splits), desc="Splits") as split_pbar:
        for split in splits:
            X_train, y_train, d_train, X_test, y_test, d_test, meta = split

            split_pbar.set_postfix({
                "type": meta.get("split_type", "-"),
                "subject": meta.get("subject", meta.get("subject_test", "-")),
                "session": meta.get("session", meta.get("session_test", "-"))
            })

            res_df, time_df = run_pipeline(
                X_train, y_train, d_train,
                X_test,  y_test,  d_test,
                fs_dict, model_dict,
                meta=meta,
                split_pbar=split_pbar
            )

            all_results.append(res_df)
            all_timing.append(time_df)

    return pd.concat(all_results, ignore_index=True), pd.concat(all_timing, ignore_index=True)


# Intra subject/session
def split_intra_subject_session(
    df,
    subject_col="subject",
    session_col="session",
    label_col="label",
    test_size=0.3
):

    feature_cols = [
        c for c in df.columns
        if c not in [subject_col, session_col, label_col]
    ]

    for subj in df[subject_col].unique():
        df_subj = df[df[subject_col] == subj]

        for sess in df_subj[session_col].unique():
            df_sess = df_subj[df_subj[session_col] == sess].copy()

            if df_sess[label_col].nunique() < 2:
                continue

            # ===== TEMPORAL SPLIT (no shuffle) =====
            n = len(df_sess)
            split_idx = int((1 - test_size) * n)

            if split_idx <= 0 or split_idx >= n:
                continue

            df_train = df_sess.iloc[:split_idx]
            df_test  = df_sess.iloc[split_idx:]

            y_train = df_train[label_col].values
            y_test  = df_test[label_col].values

            if len(np.unique(y_train)) < 2:
                continue

            X_train = df_train[feature_cols].values
            X_test  = df_test[feature_cols].values

            domains_train = df_train[session_col].values
            domains_test  = df_test[session_col].values

            meta = {
                "subject": subj,
                "session": sess,
                "split_type": "intra_session"
            }

            yield X_train, y_train, domains_train, X_test, y_test, domains_test, meta

# Inter session
def split_inter_session(
    df,
    subject_col="subject",
    session_col="session",
    label_col="label"
):

    feature_cols = [
        c for c in df.columns
        if c not in [subject_col, session_col, label_col]
    ]

    for subj in df[subject_col].unique():
        df_subj = df[df[subject_col] == subj]

        sessions = df_subj[session_col].unique()

        for test_sess in sessions:
            df_test  = df_subj[df_subj[session_col] == test_sess]
            df_train = df_subj[df_subj[session_col] != test_sess]

            if len(df_train) == 0 or len(df_test) == 0:
                continue

            if df_train[label_col].nunique() < 2:
                continue

            X_train = df_train[feature_cols].values
            y_train = df_train[label_col].values

            X_test  = df_test[feature_cols].values
            y_test  = df_test[label_col].values

            domains_train = df_train[session_col].values
            domains_test  = df_test[session_col].values

            meta = {
                "subject": subj,
                "session_test": test_sess,
                "split_type": "inter_session"
            }

            yield X_train, y_train, domains_train, X_test, y_test, domains_test, meta
            
# Inter subject
def split_inter_subject(
    df,
    subject_col="subject",
    session_col="session",
    label_col="label"
):

    feature_cols = [
        c for c in df.columns
        if c not in [subject_col, session_col, label_col]
    ]

    subjects = df[subject_col].unique()

    for test_subj in subjects:
        df_test  = df[df[subject_col] == test_subj]
        df_train = df[df[subject_col] != test_subj]

        if len(df_train) == 0 or len(df_test) == 0:
            continue

        if df_train[label_col].nunique() < 2:
            continue

        X_train = df_train[feature_cols].values
        y_train = df_train[label_col].values

        X_test  = df_test[feature_cols].values
        y_test  = df_test[label_col].values

        domains_train = df_train[session_col].values
        domains_test  = df_test[session_col].values

        meta = {
            "subject_test": test_subj,
            "split_type": "inter_subject"
        }

        yield X_train, y_train, domains_train, X_test, y_test, domains_test, meta
        
        

def split_global_mixed_subjects(
    df,
    subject_col="subject",
    session_col="session",
    label_col="label",
    test_size=0.3,
    shuffle=False,
    random_state=42
):
    feature_cols = [
        c for c in df.columns
        if c not in [subject_col, session_col, label_col]
    ]

    train_parts = []
    test_parts = []

    for subj in df[subject_col].unique():
        df_subj = df[df[subject_col] == subj].copy()

        if len(df_subj) < 2:
            continue

        if shuffle:
            df_subj = df_subj.sample(frac=1, random_state=random_state).reset_index(drop=True)

        n = len(df_subj)
        split_idx = int((1 - test_size) * n)

        if split_idx <= 0 or split_idx >= n:
            continue

        df_train_subj = df_subj.iloc[:split_idx]
        df_test_subj = df_subj.iloc[split_idx:]

        if len(df_train_subj) == 0 or len(df_test_subj) == 0:
            continue

        if df_train_subj[label_col].nunique() < 2:
            continue

        train_parts.append(df_train_subj)
        test_parts.append(df_test_subj)

    if len(train_parts) == 0 or len(test_parts) == 0:
        return

    df_train = pd.concat(train_parts, ignore_index=True)
    df_test = pd.concat(test_parts, ignore_index=True)

    X_train = df_train[feature_cols].values
    y_train = df_train[label_col].values

    X_test = df_test[feature_cols].values
    y_test = df_test[label_col].values

    # Here domains are subjects, since subjects are being mixed together
    domains_train = df_train[subject_col].values
    domains_test = df_test[subject_col].values

    meta = {
        "split_type": "global_mixed_subjects"
    }

    yield X_train, y_train, domains_train, X_test, y_test, domains_test, meta
        
# Inter dataset - TODO?

