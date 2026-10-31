
# pipeline/scenarios.py

import hashlib
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from models.scenario_split import ScenarioSplit
from utils.storage import exists, load_data, load_json, load_manifest, save_json, save_manifest
from utils.status import is_done, make_manifest, make_signature


OUTPUT_ROOT = Path("outputs/scenarios")


# ============================================================
# Helpers
# ============================================================

def _stable_seed(base_seed, *parts):
    text = "|".join(map(str, parts))
    return (int(hashlib.md5(text.encode()).hexdigest()[:8], 16) + int(base_seed)) % (2**32)


def _dataset_param(value, dataset):
    if not isinstance(value, dict):
        return value
    return value.get(dataset, value.get("default", []))


def _select_combinations(values, count, seed, max_combinations=None):
    values = sorted(values)
    if count == "all":
        return [tuple(values)]

    count = int(count)
    if count == 0:
        return [tuple()]
    if count > len(values):
        return []

    total = math.comb(len(values), count)
    if max_combinations is None or total <= max_combinations:
        return list(itertools.combinations(values, count))

    rng = np.random.default_rng(seed)
    selected = set()
    while len(selected) < max_combinations:
        selected.add(tuple(sorted(rng.choice(values, size=count, replace=False).tolist())))
    return sorted(selected)


def _load_scenario_metadata(dataset_view):
    columns = (
        ["dataset", "domain"] if dataset_view.representation == "image"
        else ["dataset", "subject", "session"]
    )
    data = load_data(dataset_view.path, keys=columns)
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, dict):
        return pd.DataFrame({col: data[col] for col in columns})
    raise TypeError("Unsupported preprocessing data type.")


def _subject_domains(df):
    return df[["dataset", "subject"]].astype(str).agg("|".join, axis=1)


def _session_domains(df):
    return df[["dataset", "subject", "session"]].astype(str).agg("|".join, axis=1)


def _make_split(scenario, source_elementary_domains,
                target_super_domain_elementary_domains,
                target_elementary_domains, target_fraction, seed):
    content = {
        "scenario": scenario,
        "source_elementary_domains": sorted(source_elementary_domains),
        "target_super_domain_elementary_domains": sorted(target_super_domain_elementary_domains),
        "target_elementary_domains": sorted(target_elementary_domains),
        "target_fraction": float(target_fraction),
        "seed": int(seed),
    }
    return ScenarioSplit(
        id=f"{scenario}_{make_signature(content)[:12]}", **content
    )


def _unique_splits(splits):
    return list({split.id: split for split in splits}.values())


# ============================================================
# Intra-subject
# ============================================================

def _generate_intra_subject(df, params):
    fraction = params.get("train_fraction", 0.8)
    base_seed = params.get("seed", 42)
    elementary = _subject_domains(df)
    splits = []

    for (dataset, subject), group in df.groupby(["dataset", "subject"]):
        targets = elementary.loc[group.index].unique().tolist()
        splits.append(_make_split(
            "intra_subject", [], [], targets, fraction,
            _stable_seed(base_seed, "intra_subject", dataset, subject),
        ))

    return splits


# ============================================================
# Cross-session
# ============================================================

def _generate_cross_session(df, params):
    counts_config = params.get("source_counts", [1, "all"])
    fractions = params.get("target_fractions", [0.0])
    max_combinations = params.get("max_source_combinations", 10)
    base_seed = params.get("seed", 42)
    elementary = _session_domains(df)
    splits = []

    for (dataset, subject), group in df.groupby(["dataset", "subject"]):
        sessions = sorted(group["session"].unique())
        if len(sessions) < 2:
            continue

        for target in sessions:
            candidates = [s for s in sessions if s != target]
            target_mask = (
                (df["dataset"] == dataset) &
                (df["subject"] == subject) &
                (df["session"] == target)
            )
            target_domains = elementary.loc[target_mask].unique().tolist()

            for count in _dataset_param(counts_config, dataset):
                seed = _stable_seed(base_seed, dataset, subject, target, count)
                limit = None if count == "all" else max_combinations
                source_sets = _select_combinations(candidates, count, seed, limit)

                for sources in source_sets:
                    mask = (
                        (df["dataset"] == dataset) &
                        (df["subject"] == subject) &
                        df["session"].isin(sources)
                    )
                    source_domains = elementary.loc[mask].unique().tolist()

                    for fraction in fractions:
                        seed = _stable_seed(
                            base_seed, "cross_session", dataset, subject,
                            sources, target, fraction
                        )
                        splits.append(_make_split(
                            "cross_session", source_domains, [],
                            target_domains, fraction, seed,
                        ))

    return _unique_splits(splits)


# ============================================================
# Cross-subject
# ============================================================

def _generate_cross_subject(df, params):
    counts_config = params.get("source_counts", [1, "all"])
    fractions = params.get("target_fractions", [0.0])
    max_combinations = params.get("max_source_combinations", 10)
    base_seed = params.get("seed", 42)
    elementary = _subject_domains(df)
    splits = []

    for dataset, group in df.groupby("dataset"):
        subjects = sorted(group["subject"].unique())
        if len(subjects) < 2:
            continue

        for target in subjects:
            candidates = [s for s in subjects if s != target]
            target_mask = (
                (df["dataset"] == dataset) & (df["subject"] == target)
            )
            target_domains = elementary.loc[target_mask].unique().tolist()

            for count in _dataset_param(counts_config, dataset):
                seed = _stable_seed(base_seed, dataset, target, count)
                limit = None if count == "all" else max_combinations
                source_sets = _select_combinations(candidates, count, seed, limit)

                for sources in source_sets:
                    mask = (
                        (df["dataset"] == dataset) &
                        df["subject"].isin(sources)
                    )
                    source_domains = elementary.loc[mask].unique().tolist()

                    for fraction in fractions:
                        seed = _stable_seed(
                            base_seed, "cross_subject", dataset,
                            sources, target, fraction
                        )
                        splits.append(_make_split(
                            "cross_subject", source_domains, [],
                            target_domains, fraction, seed,
                        ))

    return _unique_splits(splits)


# ============================================================
# Cross-dataset
# ============================================================

def _generate_cross_dataset(df, params):
    source_counts = params.get("source_dataset_counts", [1, "all"])
    target_counts = params.get("target_dataset_subject_counts", [0])
    fractions = params.get("target_subject_fractions", [0.0])
    max_source = params.get("max_source_combinations", 10)
    max_target = params.get("max_target_domain_combinations", 10)
    base_seed = params.get("seed", 42)

    elementary = _subject_domains(df)
    datasets = sorted(df["dataset"].unique())
    if len(datasets) < 2:
        return []

    splits = []

    for target_dataset in datasets:
        candidates = [d for d in datasets if d != target_dataset]
        subjects = sorted(df.loc[
            df["dataset"] == target_dataset, "subject"
        ].unique())

        for count in source_counts:
            seed = _stable_seed(
                base_seed, "source_datasets", target_dataset, count
            )
            limit = None if count == "all" else max_source
            source_sets = _select_combinations(candidates, count, seed, limit)

            for source_datasets in source_sets:
                source_domains = elementary.loc[
                    df["dataset"].isin(source_datasets)
                ].unique().tolist()

                for target in subjects:
                    target_mask = (
                        (df["dataset"] == target_dataset) &
                        (df["subject"] == target)
                    )
                    target_domains = elementary.loc[
                        target_mask
                    ].unique().tolist()
                    target_candidates = [s for s in subjects if s != target]

                    for target_count in target_counts:
                        seed = _stable_seed(
                            base_seed, "target_super_domain",
                            target_dataset, target, target_count
                        )
                        limit = (
                            None if target_count in [0, "all"]
                            else max_target
                        )
                        target_sets = _select_combinations(
                            target_candidates, target_count, seed, limit
                        )

                        for selected in target_sets:
                            mask = (
                                (df["dataset"] == target_dataset) &
                                df["subject"].isin(selected)
                            )
                            super_domains = elementary.loc[
                                mask
                            ].unique().tolist()

                            for fraction in fractions:
                                seed = _stable_seed(
                                    base_seed, "cross_dataset",
                                    source_datasets, target_dataset,
                                    selected, target, fraction
                                )
                                splits.append(_make_split(
                                    "cross_dataset", source_domains,
                                    super_domains, target_domains,
                                    fraction, seed,
                                ))

    return _unique_splits(splits)


# ============================================================
# Cross-domain (LODO)
# ============================================================

def _generate_cross_domain(df, params):
    base_seed = params.get("seed", 42)
    splits = []

    for dataset, group in df.groupby("dataset"):
        domains = sorted(group["domain"].unique())
        if len(domains) < 2:
            continue

        for target in domains:
            sources = [f"{dataset}|{d}" for d in domains if d != target]
            target_domain = [f"{dataset}|{target}"]
            seed = _stable_seed(base_seed, "cross_domain", dataset, target)

            splits.append(_make_split(
                "cross_domain", sources, [], target_domain, 0.0, seed
            ))

    return splits


# ============================================================
# Registry and persistence
# ============================================================

SCENARIO_GENERATORS = {
    "intra_subject": _generate_intra_subject,
    "cross_session": _generate_cross_session,
    "cross_subject": _generate_cross_subject,
    "cross_dataset": _generate_cross_dataset,
    "cross_domain": _generate_cross_domain,
}


def _get_output_paths(scenario, effective_params):
    signature = make_signature(effective_params)
    root = OUTPUT_ROOT / scenario / signature[:12]
    return root / "splits.json", root / "manifest.json"


def _save_splits(splits, path):
    save_json([split.to_dict() for split in splits], path)


def _load_splits(path):
    return [ScenarioSplit.from_dict(item) for item in load_json(path)]


def run_scenario(dataset_view, scenario, params):
    if scenario not in SCENARIO_GENERATORS:
        raise ValueError(f"Unknown scenario: {scenario}")

    input_manifest = load_manifest(dataset_view.manifest_path)
    effective_params = {
        "scenario": scenario,
        "params": params,
        "input_signature": input_manifest["signature"],
    }
    splits_path, manifest_path = _get_output_paths(scenario, effective_params)

    if exists(splits_path) and is_done(manifest_path, effective_params):
        return _load_splits(splits_path)

    metadata = _load_scenario_metadata(dataset_view)
    splits = SCENARIO_GENERATORS[scenario](metadata, params)
    _save_splits(splits, splits_path)

    manifest = make_manifest(status="done", params=effective_params)
    manifest["output"] = {
        "scenario": scenario,
        "n_splits": len(splits),
        "splits_path": str(splits_path),
    }
    save_manifest(manifest, manifest_path)
    return splits
