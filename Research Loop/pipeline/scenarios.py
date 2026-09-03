import hashlib
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from models.scenario_split import ScenarioSplit
from utils.storage import (
    exists,
    load_data,
    load_json,
    load_manifest,
    save_json,
    save_manifest,
)
from utils.status import (
    is_done,
    make_manifest,
    make_signature,
)


OUTPUT_ROOT = Path("outputs/scenarios")


# ============================================================
# Helpers
# ============================================================

def _stable_seed(base_seed, *parts):
    text = "|".join(str(part) for part in parts)
    digest = hashlib.md5(text.encode()).hexdigest()

    return (
        int(digest[:8], 16)
        + int(base_seed)
    ) % (2 ** 32)


def _dataset_param(value, dataset):
    """
    Allow either a global value or a dataset-specific mapping.
    """

    if not isinstance(value, dict):
        return value

    if dataset in value:
        return value[dataset]

    if "default" in value:
        return value["default"]

    return []


def _select_combinations(
    values,
    count,
    seed,
    max_combinations=None,
):
    values = sorted(list(values))

    if count == "all":
        return [tuple(values)]

    count = int(count)

    if count == 0:
        return [tuple()]

    if count > len(values):
        return []

    total = math.comb(
        len(values),
        count,
    )

    if (
        max_combinations is None
        or total <= max_combinations
    ):
        return list(
            itertools.combinations(
                values,
                count,
            )
        )

    rng = np.random.default_rng(seed)
    selected = set()

    while len(selected) < max_combinations:
        combination = tuple(
            sorted(
                rng.choice(
                    values,
                    size=count,
                    replace=False,
                ).tolist()
            )
        )

        selected.add(
            combination
        )

    return sorted(
        selected
    )


def _load_scenario_metadata(dataset_view):
    """
    Load only metadata required for scenario generation.
    """

    columns = [
        "dataset",
        "subject",
        "session",
    ]

    data = load_data(
        dataset_view.path,
        keys=columns,
    )

    if isinstance(data, pd.DataFrame):
        return data

    if isinstance(data, dict):
        return pd.DataFrame({
            column: data[column]
            for column in columns
        })

    raise TypeError(
        "Unsupported preprocessing data type."
    )


def _subject_domains(dataframe):
    return (
        dataframe[
            ["dataset", "subject"]
        ]
        .astype(str)
        .agg("|".join, axis=1)
    )


def _session_domains(dataframe):
    return (
        dataframe[
            ["dataset", "subject", "session"]
        ]
        .astype(str)
        .agg("|".join, axis=1)
    )


def _make_split(
    scenario,
    source_elementary_domains,
    target_super_domain_elementary_domains,
    target_elementary_domains,
    target_fraction,
    seed,
):
    content = {
        "scenario": scenario,

        "source_elementary_domains": sorted(
            list(source_elementary_domains)
        ),

        "target_super_domain_elementary_domains": sorted(
            list(
                target_super_domain_elementary_domains
            )
        ),

        "target_elementary_domains": sorted(
            list(target_elementary_domains)
        ),

        "target_fraction": float(
            target_fraction
        ),

        "seed": int(
            seed
        ),
    }

    split_id = (
        f"{scenario}_"
        f"{make_signature(content)[:12]}"
    )

    return ScenarioSplit(
        id=split_id,
        **content,
    )


def _unique_splits(splits):
    return list(
        {
            split.id: split
            for split in splits
        }.values()
    )


# ============================================================
# Intra-subject
# ============================================================

def _generate_intra_subject(
    dataframe,
    params,
):
    train_fraction = params.get(
        "train_fraction",
        0.8,
    )

    base_seed = params.get(
        "seed",
        42,
    )

    elementary_domains = (
        _subject_domains(
            dataframe
        )
    )

    splits = []

    for (dataset, subject), group in dataframe.groupby(
        ["dataset", "subject"]
    ):
        target_domains = (
            elementary_domains.loc[
                group.index
            ]
            .unique()
            .tolist()
        )

        seed = _stable_seed(
            base_seed,
            "intra_subject",
            dataset,
            subject,
        )

        splits.append(
            _make_split(
                scenario="intra_subject",
                source_elementary_domains=[],
                target_super_domain_elementary_domains=[],
                target_elementary_domains=target_domains,
                target_fraction=train_fraction,
                seed=seed,
            )
        )

    return splits


# ============================================================
# Cross-session
# ============================================================

def _generate_cross_session(
    dataframe,
    params,
):
    source_counts_config = params.get(
        "source_counts",
        [1, "all"],
    )

    target_fractions = params.get(
        "target_fractions",
        [0.0],
    )

    max_combinations = params.get(
        "max_source_combinations",
        10,
    )

    base_seed = params.get(
        "seed",
        42,
    )

    elementary_domains = (
        _session_domains(
            dataframe
        )
    )

    splits = []

    for (dataset, subject), group in dataframe.groupby(
        ["dataset", "subject"]
    ):
        source_counts = _dataset_param(
            source_counts_config,
            dataset,
        )

        sessions = sorted(
            group["session"].unique()
        )

        if len(sessions) < 2:
            continue

        for target_session in sessions:
            source_candidates = [
                session
                for session in sessions
                if session != target_session
            ]

            target_mask = (
                (dataframe["dataset"] == dataset)
                & (dataframe["subject"] == subject)
                & (
                    dataframe["session"]
                    == target_session
                )
            )

            target_domains = (
                elementary_domains.loc[
                    target_mask
                ]
                .unique()
                .tolist()
            )

            for source_count in source_counts:
                selection_seed = _stable_seed(
                    base_seed,
                    dataset,
                    subject,
                    target_session,
                    source_count,
                )

                limit = (
                    None
                    if source_count == "all"
                    else max_combinations
                )

                source_sets = (
                    _select_combinations(
                        source_candidates,
                        source_count,
                        selection_seed,
                        limit,
                    )
                )

                for source_sessions in source_sets:
                    source_mask = (
                        (dataframe["dataset"] == dataset)
                        & (
                            dataframe["subject"]
                            == subject
                        )
                        & dataframe["session"].isin(
                            source_sessions
                        )
                    )

                    source_domains = (
                        elementary_domains.loc[
                            source_mask
                        ]
                        .unique()
                        .tolist()
                    )

                    for target_fraction in target_fractions:
                        seed = _stable_seed(
                            base_seed,
                            "cross_session",
                            dataset,
                            subject,
                            source_sessions,
                            target_session,
                            target_fraction,
                        )

                        splits.append(
                            _make_split(
                                scenario="cross_session",
                                source_elementary_domains=source_domains,
                                target_super_domain_elementary_domains=[],
                                target_elementary_domains=target_domains,
                                target_fraction=target_fraction,
                                seed=seed,
                            )
                        )

    return _unique_splits(
        splits
    )


# ============================================================
# Cross-subject
# ============================================================

def _generate_cross_subject(
    dataframe,
    params,
):
    source_counts_config = params.get(
        "source_counts",
        [1, "all"],
    )

    target_fractions = params.get(
        "target_fractions",
        [0.0],
    )

    max_combinations = params.get(
        "max_source_combinations",
        10,
    )

    base_seed = params.get(
        "seed",
        42,
    )

    elementary_domains = (
        _subject_domains(
            dataframe
        )
    )

    splits = []

    for dataset, group in dataframe.groupby(
        "dataset"
    ):
        source_counts = _dataset_param(
            source_counts_config,
            dataset,
        )

        subjects = sorted(
            group["subject"].unique()
        )

        if len(subjects) < 2:
            continue

        for target_subject in subjects:
            source_candidates = [
                subject
                for subject in subjects
                if subject != target_subject
            ]

            target_mask = (
                (dataframe["dataset"] == dataset)
                & (
                    dataframe["subject"]
                    == target_subject
                )
            )

            target_domains = (
                elementary_domains.loc[
                    target_mask
                ]
                .unique()
                .tolist()
            )

            for source_count in source_counts:
                selection_seed = _stable_seed(
                    base_seed,
                    dataset,
                    target_subject,
                    source_count,
                )

                limit = (
                    None
                    if source_count == "all"
                    else max_combinations
                )

                source_sets = (
                    _select_combinations(
                        source_candidates,
                        source_count,
                        selection_seed,
                        limit,
                    )
                )

                for source_subjects in source_sets:
                    source_mask = (
                        (dataframe["dataset"] == dataset)
                        & dataframe["subject"].isin(
                            source_subjects
                        )
                    )

                    source_domains = (
                        elementary_domains.loc[
                            source_mask
                        ]
                        .unique()
                        .tolist()
                    )

                    for target_fraction in target_fractions:
                        seed = _stable_seed(
                            base_seed,
                            "cross_subject",
                            dataset,
                            source_subjects,
                            target_subject,
                            target_fraction,
                        )

                        splits.append(
                            _make_split(
                                scenario="cross_subject",
                                source_elementary_domains=source_domains,
                                target_super_domain_elementary_domains=[],
                                target_elementary_domains=target_domains,
                                target_fraction=target_fraction,
                                seed=seed,
                            )
                        )

    return _unique_splits(
        splits
    )


# ============================================================
# Cross-dataset
# ============================================================

def _generate_cross_dataset(
    dataframe,
    params,
):
    source_dataset_counts = params.get(
        "source_dataset_counts",
        [1, "all"],
    )

    target_dataset_subject_counts = params.get(
        "target_dataset_subject_counts",
        [0],
    )

    target_subject_fractions = params.get(
        "target_subject_fractions",
        [0.0],
    )

    max_source_combinations = params.get(
        "max_source_combinations",
        10,
    )

    max_target_combinations = params.get(
        "max_target_domain_combinations",
        10,
    )

    base_seed = params.get(
        "seed",
        42,
    )

    elementary_domains = (
        _subject_domains(
            dataframe
        )
    )

    datasets = sorted(
        dataframe[
            "dataset"
        ].unique()
    )

    if len(datasets) < 2:
        return []

    splits = []

    for target_dataset in datasets:
        source_candidates = [
            dataset
            for dataset in datasets
            if dataset != target_dataset
        ]

        target_data = dataframe[
            dataframe["dataset"]
            == target_dataset
        ]

        target_subjects = sorted(
            target_data[
                "subject"
            ].unique()
        )

        for source_count in source_dataset_counts:
            selection_seed = _stable_seed(
                base_seed,
                "source_datasets",
                target_dataset,
                source_count,
            )

            limit = (
                None
                if source_count == "all"
                else max_source_combinations
            )

            source_sets = _select_combinations(
                source_candidates,
                source_count,
                selection_seed,
                limit,
            )

            for source_datasets in source_sets:
                source_mask = dataframe[
                    "dataset"
                ].isin(
                    source_datasets
                )

                source_domains = (
                    elementary_domains.loc[
                        source_mask
                    ]
                    .unique()
                    .tolist()
                )

                for target_subject in target_subjects:
                    target_mask = (
                        (
                            dataframe["dataset"]
                            == target_dataset
                        )
                        & (
                            dataframe["subject"]
                            == target_subject
                        )
                    )

                    target_domains = (
                        elementary_domains.loc[
                            target_mask
                        ]
                        .unique()
                        .tolist()
                    )

                    target_candidates = [
                        subject
                        for subject
                        in target_subjects
                        if subject
                        != target_subject
                    ]

                    for target_count in (
                        target_dataset_subject_counts
                    ):
                        target_selection_seed = (
                            _stable_seed(
                                base_seed,
                                "target_super_domain",
                                target_dataset,
                                target_subject,
                                target_count,
                            )
                        )

                        target_limit = (
                            None
                            if target_count
                            in [0, "all"]
                            else max_target_combinations
                        )

                        target_sets = (
                            _select_combinations(
                                target_candidates,
                                target_count,
                                target_selection_seed,
                                target_limit,
                            )
                        )

                        for (
                            target_subjects_selected
                        ) in target_sets:
                            target_super_mask = (
                                (
                                    dataframe["dataset"]
                                    == target_dataset
                                )
                                & dataframe[
                                    "subject"
                                ].isin(
                                    target_subjects_selected
                                )
                            )

                            target_super_domains = (
                                elementary_domains.loc[
                                    target_super_mask
                                ]
                                .unique()
                                .tolist()
                            )

                            for target_fraction in (
                                target_subject_fractions
                            ):
                                seed = _stable_seed(
                                    base_seed,
                                    "cross_dataset",
                                    source_datasets,
                                    target_dataset,
                                    target_subjects_selected,
                                    target_subject,
                                    target_fraction,
                                )

                                splits.append(
                                    _make_split(
                                        scenario="cross_dataset",
                                        source_elementary_domains=source_domains,
                                        target_super_domain_elementary_domains=target_super_domains,
                                        target_elementary_domains=target_domains,
                                        target_fraction=target_fraction,
                                        seed=seed,
                                    )
                                )

    return _unique_splits(
        splits
    )


# ============================================================
# Registry
# ============================================================

SCENARIO_GENERATORS = {
    "intra_subject": _generate_intra_subject,
    "cross_session": _generate_cross_session,
    "cross_subject": _generate_cross_subject,
    "cross_dataset": _generate_cross_dataset,
}


# ============================================================
# Persistence
# ============================================================

def _get_output_paths(
    scenario,
    effective_params,
):
    signature = make_signature(
        effective_params
    )

    output_dir = (
        OUTPUT_ROOT
        / scenario
        / signature[:12]
    )

    return (
        output_dir / "splits.json",
        output_dir / "manifest.json",
    )


def _save_splits(
    splits,
    path,
):
    save_json(
        [
            split.to_dict()
            for split in splits
        ],
        path,
    )


def _load_splits(path):
    return [
        ScenarioSplit.from_dict(
            item
        )
        for item in load_json(
            path
        )
    ]


# ============================================================
# Public
# ============================================================

def run_scenario(
    dataset_view,
    scenario,
    params,
):
    if scenario not in SCENARIO_GENERATORS:
        raise ValueError(
            f"Unknown scenario: {scenario}"
        )

    input_manifest = load_manifest(
        dataset_view.manifest_path
    )

    effective_params = {
        "scenario": scenario,
        "params": params,
        "input_signature": input_manifest[
            "signature"
        ],
    }

    splits_path, manifest_path = (
        _get_output_paths(
            scenario,
            effective_params,
        )
    )

    if (
        exists(splits_path)
        and is_done(
            manifest_path,
            effective_params,
        )
    ):
        return _load_splits(
            splits_path
        )

    metadata = _load_scenario_metadata(
        dataset_view
    )

    splits = SCENARIO_GENERATORS[
        scenario
    ](
        metadata,
        params,
    )

    _save_splits(
        splits,
        splits_path,
    )

    manifest = make_manifest(
        status="done",
        params=effective_params,
    )

    manifest["output"] = {
        "scenario": scenario,
        "n_splits": len(splits),
        "splits_path": str(splits_path),
    }

    save_manifest(
        manifest,
        manifest_path,
    )

    return splits