from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from models.scenario_data import (
    DataGroup,
    ScenarioData,
)
from utils.storage import load_data


@dataclass
class ScenarioSplit:
    id: str
    scenario: str

    source_elementary_domains: list[str]
    target_super_domain_elementary_domains: list[str]
    target_elementary_domains: list[str]

    target_fraction: float
    seed: int

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def materialize(self, dataset_view):
        """
        Build concrete ScenarioData.

        Supports:
            features -> X [trials, features]
            signal   -> X [trials, ..., channels, time]
        """

        X, y, metadata = _load_view_data(
            dataset_view
        )

        elementary_domains = _build_elementary_domains(
            metadata,
            self.scenario,
        )

        super_domains = _build_super_domains(
            metadata,
            self.scenario,
        )

        source = _build_group(
            X=X,
            y=y,
            elementary_domains=elementary_domains,
            super_domains=super_domains,
            selected_domains=self.source_elementary_domains,
            partition="train",
        )

        target_super_domain = _build_group(
            X=X,
            y=y,
            elementary_domains=elementary_domains,
            super_domains=super_domains,
            selected_domains=(
                self.target_super_domain_elementary_domains
            ),
            partition="calibration",
        )

        target_elementary_domain = _build_target_group(
            X=X,
            y=y,
            elementary_domains=elementary_domains,
            super_domains=super_domains,
            selected_domains=self.target_elementary_domains,
            fraction=self.target_fraction,
            seed=self.seed,
            scenario=self.scenario,
        )

        return ScenarioData(
            source=source,
            target_super_domain=target_super_domain,
            target_elementary_domain=target_elementary_domain,
        )


# ============================================================
# Data loading
# ============================================================

def _load_view_data(view):
    """
    Normalize feature and signal representations into:

        X
        y
        metadata

    X always uses trials on axis 0.
    """

    data = load_data(
        view.path
    )

    representation = getattr(
        view,
        "representation",
        "features",
    )

    if representation == "features":

        X = data[
            view.feature_columns
        ].to_numpy()

        y = data[
            view.label_column
        ].to_numpy()

        metadata = data

        return X, y, metadata

    if representation == "signal":

        X = np.asarray(
            data["X"]
        )

        y = np.asarray(
            data[view.label_column]
        )

        columns = list(dict.fromkeys(
            view.domain_columns
            + view.metadata_columns
            + [view.label_column]
        ))

        metadata = pd.DataFrame({
            column: data[column]
            for column in columns
        })

        if len(X) != len(metadata):
            raise ValueError(
                "Signal and metadata sample counts do not match."
            )

        return X, y, metadata

    raise ValueError(
        f"Unknown representation: {representation}"
    )


# ============================================================
# Domain construction
# ============================================================

def _build_elementary_domains(
    metadata,
    scenario,
):
    if scenario == "cross_session":
        columns = [
            "dataset",
            "subject",
            "session",
        ]

    else:
        columns = [
            "dataset",
            "subject",
        ]

    return (
        metadata[columns]
        .astype(str)
        .agg("|".join, axis=1)
        .to_numpy()
    )


def _build_super_domains(
    metadata,
    scenario,
):
    if scenario != "cross_dataset":
        return None

    return (
        metadata["dataset"]
        .astype(str)
        .to_numpy()
    )


# ============================================================
# DataGroup helpers
# ============================================================

def _build_group(
    X,
    y,
    elementary_domains,
    super_domains,
    selected_domains,
    partition,
):
    if not selected_domains:
        return None

    mask = np.isin(
        elementary_domains,
        selected_domains,
    )

    if not np.any(mask):
        return None

    return DataGroup(
        X=X[mask],
        y=y[mask],

        elementary_domains=(
            elementary_domains[mask]
        ),

        partitions=np.full(
            np.sum(mask),
            partition,
            dtype=object,
        ),

        super_domains=(
            None
            if super_domains is None
            else super_domains[mask]
        ),
    )


def _build_target_group(
    X,
    y,
    elementary_domains,
    super_domains,
    selected_domains,
    fraction,
    seed,
    scenario,
):
    if not selected_domains:
        return None

    if not 0 <= fraction <= 1:
        raise ValueError(
            "'target_fraction' must be between 0 and 1."
        )

    mask = np.isin(
        elementary_domains,
        selected_domains,
    )

    indices = np.flatnonzero(
        mask
    )

    if len(indices) == 0:
        return None

    # --------------------------------------------------------
    # Select train/calibration samples
    # --------------------------------------------------------

    if fraction == 0:

        selected_indices = np.array(
            [],
            dtype=int,
        )

    elif fraction == 1:

        selected_indices = indices

    else:

        try:
            selected_indices, _ = train_test_split(
                indices,
                train_size=fraction,
                random_state=seed,
                stratify=y[indices],
            )

        except ValueError:
            selected_indices, _ = train_test_split(
                indices,
                train_size=fraction,
                random_state=seed,
            )

    selected_set = set(
        selected_indices
    )

    first_partition = (
        "train"
        if scenario == "intra_subject"
        else "calibration"
    )

    partitions = np.asarray(
        [
            first_partition
            if index in selected_set
            else "test"
            for index in indices
        ],
        dtype=object,
    )

    return DataGroup(
        X=X[indices],
        y=y[indices],

        elementary_domains=(
            elementary_domains[indices]
        ),

        partitions=partitions,

        super_domains=(
            None
            if super_domains is None
            else super_domains[indices]
        ),
    )