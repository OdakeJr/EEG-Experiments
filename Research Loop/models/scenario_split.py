from dataclasses import asdict, dataclass

import numpy as np
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

    # Elementary domains used as source.
    source_elementary_domains: list[str]

    # Elementary domains from the target super-domain
    # used in the first adaptation stage.
    target_super_domain_elementary_domains: list[str]

    # Final target elementary domain(s).
    target_elementary_domains: list[str]

    # Fraction of the final target elementary domain used
    # for train/calibration. The remainder is test.
    target_fraction: float

    seed: int

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def materialize(self, dataset_view):
        """
        Build the concrete ScenarioData for this split.
        """

        dataframe = load_data(
            dataset_view.path
        )

        # --------------------------------------------------
        # Domain structure
        # --------------------------------------------------

        elementary_domains = _build_elementary_domains(
            dataframe,
            self.scenario,
        )

        super_domains = _build_super_domains(
            dataframe,
            self.scenario,
        )

        # --------------------------------------------------
        # Source
        # --------------------------------------------------

        source = _build_group(
            dataframe=dataframe,
            elementary_domains=elementary_domains,
            super_domains=super_domains,
            selected_domains=self.source_elementary_domains,
            feature_columns=dataset_view.feature_columns,
            label_column=dataset_view.label_column,
            partition="train",
        )

        # --------------------------------------------------
        # Target super-domain adaptation
        # --------------------------------------------------

        target_super_domain = _build_group(
            dataframe=dataframe,
            elementary_domains=elementary_domains,
            super_domains=super_domains,
            selected_domains=(
                self.target_super_domain_elementary_domains
            ),
            feature_columns=dataset_view.feature_columns,
            label_column=dataset_view.label_column,
            partition="calibration",
        )

        # --------------------------------------------------
        # Final target elementary domain
        # --------------------------------------------------

        target_elementary_domain = _build_target_group(
            dataframe=dataframe,
            elementary_domains=elementary_domains,
            super_domains=super_domains,
            selected_domains=self.target_elementary_domains,
            feature_columns=dataset_view.feature_columns,
            label_column=dataset_view.label_column,
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
# Domain construction
# ============================================================

def _build_elementary_domains(
    dataframe,
    scenario,
):
    """
    Define the elementary domain according to the scenario.

    Intra-subject  -> subject
    Cross-session  -> session
    Cross-subject  -> subject
    Cross-dataset  -> subject
    """

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
        dataframe[columns]
        .astype(str)
        .agg("|".join, axis=1)
    )


def _build_super_domains(
    dataframe,
    scenario,
):
    """
    Define the higher-level domain structure.

    Currently only cross-dataset uses super-domains,
    where each dataset is one super-domain.
    """

    if scenario != "cross_dataset":
        return None

    return (
        dataframe["dataset"]
        .astype(str)
    )


# ============================================================
# DataGroup helpers
# ============================================================

def _build_group(
    dataframe,
    elementary_domains,
    super_domains,
    selected_domains,
    feature_columns,
    label_column,
    partition,
):
    """
    Build a DataGroup with a single partition.
    """

    if not selected_domains:
        return None

    mask = elementary_domains.isin(
        selected_domains
    )

    selected = dataframe.loc[
        mask
    ]

    if selected.empty:
        return None

    if super_domains is None:
        selected_super_domains = None

    else:
        selected_super_domains = (
            super_domains.loc[
                mask
            ].to_numpy()
        )

    return DataGroup(
        X=selected[
            feature_columns
        ].to_numpy(),

        y=selected[
            label_column
        ].to_numpy(),

        elementary_domains=(
            elementary_domains.loc[
                mask
            ].to_numpy()
        ),

        super_domains=(
            selected_super_domains
        ),

        partitions=np.full(
            len(selected),
            partition,
            dtype=object,
        ),
    )


def _build_target_group(
    dataframe,
    elementary_domains,
    super_domains,
    selected_domains,
    feature_columns,
    label_column,
    fraction,
    seed,
    scenario,
):
    """
    Build the final target elementary-domain group and divide
    its samples between train/calibration and test.
    """

    if not selected_domains:
        return None

    if not 0 <= fraction <= 1:
        raise ValueError(
            "'target_fraction' must be between 0 and 1."
        )

    mask = elementary_domains.isin(
        selected_domains
    )

    selected = dataframe.loc[
        mask
    ]

    if selected.empty:
        return None

    indices = selected.index.to_numpy()

    # --------------------------------------------------
    # Select train/calibration samples
    # --------------------------------------------------

    if fraction == 0:

        selected_indices = np.array(
            [],
            dtype=indices.dtype,
        )

    elif fraction == 1:

        selected_indices = indices

    else:

        labels = dataframe.loc[
            indices,
            label_column,
        ]

        try:

            selected_indices, _ = train_test_split(
                indices,
                train_size=fraction,
                random_state=seed,
                stratify=labels,
            )

        except ValueError:

            selected_indices, _ = train_test_split(
                indices,
                train_size=fraction,
                random_state=seed,
            )

    selected_indices = set(
        selected_indices
    )

    # --------------------------------------------------
    # Partition names
    # --------------------------------------------------

    first_partition = (
        "train"
        if scenario == "intra_subject"
        else "calibration"
    )

    partitions = np.asarray(
        [
            first_partition
            if index in selected_indices
            else "test"
            for index in indices
        ],
        dtype=object,
    )

    if super_domains is None:
        selected_super_domains = None

    else:
        selected_super_domains = (
            super_domains.loc[
                mask
            ].to_numpy()
        )

    return DataGroup(
        X=selected[
            feature_columns
        ].to_numpy(),

        y=selected[
            label_column
        ].to_numpy(),

        elementary_domains=(
            elementary_domains.loc[
                mask
            ].to_numpy()
        ),

        super_domains=(
            selected_super_domains
        ),

        partitions=partitions,
    )