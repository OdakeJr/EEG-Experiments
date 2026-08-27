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
    # Example:
    #   cross-subject -> subjects
    #   cross-dataset -> subjects from source datasets
    source_elementary_domains: list[str]

    # Elementary domains from the target super-domain
    # used in the first adaptation stage.
    # Example:
    #   cross-dataset -> other subjects from held-out dataset
    target_super_domain_elementary_domains: list[str]

    # Final target elementary domain(s).
    # Example:
    #   cross-subject -> held-out subject
    #   cross-dataset -> held-out subject from held-out dataset
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

        In cross-dataset experiments:
            elementary domain = subject
            super domain      = dataset

        Therefore, source data can also have both:
            source.elementary_domains = source subjects
            source.super_domains      = source datasets
        """

        dataframe = load_data(
            dataset_view.path
        )

        # --------------------------------------------------
        # Domain structure
        # --------------------------------------------------

        elementary_domains = _build_elementary_domains(
            dataframe=dataframe,
            scenario=self.scenario,
        )

        super_domains = _build_super_domains(
            dataframe=dataframe,
            scenario=self.scenario,
        )

        # --------------------------------------------------
        # Source block
        # --------------------------------------------------
        # For cross-dataset:
        #   elementary_domains -> source subjects
        #   super_domains      -> source datasets
        #
        # For other scenarios:
        #   super_domains is None.

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
        # Optional target super-domain block
        # --------------------------------------------------
        # Mainly useful for future two-stage adaptation.
        # For the current simplified cross-dataset benchmark,
        # this can simply be empty.

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
        # Final target elementary-domain block
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

    Intra-subject  -> dataset | subject
    Cross-session  -> dataset | subject | session
    Cross-subject  -> dataset | subject
    Cross-dataset  -> dataset | subject

    In cross-dataset, the elementary domain is still the
    subject, but the dataset prefix keeps subject IDs unique
    across datasets.
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

    Cross-dataset:
        super-domain = dataset

    Other scenarios:
        no explicit super-domain is needed.
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

def _select_super_domains(
    super_domains,
    mask,
):
    """
    Select super-domain labels when available.
    """

    if super_domains is None:
        return None

    return (
        super_domains.loc[
            mask
        ].to_numpy()
    )


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

    The same function is used for source and target blocks.
    Therefore, in cross-dataset experiments, both source and
    target blocks receive their dataset-level super-domain labels.
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

    selected_super_domains = _select_super_domains(
        super_domains=super_domains,
        mask=mask,
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

        partitions=np.full(
            len(selected),
            partition,
            dtype=object,
        ),

        super_domains=selected_super_domains,
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

    selected_super_domains = _select_super_domains(
        super_domains=super_domains,
        mask=mask,
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

        partitions=partitions,

        super_domains=selected_super_domains,
    )