# plugins/data_manipulation/result.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


# ============================================================
# Domain-defined dataset
# ============================================================

@dataclass
class DomainData:
    """
    Dataset after elementary domains have been defined.

    Example domain definition:
        dataset + subject + session

    The DataFrame still contains all samples. No source/target
    assignment has been made yet.
    """

    data: pd.DataFrame

    # Columns used to define an elementary domain.
    domain_columns: List[str]

    # Column containing the generated domain identifier.
    domain_id_column: str = "domain_id"

    # Feature/target information propagated from the input.
    feature_columns: Optional[List[str]] = None
    label_column: str = "label"

    metadata: Dict = field(default_factory=dict)


# ============================================================
# Source / target domain specification
# ============================================================

@dataclass
class DomainSplit:
    """
    One source-target domain configuration.

    This describes which domains belong to source and target,
    without yet performing train/test or adaptation splitting.
    """

    id: str

    source_domain_ids: List[str]
    target_domain_ids: List[str]

    # Useful for cases where source and target correspond to the
    # same elementary domain and samples must be separated later.
    requires_sample_split: bool = False

    metadata: Dict = field(default_factory=dict)


# ============================================================
# Collection of source / target configurations
# ============================================================

@dataclass
class DomainSplitCollection:
    """
    Output of a domain-construction strategy.

    A single strategy may generate many configurations, e.g.
    LOSO may generate one DomainSplit per held-out subject.
    """

    domain_data: DomainData

    splits: List[DomainSplit]

    strategy: str

    metadata: Dict = field(default_factory=dict)


# ============================================================
# Materialized train/adaptation/test data
# ============================================================

@dataclass
class SplitData:
    """
    Materialized data for one experimental configuration.

    Data remain as DataFrames so metadata such as dataset,
    subject, session and domain_id are preserved.

    The learning stage can later convert these into X/y arrays.
    """

    id: str

    # Main labeled source data used for training.
    source_train: pd.DataFrame

    # Optional source holdout data.
    source_test: Optional[pd.DataFrame] = None

    # Target samples made available to the learning algorithm.
    # Depending on the scenario these may be labeled or unlabeled.
    target_adapt: Optional[pd.DataFrame] = None

    # Completely held-out target evaluation data.
    target_test: Optional[pd.DataFrame] = None

    # Whether labels in target_adapt are allowed to be used.
    target_labels_available: bool = False

    feature_columns: Optional[List[str]] = None
    label_column: str = "label"
    domain_id_column: str = "domain_id"

    metadata: Dict = field(default_factory=dict)


# ============================================================
# Collection of materialized splits
# ============================================================

@dataclass
class SplitDataCollection:
    """
    Collection produced when one domain strategy creates many
    train/adaptation/test configurations.
    """

    splits: List[SplitData]

    strategy: Optional[str] = None

    metadata: Dict = field(default_factory=dict)