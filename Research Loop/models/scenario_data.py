# models/scenario_data.py

from dataclasses import dataclass

import numpy as np


@dataclass
class DataGroup:
    """
    Concrete data belonging to one stage of a scenario.
    """

    X: np.ndarray
    y: np.ndarray

    elementary_domains: np.ndarray
    partitions: np.ndarray

    # Used when a higher-level domain structure exists,
    # mainly in cross-dataset experiments.
    super_domains: np.ndarray | None = None


@dataclass
class ScenarioData:
    """
    Materialized data for one concrete scenario split.
    """

    source: DataGroup | None

    target_super_domain: DataGroup | None
    target_elementary_domain: DataGroup | None