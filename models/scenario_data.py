
# models/scenario_data.py

from dataclasses import dataclass

import numpy as np


@dataclass
class DataGroup:
    # EEG: numerical arrays; images: file paths.
    X: np.ndarray
    y: np.ndarray
    elementary_domains: np.ndarray
    partitions: np.ndarray
    super_domains: np.ndarray | None = None


@dataclass
class ScenarioData:
    source: DataGroup | None
    target_super_domain: DataGroup | None
    target_elementary_domain: DataGroup | None
