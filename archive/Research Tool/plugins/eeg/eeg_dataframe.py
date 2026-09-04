# plugins/eeg/eeg_dataframe.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class EEGDataFrame:
    """Standardized output produced by EEG dataset preparation pipes."""

    # Final feature-level dataset used by downstream pipeline stages.
    data: pd.DataFrame

    # Dataset identification.
    dataset_name: str

    # EEG configuration used to generate the dataset.
    channels: List[str]
    sampling_rate: float

    # Columns containing extracted features.
    feature_columns: Optional[List[str]] = None

    # Name of the target/label column.
    label_column: str = "label"

    # Dataset-specific or preprocessing-specific information.
    metadata: Dict = field(default_factory=dict)