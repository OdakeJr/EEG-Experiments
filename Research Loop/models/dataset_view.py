from dataclasses import dataclass
from typing import Any


@dataclass
class DatasetView:

    path: str
    feature_columns: list[str]
    label_column: str
    domain_columns: list[str]
    metadata_columns: list[str]
    manifest_path: str

    # Clean preprocessing trace
    preprocessing_signature: str | None = None
    preprocessing_config_label: str | None = None
    preprocessing_params: dict[str, Any] | list[dict[str, Any]] | None = None