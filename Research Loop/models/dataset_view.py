from dataclasses import dataclass


@dataclass
class DatasetView:
    path: str
    feature_columns: list[str]
    label_column: str
    domain_columns: list[str]
    metadata_columns: list[str]
    manifest_path: str