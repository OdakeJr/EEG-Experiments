
# models/dataset_view.py

from dataclasses import dataclass
from typing import Any


@dataclass
class DatasetView:
    path: str
    representation: str  # "features", "signal", "image"
    feature_columns: list[str] | None

    label_column: str
    domain_columns: list[str]
    metadata_columns: list[str]
    manifest_path: str

    # Input dimensions, excluding the sample axis.
    # Images: None until transformations are applied.
    input_shape: tuple[int, ...] | None = None

    # EEG-specific metadata.
    channel_names: list[str] | None = None
    band_labels: list[tuple[float, float]] | None = None

    # Image-specific metadata.
    image_path_column: str | None = "image_path"

    # Preprocessing trace.
    preprocessing_signature: str | None = None
    preprocessing_config_label: str | None = None
    preprocessing_params: dict[str, Any] | list[dict[str, Any]] | None = None
