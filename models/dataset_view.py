from dataclasses import dataclass
from typing import Any


@dataclass
class DatasetView:

    path: str

    # Representation stored at path:
    # "features" -> X shape [trials, features]
    # "signal"   -> X shape [trials, bands, channels, time]
    representation: str

    # Only used for feature representation.
    # None when representation == "signal".
    feature_columns: list[str] | None

    label_column: str
    domain_columns: list[str]
    metadata_columns: list[str]

    manifest_path: str

    # Shape of one input sample, excluding the trial axis.
    # Examples:
    # features -> (250,)
    # signal   -> (2, 22, 384)
    input_shape: tuple[int, ...] | None = None

    # Useful for signal-based transformations such as CSP.
    channel_names: list[str] | None = None

    # Example: [(8, 12), (13, 30)]
    band_labels: list[tuple[float, float]] | None = None

    # Clean preprocessing trace
    preprocessing_signature: str | None = None
    preprocessing_config_label: str | None = None
    preprocessing_params: (
        dict[str, Any]
        | list[dict[str, Any]]
        | None
    ) = None