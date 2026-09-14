# models/model_artifact.py

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelArtifact:
    split_id: str
    representation_signature: str

    learning_method: str
    model_name: str

    model_path: str
    manifest_path: str

    signature: str

    input_representation: str | None = None
    output_representation: str | None = None
    model_input_representation: str | None = None

    representation_method: str | None = None
    representation_params: dict[str, Any] | None = None
    representation_config_label: str | None = None

    signal_transform_method: str | None = None
    signal_transform_params: dict[str, Any] | None = None
    signal_transform_config_label: str | None = None

    feature_extraction_method: str | None = None
    feature_extraction_params: dict[str, Any] | None = None
    feature_extraction_config_label: str | None = None

    feature_selection_method: str | None = None
    feature_selection_params: dict[str, Any] | None = None
    feature_selection_config_label: str | None = None

    preprocessing_signature: str | None = None
    preprocessing_config_label: str | None = None