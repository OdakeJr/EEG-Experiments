from dataclasses import dataclass
from typing import Any


@dataclass
class ModelArtifact:
    split_id: str
    feature_selection_signature: str

    learning_method: str
    model_name: str

    model_path: str
    manifest_path: str

    signature: str

    # Clean feature-selection trace
    feature_selection_method: str | None = None

    feature_selection_params: dict[str, Any] | None = None

    feature_selection_config_label: str | None = None

    # Upstream preprocessing trace
    preprocessing_signature: str | None = None

    preprocessing_config_label: str | None = None