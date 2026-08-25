from dataclasses import dataclass


@dataclass
class ModelArtifact:
    split_id: str
    feature_selection_signature: str

    learning_method: str
    model_name: str

    model_path: str
    manifest_path: str

    signature: str