from dataclasses import dataclass


@dataclass
class FeatureSelectionArtifact:
    split_id: str
    method: str

    transformer_path: str
    manifest_path: str

    signature: str