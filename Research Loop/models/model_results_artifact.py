# models/model_results_artifact.py

from dataclasses import dataclass


@dataclass
class ModelResultsArtifact:
    path: str
    manifest_path: str
    signature: str
    n_rows: int