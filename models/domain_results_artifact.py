# models/domain_results_artifact.py

from dataclasses import dataclass


@dataclass
class DomainResultsArtifact:
    path: str
    manifest_path: str
    signature: str
    n_rows: int