from dataclasses import dataclass, asdict


@dataclass
class AnalysisArtifact:
    # Analysis identity
    name: str

    # Output location
    output_dir: str

    # Named persistent outputs
    tables: dict[str, str]
    figures: dict[str, str]

    # Execution information
    manifest_path: str
    signature: str

    def to_dict(self):
        return asdict(self)