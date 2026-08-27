from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class DomainResult:
    # Experiment identity
    split_id: str
    scenario: str
    group: str
    seed: int

    # Scenario configuration
    target_fraction: float

    # Pairwise comparison
    comparison: str

    left_group: str
    right_group: str

    left_domains: str
    right_domains: str

    n_left_domains: int
    n_right_domains: int

    # Discrepancy definition
    metric: str
    metric_signature: str

    representation: str
    class_label: Optional[str]

    # Measured discrepancy
    value: float

    # Samples used
    n_left_samples: int
    n_right_samples: int

    def to_dict(self):
        return asdict(self)