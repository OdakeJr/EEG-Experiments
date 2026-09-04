from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ModelResult:
    # Experiment identity
    split_id: str
    scenario: str
    group: str

    # Scenario configuration
    n_source_domains: int
    n_target_super_domains: int
    target_fraction: float
    split_seed: int

    # Domain identity
    source_domains: str
    target_super_domains: str
    target_domains: str

    # Training configuration
    feature_selection_signature: str
    learning_method: str
    model_name: str
    model_signature: str

    # Evaluation location
    evaluation_group: str
    partition: str
    n_samples: int

    # Optional training information
    training_seed: Optional[int] = None

    # Predictive performance
    accuracy: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    auc: Optional[float] = None

    # Computational information
    training_time: Optional[float] = None
    inference_time: Optional[float] = None
    inference_time_per_sample: Optional[float] = None
    model_size_bytes: Optional[int] = None
    n_parameters: Optional[int] = None

    def to_dict(self):
        return asdict(self)
