# models/model_result.py

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class ModelResult:
    split_id: str
    scenario: str
    group: str

    n_source_domains: int
    n_target_super_domains: int
    target_fraction: float
    split_seed: int

    source_domains: str
    target_super_domains: str
    target_domains: str

    representation_signature: str
    learning_method: str
    model_name: str
    model_signature: str

    evaluation_group: str
    partition: str
    n_samples: int

    training_seed: Optional[int] = None

    accuracy: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    auc: Optional[float] = None

    training_time: Optional[float] = None
    inference_time: Optional[float] = None
    inference_time_per_sample: Optional[float] = None
    model_size_bytes: Optional[int] = None
    n_parameters: Optional[int] = None

    def to_dict(self):
        return asdict(self)