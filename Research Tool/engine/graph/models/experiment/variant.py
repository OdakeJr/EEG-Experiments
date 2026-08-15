# graph/models/experiment/variant.py

from dataclasses import dataclass, field
from typing import Any

@dataclass
class Variant:
    id: str
    pipe: str
    params: dict = field(default_factory=dict)