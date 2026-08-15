# graph/models/experiment/block.py

from dataclasses import dataclass, field
from typing import Any

from .variant import Variant

@dataclass
class Block:
    """Defines one configurable step of an experiment."""

    id: str
    inputs: list[str]
    variants: list[Variant]