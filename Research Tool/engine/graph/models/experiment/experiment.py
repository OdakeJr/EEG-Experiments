# graph/models/experiment/experiment.py

from dataclasses import dataclass, field

from .block import Block


#@dataclass
#class Experiment:
#    """Defines a complete experiment configuration."""

#    id: str
#    blocks: list[Block]
#    name: str | None = None
#    metadata: dict = field(default_factory=dict)
    
    
from typing import Optional

@dataclass
class Experiment:
    """Defines a complete experiment configuration."""

    id: str
    blocks: list[Block]
    name: Optional[str] = None
    metadata: dict = field(default_factory=dict)