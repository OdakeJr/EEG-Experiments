from dataclasses import dataclass, field

@dataclass
class Node:
    id: str
    block_id: str
    variant_id: str
    pipe: str
    params: dict = field(default_factory=dict)