from dataclasses import dataclass

from .node import Node
from .edge import Edge

@dataclass
class Graph:
    id: str
    experiment_id: str
    nodes: list[Node]
    edges: list[Edge]

