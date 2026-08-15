# tests/test_graph.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from engine.graph.builder import build_graph
from engine.graph.models.experiment.experiment import Experiment
from engine.graph.models.experiment.block import Block
from engine.graph.models.experiment.variant import Variant
from engine.registry.registry import Registry


registry = Registry(
    PROJECT_ROOT / "config/registry/pipes.json"
)

experiment = Experiment(
    id="dummy_experiment",
    name="Dummy graph test",
    blocks=[
        Block(
            id="source",
            inputs=[],
            variants=[
                Variant(
                    id="source_variant",
                    pipe="dummy.source",
                    params={"values": [1, 2]}
                )
            ]
        ),
        Block(
            id="process",
            inputs=["source"],
            variants=[
                Variant(
                    id="process_variant",
                    pipe="dummy.one_to_one",
                    params={"example": True}
                )
            ]
        ),
        Block(
            id="collect",
            inputs=["process"],
            variants=[
                Variant(
                    id="collect_variant",
                    pipe="dummy.many_to_one",
                    params={}
                )
            ]
        )
    ]
)

graph = build_graph(experiment, registry)

print("\nNODES")
for node in graph.nodes:
    print(node)

print("\nEDGES")
for edge in graph.edges:
    print(edge)