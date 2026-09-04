# tests/test_orchestration.py

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from engine.graph.builder import build_graph
from engine.graph.models.experiment.experiment import Experiment
from engine.graph.models.experiment.block import Block
from engine.graph.models.experiment.variant import Variant

from engine.registry.registry import Registry
from engine.execution.executor import Executor
from engine.orchestration.scheduler import Scheduler
from engine.orchestration.orchestrator import Orchestrator
from engine.storage.local import LocalStorage

from engine.visualization.execution import visualize_execution

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------

registry = Registry(
    PROJECT_ROOT / "config/registry/pipes.json"
)

storage = LocalStorage(
    PROJECT_ROOT / "tests/outputs"
)

executor = Executor(registry)
scheduler = Scheduler()

orchestrator = Orchestrator(
    scheduler=scheduler,
    executor=executor,
    storage=storage
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def create_experiment(example_value=True):
    """Create the dummy experiment with configurable process parameters."""

    return Experiment(
        id="dummy_experiment",
        name="Dummy orchestration test",
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
                        params={"example": example_value}
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


def get_node_times(graph, state):
    """Return modification times for all stored node outputs."""

    times = {}

    for node in graph.nodes:

        key = state.get_output(node.id)

        if key is not None:
            path = storage.root_path / f"{key}.pkl"
            times[node.id] = path.stat().st_mtime

    return times


# ------------------------------------------------------------
# RUN 1 - Initial execution
# ------------------------------------------------------------

print("\n--- RUN 1: INITIAL EXECUTION ---")

experiment_1 = create_experiment(example_value=True)
graph_1 = build_graph(experiment_1, registry)
state_1 = orchestrator.run(graph_1)

times_1 = get_node_times(graph_1, state_1)

for node_id, timestamp in times_1.items():
    print(node_id, "->", timestamp)


# Ensure modification times can differ.
time.sleep(1)


# ------------------------------------------------------------
# RUN 2 - Identical experiment
# ------------------------------------------------------------

print("\n--- RUN 2: IDENTICAL EXPERIMENT ---")

experiment_2 = create_experiment(example_value=True)
graph_2 = build_graph(experiment_2, registry)
state_2 = orchestrator.run(graph_2)

times_2 = get_node_times(graph_2, state_2)

for node_id in times_1:

    reused = times_1[node_id] == times_2[node_id]

    print(
        node_id,
        "->",
        "REUSED" if reused else "RECOMPUTED"
    )


time.sleep(1)


# ------------------------------------------------------------
# RUN 3 - Change process parameters
# ------------------------------------------------------------

print("\n--- RUN 3: MODIFIED PROCESS ---")

experiment_3 = create_experiment(example_value=False)
graph_3 = build_graph(experiment_3, registry)
state_3 = orchestrator.run(graph_3)

times_3 = get_node_times(graph_3, state_3)

for node_id in times_2:

    reused = times_2[node_id] == times_3[node_id]

    print(
        node_id,
        "->",
        "REUSED" if reused else "RECOMPUTED"
    )


# ------------------------------------------------------------
# Expected behavior
# ------------------------------------------------------------

print("\nEXPECTED:")
print("Run 2: all nodes should be REUSED.")
print("Run 3: source nodes should be REUSED.")
print("Run 3: process and collect nodes should be RECOMPUTED.")

visualize_execution(graph_3, state_3)

