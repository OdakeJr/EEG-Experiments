# engine/orchestration/orchestrator.py

from engine.orchestration.models.state import RunState
from engine.orchestration.reconciliation import reconcile


class Orchestrator:
    """Coordinates the execution and persistence of a complete graph."""

    def __init__(self, scheduler, executor, storage):
        self.scheduler = scheduler
        self.executor = executor
        self.storage = storage

    def run(self, graph):
        """Execute a graph, reusing valid results from previous executions."""

        graph_key = self._build_graph_key(graph)
        state_key = self._build_state_key(graph)

        # -----------------------------------------------------
        # Load previous execution, if available
        # -----------------------------------------------------

        if (
            self.storage.exists(graph_key)
            and self.storage.exists(state_key)
        ):
            old_graph = self.storage.load(graph_key)
            old_state = self.storage.load(state_key)

            state = reconcile(
                old_graph=old_graph,
                old_state=old_state,
                new_graph=graph
            )

        else:
            state = RunState(run_id=graph.id)

        # Store the current graph and reconciled state.
        self.storage.save(graph_key, graph)
        self.storage.save(state_key, state)

        # -----------------------------------------------------
        # Execute graph
        # -----------------------------------------------------

        while True:

            ready_nodes = self.scheduler.get_ready_nodes(
                graph,
                state
            )

            if not ready_nodes:
                break

            for node in ready_nodes:

                state.set_status(node.id, "running")
                self.storage.save(state_key, state)

                inputs = self._load_inputs(
                    graph,
                    node,
                    state
                )

                result = self.executor.execute(
                    node,
                    inputs
                )

                if result.status == "success":

                    output_key = self._build_output_key(
                        graph,
                        node
                    )

                    self.storage.save(
                        output_key,
                        result.output
                    )

                    state.set_output(
                        node.id,
                        output_key
                    )

                    state.set_status(
                        node.id,
                        "success"
                    )

                else:

                    state.set_error(
                        node.id,
                        result.error
                    )

                    state.set_status(
                        node.id,
                        "failed"
                    )

                # Persist progress after every node.
                self.storage.save(state_key, state)

        return state

    def _load_inputs(self, graph, node, state):
        """Load outputs from all parent nodes."""

        input_node_ids = [
            edge.source
            for edge in graph.edges
            if edge.target == node.id
        ]

        inputs = []

        for node_id in input_node_ids:

            output_key = state.get_output(node_id)

            if output_key is None:
                raise ValueError(
                    f"No stored output found for input node '{node_id}'."
                )

            inputs.append(
                self.storage.load(output_key)
            )

        return inputs

    def _build_output_key(self, graph, node):
        """Build the storage key for a node output."""

        return (
            f"{graph.experiment_id}/"
            f"{graph.id}/"
            f"nodes/{node.id}"
        )

    def _build_graph_key(self, graph):
        """Build the storage key for the graph."""

        return (
            f"{graph.experiment_id}/"
            f"{graph.id}/graph"
        )

    def _build_state_key(self, graph):
        """Build the storage key for the execution state."""

        return (
            f"{graph.experiment_id}/"
            f"{graph.id}/state"
        )