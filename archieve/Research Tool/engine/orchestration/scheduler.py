# engine/orchestration/scheduler.py

class Scheduler:
    """Determines which graph nodes are ready to execute."""

    def get_ready_nodes(self, graph, state):
        """
        Return all pending nodes whose dependencies have completed successfully.
        """

        ready_nodes = []

        for node in graph.nodes:

            # Skip nodes that are not pending.
            if state.get_status(node.id) != "pending":
                continue

            dependencies = self._get_dependencies(graph, node.id)

            # A node is ready if all dependency nodes succeeded.
            if all(
                state.get_status(dependency_id) == "success"
                for dependency_id in dependencies
            ):
                ready_nodes.append(node)

        return ready_nodes

    def _get_dependencies(self, graph, node_id):
        """Return the IDs of the nodes that feed into the given node."""

        return [
            edge.source
            for edge in graph.edges
            if edge.target == node_id
        ]