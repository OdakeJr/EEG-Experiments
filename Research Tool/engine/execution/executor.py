# engine/execution/executor.py

from engine.execution.models.result import ExecutionResult


class Executor:
    """Executes individual graph nodes."""

    def __init__(self, registry):
        self.registry = registry

    def execute(self, node, inputs):
        """Execute one node and return a standardized result."""

        try:
            # Resolve the pipe implementation.
            pipe = self.registry.get(node.pipe)

            # Execute the node.
            output = pipe.run(
                inputs=inputs,
                params=node.params
            )

            return ExecutionResult(
                node_id=node.id,
                status="success",
                output=output
            )

        except Exception as error:
            return ExecutionResult(
                node_id=node.id,
                status="failed",
                output=None,
                error=str(error)
            )