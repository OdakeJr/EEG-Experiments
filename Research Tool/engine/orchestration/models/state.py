from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class RunState:
    """Tracks the execution state of nodes within one run."""

    run_id: str

    node_status: Dict[str, str] = field(default_factory=dict)
    output_keys: Dict[str, str] = field(default_factory=dict)
    errors: Dict[str, Optional[str]] = field(default_factory=dict)

    def set_status(self, node_id, status):
        """Update the execution status of a node."""
        self.node_status[node_id] = status

    def set_output(self, node_id, key):
        """Associate a stored output with a node."""
        self.output_keys[node_id] = key

    def set_error(self, node_id, error):
        """Associate an execution error with a node."""
        self.errors[node_id] = error

    def get_status(self, node_id):
        """Return the current status of a node."""
        return self.node_status.get(node_id, "pending")

    def get_output(self, node_id):
        """Return the stored output key associated with a node."""
        return self.output_keys.get(node_id)