# engine/orchestration/reconciliation.py

from engine.orchestration.models.state import RunState


def reconcile(old_graph, old_state, new_graph):
    """
    Reconcile a previous execution with a newly generated graph.

    Unchanged successful nodes keep their status and output reference.
    Changed/new nodes and all their downstream dependencies become pending.
    """

    new_state = RunState(run_id=new_graph.id)

    old_nodes = {
        node.id: node
        for node in old_graph.nodes
    }

    new_nodes = {
        node.id: node
        for node in new_graph.nodes
    }

    invalid_nodes = set()

    # ---------------------------------------------------------
    # Detect new or changed nodes
    # ---------------------------------------------------------

    for node_id, new_node in new_nodes.items():

        old_node = old_nodes.get(node_id)

        # New node.
        if old_node is None:
            invalid_nodes.add(node_id)
            continue

        # Existing node whose definition changed.
        if _node_changed(
            old_node,
            new_node,
            old_graph,
            new_graph
        ):
            invalid_nodes.add(node_id)

    # ---------------------------------------------------------
    # Invalidate everything downstream of changed nodes
    # ---------------------------------------------------------

    invalid_nodes = _expand_downstream(
        new_graph,
        invalid_nodes
    )

    # ---------------------------------------------------------
    # Build the new state
    # ---------------------------------------------------------

    for node_id in new_nodes:

        # Changed/new/downstream nodes remain pending.
        if node_id in invalid_nodes:
            continue

        old_status = old_state.get_status(node_id)
        old_output = old_state.get_output(node_id)

        # Only completed nodes with an output can be reused.
        if old_status == "success" and old_output is not None:
            new_state.set_status(node_id, "success")
            new_state.set_output(node_id, old_output)

    return new_state


def _node_changed(old_node, new_node, old_graph, new_graph):
    """Check whether an existing node changed."""

    if old_node.block_id != new_node.block_id:
        return True

    if old_node.variant_id != new_node.variant_id:
        return True

    if old_node.pipe != new_node.pipe:
        return True

    if old_node.params != new_node.params:
        return True

    # Changes in incoming connections also invalidate the node.
    old_inputs = _get_inputs(old_graph, old_node.id)
    new_inputs = _get_inputs(new_graph, new_node.id)

    if old_inputs != new_inputs:
        return True

    return False


def _get_inputs(graph, node_id):
    """Return the set of nodes connected into a node."""

    return {
        edge.source
        for edge in graph.edges
        if edge.target == node_id
    }


def _expand_downstream(graph, invalid_nodes):
    """Recursively invalidate all nodes downstream of invalid nodes."""

    invalid_nodes = set(invalid_nodes)

    changed = True

    while changed:
        changed = False

        for edge in graph.edges:

            if (
                edge.source in invalid_nodes
                and edge.target not in invalid_nodes
            ):
                invalid_nodes.add(edge.target)
                changed = True

    return invalid_nodes