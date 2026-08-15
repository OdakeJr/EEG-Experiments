# engine/graph/validation.py


def validate_experiment(experiment):
    """Validate the structure and dependencies of an experiment."""

    if not experiment.blocks:
        raise ValueError("Experiment must contain at least one block.")

    block_ids = [block.id for block in experiment.blocks]

    # Block IDs must be unique.
    if len(block_ids) != len(set(block_ids)):
        raise ValueError("Block IDs must be unique.")

    block_id_set = set(block_ids)

    for block in experiment.blocks:

        if not block.id:
            raise ValueError("Block ID cannot be empty.")

        # Variant IDs must be unique inside each block.
        variant_ids = [variant.id for variant in block.variants]

        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError(
                f"Variant IDs must be unique inside block '{block.id}'."
            )

        if not block.variants:
            raise ValueError(
                f"Block '{block.id}' must contain at least one variant."
            )

        for variant in block.variants:

            if not variant.id:
                raise ValueError(
                    f"Variant ID cannot be empty in block '{block.id}'."
                )

            if not variant.pipe:
                raise ValueError(
                    f"Variant '{variant.id}' in block '{block.id}' "
                    "must define a pipe."
                )

        # Validate block inputs.
        for input_id in block.inputs:

            if input_id == block.id:
                raise ValueError(
                    f"Block '{block.id}' cannot depend on itself."
                )

            if input_id not in block_id_set:
                raise ValueError(
                    f"Block '{block.id}' references unknown input "
                    f"block '{input_id}'."
                )

    _validate_block_cycles(experiment.blocks)


def validate_graph(graph):
    """Validate nodes, edges, and dependencies of a generated graph."""

    if not graph.nodes:
        raise ValueError("Graph must contain at least one node.")

    node_ids = [node.id for node in graph.nodes]

    # Node IDs must be unique.
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Node IDs must be unique.")

    node_id_set = set(node_ids)

    for edge in graph.edges:

        if edge.source not in node_id_set:
            raise ValueError(
                f"Edge references unknown source node '{edge.source}'."
            )

        if edge.target not in node_id_set:
            raise ValueError(
                f"Edge references unknown target node '{edge.target}'."
            )

        if edge.source == edge.target:
            raise ValueError(
                f"Node '{edge.source}' cannot have an edge to itself."
            )

    _validate_graph_cycles(graph)


def _validate_block_cycles(blocks):
    """Check whether block dependencies contain a cycle."""

    dependencies = {
        block.id: list(block.inputs)
        for block in blocks
    }

    _detect_cycles(dependencies, "experiment blocks")


def _validate_graph_cycles(graph):
    """Check whether graph edges contain a cycle."""

    dependencies = {
        node.id: []
        for node in graph.nodes
    }

    for edge in graph.edges:
        dependencies[edge.target].append(edge.source)

    _detect_cycles(dependencies, "graph nodes")


def _detect_cycles(dependencies, structure_name):
    """Detect cycles in a dependency dictionary using depth-first search."""

    visited = set()
    visiting = set()

    def visit(item):

        if item in visiting:
            raise ValueError(
                f"Cycle detected in {structure_name} involving '{item}'."
            )

        if item in visited:
            return

        visiting.add(item)

        for dependency in dependencies[item]:
            visit(dependency)

        visiting.remove(item)
        visited.add(item)

    for item in dependencies:
        visit(item)