# visualization/execution.py

import matplotlib.pyplot as plt
import networkx as nx


def visualize_execution(graph, state):
    """Display the graph together with the execution status of each node."""

    network = nx.DiGraph()

    status_colors = {
        "pending": "lightgray",
        "running": "gold",
        "success": "lightgreen",
        "failed": "lightcoral"
    }

    # Add nodes.
    for node in graph.nodes:

        status = state.get_status(node.id)

        network.add_node(
            node.id,
            label=f"{node.block_id}\n{node.variant_id}\n[{status}]",
            status=status
        )

    # Add edges.
    for edge in graph.edges:
        network.add_edge(edge.source, edge.target)

    positions = nx.spring_layout(network, seed=42)

    labels = {
        node_id: data["label"]
        for node_id, data in network.nodes(data=True)
    }

    node_colors = [
        status_colors.get(
            data["status"],
            "lightgray"
        )
        for _, data in network.nodes(data=True)
    ]

    nx.draw(
        network,
        positions,
        labels=labels,
        with_labels=True,
        node_color=node_colors,
        node_size=2800,
        font_size=8,
        arrows=True
    )

    plt.tight_layout()
    plt.show()