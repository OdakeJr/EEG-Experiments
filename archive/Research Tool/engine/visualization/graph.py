# engine/graph/visualization.py

import matplotlib.pyplot as plt
import networkx as nx


def visualize_graph(graph):
    """Display a simple visualization of the experiment graph."""

    network = nx.DiGraph()

    # Add nodes.
    for node in graph.nodes:
        network.add_node(
            node.id,
            label=f"{node.block_id}\n{node.variant_id}"
        )

    # Add edges.
    for edge in graph.edges:
        network.add_edge(edge.source, edge.target)

    # Compute positions automatically.
    positions = nx.spring_layout(network, seed=42)

    labels = {
        node_id: data["label"]
        for node_id, data in network.nodes(data=True)
    }

    nx.draw(
        network,
        positions,
        labels=labels,
        with_labels=True,
        node_size=2500,
        font_size=8,
        arrows=True
    )

    plt.tight_layout()
    plt.show()
    
    
