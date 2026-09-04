# plugins/dummy/dummy.py

from plugins.pipe import Pipe


class SourcePipe(Pipe):
    """Create initial nodes without upstream inputs."""

    def expand(self, input_nodes, params):
        return [
            {
                "inputs": [],
                "params": {"value": value}
            }
            for value in params.get("values", [])
        ]

    def run(self, inputs, params):
        return {
            "type": "source",
            "value": params["value"]
        }


class OneToOnePipe(Pipe):
    """Create one node for each input node."""

    def expand(self, input_nodes, params):
        return [
            {
                "inputs": [node.id],
                "params": params.copy()
            }
            for node in input_nodes
        ]

    def run(self, inputs, params):
        return {
            "type": "one_to_one",
            "inputs": inputs,
            "params": params
        }


class OneToManyPipe(Pipe):
    """Create several nodes for each input node."""

    def expand(self, input_nodes, params):
        values = params.get("values", [])
        expanded = []

        for node in input_nodes:
            for value in values:
                node_params = params.copy()
                node_params["value"] = value

                expanded.append({
                    "inputs": [node.id],
                    "params": node_params
                })

        return expanded

    def run(self, inputs, params):
        return {
            "type": "one_to_many",
            "inputs": inputs,
            "params": params
        }


class ManyToOnePipe(Pipe):
    """Create one node that receives all input nodes."""

    def expand(self, input_nodes, params):
        return [
            {
                "inputs": [node.id for node in input_nodes],
                "params": params.copy()
            }
        ]

    def run(self, inputs, params):
        return {
            "type": "many_to_one",
            "inputs": inputs,
            "params": params
        }