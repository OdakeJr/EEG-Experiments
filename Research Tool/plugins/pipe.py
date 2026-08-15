from abc import ABC, abstractmethod


class Pipe(ABC):

    @abstractmethod
    def expand(self, input_nodes, params):
        """
        Define the concrete nodes that should be created.

        Returns a list of dictionaries containing:
        - inputs: IDs of the input nodes
        - params: resolved parameters for this concrete node
        """
        pass

    @abstractmethod
    def run(self, inputs, params):
        """Execute one concrete node."""
        pass