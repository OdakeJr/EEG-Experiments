import importlib
import json
from pathlib import Path


class Registry:
    """Maps pipe names to their Python implementations."""

    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self._mapping = self._load_mapping()

    def _load_mapping(self):
        """Load the pipe mapping from JSON."""

        with open(self.config_path, "r") as file:
            return json.load(file)

    def get(self, pipe_name):
        """Return an instance of the requested pipe."""

        if pipe_name not in self._mapping:
            raise KeyError(f"Pipe not registered: {pipe_name}")

        path = self._mapping[pipe_name]

        module_path, class_name = path.rsplit(".", 1)

        module = importlib.import_module(module_path)
        pipe_class = getattr(module, class_name)

        return pipe_class()