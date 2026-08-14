# engine/storage/local.py

import pickle
from pathlib import Path

from .base import Storage


class LocalStorage(Storage):

    def __init__(self, root_path):
        """Define the root directory where all data will be stored."""
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key):
        """Convert a storage key into a local file path."""
        return self.root_path / f"{key}.pkl"

    def save(self, key, data):
        """Store data locally using pickle."""
        path = self._get_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as file:
            pickle.dump(data, file)

    def load(self, key):
        """Load data associated with a key."""
        path = self._get_path(key)

        if not path.exists():
            raise KeyError(f"Key not found: {key}")

        with open(path, "rb") as file:
            return pickle.load(file)

    def exists(self, key):
        """Check whether a key exists."""
        return self._get_path(key).exists()

    #def delete(self, key):
    #    """Delete data associated with a key."""
    #    path = self._get_path(key)

    #    if path.exists():
    #        path.unlink()
            
    def delete(self, key):
        """Delete stored data."""
        raise NotImplementedError("Delete is intentionally disabled for safety.")

    def list(self, prefix=None):
        """List all stored keys, optionally filtered by prefix."""
        keys = []

        for path in self.root_path.rglob("*.pkl"):
            key = path.relative_to(self.root_path).with_suffix("").as_posix()

            if prefix is None or key.startswith(prefix):
                keys.append(key)

        return keys