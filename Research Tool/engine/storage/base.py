from abc import ABC, abstractmethod


class Storage(ABC):

    @abstractmethod
    def save(self, key, data):
        """Store data using a unique key."""
        pass

    @abstractmethod
    def load(self, key):
        """Retrieve data associated with a key."""
        pass

    @abstractmethod
    def exists(self, key):
        """Check whether a key already exists."""
        pass

    @abstractmethod
    def delete(self, key):
        """Remove the data associated with a key."""
        pass

    @abstractmethod
    def list(self, prefix=None):
        """List stored keys, optionally filtered by a prefix."""
        pass