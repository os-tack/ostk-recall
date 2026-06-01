"""A tiny LRU cache, no external deps."""

from collections import OrderedDict


class LRUCache:
    """Fixed-capacity least-recently-used cache.

    get/put are O(1); the least-recently-used entry is evicted when the
    cache is full and a new key is inserted.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._store: "OrderedDict[object, object]" = OrderedDict()

    def get(self, key):
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key, value) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)
