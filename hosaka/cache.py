import threading
from collections import OrderedDict


class PcmCache:
    """Bounded, thread-safe LRU cache of PCM byte blobs, keyed by a hashable key.

    Sized by *total bytes* (not entry count) since PCM blobs vary widely in
    length. On insert past the budget the least-recently-used entries are evicted
    until it fits. A value larger than the whole budget is simply not stored (it
    would evict everything and still not fit). `max_bytes <= 0` disables caching:
    get always misses, put is a no-op.

    The server touches this from GPU worker threads serialized by the single GPU
    slot, so contention is near nil -- but a lock keeps get/put correct regardless.
    """

    def __init__(self, max_bytes: int):
        self._max = int(max_bytes)
        self._store: OrderedDict[object, bytes] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key) -> bytes | None:
        if self._max <= 0:
            return None
        with self._lock:
            if key not in self._store:
                return None
            self._store.move_to_end(key)  # mark most-recently-used
            return self._store[key]

    def put(self, key, value: bytes) -> None:
        if self._max <= 0 or len(value) > self._max:
            return
        with self._lock:
            if key in self._store:
                self._bytes -= len(self._store.pop(key))
            self._store[key] = value
            self._bytes += len(value)
            while self._bytes > self._max and len(self._store) > 1:
                _, evicted = self._store.popitem(last=False)  # drop LRU
                self._bytes -= len(evicted)

    def __len__(self) -> int:
        return len(self._store)

    @property
    def nbytes(self) -> int:
        return self._bytes
