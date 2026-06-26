import hashlib
import os
import threading
from collections import OrderedDict
from pathlib import Path


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


class SourceCache:
    """RAM-hot, disk-durable PCM cache (the RvcEngine source cache).

    A small in-RAM `PcmCache` is the hot tier; the durable copy lives on disk
    under `dir_path` as one `<sha256(key)>.f32` blob per entry, bounded by
    `disk_max_bytes` and evicted LRU by file mtime. `get` checks RAM, then reads
    the disk blob into RAM, then misses; `put` writes RAM plus an atomic disk
    write (tmp + rename). Survives process restarts. `version` is folded into the
    key hash so a source-gen / voice-weight change invalidates cleanly (stale
    blobs just stop being addressed and age out by LRU).

    Disk is disabled when `dir_path` is None or `disk_max_bytes <= 0` -- then this
    is a plain RAM cache (used by tests; prod injects a real dir).
    """

    def __init__(self, dir_path, ram_max_bytes, disk_max_bytes, version=""):
        self._ram = PcmCache(ram_max_bytes)
        self._dir = Path(dir_path) if dir_path else None
        self._disk_max = int(disk_max_bytes)
        self._version = str(version)
        self._lock = threading.Lock()
        if self._disk_enabled():
            self._dir.mkdir(parents=True, exist_ok=True)

    def _disk_enabled(self) -> bool:
        return self._dir is not None and self._disk_max > 0

    def _path_for(self, key) -> Path:
        digest = hashlib.sha256(f"{self._version}\x00{key!r}".encode()).hexdigest()
        return self._dir / f"{digest}.f32"

    def get(self, key) -> bytes | None:
        hit = self._ram.get(key)
        if hit is not None:
            return hit
        if not self._disk_enabled():
            return None
        path = self._path_for(key)
        try:
            data = path.read_bytes()
        except OSError:
            return None
        try:
            os.utime(path, None)  # touch mtime -> most-recently-used for LRU
        except OSError:
            pass
        self._ram.put(key, data)
        return data

    def put(self, key, value: bytes) -> None:
        self._ram.put(key, value)
        if not self._disk_enabled() or len(value) > self._disk_max:
            return
        path = self._path_for(key)
        with self._lock:
            tmp = path.with_suffix(".tmp")
            try:
                tmp.write_bytes(value)
                tmp.replace(path)  # atomic
            except OSError:
                return
            self._evict_disk()

    def _evict_disk(self) -> None:
        entries = []
        total = 0
        for f in self._dir.glob("*.f32"):
            try:
                st = f.stat()
            except OSError:
                continue
            entries.append((st.st_mtime, st.st_size, f))
            total += st.st_size
        if total <= self._disk_max:
            return
        for _mtime, size, f in sorted(entries):  # oldest mtime first
            try:
                f.unlink()
            except OSError:
                continue
            total -= size
            if total <= self._disk_max:
                break

    @property
    def nbytes(self) -> int:
        return self._ram.nbytes
