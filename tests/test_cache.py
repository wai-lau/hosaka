from hosaka.cache import PcmCache, SourceCache


def test_get_miss_then_hit():
    c = PcmCache(1024)
    assert c.get("k") is None
    c.put("k", b"abc")
    assert c.get("k") == b"abc"
    assert len(c) == 1
    assert c.nbytes == 3


def test_lru_evicts_oldest_by_bytes():
    c = PcmCache(10)
    c.put("a", b"aaaa")  # 4
    c.put("b", b"bbbb")  # 8
    c.get("a")  # touch a -> b is now LRU
    c.put("c", b"cccc")  # 12 > 10 -> evict LRU (b)
    assert c.get("a") == b"aaaa"
    assert c.get("c") == b"cccc"
    assert c.get("b") is None
    assert c.nbytes == 8


def test_put_same_key_replaces_and_adjusts_bytes():
    c = PcmCache(1024)
    c.put("k", b"aaaa")
    c.put("k", b"bb")
    assert c.get("k") == b"bb"
    assert len(c) == 1
    assert c.nbytes == 2


def test_value_larger_than_budget_is_not_stored():
    c = PcmCache(4)
    c.put("big", b"aaaaaa")  # 6 > 4
    assert c.get("big") is None
    assert len(c) == 0
    assert c.nbytes == 0


def test_disabled_cache_never_stores():
    c = PcmCache(0)
    c.put("k", b"abc")
    assert c.get("k") is None
    assert len(c) == 0


def test_tuple_keys_distinguish_params():
    c = PcmCache(1024)
    c.put(("v", "text", "cb", (("exaggeration", 0.4),)), b"x")
    assert c.get(("v", "text", "cb", (("exaggeration", 0.4),))) == b"x"
    assert c.get(("v", "text", "cb", (("exaggeration", 0.5),))) is None


# --- SourceCache (disk-backed) ---


def test_sourcecache_persists_across_instances(tmp_path):
    key = ("clip", "Hello.", "chatterbox", (("exaggeration", 0.4),))
    a = SourceCache(tmp_path, 10_000, 10_000)
    a.put(key, b"audio-bytes")
    # a fresh instance over the same dir (a restart) still has it
    b = SourceCache(tmp_path, 10_000, 10_000)
    assert b.get(key) == b"audio-bytes"


def test_sourcecache_ram_miss_falls_through_to_disk(tmp_path):
    key = ("clip", "x", "chatterbox", ())
    a = SourceCache(tmp_path, 10_000, 10_000)
    a.put(key, b"abc")
    # tiny RAM tier that can't hold it -> served from disk
    b = SourceCache(tmp_path, 1, 10_000)
    assert b.get(key) == b"abc"


def test_sourcecache_version_change_invalidates(tmp_path):
    key = ("clip", "x", "chatterbox", ())
    SourceCache(tmp_path, 10_000, 10_000, version="1").put(key, b"v1")
    assert SourceCache(tmp_path, 10_000, 10_000, version="2").get(key) is None


def test_sourcecache_evicts_disk_by_budget(tmp_path):
    import os
    import time

    c = SourceCache(tmp_path, 10_000, 10)  # 10-byte disk budget
    c.put(("k", "a", "b", ()), b"aaaa")  # 4
    pa = c._path_for(("k", "a", "b", ()))
    os.utime(pa, (1, 1))  # force 'a' to be the oldest
    time.sleep(0.01)
    c.put(("k", "b", "b", ()), b"bbbb")  # 8
    c.put(("k", "c", "b", ()), b"cccc")  # 12 > 10 -> evict oldest ('a')
    blobs = list(tmp_path.glob("*.f32"))
    assert sum(f.stat().st_size for f in blobs) <= 10
    assert not pa.exists()  # the oldest blob was evicted


def test_sourcecache_disk_disabled_is_ram_only(tmp_path):
    key = ("k", "x", "b", ())
    a = SourceCache(None, 10_000, 0)  # no dir -> RAM only
    a.put(key, b"abc")
    assert a.get(key) == b"abc"
    # nothing written to disk; a fresh instance has nothing
    assert list(tmp_path.glob("*.f32")) == []
