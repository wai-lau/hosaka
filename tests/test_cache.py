from hosaka.cache import PcmCache


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
