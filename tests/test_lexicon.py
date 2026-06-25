from hosaka.lexicon import (
    Lexicon,
    add_entry,
    apply_lexicon,
    load_map,
    remove_entry,
    save_map,
)


def test_empty_map_is_identity():
    assert apply_lexicon("Wai builds tools", {}) == "Wai builds tools"


def test_respells_whole_word():
    assert apply_lexicon("Hi Wai there", {"Wai": "Way"}) == "Hi Way there"


def test_case_insensitive_match_emits_replacement_verbatim():
    assert apply_lexicon("wai and WAI", {"Wai": "Way"}) == "Way and Way"


def test_whole_word_only_leaves_substrings_intact():
    assert apply_lexicon("Waitress await Wai", {"Wai": "Way"}) == "Waitress await Way"


def test_apostrophe_is_a_boundary():
    assert apply_lexicon("Wai's tool", {"Wai": "Way"}) == "Way's tool"


def test_symbol_key_matches_without_word_boundary():
    # A punctuation key like "~" has no word boundary to anchor `\b` to, so it
    # must match anywhere -- even glued to a digit ("~30") or before a space.
    m = {"~": "about"}
    assert apply_lexicon("~30", m) == "about30"
    assert apply_lexicon("~ 30cm", m) == "about 30cm"
    assert apply_lexicon("it is ~5 m", m) == "it is about5 m"


def test_symbol_key_alongside_word_key_keeps_word_boundary():
    # Mixing a symbol key and a word key: the word key still stays whole-word.
    m = {"~": "about", "Wai": "Way"}
    assert apply_lexicon("Wai ~30 Waitress", m) == "Way about30 Waitress"


def test_multiword_key_matches_and_wins_longest_first():
    m = {"York": "Yorke", "New York": "Noo York"}
    assert apply_lexicon("I love New York", m) == "I love Noo York"


def test_multiple_distinct_keys():
    m = {"Wai": "Way", "hosaka": "ho sock uh"}
    assert apply_lexicon("Wai made hosaka", m) == "Way made ho sock uh"


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "lexicon.json"
    save_map(p, {"Wai": "Way"})
    assert load_map(p) == {"Wai": "Way"}


def test_load_missing_file_is_empty(tmp_path):
    assert load_map(tmp_path / "nope.json") == {}


def test_save_creates_parent_dir(tmp_path):
    p = tmp_path / "deep" / "dir" / "lexicon.json"
    save_map(p, {"a": "b"})
    assert load_map(p) == {"a": "b"}


def test_add_entry_persists(tmp_path):
    p = tmp_path / "lexicon.json"
    add_entry(p, "Wai", "Way")
    assert load_map(p) == {"Wai": "Way"}
    add_entry(p, "hosaka", "ho sock uh")
    assert load_map(p) == {"Wai": "Way", "hosaka": "ho sock uh"}


def test_remove_entry_case_insensitive(tmp_path):
    p = tmp_path / "lexicon.json"
    add_entry(p, "Wai", "Way")
    mapping, removed = remove_entry(p, "wai")
    assert removed is True
    assert mapping == {}
    assert load_map(p) == {}


def test_remove_missing_entry_reports_false(tmp_path):
    p = tmp_path / "lexicon.json"
    save_map(p, {"Wai": "Way"})
    mapping, removed = remove_entry(p, "nope")
    assert removed is False
    assert mapping == {"Wai": "Way"}


def test_lexicon_class_applies(tmp_path):
    p = tmp_path / "lexicon.json"
    save_map(p, {"Wai": "Way"})
    lex = Lexicon(p)
    assert lex.apply("Hi Wai") == "Hi Way"


def test_lexicon_class_missing_file_is_identity(tmp_path):
    lex = Lexicon(tmp_path / "nope.json")
    assert lex.apply("Hi Wai") == "Hi Wai"


def test_lexicon_class_reloads_on_mtime_change(tmp_path):
    p = tmp_path / "lexicon.json"
    save_map(p, {"Wai": "Way"})
    lex = Lexicon(p)
    assert lex.apply("Wai") == "Way"
    # Edit the file: a later mtime must invalidate the cached regex.
    import os

    st = p.stat()
    save_map(p, {"Wai": "Wye"})
    os.utime(p, (st.st_atime, st.st_mtime + 10))
    assert lex.apply("Wai") == "Wye"
