from hosaka.chunking import split_fragments


def test_single_short_sentence_is_one_fragment():
    assert split_fragments("Hello there.") == ["Hello there."]


def test_first_fragment_breaks_at_first_sentence():
    out = split_fragments("Hi. This is the rest of a longer thought here.")
    assert out[0] == "Hi."
    assert "".join(out) == "Hi. This is the rest of a longer thought here."


def test_long_first_clause_breaks_on_word_boundary():
    text = "x " * 80  # 160 chars, no punctuation
    out = split_fragments(text.strip(), first_max_chars=60)
    assert len(out[0]) <= 60
    assert not out[0].endswith(" ")
    assert "".join(s if s.endswith(" ") else s + " " for s in out).strip() == text.strip()


def test_empty_text_returns_empty_list():
    assert split_fragments("   ") == []


def test_every_fragment_capped_for_long_run_on_sentence():
    # One long comma-spliced sentence (single period) must not produce any
    # fragment over max_chars -- that is what blew past Chatterbox's limit.
    text = (
        "the whole graph for her career and finances, her health "
        "and gender, her family and chosen kin, her crafting and "
        "hobbies, her technical work, her history, and her own "
        "introspection and worldview, all at once and in detail."
    )
    out = split_fragments(text, max_chars=120)
    assert all(len(f) <= 120 for f in out)
    assert len(out) > 1


def test_long_non_first_sentence_is_also_capped():
    text = "Hi. " + "word " * 100  # second sentence ~500 chars
    out = split_fragments(text.strip(), max_chars=120)
    assert out[0] == "Hi."
    assert all(len(f) <= 120 for f in out)
