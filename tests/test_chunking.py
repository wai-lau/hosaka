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


def test_em_dash_becomes_a_pause():
    # -- / em-dash / en-dash are not pauses to the model; normalize to a comma.
    for dash in ("Wait -- no.", "Wait--no.", "Wait — no.", "Wait – no."):
        out = " ".join(split_fragments(dash))
        assert "--" not in out and "—" not in out and "–" not in out
        assert "Wait, no." in out


def test_word_hyphen_is_left_alone():
    assert split_fragments("well-being matters.") == ["well-being matters."]


def test_long_single_sentence_not_cut_mid_phrase():
    # A 71-char sentence under max_chars stays whole -- no first-fragment seam.
    t = "You were told repeatedly and in no uncertain terms to never touch that."
    assert split_fragments(t) == [t]


def test_first_max_chars_still_shrinks_when_requested():
    # The streaming opt-in still works when explicitly asked for.
    t = "You were told repeatedly and in no uncertain terms to never touch that."
    out = split_fragments(t, first_max_chars=40)
    assert len(out[0]) <= 40


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


def test_wrap_prefers_clause_boundary_over_word_boundary():
    # A long sentence split mid-clause should break after a comma, not mid-phrase.
    text = "alpha beta gamma, delta epsilon zeta eta theta phrase here."
    out = split_fragments(text, first_max_chars=200, max_chars=30)
    assert out[0].rstrip().endswith(",")


def test_long_non_first_sentence_is_also_capped():
    text = "Hi. " + "word " * 100  # second sentence ~500 chars
    out = split_fragments(text.strip(), max_chars=120)
    assert out[0] == "Hi."
    assert all(len(f) <= 120 for f in out)
