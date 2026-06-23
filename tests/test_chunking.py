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


def _schedule_underrun(sizes, rtf=0.80, overhead=0.45, cr=16.0, lead_s=1.5):
    """Worst seconds a fragment is delivered LATE for gapless playback, given
    Chatterbox delivers each fragment whole at the measured RTF ~0.8. >0 = gap.
    Encodes the design guarantee behind the ramp (see config.FRAGMENT_GROWTH)."""
    durs = [max(c / cr, 0.2) for c in sizes]
    gens = [rtf * d + overhead for d in durs]
    deliver, t = [], 0.0
    for g in gens:
        t += g
        deliver.append(t)
    cum, t_start = 0.0, deliver[-1]
    for k, d in enumerate(durs):
        cum += d
        if cum >= lead_s:  # playback starts when the lead buffer fills
            t_start = deliver[k]
            break
    worst, need = -1e9, t_start
    for k in range(len(durs)):
        worst = max(worst, deliver[k] - need)
        need += durs[k]
    return worst


def test_ramp_first_fragment_is_small_for_fast_first_audio():
    out = split_fragments("word " * 120, first_max_chars=64, growth=1.1)
    assert len(out[0]) <= 64
    assert len(out) > 3  # a long input ramps into several fragments


def test_ramp_keeps_every_fragment_within_max():
    out = split_fragments("word " * 200, first_max_chars=64, max_chars=280, growth=1.1)
    assert all(len(f) <= 280 for f in out)


def test_ramp_caps_grow_after_the_first():
    out = split_fragments("word " * 200, first_max_chars=64, growth=1.1)
    assert max(len(f) for f in out) > len(out[0])


def test_ramp_preserves_all_words():
    t = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu."
    out = split_fragments(t, first_max_chars=32, growth=1.1)
    # Fragments are spoken separately; no word may be dropped or split mid-word.
    assert [w for frag in out for w in frag.split()] == t.split()


def test_ramp_short_input_stays_single_fragment():
    # Inputs under first_max_chars are untouched -- no needless seams.
    out = split_fragments("This is a short prompt.", first_max_chars=64, growth=1.1)
    assert out == ["This is a short prompt."]


def test_ramp_schedule_gapless_where_legacy_gaps():
    # A long run-on sentence: legacy leaves a big mid-utterance fragment that
    # arrives far too late; the ramp keeps every fragment inside the budget.
    t = ("word " * 120).strip()  # ~600 chars, one sentence
    ramp = [len(f) for f in split_fragments(t, first_max_chars=64, max_chars=280, growth=1.1)]
    legacy = [len(f) for f in split_fragments(t, first_max_chars=64)]
    assert _schedule_underrun(ramp) < 1.0  # effectively gapless
    assert _schedule_underrun(legacy) > 3.0  # legacy underruns hard
