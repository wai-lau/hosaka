import pytest

from hosaka.config import RVC_KNOBS, RVC_VOICES, resolve_source


def test_charlie_registered_female_american():
    c = RVC_VOICES["charlie"]
    assert c["gender"] == "female"
    assert c["accent"] == "american"
    assert c["model_sr"] == 32000
    assert c["source"] == "af_aoede"  # match-the-character pick
    assert c["transpose"] == 1


def test_resolve_source_returns_the_characters_source():
    # match-the-character: the per-voice source is authoritative (not a fixed
    # neutral default), validated only for gender + accent.
    assert resolve_source(RVC_VOICES["charlie"]) == "af_aoede"


def test_resolve_source_accepts_any_matching_gender_accent_voice():
    # Any American-female (af_) voice is valid for a female/american character.
    cfg = dict(RVC_VOICES["charlie"], source="af_bella")
    assert resolve_source(cfg) == "af_bella"


def test_resolve_source_rejects_wrong_gender():
    # A male (am_) source for a female character is a loud error.
    bad = dict(RVC_VOICES["charlie"], source="am_michael")
    with pytest.raises(ValueError, match="not female/american"):
        resolve_source(bad)


def test_resolve_source_unknown_tuple_raises():
    # Kokoro has no Australian base -> no matching source (hard stop).
    bad = {"gender": "female", "accent": "australian", "source": "af_aoede"}
    with pytest.raises(KeyError):
        resolve_source(bad)


def test_knobs_are_fixed_rmvpe():
    assert RVC_KNOBS["f0_method"] == "rmvpe"
    assert set(RVC_KNOBS) == {"index_rate", "f0_method", "protect", "rms_mix_rate"}
