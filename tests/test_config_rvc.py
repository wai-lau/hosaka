import pytest

from hosaka.config import RVC_KNOBS, RVC_VOICES, resolve_source

# A Kokoro-sourced voice cfg, for exercising the gender/accent prefix validation
# (Charlie itself now uses a Chatterbox-clone source, which skips that path).
KOKORO_CFG = {"gender": "female", "accent": "american", "source": "af_aoede"}


def test_charlie_is_chatterbox_clone_hybrid():
    c = RVC_VOICES["charlie"]
    assert c["model_sr"] == 32000
    assert c["source_backend"] == "chatterbox"
    assert c["source"] == "charlie_cb"
    assert c["transpose"] == 0
    assert c["gate"] is True


def test_resolve_source_non_kokoro_returns_source_as_is():
    # A Chatterbox-clone source is a library voice id, not a Kokoro preset, so the
    # gender/accent prefix rule does not apply.
    assert resolve_source(RVC_VOICES["charlie"]) == "charlie_cb"
    assert resolve_source(dict(RVC_VOICES["charlie"], source="anything")) == "anything"


def test_resolve_source_kokoro_accepts_matching_gender_accent():
    assert resolve_source(KOKORO_CFG) == "af_aoede"
    assert resolve_source(dict(KOKORO_CFG, source="af_bella")) == "af_bella"


def test_resolve_source_kokoro_rejects_wrong_gender():
    with pytest.raises(ValueError, match="not female/american"):
        resolve_source(dict(KOKORO_CFG, source="am_michael"))


def test_resolve_source_unknown_tuple_raises():
    # Kokoro has no Australian base -> hard stop.
    with pytest.raises(KeyError):
        resolve_source({"gender": "female", "accent": "australian", "source": "af_aoede"})


def test_knobs_reduce_silence_hallucination():
    assert RVC_KNOBS["f0_method"] == "rmvpe"
    assert RVC_KNOBS["protect"] == 0.5  # high protect for the source's silent gaps
    assert RVC_KNOBS["index_rate"] == 0.3
    assert set(RVC_KNOBS) == {"index_rate", "f0_method", "protect", "rms_mix_rate"}
