import pytest

from hosaka.config import RVC_KNOBS, RVC_VOICES, SOURCE_PRESETS, resolve_source


def test_charlie_registered_female_american():
    c = RVC_VOICES["charlie"]
    assert c["gender"] == "female"
    assert c["accent"] == "american"
    assert c["model_sr"] == 40000


def test_resolve_source_matches_tuple():
    assert resolve_source(RVC_VOICES["charlie"]) == "af_sarah"
    assert SOURCE_PRESETS[("female", "american")] == "af_sarah"


def test_resolve_source_rejects_mismatch():
    bad = dict(RVC_VOICES["charlie"], source="am_michael")
    with pytest.raises(ValueError, match="source"):
        resolve_source(bad)


def test_resolve_source_unknown_tuple_raises():
    # Kokoro has no Australian base -> no matching source (hard stop).
    bad = {"gender": "female", "accent": "australian", "source": "x"}
    with pytest.raises(KeyError):
        resolve_source(bad)


def test_knobs_are_fixed_rmvpe():
    assert RVC_KNOBS["f0_method"] == "rmvpe"
    assert set(RVC_KNOBS) == {"index_rate", "f0_method", "protect", "rms_mix_rate"}
