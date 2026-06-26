import json
import sys
from pathlib import Path

import numpy as np
import pytest

from hosaka.server.engines.rvc_engine import RvcEngine
from hosaka.server.engines.rvc_proto import RvcProtocolError, RvcSidecarError

FAKE = [sys.executable, str(Path(__file__).parent / "fake_rvc_sidecar.py")]

KNOBS = {"index_rate": 0.5, "f0_method": "rmvpe", "protect": 0.33, "rms_mix_rate": 0.25}


class FakeSource:
    """Stand-in Kokoro: yields two deterministic chunks so a test can check the
    engine concatenates the whole fragment before sending it to the sidecar."""

    def __init__(self):
        self.calls = []

    def stream(self, text, voice, params):
        self.calls.append((text, voice, params))
        yield np.full(50, 0.1, dtype=np.float32)
        yield np.full(30, 0.2, dtype=np.float32)

    def warmup(self):
        pass


def _voices():
    return {
        "charlie": {"source": "af_sarah", "transpose": 2},
        "boom": {"source": "af_sarah", "transpose": 0},
        "die": {"source": "af_sarah", "transpose": 0},
        "echo": {"source": "af_sarah", "transpose": 7, "passes": 2, "gate": True, "speed": 1.1},
        "cbvoice": {
            "source_backend": "chatterbox",
            "source": "clip",
            "source_params": {"exaggeration": 0.7},
            "transpose": 0,
        },
    }


def _engine(src=None):
    return RvcEngine({"kokoro": src or FakeSource()}, FAKE, voices=_voices(), knobs=dict(KNOBS))


def test_voice_ids_lists_configured_voices():
    assert set(_engine().voice_ids) == {"charlie", "boom", "die", "echo", "cbvoice"}


def test_stream_converts_and_round_trips_source():
    # The fake echoes the source PCM back; the engine must have accumulated BOTH
    # source chunks (80 float32 samples) before sending.
    eng = _engine()
    out = np.concatenate(list(eng.stream("Hi there.", "charlie", {"speed": 1.0})))
    assert out.dtype == np.float32
    assert len(out) == 80
    assert np.allclose(out[:50], 0.1) and np.allclose(out[50:], 0.2)
    eng.close()


def test_stream_uses_voices_source_preset():
    src = FakeSource()
    eng = _engine(src)
    list(eng.stream("Hi.", "charlie", {"speed": 1.0}))
    # Kokoro was asked for the configured source preset, not the rvc voice id.
    assert src.calls[0][1] == "af_sarah"
    eng.close()


def test_stream_routes_to_per_voice_source_engine():
    # A chatterbox-backed voice generates its source via the chatterbox engine
    # with its own params (exaggeration), not Kokoro.
    kok, cb = FakeSource(), FakeSource()
    eng = RvcEngine({"kokoro": kok, "chatterbox": cb}, FAKE, voices=_voices(), knobs=dict(KNOBS))
    list(eng.stream("Hi.", "cbvoice", {"speed": 1.0}))
    assert not kok.calls  # Kokoro untouched
    assert cb.calls[0][1] == "clip"
    assert cb.calls[0][2] == {"exaggeration": 0.7}
    eng.close()


def test_chatterbox_source_merges_request_cb_knobs():
    # A chatterbox-clone source starts from its pinned source_params, then the
    # request's live cb knobs override them. speed is NOT a cb source knob (for
    # RVC it is an output tempo stretch), so it must not leak into the source.
    kok, cb = FakeSource(), FakeSource()
    eng = RvcEngine({"kokoro": kok, "chatterbox": cb}, FAKE, voices=_voices(), knobs=dict(KNOBS))
    list(eng.stream("Hi.", "cbvoice", {"exaggeration": 0.2, "temperature": 1.5, "speed": 1.3}))
    sp = cb.calls[0][2]
    assert sp["exaggeration"] == 0.2  # overridden from the pinned 0.7
    assert sp["temperature"] == 1.5  # added from the request
    assert "speed" not in sp  # speed never crosses into the chatterbox source
    eng.close()


def test_source_pcm_cached_when_only_downstream_speed_changes():
    # For a chatterbox source, speed is the RVC output stretch (downstream), not a
    # source-gen knob -- so an identical fragment with only :speed changed reuses
    # the cached source instead of re-generating it (the 89% win).
    cb = FakeSource()
    eng = RvcEngine({"chatterbox": cb}, FAKE, voices=_voices(), knobs=dict(KNOBS))
    list(eng.stream("Hi.", "cbvoice", {"speed": 1.0}))
    list(eng.stream("Hi.", "cbvoice", {"speed": 1.5}))  # speed is downstream
    assert len(cb.calls) == 1  # source generated only once
    eng.close()


def test_source_cache_keys_on_cb_knobs():
    # Changing a cb knob (which DOES change the source) is a cache miss -> re-gen.
    cb = FakeSource()
    eng = RvcEngine({"chatterbox": cb}, FAKE, voices=_voices(), knobs=dict(KNOBS))
    list(eng.stream("Hi.", "cbvoice", {"exaggeration": 0.2}))
    list(eng.stream("Hi.", "cbvoice", {"exaggeration": 0.9}))
    assert len(cb.calls) == 2  # different source params -> two gens
    eng.close()


def test_source_cache_disabled_regenerates():
    src = FakeSource()
    eng = RvcEngine({"kokoro": src}, FAKE, voices=_voices(), knobs=dict(KNOBS), cache_max_bytes=0)
    list(eng.stream("Hi.", "charlie", {"speed": 1.0}))
    list(eng.stream("Hi.", "charlie", {"speed": 1.0}))
    assert len(src.calls) == 2  # cache off -> source each time
    eng.close()


def test_stream_sends_transpose_and_knobs_to_sidecar():
    eng = _engine()
    # No request speed -> the voice's configured default (1.1) reaches the sidecar.
    with pytest.raises(RvcSidecarError) as exc:
        list(eng.stream("Hi.", "echo", {}))
    params = json.loads(str(exc.value))
    assert params["voice"] == "echo"
    assert params["transpose"] == 7
    assert params["index_rate"] == 0.5
    assert params["f0_method"] == "rmvpe"
    assert params["passes"] == 2
    assert params["gate"] is True
    assert params["speed"] == 1.1
    eng.close()


def test_request_speed_overrides_config_default():
    # A request speed wins over the voice's configured speed (the REPL preloads
    # that default, so :speed tunes the output stretch live).
    eng = _engine()
    with pytest.raises(RvcSidecarError) as exc:
        list(eng.stream("Hi.", "echo", {"speed": 1.5}))
    params = json.loads(str(exc.value))
    assert params["speed"] == 1.5
    eng.close()


def test_stream_surfaces_sidecar_error_and_stays_alive():
    eng = _engine()
    with pytest.raises(RvcSidecarError, match="boom"):
        list(eng.stream("Hi.", "boom", {"speed": 1.0}))
    # A normal synth error leaves the sidecar alive: the next call still works.
    assert len(np.concatenate(list(eng.stream("Hi.", "charlie", {"speed": 1.0})))) == 80
    eng.close()


def test_stream_respawns_after_sidecar_death():
    eng = _engine()
    with pytest.raises(RvcProtocolError):
        list(eng.stream("Hi.", "die", {"speed": 1.0}))
    assert len(np.concatenate(list(eng.stream("Hi.", "charlie", {"speed": 1.0})))) == 80
    eng.close()


def test_warmup_does_not_raise():
    eng = RvcEngine(
        {"kokoro": FakeSource()},
        FAKE,
        voices={"charlie": {"source": "af_sarah", "transpose": 0}},
        knobs=dict(KNOBS),
    )
    eng.warmup()
    eng.close()
