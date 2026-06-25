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
        "echo": {"source": "af_sarah", "transpose": 7, "passes": 2},
    }


def _engine(src=None):
    return RvcEngine(src or FakeSource(), FAKE, voices=_voices(), knobs=dict(KNOBS))


def test_voice_ids_lists_configured_voices():
    assert set(_engine().voice_ids) == {"charlie", "boom", "die", "echo"}


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


def test_stream_sends_transpose_and_knobs_to_sidecar():
    eng = _engine()
    with pytest.raises(RvcSidecarError) as exc:
        list(eng.stream("Hi.", "echo", {"speed": 1.0}))
    params = json.loads(str(exc.value))
    assert params["voice"] == "echo"
    assert params["transpose"] == 7
    assert params["index_rate"] == 0.5
    assert params["f0_method"] == "rmvpe"
    assert params["passes"] == 2
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
        FakeSource(),
        FAKE,
        voices={"charlie": {"source": "af_sarah", "transpose": 0}},
        knobs=dict(KNOBS),
    )
    eng.warmup()
    eng.close()
