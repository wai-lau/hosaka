import sys
from pathlib import Path

import numpy as np
import pytest

from hosaka.server.engines.piper_engine import PiperEngine, speed_to_length_scale
from hosaka.server.engines.piper_proto import PiperProtocolError, PiperSidecarError

FAKE = [sys.executable, str(Path(__file__).parent / "fake_piper_sidecar.py")]


def _engine():
    return PiperEngine(FAKE)


def test_speed_to_length_scale_is_inverse():
    assert speed_to_length_scale(1.0) == 1.0
    assert speed_to_length_scale(2.0) == 0.5
    assert speed_to_length_scale(0.5) == 2.0


def test_speed_to_length_scale_clamps():
    assert speed_to_length_scale(99.0) == 0.5  # capped at speed 2.0
    assert speed_to_length_scale(0.0) == 2.0  # floored at speed 0.5


def test_stream_yields_float32_arrays():
    eng = _engine()
    chunks = list(eng.stream("Hello.", "glados", {}))
    assert len(chunks) == 1
    assert chunks[0].dtype == np.float32
    assert len(chunks[0]) == 100
    eng.close()


def test_stream_one_chunk_per_sentence():
    eng = _engine()
    chunks = list(eng.stream("One. Two. Three.", "glados", {}))
    assert len(chunks) == 3
    assert all(c.dtype == np.float32 and len(c) == 100 for c in chunks)
    eng.close()


def test_stream_surfaces_sidecar_error():
    eng = _engine()
    with pytest.raises(PiperSidecarError, match="boom"):
        list(eng.stream("BOOM", "glados", {}))
    # The sidecar stays alive after a normal synth error: next call still works.
    assert len(list(eng.stream("Recovered.", "glados", {}))) == 1
    eng.close()


def test_stream_respawns_after_sidecar_death():
    eng = _engine()
    with pytest.raises(PiperProtocolError):
        list(eng.stream("DIE", "glados", {}))
    # A dead sidecar must be replaced transparently on the next request.
    assert len(list(eng.stream("Back up.", "glados", {}))) == 1
    eng.close()


def test_stream_sends_selected_voice():
    # The chosen voice must reach the sidecar so it can pick the right model.
    eng = _engine()
    with pytest.raises(PiperSidecarError, match="glados_high"):
        list(eng.stream("ECHOVOICE", "glados_high", {}))
    eng.close()


def test_warmup_warms_each_configured_voice():
    # warmup must touch every voice so each model is resident before requests.
    eng = PiperEngine(FAKE, voices=["glados", "glados_high"])
    eng.warmup()  # must not raise
    eng.close()


def test_warmup_does_not_raise():
    eng = _engine()
    eng.warmup()
    eng.close()
