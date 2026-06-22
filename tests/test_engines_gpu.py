import numpy as np
import pytest
from hosaka.server.engines.base import EngineRegistry


class FakeEngine:
    def __init__(self, tag):
        self.tag = tag
        self.warmed = False

    def stream(self, text, voice, params):
        yield np.zeros(8, dtype=np.float32)

    def warmup(self):
        self.warmed = True


def test_registry_routes_by_backend():
    reg = EngineRegistry(kokoro=FakeEngine("k"), chatterbox=FakeEngine("c"))
    assert reg.get("kokoro").tag == "k"
    assert reg.get("chatterbox").tag == "c"


def test_registry_unknown_backend_raises():
    reg = EngineRegistry(kokoro=FakeEngine("k"), chatterbox=FakeEngine("c"))
    with pytest.raises(KeyError):
        reg.get("bogus")


def test_warmup_all_warms_both():
    k, c = FakeEngine("k"), FakeEngine("c")
    EngineRegistry(kokoro=k, chatterbox=c).warmup_all()
    assert k.warmed and c.warmed


@pytest.mark.gpu
def test_kokoro_streams_audio():
    from hosaka.server.engines.kokoro_engine import KokoroEngine
    eng = KokoroEngine()
    eng.warmup()
    chunks = list(eng.stream("Hello there.", "af_heart", {"speed": 1.0}))
    assert chunks, "no audio produced"
    assert chunks[0].dtype == np.float32
    assert chunks[0].ndim == 1


@pytest.mark.gpu
def test_chatterbox_clones_from_seed(tmp_path):
    import soundfile as sf
    from hosaka.config import SAMPLE_RATE
    from hosaka.library import VoiceLibrary
    from hosaka.server.engines.kokoro_engine import KokoroEngine
    from hosaka.server.engines.chatterbox_engine import ChatterboxEngine

    seed = np.concatenate(list(
        KokoroEngine().stream(
            "This is a seed clip for cloning, spoken clearly.",
            "af_heart", {"speed": 1.0})))
    seed_path = tmp_path / "seed.wav"
    sf.write(seed_path, seed, SAMPLE_RATE)

    lib = VoiceLibrary(tmp_path / "voices")
    lib.add("seed1", seed_path, source="kokoro")

    eng = ChatterboxEngine(lib)
    chunks = list(eng.stream("Now cloned.", "seed1",
                             {"exaggeration": 0.5, "cfg_weight": 0.4,
                              "temperature": 0.8}))
    assert chunks and chunks[0].dtype == np.float32
