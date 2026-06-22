import numpy as np
from fastapi.testclient import TestClient
from hosaka.server.engines.base import EngineRegistry
from hosaka.library import VoiceLibrary
from hosaka.server.app import create_app


class FakeEngine:
    def stream(self, text, voice, params):
        yield np.zeros(1200, dtype=np.float32)

    def warmup(self):
        pass


def _client(tmp_path):
    reg = EngineRegistry(kokoro=FakeEngine(), chatterbox=FakeEngine())
    lib = VoiceLibrary(tmp_path / "voices")
    return TestClient(create_app(reg, lib, do_warmup=False)), lib


def test_health_ok(tmp_path):
    client, _ = _client(tmp_path)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_voices_lists_presets_and_library(tmp_path):
    client, lib = _client(tmp_path)
    seed = tmp_path / "s.wav"
    seed.write_bytes(b"RIFFfake")
    lib.add("myclone", seed, source="recording")
    ids = {v["id"] for v in client.get("/v1/voices").json()}
    assert "af_heart" in ids        # a preset
    assert "myclone" in ids         # a library clip


def test_speech_streams_pcm_bytes(tmp_path):
    client, _ = _client(tmp_path)
    r = client.post("/v1/audio/speech",
                    json={"input": "Hello. World.", "backend": "kokoro",
                          "voice": "af_heart"})
    assert r.status_code == 200
    assert len(r.content) > 0
    assert len(r.content) % 4 == 0      # float32 == 4 bytes/sample


def test_speech_unknown_backend_is_400(tmp_path):
    client, _ = _client(tmp_path)
    r = client.post("/v1/audio/speech",
                    json={"input": "hi", "backend": "bogus"})
    assert r.status_code == 400


def test_shutdown_route_exists(tmp_path):
    client, _ = _client(tmp_path)
    import hosaka.server.app as appmod
    called = {}
    appmod._do_shutdown = lambda: called.setdefault("hit", True)
    r = client.post("/shutdown")
    assert r.status_code == 200
    assert called.get("hit")
