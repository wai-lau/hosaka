import asyncio
import json
import threading

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from hosaka.library import VoiceLibrary
from hosaka.server.app import create_app
from hosaka.server.engines.base import EngineRegistry


class FakeEngine:
    def stream(self, text, voice, params):
        yield np.zeros(1200, dtype=np.float32)

    def warmup(self):
        pass


def _client(tmp_path):
    reg = EngineRegistry(kokoro=FakeEngine(), chatterbox=FakeEngine())
    lib = VoiceLibrary(tmp_path / "voices")
    return TestClient(create_app(reg, lib, do_warmup=False)), lib


class PiperFakeEngine(FakeEngine):
    # Two voices to prove multi-voice listing/routing; the second is a test-only
    # id (prod config currently ships just "glados").
    voice_ids = ["glados", "glados2"]


def _client_with_piper(tmp_path):
    reg = EngineRegistry(kokoro=FakeEngine(), chatterbox=FakeEngine(), piper=PiperFakeEngine())
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
    assert "nicole" in ids  # a preset
    assert "myclone" in ids  # a library clip


def test_voices_include_descriptions(tmp_path):
    client, lib = _client(tmp_path)
    seed = tmp_path / "s.wav"
    seed.write_bytes(b"RIFFfake")
    lib.add("baked", seed, source="bake", params={"description": "a gruff narrator"})
    voices = {v["id"]: v for v in client.get("/v1/voices").json()}
    assert voices["nicole"]["description"]  # presets get a hardcoded blurb
    assert voices["baked"]["description"] == "a gruff narrator"  # from bake params


def test_voices_cb_flag(tmp_path):
    # cb = generation runs through Chatterbox (the cb knobs apply). Kokoro presets
    # are non-cb; library/chatterbox clips are cb.
    client, lib = _client(tmp_path)
    seed = tmp_path / "s.wav"
    seed.write_bytes(b"RIFFfake")
    lib.add("clip", seed, source="recording")
    voices = {v["id"]: v for v in client.get("/v1/voices").json()}
    assert voices["nicole"]["cb"] is False
    assert voices["clip"]["cb"] is True


def test_voices_rvc_cb_from_source_backend(tmp_path):
    # An rvc voice is cb iff its source engine is chatterbox (per RVC_VOICES).
    client, _ = _client_with_rvc(tmp_path)
    voices = {v["id"]: v for v in client.get("/v1/voices").json()}
    assert voices["charlie"]["cb"] is True  # RVC_VOICES["charlie"] sources from chatterbox
    assert voices["charlie2"]["cb"] is False  # not in RVC_VOICES -> default


def test_speech_streams_pcm_bytes(tmp_path):
    client, _ = _client(tmp_path)
    r = client.post(
        "/v1/audio/speech",
        json={"input": "Hello. World.", "backend": "kokoro", "voice": "nicole"},
    )
    assert r.status_code == 200
    assert len(r.content) > 0
    assert len(r.content) % 4 == 0  # float32 == 4 bytes/sample


def test_speech_unknown_backend_is_400(tmp_path):
    client, _ = _client(tmp_path)
    r = client.post("/v1/audio/speech", json={"input": "hi", "backend": "bogus"})
    assert r.status_code == 400


def test_speech_unknown_kokoro_voice_is_400(tmp_path):
    # An unoffered voice is rejected before the stream opens.
    client, _ = _client(tmp_path)
    r = client.post(
        "/v1/audio/speech",
        json={"input": "hi", "backend": "kokoro", "voice": "bogus_voice"},
    )
    assert r.status_code == 400
    assert "bogus_voice" in r.json()["detail"]


def test_speech_unknown_chatterbox_voice_is_400(tmp_path):
    client, _ = _client(tmp_path)
    r = client.post(
        "/v1/audio/speech",
        json={"input": "hi", "backend": "chatterbox", "voice": "ghost"},
    )
    assert r.status_code == 400


def test_speech_chatterbox_empty_voice_ok(tmp_path):
    # "" is the model's own default voice and must stream, not 400.
    client, _ = _client(tmp_path)
    r = client.post(
        "/v1/audio/speech",
        json={"input": "hi", "backend": "chatterbox", "voice": ""},
    )
    assert r.status_code == 200


def test_voices_list_piper_when_available(tmp_path):
    from hosaka.config import PIPER_VOICES

    client, _ = _client_with_piper(tmp_path)
    voices = {v["id"]: v for v in client.get("/v1/voices").json()}
    # Engine advertises every voice it serves (multi-voice), each tagged piper.
    assert voices["glados"]["backend"] == "piper"
    assert voices["glados2"]["backend"] == "piper"
    assert voices["glados"]["description"] == PIPER_VOICES["glados"]["description"]


def test_voices_omit_piper_when_unavailable(tmp_path):
    client, _ = _client(tmp_path)  # registry has no piper engine
    backends = {v["backend"] for v in client.get("/v1/voices").json()}
    assert "piper" not in backends


def test_speech_piper_voice_streams(tmp_path):
    client, _ = _client_with_piper(tmp_path)
    r = client.post(
        "/v1/audio/speech",
        json={"input": "hi", "backend": "piper", "voice": "glados"},
    )
    assert r.status_code == 200
    assert len(r.content) > 0
    assert len(r.content) % 4 == 0


def test_speech_unknown_piper_voice_is_400(tmp_path):
    client, _ = _client_with_piper(tmp_path)
    r = client.post(
        "/v1/audio/speech",
        json={"input": "hi", "backend": "piper", "voice": "bogus"},
    )
    assert r.status_code == 400
    assert "bogus" in r.json()["detail"]


def test_speech_piper_unavailable_is_400(tmp_path):
    # No piper engine in the registry -> the backend is rejected up front.
    client, _ = _client(tmp_path)
    r = client.post(
        "/v1/audio/speech",
        json={"input": "hi", "backend": "piper", "voice": "glados"},
    )
    assert r.status_code == 400


class BoomEngine:
    def stream(self, text, voice, params):
        yield np.zeros(100, dtype=np.float32)
        raise RuntimeError("engine boom")

    def warmup(self):
        pass


def test_speech_engine_error_is_not_silent(tmp_path):
    # A mid-stream engine failure must surface, not return a clean truncated 200.
    reg = EngineRegistry(kokoro=BoomEngine(), chatterbox=BoomEngine())
    lib = VoiceLibrary(tmp_path / "voices")
    client = TestClient(create_app(reg, lib, do_warmup=False))
    with pytest.raises(RuntimeError, match="engine boom"):
        client.post(
            "/v1/audio/speech",
            json={"input": "hi", "backend": "kokoro", "voice": "nicole"},
        )


class CudaBoomEngine:
    def stream(self, text, voice, params):
        yield np.zeros(100, dtype=np.float32)
        raise RuntimeError("CUDA error: device-side assert triggered")

    def warmup(self):
        pass


def test_fatal_cuda_error_triggers_shutdown(tmp_path, monkeypatch):
    # A device-side assert poisons the GPU context; the server must exit so the
    # next launch respawns clean instead of failing every later request.
    import hosaka.server.app as appmod

    hits = {}
    monkeypatch.setattr(appmod, "_do_shutdown", lambda: hits.setdefault("hit", True))
    reg = EngineRegistry(kokoro=CudaBoomEngine(), chatterbox=CudaBoomEngine())
    lib = VoiceLibrary(tmp_path / "voices")
    client = TestClient(create_app(reg, lib, do_warmup=False))
    with pytest.raises(RuntimeError, match="device-side assert"):
        client.post(
            "/v1/audio/speech",
            json={"input": "hi", "backend": "kokoro", "voice": "nicole"},
        )
    assert hits.get("hit")


class HangEngine:
    """Engine whose stream() blocks without ever yielding -- simulates a wedged
    GPU call that can't be cancelled, so the watchdog must fire."""

    def __init__(self):
        self.release = threading.Event()

    def stream(self, text, voice, params):
        self.release.wait(5)  # never released within the test -> past the watchdog
        yield np.zeros(10, dtype=np.float32)

    def warmup(self):
        pass


def test_wedged_generation_triggers_shutdown(tmp_path, monkeypatch):
    # A generation that makes no progress for GEN_TIMEOUT_S holds the GPU slot
    # forever; the watchdog must exit (systemd respawns) instead of hanging every
    # later request behind the dead slot.
    import hosaka.server.app as appmod

    monkeypatch.setattr(appmod, "GEN_TIMEOUT_S", 0.2)
    hits = {}
    monkeypatch.setattr(appmod, "_do_shutdown", lambda: hits.setdefault("hit", True))
    eng = HangEngine()
    reg = EngineRegistry(kokoro=eng, chatterbox=eng)
    client = TestClient(create_app(reg, VoiceLibrary(tmp_path / "v"), do_warmup=False))
    try:
        r = client.post("/v1/audio/speech", json=KOKORO_REQ)
        # The stream ends (truncated) rather than hanging forever, and shutdown fired.
        assert r.status_code == 200
        assert hits.get("hit")
    finally:
        eng.release.set()  # let the orphaned worker thread exit promptly


def test_ordinary_engine_error_does_not_shutdown(tmp_path, monkeypatch):
    # A non-CUDA failure must surface but leave the server running.
    import hosaka.server.app as appmod

    hits = {}
    monkeypatch.setattr(appmod, "_do_shutdown", lambda: hits.setdefault("hit", True))
    reg = EngineRegistry(kokoro=BoomEngine(), chatterbox=BoomEngine())
    lib = VoiceLibrary(tmp_path / "voices")
    client = TestClient(create_app(reg, lib, do_warmup=False))
    with pytest.raises(RuntimeError, match="engine boom"):
        client.post(
            "/v1/audio/speech",
            json={"input": "hi", "backend": "kokoro", "voice": "nicole"},
        )
    assert not hits.get("hit")


def test_lock_released_after_request(tmp_path):
    # After a normal request completes, the GPU slot must be free for the next.
    client, _ = _client(tmp_path)
    for _ in range(3):
        r = client.post(
            "/v1/audio/speech",
            json={"input": "Hello.", "backend": "kokoro", "voice": "nicole"},
        )
        assert r.status_code == 200


def test_shutdown_route_exists(tmp_path):
    client, _ = _client(tmp_path)
    import hosaka.server.app as appmod

    called = {}
    appmod._do_shutdown = lambda: called.setdefault("hit", True)
    r = client.post("/shutdown")
    assert r.status_code == 200
    assert called.get("hit")


class GatedEngine:
    """Engine whose stream() blocks until released, so a test can pin the GPU
    slot and observe how a concurrent request behaves (queued vs rejected).

    stream() runs in the server's worker thread, so the gate is a threading
    primitive; ``entered`` is set the moment a stream starts running."""

    def __init__(self):
        self.entered = threading.Event()
        self.gate = threading.Event()

    def stream(self, text, voice, params):
        self.entered.set()
        self.gate.wait(5)
        yield np.zeros(1200, dtype=np.float32)

    def warmup(self):
        pass


KOKORO_REQ = {"input": "hi", "backend": "kokoro", "voice": "nicole"}


async def _wait(flag: threading.Event):
    # Bridge a threading.Event into the event loop without blocking it.
    while not flag.is_set():
        await asyncio.sleep(0.01)


def _async_client(app):
    # Drive the ASGI app in-process over one event loop, so concurrent requests
    # share the app's asyncio.Semaphore correctly (a single sync TestClient
    # across threads deadlocks once a request actually awaits the slot).
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def test_concurrent_request_queues_instead_of_503(tmp_path):
    # While one request holds the GPU slot, a second must WAIT in line and then
    # succeed -- not get rejected with 503 (the old single-slot behavior).
    eng = GatedEngine()
    reg = EngineRegistry(kokoro=eng, chatterbox=eng)
    app = create_app(reg, VoiceLibrary(tmp_path / "v"), do_warmup=False)

    async def scenario():
        async with _async_client(app) as ac:
            a = asyncio.create_task(ac.post("/v1/audio/speech", json=KOKORO_REQ))
            await _wait(eng.entered)  # A is inside the engine, holding the slot
            b = asyncio.create_task(ac.post("/v1/audio/speech", json=KOKORO_REQ))
            await asyncio.sleep(0.2)
            assert not b.done()  # B is queued (waiting), not already rejected

            eng.gate.set()
            ra, rb = await asyncio.gather(a, b)
            assert ra.status_code == 200
            assert rb.status_code == 200

    asyncio.run(scenario())


def test_queue_cap_returns_503_when_full(tmp_path):
    # The queue is bounded: once admitted (waiting + running) hits the cap, the
    # next request is rejected with 503 instead of growing the backlog forever.
    eng = GatedEngine()
    reg = EngineRegistry(kokoro=eng, chatterbox=eng)
    app = create_app(reg, VoiceLibrary(tmp_path / "v"), do_warmup=False, max_queue=2)

    async def scenario():
        async with _async_client(app) as ac:
            a = asyncio.create_task(ac.post("/v1/audio/speech", json=KOKORO_REQ))  # runs
            await _wait(eng.entered)
            b = asyncio.create_task(ac.post("/v1/audio/speech", json=KOKORO_REQ))  # queued
            await asyncio.sleep(0.2)  # let B reach the slot (depth 2 == cap)
            rc = await ac.post("/v1/audio/speech", json=KOKORO_REQ)  # over cap -> 503
            assert rc.status_code == 503

            eng.gate.set()
            ra, rb = await asyncio.gather(a, b)
            assert ra.status_code == 200
            assert rb.status_code == 200

    asyncio.run(scenario())


def _drain_ws(ws):
    """Read one utterance off the socket: a start marker, PCM binary frames,
    then an end marker. Return the concatenated PCM bytes."""
    assert ws.receive_json()["type"] == "start"
    chunks = []
    while True:
        msg = ws.receive()
        if msg.get("bytes") is not None:
            chunks.append(msg["bytes"])
        else:
            assert json.loads(msg["text"])["type"] == "end"
            return b"".join(chunks)


def test_ws_streams_pcm_frames(tmp_path):
    client, _ = _client(tmp_path)
    with client.websocket_connect("/v1/audio/stream") as ws:
        ws.send_json({"input": "Hello.", "backend": "kokoro", "voice": "nicole"})
        pcm = _drain_ws(ws)
    assert len(pcm) > 0
    assert len(pcm) % 4 == 0  # float32 == 4 bytes/sample


def test_ws_unknown_voice_sends_error_and_stays_open(tmp_path):
    client, _ = _client(tmp_path)
    with client.websocket_connect("/v1/audio/stream") as ws:
        ws.send_json({"input": "hi", "backend": "kokoro", "voice": "bogus_voice"})
        err = ws.receive_json()
        assert err["type"] == "error"
        assert "bogus_voice" in err["detail"]
        # connection survives a bad request: a good one still streams.
        ws.send_json({"input": "hi", "backend": "kokoro", "voice": "nicole"})
        assert len(_drain_ws(ws)) > 0


def test_ws_multiple_utterances_one_connection(tmp_path):
    client, _ = _client(tmp_path)
    with client.websocket_connect("/v1/audio/stream") as ws:
        for _ in range(3):
            ws.send_json({"input": "hi", "backend": "kokoro", "voice": "nicole"})
            assert len(_drain_ws(ws)) > 0


def test_app_static_page_served(tmp_path):
    # The bundled browser demo client is served at /app/.
    client, _ = _client(tmp_path)
    r = client.get("/app/")
    assert r.status_code == 200
    assert "hosaka" in r.text.lower()
    assert client.get("/app/pcm-player.js").status_code == 200


class RvcFakeEngine(FakeEngine):
    voice_ids = ["charlie", "charlie2"]


def _client_with_rvc(tmp_path):
    reg = EngineRegistry(kokoro=FakeEngine(), chatterbox=FakeEngine(), rvc=RvcFakeEngine())
    lib = VoiceLibrary(tmp_path / "voices")
    return TestClient(create_app(reg, lib, do_warmup=False)), lib


def test_voices_list_rvc_when_available(tmp_path):
    from hosaka.config import RVC_VOICES

    client, _ = _client_with_rvc(tmp_path)
    voices = {v["id"]: v for v in client.get("/v1/voices").json()}
    assert voices["charlie"]["backend"] == "rvc"
    assert voices["charlie2"]["backend"] == "rvc"
    assert voices["charlie"]["description"] == RVC_VOICES["charlie"]["description"]


def test_voices_rvc_cb_params(tmp_path):
    # A cb (chatterbox-sourced) rvc voice carries its tuned cb-knob defaults so a
    # client can preload them on :voice; voices not in RVC_VOICES carry None.
    from hosaka.config import RVC_VOICES

    client, _ = _client_with_rvc(tmp_path)
    voices = {v["id"]: v for v in client.get("/v1/voices").json()}
    assert voices["charlie"]["cb_params"] == RVC_VOICES["charlie"]["source_params"]
    assert voices["charlie2"]["cb_params"] is None


def test_voices_rvc_speed_default(tmp_path):
    # A cb rvc voice ships its configured output speed so a client can preload it
    # on :voice (:speed then tunes it live); voices not in RVC_VOICES default 1.0.
    from hosaka.config import RVC_VOICES

    client, _ = _client_with_rvc(tmp_path)
    voices = {v["id"]: v for v in client.get("/v1/voices").json()}
    assert voices["charlie"]["speed"] == RVC_VOICES["charlie"]["speed"]
    assert voices["charlie2"]["speed"] == 1.0


def test_voices_omit_rvc_when_unavailable(tmp_path):
    client, _ = _client(tmp_path)
    backends = {v["backend"] for v in client.get("/v1/voices").json()}
    assert "rvc" not in backends


def test_speech_rvc_voice_streams(tmp_path):
    client, _ = _client_with_rvc(tmp_path)
    r = client.post(
        "/v1/audio/speech",
        json={"input": "hi", "backend": "rvc", "voice": "charlie"},
    )
    assert r.status_code == 200
    assert len(r.content) > 0 and len(r.content) % 4 == 0


def test_speech_unknown_rvc_voice_is_400(tmp_path):
    client, _ = _client_with_rvc(tmp_path)
    r = client.post(
        "/v1/audio/speech",
        json={"input": "hi", "backend": "rvc", "voice": "bogus"},
    )
    assert r.status_code == 400
    assert "bogus" in r.json()["detail"]


def test_speech_rvc_unavailable_is_400(tmp_path):
    client, _ = _client(tmp_path)
    r = client.post(
        "/v1/audio/speech",
        json={"input": "hi", "backend": "rvc", "voice": "charlie"},
    )
    assert r.status_code == 400
