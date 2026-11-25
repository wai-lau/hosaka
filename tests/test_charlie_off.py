# Always-on nicole: the server can run with Charlie's engines (Chatterbox + RVC)
# absent. Then it must not stop ollama at startup, and a Charlie request must say
# why it failed (GPU mode is not homo) instead of "unknown backend".
import numpy as np
from fastapi.testclient import TestClient

import hosaka.server.app as app_mod
from hosaka import charlie_state
from hosaka.library import VoiceLibrary
from hosaka.server.app import create_app
from hosaka.server.engines.base import EngineRegistry


class FakeEngine:
    voice_ids = ["charlie"]

    def stream(self, text, voice, params):
        yield np.zeros(1200, dtype=np.float32)

    def warmup(self):
        pass


def _app(tmp_path, monkeypatch, **engines):
    calls = []
    monkeypatch.setattr(app_mod, "stop_llm", lambda: calls.append(1))
    reg = EngineRegistry(**engines)
    return create_app(reg, VoiceLibrary(tmp_path / "voices"), do_warmup=True), calls


def test_nicole_only_does_not_stop_ollama(tmp_path, monkeypatch):
    app, calls = _app(tmp_path, monkeypatch, kokoro=FakeEngine())
    with TestClient(app):
        pass
    assert calls == []


def test_charlie_loaded_stops_ollama(tmp_path, monkeypatch):
    app, calls = _app(tmp_path, monkeypatch, kokoro=FakeEngine(), chatterbox=FakeEngine())
    with TestClient(app):
        pass
    assert calls == [1]


def test_charlie_request_while_off_names_homo(tmp_path, monkeypatch):
    app, _ = _app(tmp_path, monkeypatch, kokoro=FakeEngine())
    with TestClient(app) as c:
        for backend, voice in (("rvc", "charlie"), ("chatterbox", "")):
            r = c.post("/v1/audio/speech", json={"input": "hi", "backend": backend, "voice": voice})
            assert r.status_code == 400
            assert "homo" in r.json()["detail"], r.json()
        assert (
            c.post(
                "/v1/audio/speech", json={"input": "hi", "backend": "nope", "voice": "x"}
            ).json()["detail"]
            == "unknown backend: nope"
        )
        ok = c.post(
            "/v1/audio/speech", json={"input": "hi", "backend": "kokoro", "voice": "nicole"}
        )
        assert ok.status_code == 200


def test_state_defaults_off(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.delenv("HOSAKA_CHARLIE", raising=False)
    assert charlie_state.wanted() is False


def test_state_file_on(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.delenv("HOSAKA_CHARLIE", raising=False)
    (tmp_path / "hosaka").mkdir()
    (tmp_path / "hosaka" / "charlie").write_text("on\n")
    assert charlie_state.wanted() is True
    (tmp_path / "hosaka" / "charlie").write_text("off\n")
    assert charlie_state.wanted() is False


def test_state_env_overrides_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    (tmp_path / "hosaka").mkdir()
    (tmp_path / "hosaka" / "charlie").write_text("on\n")
    monkeypatch.setenv("HOSAKA_CHARLIE", "0")
    assert charlie_state.wanted() is False
    monkeypatch.setenv("HOSAKA_CHARLIE", "1")
    (tmp_path / "hosaka" / "charlie").write_text("off\n")
    assert charlie_state.wanted() is True
