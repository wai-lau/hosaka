import sys

from fastapi.testclient import TestClient


def test_main_piper_imports_without_torch_and_serves_only_piper():
    # Importing the piper-only root must not pull torch / kokoro / chatterbox.
    import hosaka.server.main_piper as mp

    assert "torch" not in sys.modules
    client = TestClient(mp.app)
    r = client.get("/v1/voices")
    assert r.status_code == 200
    backends = {v["backend"] for v in r.json()}
    # No model is present in the test env -> piper builds nothing -> empty list.
    # If a model IS present, the only backend may be piper. Never kokoro/chatterbox.
    assert backends <= {"piper"}


def test_build_piper_registry_uses_current_interpreter(monkeypatch, tmp_path):
    import hosaka.server.main_piper as mp
    from hosaka import config

    model = tmp_path / "glados.onnx"
    model.write_bytes(b"\x00")  # presence is all build_piper_registry checks
    monkeypatch.setattr(config, "PIPER_VOICES", {"glados": {"model": model, "description": "x"}})
    # main_piper imported PIPER_VOICES by value; patch the module's own reference too.
    monkeypatch.setattr(mp, "PIPER_VOICES", config.PIPER_VOICES, raising=False)

    reg = mp.build_piper_registry()
    assert reg.kokoro is None and reg.chatterbox is None and reg.rvc is None
    assert reg.piper is not None
    assert reg.piper.voice_ids == ["glados"]
    # The sidecar runs under THIS interpreter (no .venv-piper in the container).
    assert reg.piper._cmd[0] == sys.executable
