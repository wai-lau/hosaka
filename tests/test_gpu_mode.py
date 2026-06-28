from fastapi.testclient import TestClient

from hosaka.server.main_gpu_mode import create_gpu_mode_app

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class FakeRunner:
    """Records actions; returns a scripted mode. `status` returns self.mode."""

    def __init__(self, mode="idle"):
        self.mode = mode
        self.calls = []

    def __call__(self, action):
        self.calls.append(action)
        if action == "status":
            return self.mode
        # action verbs settle into the matching mode
        self.mode = {"homo": "homo", "emo": "emo", "idle": "idle"}[action]
        return self.mode


def _client(runner):
    return TestClient(create_gpu_mode_app(runner=runner, token=TOKEN))


def test_mode_returns_status():
    c = _client(FakeRunner(mode="homo"))
    r = c.get("/mode", headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {"mode": "homo"}


def test_post_emo_dispatches_and_returns_mode():
    runner = FakeRunner(mode="homo")
    r = _client(runner).post("/emo", headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {"mode": "emo"}
    assert "emo" in runner.calls


def test_post_idle_dispatches():
    runner = FakeRunner(mode="homo")
    r = _client(runner).post("/idle", headers=AUTH)
    assert r.json() == {"mode": "idle"}
    assert "idle" in runner.calls


def test_post_homo_dispatches():
    runner = FakeRunner(mode="idle")
    r = _client(runner).post("/homo", headers=AUTH)
    assert r.json() == {"mode": "homo"}
    assert "homo" in runner.calls


def test_missing_token_is_401():
    assert _client(FakeRunner()).get("/mode").status_code == 401


def test_bad_token_is_401():
    c = _client(FakeRunner())
    assert c.get("/mode", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_runner_failure_is_500():
    def boom(action):
        raise RuntimeError("systemctl exploded")

    r = _client(boom).post("/emo", headers=AUTH)
    assert r.status_code == 500


def test_mixed_is_repaired_to_idle_label():
    # parse_mode collapses the both-up invariant violation so it is never shown.
    from hosaka.gpu_mode import parse_mode

    assert parse_mode("mixed") == "idle"
    assert parse_mode("homo") == "homo"
