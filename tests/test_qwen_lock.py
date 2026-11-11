# tests/test_qwen_lock.py
import subprocess
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _check(state: Path):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(state / "home"), "XDG_STATE_HOME": str(state)}
    return subprocess.run(
        ["bash", "-c", f". {SCRIPTS}/qwen_lock.sh; refuse_if_qwen_holds_gpu"],
        capture_output=True,
        text=True,
        env=env,
    )


def _lock(state: Path, pid: int) -> Path:
    d = state / "qwen" / "locks"
    d.mkdir(parents=True, exist_ok=True)
    f = d / str(pid)
    f.write_text("opencode\n")
    return f


def _dead_pid() -> int:
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


def test_no_locks_dir_starts(tmp_path):
    assert _check(tmp_path).returncode == 0


def test_live_lock_refuses(tmp_path):
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        f = _lock(tmp_path, sleeper.pid)
        r = _check(tmp_path)
        assert r.returncode == 1
        assert "qwen is using the GPU (1 session(s)); close them or run `qwen stop`" in r.stderr
        assert f.exists()
    finally:
        sleeper.kill()
        sleeper.wait()


def test_dead_lock_is_cleared_and_start_goes_on(tmp_path):
    f = _lock(tmp_path, _dead_pid())
    r = _check(tmp_path)
    assert r.returncode == 0
    assert not f.exists()


def test_live_lock_blocks_even_with_stale_one(tmp_path):
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        _lock(tmp_path, sleeper.pid)
        stale = _lock(tmp_path, _dead_pid())
        r = _check(tmp_path)
        assert r.returncode == 1
        assert "(1 session(s))" in r.stderr
        assert not stale.exists()
    finally:
        sleeper.kill()
        sleeper.wait()
