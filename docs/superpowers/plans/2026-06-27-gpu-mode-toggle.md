# GPU-mode toggle (homo/emo/idle) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A 3-button segmented control on the owner-only exec-fn `/hosaka` page that switches the home-box GPU between hosaka TTS (`homo`), the ollama LLM (`emo`), and nothing (`idle`).

**Architecture:** A tiny always-on home-box FastAPI service (`gpu-mode`, port 8124, GPU-free, runs under `.venv-dev`) shells out to `scripts/gpu_mode.sh` to drive the two systemd units; it is reverse-tunneled to the droplet alongside the existing hosaka tunnel. exec-fn adds owner-only proxy routes + UI that call it, guarding any hosaka-stopping switch behind an active-user confirm.

**Tech Stack:** bash + systemd (home box), FastAPI + httpx (both services), vanilla JS (UI), pytest (`.venv-dev` for hosaka, `tests/` pure-fn + smoke for exec-fn).

**Spec:** `docs/superpowers/specs/2026-06-27-gpu-mode-toggle-design.md`

**Two repos:** Phase A is the `hosaka` repo (`/home/wai/src/hosaka`). Phase B is the `exec-fn` repo (`/home/wai/src/exec-fn`). Phase A is independently shippable and testable (curl `:8124/mode`); do it first. Phase B depends on A's HTTP contract.

## Global Constraints

- **No Unicode emoji** anywhere (code, comments, commits, docs). Plain text only.
- **`.venv-dev` has NO torch.** The gpu-mode service must import only fastapi + stdlib + subprocess. Never import `hosaka.server.engines.*` or torch from it.
- **Ports:** hosaka TTS = `127.0.0.1:8123` (unchanged). gpu-mode = `127.0.0.1:8124`. Tunnel binds the droplet **docker bridge** `172.17.0.1`, not loopback.
- **Four modes, XOR:** `homo` (hosaka up), `emo` (ollama up), `idle` (both down, reachable), `gone` (exec-fn cannot reach the home service). Both-up is an invariant violation, never a displayed mode.
- **Active-user guard:** actions that stop hosaka-server (`emo`, `idle`) require confirmation when `len(_presence) > 0`; `homo` never does.
- **Auth:** exec-fn routes are owner-only (`require_auth` / `protected` router), NOT guest. The home service requires `Authorization: Bearer $GPU_MODE_TOKEN` on every route.
- **sudo:** only `systemctl start ollama` and `systemctl stop ollama` are NOPASSWD; nothing else.
- **hosaka tests:** `.venv-dev/bin/python -m pytest -m "not gpu"`. Lint before commit: `.venv-dev/bin/ruff check --fix` + `.venv-dev/bin/ruff format`.
- **Both repos: push immediately after every commit** (`git push origin main`). hosaka and exec-fn are both prod.
- **Commit doc-sync:** every commit, check whether ARCHITECTURE.md / CLAUDE.md need updating; update in the same commit (covered explicitly by Task B4).

---

# Phase A -- hosaka home-box `gpu-mode` service

## File structure (Phase A)

- Create `scripts/gpu_mode.sh` -- shell, single source of truth for the systemctl logic (`homo|emo|idle|status`).
- Create `hosaka/gpu_mode.py` -- pure mode-parsing helper (no I/O).
- Create `hosaka/server/main_gpu_mode.py` -- the FastAPI app (`create_gpu_mode_app(runner, token)` + module `app`), mirrors `main_piper.py`.
- Create `scripts/start_gpu_mode.sh` -- execs uvicorn on the app under `.venv-dev`.
- Create `scripts/systemd/gpu-mode.service` -- systemd user unit.
- Create `scripts/systemd/hosaka-tunnel.service` -- canonical tunnel unit (currently only installed, not in the repo) with BOTH `-R` forwards.
- Create `deploy/gpu-mode.sudoers` -- the scoped NOPASSWD rule.
- Create `tests/test_gpu_mode_sh.py` -- drives `gpu_mode.sh` against a fake `systemctl`/`sudo` on PATH.
- Create `tests/test_gpu_mode.py` -- TestClient tests of the FastAPI app with an injected fake runner.
- Modify `~/bin/homo`, `~/bin/emo` -- rewrite as one-line wrappers (outside the repo; not git-tracked).
- Modify `README.md` -- install/ops notes for the service, tunnel, sudoers.

---

### Task A1: `gpu_mode.sh` + `~/bin` wrappers

**Files:**
- Create: `scripts/gpu_mode.sh`
- Modify: `/home/wai/bin/homo`, `/home/wai/bin/emo` (not git-tracked)
- Test: `tests/test_gpu_mode_sh.py`

**Interfaces:**
- Produces: `scripts/gpu_mode.sh <homo|emo|idle|status>`. `status` prints exactly one of `homo|emo|idle` to stdout (both-up prints `mixed` so callers can detect the invariant violation). Action verbs print a human line and exit 0; idempotent (already in target mode -> print + exit 0).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gpu_mode_sh.py
import subprocess
import textwrap
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gpu_mode.sh"


def _fake_bin(tmp_path: Path, ollama: str, hosaka: str) -> Path:
    """A bin dir with fake `systemctl` + `sudo` on PATH.

    `systemctl is-active ollama`            -> $ollama (from state file)
    `systemctl --user is-active hosaka...`  -> $hosaka (from state file)
    start/stop mutate the state files and log the argv.
    `sudo X...`                             -> exec X... (so sudo is transparent)
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (tmp_path / "ollama.state").write_text(ollama)
    (tmp_path / "hosaka.state").write_text(hosaka)
    sysctl = bindir / "systemctl"
    sysctl.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        args=("$@"); [ "${{args[0]}}" = "--user" ] && unset 'args[0]' && args=("${{args[@]}}")
        verb="${{args[0]}}"; unit="${{args[1]}}"
        case "$unit" in *hosaka*) sf="{tmp_path}/hosaka.state";; *) sf="{tmp_path}/ollama.state";; esac
        case "$verb" in
          is-active) cat "$sf";;
          start) echo active > "$sf"; echo "start $unit" >> "{tmp_path}/calls.log";;
          stop)  echo inactive > "$sf"; echo "stop $unit"  >> "{tmp_path}/calls.log";;
        esac
    """))
    sudo = bindir / "sudo"
    sudo.write_text('#!/usr/bin/env bash\nexec "$@"\n')
    for f in (sysctl, sudo):
        f.chmod(0o755)
    return bindir


def _run(tmp_path, bindir, arg):
    env = {"PATH": f"{bindir}:/usr/bin:/bin"}
    return subprocess.run([str(SCRIPT), arg], capture_output=True, text=True, env=env)


def test_status_homo(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="inactive", hosaka="active")
    r = _run(tmp_path, bindir, "status")
    assert r.stdout.strip() == "homo"


def test_status_emo(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="active", hosaka="inactive")
    assert _run(tmp_path, bindir, "status").stdout.strip() == "emo"


def test_status_idle(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="inactive", hosaka="inactive")
    assert _run(tmp_path, bindir, "status").stdout.strip() == "idle"


def test_emo_stops_hosaka_starts_ollama(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="inactive", hosaka="active")
    r = _run(tmp_path, bindir, "emo")
    assert r.returncode == 0
    log = (tmp_path / "calls.log").read_text()
    assert "stop hosaka-server" in log
    assert "start ollama" in log
    assert _run(tmp_path, bindir, "status").stdout.strip() == "emo"


def test_idle_stops_both(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="active", hosaka="active")
    _run(tmp_path, bindir, "idle")
    assert _run(tmp_path, bindir, "status").stdout.strip() == "idle"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-dev/bin/python -m pytest tests/test_gpu_mode_sh.py -v`
Expected: FAIL (script does not exist / non-zero exit).

- [ ] **Step 3: Write `scripts/gpu_mode.sh`**

```bash
#!/usr/bin/env bash
# gpu_mode.sh -- arbitrate the home-box GPU between hosaka TTS and ollama.
# Single source of truth for the systemctl logic; ~/bin/homo, ~/bin/emo and the
# gpu-mode service all call this. Idempotent.
#
#   homo  -> hosaka holds the GPU  (ollama off, hosaka-server on)
#   emo   -> ollama holds the GPU  (hosaka-server off, ollama on)
#   idle  -> nothing on the GPU    (both off)
#   status-> print one of: homo | emo | idle | mixed
set -euo pipefail

ollama_active()  { [ "$(systemctl is-active ollama 2>/dev/null)" = active ]; }
hosaka_active()  { [ "$(systemctl --user is-active hosaka-server 2>/dev/null)" = active ]; }

status() {
  local o=0 h=0
  ollama_active && o=1
  hosaka_active && h=1
  if   [ "$h" = 1 ] && [ "$o" = 0 ]; then echo homo
  elif [ "$o" = 1 ] && [ "$h" = 0 ]; then echo emo
  elif [ "$o" = 0 ] && [ "$h" = 0 ]; then echo idle
  else echo mixed   # invariant violation: both up
  fi
}

case "${1:-status}" in
  homo)
    [ "$(status)" = homo ] && { echo "already homo"; exit 0; }
    sudo systemctl stop ollama
    systemctl --user start hosaka-server
    echo "-> homo (ollama off, hosaka up)"
    ;;
  emo)
    [ "$(status)" = emo ] && { echo "already emo"; exit 0; }
    systemctl --user stop hosaka-server
    sudo systemctl start ollama
    echo "-> emo (hosaka off, ollama up)"
    ;;
  idle)
    [ "$(status)" = idle ] && { echo "already idle"; exit 0; }
    systemctl --user stop hosaka-server
    sudo systemctl stop ollama
    echo "-> idle (gpu free)"
    ;;
  status) status ;;
  *) echo "usage: gpu_mode.sh {homo|emo|idle|status}" >&2; exit 2 ;;
esac
```

Then `chmod +x scripts/gpu_mode.sh`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-dev/bin/python -m pytest tests/test_gpu_mode_sh.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Rewrite the `~/bin` wrappers**

```bash
printf '#!/bin/sh\nexec /home/wai/src/hosaka/scripts/gpu_mode.sh homo\n' > /home/wai/bin/homo
printf '#!/bin/sh\nexec /home/wai/src/hosaka/scripts/gpu_mode.sh emo\n'  > /home/wai/bin/emo
chmod +x /home/wai/bin/homo /home/wai/bin/emo
```

(These files are outside the repo; no commit. They now share the script's logic so CLI and service never diverge.)

- [ ] **Step 6: Commit**

```bash
cd /home/wai/src/hosaka
.venv-dev/bin/ruff format tests/test_gpu_mode_sh.py && .venv-dev/bin/ruff check --fix tests/test_gpu_mode_sh.py
git add scripts/gpu_mode.sh tests/test_gpu_mode_sh.py
git commit -m "feat(gpu-mode): gpu_mode.sh homo/emo/idle/status arbitration"
git push origin main
```

---

### Task A2: gpu-mode FastAPI app

**Files:**
- Create: `hosaka/gpu_mode.py`
- Create: `hosaka/server/main_gpu_mode.py`
- Test: `tests/test_gpu_mode.py`

**Interfaces:**
- Consumes: `scripts/gpu_mode.sh` from Task A1 (the default runner shells out to it).
- Produces:
  - `hosaka/gpu_mode.py`: `parse_mode(raw: str) -> str` -- maps the script's `homo|emo|idle|mixed` stdout to a display mode, collapsing/repairing `mixed`. `VALID_ACTIONS = ("homo", "emo", "idle")`.
  - `hosaka/server/main_gpu_mode.py`: `create_gpu_mode_app(runner: Callable[[str], str], token: str) -> FastAPI` and module-level `app`. Routes: `GET /mode` -> `{"mode": str}`, `POST /homo|/emo|/idle` -> `{"mode": str}`. `runner(action)` returns the script's stdout for that action; `runner("status")` returns current mode. All routes require `Authorization: Bearer <token>`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gpu_mode.py
import pytest
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-dev/bin/python -m pytest tests/test_gpu_mode.py -v`
Expected: FAIL with `ModuleNotFoundError: hosaka.server.main_gpu_mode`.

- [ ] **Step 3: Write `hosaka/gpu_mode.py`**

```python
"""Pure mode helpers for the gpu-mode service. No I/O, no torch -- safe to
import under .venv-dev."""

VALID_ACTIONS = ("homo", "emo", "idle")
_DISPLAY = {"homo": "homo", "emo": "emo", "idle": "idle"}


def parse_mode(raw: str) -> str:
    """Map gpu_mode.sh stdout to a display mode. `mixed` (both services up) is an
    invariant violation that must never reach the UI; collapse it to `idle` --
    the next status read after a settled action returns the real mode."""
    return _DISPLAY.get((raw or "").strip(), "idle")
```

- [ ] **Step 4: Write `hosaka/server/main_gpu_mode.py`**

```python
"""Always-on, GPU-free home-box service that arbitrates the GPU between hosaka
TTS and ollama by shelling out to scripts/gpu_mode.sh. Runs under .venv-dev on
127.0.0.1:8124 and is reverse-tunneled to the droplet. Imports NO torch / no
hosaka engines -- keep it that way.

Auth: every route requires `Authorization: Bearer $GPU_MODE_TOKEN`. The primary
boundary is loopback + the SSH tunnel; the token is defense-in-depth on the
tunnel hop."""

import os
import subprocess
from pathlib import Path
from typing import Callable

from fastapi import Depends, FastAPI, Header, HTTPException

from hosaka.gpu_mode import VALID_ACTIONS, parse_mode

_SCRIPT = Path(__file__).resolve().parent.parent.parent / "scripts" / "gpu_mode.sh"


def _shell_runner(action: str) -> str:
    """Default runner: run gpu_mode.sh <action>, return current mode.

    Action verbs change state then we re-read `status` so the response always
    reflects settled reality, not the verb we asked for."""
    subprocess.run([str(_SCRIPT), action], check=True, capture_output=True, text=True)
    out = subprocess.run([str(_SCRIPT), "status"], check=True, capture_output=True, text=True)
    return out.stdout


def create_gpu_mode_app(runner: Callable[[str], str] = _shell_runner, token: str | None = None) -> FastAPI:
    token = token if token is not None else os.environ.get("GPU_MODE_TOKEN", "")
    app = FastAPI()

    def require_token(authorization: str | None = Header(default=None)):
        expected = f"Bearer {token}"
        if not token or authorization != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

    def _mode(action: str) -> dict:
        try:
            return {"mode": parse_mode(runner(action))}
        except Exception as e:  # subprocess.CalledProcessError or anything else
            raise HTTPException(status_code=500, detail=str(e)[:200])

    @app.get("/mode", dependencies=[Depends(require_token)])
    def get_mode():
        return _mode("status")

    for action in VALID_ACTIONS:
        # bind `action` per-iteration via default arg
        @app.post(f"/{action}", dependencies=[Depends(require_token)])
        def do_action(action=action):
            return _mode(action)

    return app


app = create_gpu_mode_app()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv-dev/bin/python -m pytest tests/test_gpu_mode.py -v`
Expected: PASS (7 passed).

- [ ] **Step 6: Commit**

```bash
cd /home/wai/src/hosaka
.venv-dev/bin/ruff format hosaka/gpu_mode.py hosaka/server/main_gpu_mode.py tests/test_gpu_mode.py
.venv-dev/bin/ruff check --fix hosaka/gpu_mode.py hosaka/server/main_gpu_mode.py tests/test_gpu_mode.py
git add hosaka/gpu_mode.py hosaka/server/main_gpu_mode.py tests/test_gpu_mode.py
git commit -m "feat(gpu-mode): FastAPI service (mode/homo/emo/idle) with bearer auth"
git push origin main
```

---

### Task A3: start script + systemd unit + uvicorn dep

**Files:**
- Create: `scripts/start_gpu_mode.sh`
- Create: `scripts/systemd/gpu-mode.service`
- (env) Ensure `uvicorn` is installed in `.venv-dev`.

**Interfaces:**
- Consumes: `hosaka.server.main_gpu_mode:app` (Task A2), `GPU_MODE_TOKEN` from the environment.
- Produces: a running service answering `http://127.0.0.1:8124/mode`.

- [ ] **Step 1: Ensure uvicorn in `.venv-dev`**

Run: `.venv-dev/bin/python -c "import uvicorn" 2>/dev/null || .venv-dev/bin/pip install uvicorn`
Expected: exits 0; `uvicorn` importable. (`.venv-dev` already has fastapi + httpx; this adds the ASGI server. No torch.)

- [ ] **Step 2: Write `scripts/start_gpu_mode.sh`**

```bash
#!/usr/bin/env bash
# Launch the always-on gpu-mode service on 127.0.0.1:8124 under .venv-dev
# (GPU-free). Reads GPU_MODE_TOKEN from the environment (set by the systemd unit
# via EnvironmentFile). Run by hand the same way: bash scripts/start_gpu_mode.sh
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv-dev/bin/python -m uvicorn hosaka.server.main_gpu_mode:app \
  --host 127.0.0.1 --port 8124
```

Then `chmod +x scripts/start_gpu_mode.sh`.

- [ ] **Step 3: Write `scripts/systemd/gpu-mode.service`**

```ini
# Always-on, GPU-free home-box service that arbitrates the GPU between hosaka
# TTS and ollama. Reverse-tunneled to the droplet alongside hosaka-tunnel.
#
# Install (one time):
#   mkdir -p ~/.config/systemd/user ~/.config/hosaka
#   printf 'GPU_MODE_TOKEN=%s\n' "$(openssl rand -hex 32)" > ~/.config/hosaka/gpu-mode.env
#   chmod 600 ~/.config/hosaka/gpu-mode.env
#   cp scripts/systemd/gpu-mode.service ~/.config/systemd/user/
#   systemctl --user daemon-reload
#   systemctl --user enable --now gpu-mode.service
#   loginctl enable-linger "$USER"
#   curl -s -H "Authorization: Bearer $(. ~/.config/hosaka/gpu-mode.env; echo $GPU_MODE_TOKEN)" \
#        localhost:8124/mode
[Unit]
Description=hosaka gpu-mode switch (homo/emo/idle, 127.0.0.1:8124, GPU-free)
After=default.target

[Service]
Type=simple
WorkingDirectory=/home/wai/src/hosaka
# /usr/bin so `systemctl` and `sudo` resolve; gpu_mode.sh needs both.
Environment=PATH=/usr/bin:/bin
EnvironmentFile=%h/.config/hosaka/gpu-mode.env
ExecStart=/home/wai/src/hosaka/scripts/start_gpu_mode.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

- [ ] **Step 4: Install + start + verify**

```bash
mkdir -p ~/.config/systemd/user ~/.config/hosaka
[ -f ~/.config/hosaka/gpu-mode.env ] || { printf 'GPU_MODE_TOKEN=%s\n' "$(openssl rand -hex 32)" > ~/.config/hosaka/gpu-mode.env; chmod 600 ~/.config/hosaka/gpu-mode.env; }
cp scripts/systemd/gpu-mode.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now gpu-mode.service
sleep 2
TOKEN=$(. ~/.config/hosaka/gpu-mode.env; echo "$GPU_MODE_TOKEN")
curl -s -H "Authorization: Bearer $TOKEN" localhost:8124/mode
```

Expected: `{"mode":"homo"}` or `{"mode":"emo"}` / `{"mode":"idle"}` depending on current state, AND `curl -s localhost:8124/mode` (no header) returns `{"detail":"Unauthorized"}` with 401.

NOTE: actions (`POST /emo` etc.) will fail at the `sudo systemctl ... ollama` step until Task A5 installs the NOPASSWD rule. `GET /mode` works now. Do not "fix" a sudo password prompt here -- A5 is the fix.

- [ ] **Step 5: Commit**

```bash
cd /home/wai/src/hosaka
git add scripts/start_gpu_mode.sh scripts/systemd/gpu-mode.service
git commit -m "feat(gpu-mode): start script + systemd user unit (port 8124)"
git push origin main
```

---

### Task A4: tunnel forward (add 8124)

**Files:**
- Create: `scripts/systemd/hosaka-tunnel.service` (canonical copy; currently only installed, not tracked).
- Modify: installed `~/.config/systemd/user/hosaka-tunnel.service`.

**Interfaces:**
- Produces: droplet `172.17.0.1:8124` reverse-forwarded to home `127.0.0.1:8124`.

- [ ] **Step 1: Write `scripts/systemd/hosaka-tunnel.service`**

Copy the currently-installed unit and add the second `-R` forward. The full file:

```ini
[Unit]
Description=hosaka reverse SSH tunnel (home :8123 + :8124 -> droplet 172.17.0.1)
After=network-online.target
Wants=network-online.target
# Never stop retrying: while the key is unauthorized or the link is down, ssh
# exits fast; without this systemd would give up after a few quick restarts.
StartLimitIntervalSec=0

[Service]
# Two forwards over one tunnel: 8123 = hosaka TTS, 8124 = gpu-mode switch.
# Config supplies ServerAliveInterval / ExitOnForwardFailure, so a dead link
# makes ssh exit and systemd reconnect.
ExecStart=/usr/bin/ssh -NT \
  -R 172.17.0.1:8123:127.0.0.1:8123 \
  -R 172.17.0.1:8124:127.0.0.1:8124 \
  wai-lau-tunnel
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
```

- [ ] **Step 2: Install + restart**

```bash
cp scripts/systemd/hosaka-tunnel.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user restart hosaka-tunnel.service
sleep 3
systemctl --user is-active hosaka-tunnel.service
```

Expected: `active`.

- [ ] **Step 3: Verify the forward from the droplet**

Run (from the home box; SSHes to the droplet and curls the bridge):
```bash
ssh wai-lau.net 'curl -s -o /dev/null -w "%{http_code}\n" http://172.17.0.1:8124/mode'
```
Expected: `401` (reached the service, rejected for missing token -- proves the forward works end-to-end). If `000`/timeout, the tunnel forward is not up -- re-check Step 2; do not proceed.

- [ ] **Step 4: Commit**

```bash
cd /home/wai/src/hosaka
git add scripts/systemd/hosaka-tunnel.service
git commit -m "feat(gpu-mode): tunnel gpu-mode :8124 to the droplet (track tunnel unit)"
git push origin main
```

---

### Task A5: scoped NOPASSWD sudoers

**Files:**
- Create: `deploy/gpu-mode.sudoers`

**Interfaces:**
- Produces: passwordless `sudo systemctl start ollama` / `sudo systemctl stop ollama` for user `wai`, enabling the `emo`/`homo`/`idle` actions from the daemon.

- [ ] **Step 1: Write `deploy/gpu-mode.sudoers`**

```
# Lets the gpu-mode service (running as wai) start/stop the ollama system unit
# without a password. Scope is exactly these two argv -- nothing else.
# Install: sudo install -m 0440 -o root -g root deploy/gpu-mode.sudoers /etc/sudoers.d/gpu-mode
# Validate BEFORE trusting: sudo visudo -cf /etc/sudoers.d/gpu-mode
wai ALL=(root) NOPASSWD: /usr/bin/systemctl start ollama, /usr/bin/systemctl stop ollama
```

- [ ] **Step 2: Install (manual, needs root)**

The implementer cannot run interactive sudo. Hand the user this command to run via the session `!` prefix:

```
! sudo install -m 0440 -o root -g root deploy/gpu-mode.sudoers /etc/sudoers.d/gpu-mode && sudo visudo -cf /etc/sudoers.d/gpu-mode
```

Expected: `/etc/sudoers.d/gpu-mode: parsed OK`.

- [ ] **Step 3: Verify passwordless + an end-to-end switch**

```bash
sudo -n systemctl is-active ollama   # must NOT prompt for a password
TOKEN=$(. ~/.config/hosaka/gpu-mode.env; echo "$GPU_MODE_TOKEN")
curl -s -X POST -H "Authorization: Bearer $TOKEN" localhost:8124/emo
curl -s    -H "Authorization: Bearer $TOKEN" localhost:8124/mode
curl -s -X POST -H "Authorization: Bearer $TOKEN" localhost:8124/homo
```

Expected: `sudo -n` prints `active`/`inactive` without prompting; `/emo` -> `{"mode":"emo"}`; `/mode` -> `{"mode":"emo"}`; `/homo` -> `{"mode":"homo"}`.

CAUTION: this switch is live -- it will stop hosaka-server / start ollama on the real box. Run it when no one is using the TTS server.

- [ ] **Step 4: Commit**

```bash
cd /home/wai/src/hosaka
git add deploy/gpu-mode.sudoers
git commit -m "feat(gpu-mode): scoped NOPASSWD sudoers for systemctl start/stop ollama"
git push origin main
```

**Phase A is now independently shippable: the home box exposes a working, tunneled, authenticated mode switch.**

---

# Phase B -- exec-fn protected route + UI

Work in `/home/wai/src/exec-fn`. Phase B depends on Phase A being deployed (the home `:8124` service reachable at `172.17.0.1:8124` from the droplet).

## File structure (Phase B)

- Create `api/gpu_mode_client.py` -- pure guard helper + async proxy helpers (`needs_user_confirm`, `fetch_mode`, `switch_mode`).
- Modify `api/routes_tts.py` -- add owner-only `GET`/`POST /api/hosaka/mode` on the `protected` router; import `protected`; read `GPU_MODE_UPSTREAM` / `GPU_MODE_TOKEN`.
- Modify `api/templates/tts.html` -- add the 3-button segmented control markup; bump `tts.js` version.
- Modify `api/static/tts.js` -- mode control logic (fetch/render/click/confirm).
- Create `tests/test_gpu_mode_client.py` -- pure-fn tests for `needs_user_confirm`.
- Modify `tests/test_smoke.py` -- a smoke check that the authed mode route answers with a known mode.
- Modify `ARCHITECTURE.md` (both repos) + `CLAUDE.md` (hosaka) -- Task B4.

---

### Task B1: mode client (pure guard + proxy helpers)

**Files:**
- Create: `api/gpu_mode_client.py`
- Test: `tests/test_gpu_mode_client.py`

**Interfaces:**
- Produces:
  - `needs_user_confirm(action: str, presence_count: int, force: bool) -> bool` -- True iff `action in {"emo","idle"}` and `presence_count > 0` and not `force`.
  - `async fetch_mode(upstream: str, token: str) -> str` -- GET `http://{upstream}/mode`; returns the mode string, or `"gone"` on any failure.
  - `async switch_mode(upstream: str, token: str, action: str) -> str` -- POST `http://{upstream}/{action}`; returns the resulting mode, or `"gone"` on failure.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gpu_mode_client.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from gpu_mode_client import needs_user_confirm  # noqa: E402


def test_emo_with_users_needs_confirm():
    assert needs_user_confirm("emo", presence_count=2, force=False) is True


def test_idle_with_users_needs_confirm():
    assert needs_user_confirm("idle", presence_count=1, force=False) is True


def test_homo_never_needs_confirm():
    assert needs_user_confirm("homo", presence_count=5, force=False) is False


def test_no_users_no_confirm():
    assert needs_user_confirm("emo", presence_count=0, force=False) is False


def test_force_bypasses_confirm():
    assert needs_user_confirm("emo", presence_count=3, force=True) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/wai/src/exec-fn && python -m pytest tests/test_gpu_mode_client.py -v`
Expected: FAIL with `ModuleNotFoundError: gpu_mode_client`.

- [ ] **Step 3: Write `api/gpu_mode_client.py`**

```python
"""Owner-only client for the home-box gpu-mode service (port 8124, reached over
the SSH reverse tunnel at 172.17.0.1:8124). Pure guard logic + thin async
proxies; the route layer (routes_tts.py) owns auth + the _presence count."""

import httpx

_STOP_HOSAKA = {"emo", "idle"}  # actions that kill hosaka-server -> guard them


def needs_user_confirm(action: str, presence_count: int, force: bool) -> bool:
    """True iff this switch would cut off connected users and the caller has not
    already confirmed. homo (which starts hosaka) never needs it."""
    return action in _STOP_HOSAKA and presence_count > 0 and not force


async def fetch_mode(upstream: str, token: str) -> str:
    """Current mode, or 'gone' if the home service / tunnel is unreachable."""
    try:
        async with httpx.AsyncClient(timeout=4) as client:
            r = await client.get(
                f"http://{upstream}/mode",
                headers={"Authorization": f"Bearer {token}"},
            )
            r.raise_for_status()
            return r.json()["mode"]
    except Exception:
        return "gone"


async def switch_mode(upstream: str, token: str, action: str) -> str:
    """Run an action; return the resulting mode, or 'gone' on failure."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                f"http://{upstream}/{action}",
                headers={"Authorization": f"Bearer {token}"},
            )
            r.raise_for_status()
            return r.json()["mode"]
    except Exception:
        return "gone"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/wai/src/exec-fn && python -m pytest tests/test_gpu_mode_client.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
cd /home/wai/src/exec-fn
git add api/gpu_mode_client.py tests/test_gpu_mode_client.py
git commit -m "feat(gpu-mode): exec-fn mode client (guard + proxy helpers)"
git push origin main
```

---

### Task B2: owner-only mode routes

**Files:**
- Modify: `api/routes_tts.py`

**Interfaces:**
- Consumes: `needs_user_confirm`, `fetch_mode`, `switch_mode` (B1); `_presence` (existing, same module); `protected` router (from `routers`).
- Produces: `GET /api/hosaka/mode` -> `{"mode": str}`; `POST /api/hosaka/mode {action, force?}` -> `{"mode": str}` or `409 {"detail":"active_users","count": int}`.

- [ ] **Step 1: Add imports + env + routes to `api/routes_tts.py`**

At the top, alongside the existing `from routers import guest_protected, public`, add `protected`:

```python
from routers import guest_protected, protected, public
```

Add to the imports block:

```python
from fastapi import HTTPException
from gpu_mode_client import fetch_mode, needs_user_confirm, switch_mode
```

Near the `_UPSTREAM` / `_PIPER_UPSTREAM` env block, add:

```python
# Home-box gpu-mode switch (homo/emo/idle), reached over the same SSH tunnel as
# the TTS upstream. Owner-only; bearer-authed on the home side.
_GPU_MODE_UPSTREAM = os.environ.get("GPU_MODE_UPSTREAM", "172.17.0.1:8124")
_GPU_MODE_TOKEN = os.environ.get("GPU_MODE_TOKEN", "")
```

Then add the two routes (place them after `tts_health`, before the websocket helpers):

```python
@protected.get("/api/hosaka/mode")
async def gpu_mode_get():
    """Current GPU mode (homo/emo/idle), or 'gone' if the home box is
    unreachable. Owner-only: guests never see the control."""
    return JSONResponse({"mode": await fetch_mode(_GPU_MODE_UPSTREAM, _GPU_MODE_TOKEN)})


@protected.post("/api/hosaka/mode")
async def gpu_mode_post(request: Request):
    body = await request.json()
    action = body.get("action")
    force = bool(body.get("force"))
    if action not in ("homo", "emo", "idle"):
        raise HTTPException(status_code=400, detail="bad action")
    if needs_user_confirm(action, len(_presence), force):
        raise HTTPException(status_code=409, detail={"detail": "active_users", "count": len(_presence)})
    return JSONResponse({"mode": await switch_mode(_GPU_MODE_UPSTREAM, _GPU_MODE_TOKEN, action)})
```

- [ ] **Step 2: Add a smoke check to `tests/test_smoke.py`**

Append a test that hits the live authed route (Bearer admin auth, per the conftest pattern). It asserts the route is wired and returns a known mode:

```python
def test_hosaka_mode_route_authed(base_url, admin_headers):
    r = httpx.get(f"{base_url}/api/hosaka/mode", headers=admin_headers, timeout=10)
    assert r.status_code == 200
    assert r.json()["mode"] in {"homo", "emo", "idle", "gone"}


def test_hosaka_mode_route_requires_auth(base_url):
    # no cookie / no bearer -> 401 from require_auth
    r = httpx.get(f"{base_url}/api/hosaka/mode", timeout=10)
    assert r.status_code == 401
```

(Use whatever the existing `admin_headers`/Bearer fixture in `tests/conftest.py` is named; match `test_smoke.py`'s existing authed tests.)

- [ ] **Step 3: Run the pure suite + (if a container is up) the smoke check**

Run: `cd /home/wai/src/exec-fn && python -m pytest tests/test_gpu_mode_client.py tests/test_tts_routing.py -v`
Expected: PASS. (The smoke tests need the live container + a deployed Phase A; run them after deploy in Task B4.)

- [ ] **Step 4: Commit**

```bash
cd /home/wai/src/exec-fn
git add api/routes_tts.py tests/test_smoke.py
git commit -m "feat(gpu-mode): owner-only /api/hosaka/mode get+post with active-user guard"
git push origin main
```

---

### Task B3: the 3-button UI

**Files:**
- Modify: `api/templates/tts.html`
- Modify: `api/static/tts.js`

**Interfaces:**
- Consumes: `GET`/`POST /api/hosaka/mode` (B2).
- Produces: a segmented control `emo | idle | homo` shown only to owners (the GET 401s for guests -> control stays hidden).

- [ ] **Step 1: Add the control markup to `api/templates/tts.html`**

Inside the existing `<div class="tts-presence-row">` (which holds `#tts-status` and `#tts-presence`), add the control (hidden until JS confirms owner):

```html
  <div id="tts-mode" class="tts-mode" hidden>
    <button type="button" class="tts-mode-btn" data-mode="emo">emo</button>
    <button type="button" class="tts-mode-btn" data-mode="idle">idle</button>
    <button type="button" class="tts-mode-btn" data-mode="homo">homo</button>
  </div>
```

Add the styling (in the page's existing `<style>` block, or a `<style>` near the top of the template):

```html
<style>
  .tts-mode { display: inline-flex; gap: 0; border-radius: 6px; overflow: hidden; }
  .tts-mode-btn {
    padding: 2px 10px; font: inherit; cursor: pointer;
    background: transparent; color: #ddd; border: 1px solid #888;
    border-left-width: 0;
  }
  .tts-mode-btn:first-child { border-left-width: 1px; border-radius: 6px 0 0 6px; }
  .tts-mode-btn:last-child  { border-radius: 0 6px 6px 0; }
  /* current mode: filled light bg, dark text, not clickable */
  .tts-mode-btn.active { background: #eee; color: #111; cursor: default; }
  /* gone: all greyed, deactivated */
  .tts-mode.gone .tts-mode-btn { color: #666; border-color: #444; cursor: not-allowed; background: transparent; }
</style>
```

Bump the script version so browsers reload it: change `<script src="/tts.js?v=20"></script>` to `<script src="/tts.js?v=21"></script>`.

- [ ] **Step 2: Add mode logic to `api/static/tts.js`**

Append this self-contained block (it does not depend on the rest of tts.js):

```javascript
// --- GPU mode control (owner-only). The GET 401s for guests, so the control
// stays hidden for them. ---
(function () {
  const el = document.getElementById("tts-mode");
  if (!el) return;
  const buttons = Array.from(el.querySelectorAll(".tts-mode-btn"));

  function render(mode) {
    el.hidden = false;
    el.classList.toggle("gone", mode === "gone");
    for (const b of buttons) {
      const isActive = b.dataset.mode === mode;
      b.classList.toggle("active", isActive);
      // active button + gone state are non-interactive; others clickable
      b.disabled = isActive || mode === "gone";
    }
  }

  async function load() {
    try {
      const r = await fetch("/api/hosaka/mode");
      if (r.status === 401) { el.hidden = true; return; }  // guest: no control
      render((await r.json()).mode);
    } catch { el.hidden = true; }
  }

  async function post(action, force) {
    const r = await fetch("/api/hosaka/mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, force: !!force }),
    });
    if (r.status === 409) {
      const info = (await r.json()).detail || {};
      const n = info.count != null ? info.count : "some";
      if (confirm(n + " user(s) streaming -- switch anyway?")) return post(action, true);
      return;  // cancelled
    }
    render((await r.json()).mode);
  }

  for (const b of buttons) {
    b.addEventListener("click", () => { if (!b.disabled) post(b.dataset.mode, false); });
  }
  load();
})();
```

- [ ] **Step 3: Manual verification (browser)**

There is no headless unit test for the rendered control; verify by hand against the live deployment (after Task B4 deploy):

1. As the owner (full session cookie), open `https://wai-lau.net/hosaka`. Confirm the `emo | idle | homo` control shows; the button for the current mode is filled and not clickable; the other two are outlined and clickable.
2. Click an outlined mode (when no one else is streaming). Confirm it switches and the filled button moves.
3. With a second tab/device connected to `/hosaka`, click `emo` or `idle`; confirm the "N user(s) streaming -- switch anyway?" dialog appears; cancel leaves the mode unchanged, OK switches.
4. Stop `gpu-mode.service` on the home box; reload the page; confirm all three buttons are greyed/deactivated (`gone`).
5. As a guest (Turnstile session, not full owner), open `/hosaka`; confirm the control is NOT shown.

- [ ] **Step 4: Commit**

```bash
cd /home/wai/src/exec-fn
git add api/templates/tts.html api/static/tts.js
git commit -m "feat(gpu-mode): owner-only emo/idle/homo segmented control on /hosaka"
git push origin main
```

---

### Task B4: deploy env + docs

**Files:**
- (deploy) droplet compose / `.env` -- add `GPU_MODE_UPSTREAM`, `GPU_MODE_TOKEN`.
- Modify: `hosaka` `ARCHITECTURE.md`, `hosaka` `CLAUDE.md`.
- Modify: `exec-fn` `ARCHITECTURE.md`.

**Interfaces:**
- Produces: the deployed feature + docs reflecting it.

- [ ] **Step 1: Provision the droplet env**

The `GPU_MODE_TOKEN` must MATCH the home box's `~/.config/hosaka/gpu-mode.env`. Print the home value and set it on the droplet (hand the user these via `!`):

```
! cat ~/.config/hosaka/gpu-mode.env
```

Then on the droplet (per the exec-fn deploy layout -- compose env / `.env` at `/exec-fn`), add:
```
GPU_MODE_UPSTREAM=172.17.0.1:8124
GPU_MODE_TOKEN=<the value from the home env file>
```
and recreate the api container so it picks up the env (the exec-fn deploy uses `docker compose up`/`--reload`; follow the project's documented redeploy). Verify:
```
! ssh wai-lau.net 'curl -s -o /dev/null -w "%{http_code}\n" http://172.17.0.1:8124/mode'
```
Expected: `401` (reachable). Then run the exec-fn smoke tests from Task B2 against the live URL.

- [ ] **Step 2: Update hosaka `ARCHITECTURE.md` + `CLAUDE.md`**

Add a short section to hosaka `ARCHITECTURE.md` describing the gpu-mode service: port 8124, GPU-free `.venv-dev`, shells to `gpu_mode.sh`, the four modes, the second tunnel forward, and the NOPASSWD sudoers. Add a bullet to hosaka `CLAUDE.md` (e.g. under Conventions) noting the always-on `gpu-mode.service` and that `~/bin/homo`/`emo` are wrappers over `scripts/gpu_mode.sh`.

- [ ] **Step 3: Update exec-fn `ARCHITECTURE.md`**

Add a short note: owner-only `/api/hosaka/mode` proxies the home gpu-mode service over the tunnel (`172.17.0.1:8124`, bearer `GPU_MODE_TOKEN`); the `/hosaka` page shows the `emo|idle|homo` control to owners only; `emo`/`idle` confirm against `_presence`.

- [ ] **Step 4: Commit (both repos)**

```bash
cd /home/wai/src/hosaka
git add ARCHITECTURE.md CLAUDE.md
git commit -m "docs(gpu-mode): document the gpu-mode service, tunnel, sudoers"
git push origin main

cd /home/wai/src/exec-fn
git add ARCHITECTURE.md
git commit -m "docs(gpu-mode): document the owner-only mode route + control"
git push origin main
```

---

## Self-review notes

- **Spec coverage:** gpu_mode.sh + idle action (A1) · service GET/POST + bearer (A2) · start/unit/uvicorn (A3) · tunnel 8124 (A4) · sudoers (A5) · guard helper + modes incl. gone (B1) · owner-only routes + 409 guard (B2) · 3-button segmented control with active/outlined/grey states (B3) · env + docs (B4). All spec sections map to a task.
- **Both-up invariant:** handled in `gpu_mode.sh` (`mixed`) and collapsed in `parse_mode` so it never displays.
- **Type consistency:** `runner(action)->str` (A2) matches the FakeRunner + `_shell_runner`; `{"mode": str}` shape consistent across A2/B1/B2/JS; `needs_user_confirm(action, presence_count, force)` signature identical in B1 def, B1 test, and B2 call.
- **Known soft spot:** exec-fn route handlers (B2) are covered only by live smoke tests (the project's conftest runs against a live container, not an in-process app); the pure guard logic carries the unit coverage (B1). Acceptable per the existing exec-fn test architecture.
