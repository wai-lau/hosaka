# GLaDOS on the Droplet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve the CPU-only GLaDOS (Piper) voice from an always-on droplet container so it has 100% uptime independent of the home GPU box.

**Architecture:** A new piper-only composition root runs the existing hosaka server with `kokoro/chatterbox/rvc = None`, so it imports no torch and builds only the Piper engine. It ships as a `hosaka-piper` Docker image (built from the hosaka repo) running on the droplet's docker network. The exec-fn proxy stops being a blind byte-pump and routes per utterance: glados -> `hosaka-piper`, all other backends -> the home SSH tunnel (unchanged). The web UI greys out GPU voices when the home box is down while keeping glados live.

**Tech Stack:** Python 3.12, FastAPI/uvicorn, hosaka (`create_app`, Piper sidecar over a pipe), piper-tts + onnxruntime (CPU), Docker Compose, vanilla JS UI.

## Global Constraints

- Audio is float32 LE PCM, 24 kHz, mono everywhere (unchanged; the Piper sidecar already resamples 22.05k->24k).
- No Unicode emoji anywhere (code, comments, commits, docs). Plain text only.
- No torch / Kokoro / Chatterbox / RVC import in any module the piper-only root loads. The container installs NO torch.
- `Dockerfile.piper` lives in the **hosaka** repo; the Docker build is the code-transit (no submodule / vendor / pip-from-git).
- GLaDOS `.onnx` is baked into the image (not host-mounted). Reproducible; the droplet has no `~/.local/share/hosaka`.
- The home box server, GPU queue, and local REPL are OUT OF SCOPE — do not touch them.
- hosaka guards added to `app.py` must be benign on the full home server (engines never None there).
- hosaka non-GPU suite stays green: `.venv-dev/bin/python -m pytest -m "not gpu"`.
- hosaka: run `.venv-dev/bin/ruff check --fix` and `.venv-dev/bin/ruff format` before each hosaka commit. exec-fn: its own `ruff` config.
- hosaka is prod: `git push origin main` immediately after every hosaka commit.

---

### Task 1: hosaka app.py guards for absent kokoro/chatterbox

The full server always has Kokoro + Chatterbox; two paths assume they exist and misbehave when `None`. Fix both so a piper-only registry is correct. These are benign on the home server.

**Files:**
- Modify: `hosaka/server/app.py` (the `/v1/voices` builder around lines 241-292; `_resolve` around lines 65-90)
- Test: `tests/test_server.py` (add cases; reuse existing `FakeEngine` / `PiperFakeEngine`)

**Interfaces:**
- Consumes: `EngineRegistry(kokoro=None, chatterbox=None, piper=PiperFakeEngine())`, `create_app`, `TestClient` (all already imported in `tests/test_server.py`).
- Produces: a `/v1/voices` that emits kokoro voices only when `registry.kokoro is not None` and chatterbox voices only when `registry.chatterbox is not None`; a `_resolve` that returns `(None, "backend unavailable: <backend>")` when the resolved engine is `None`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_server.py`:

```python
def _client_piper_only(tmp_path):
    reg = EngineRegistry(kokoro=None, chatterbox=None, piper=PiperFakeEngine())
    lib = VoiceLibrary(tmp_path / "voices")
    return TestClient(create_app(reg, lib, do_warmup=False)), lib


def test_piper_only_voices_lists_only_piper(tmp_path):
    client, lib = _client_piper_only(tmp_path)
    # A library clip must NOT be advertised as chatterbox when chatterbox is None.
    seed = tmp_path / "s.wav"
    seed.write_bytes(b"RIFFfake")
    lib.add("myclone", seed, source="recording")
    voices = client.get("/v1/voices").json()
    backends = {v["backend"] for v in voices}
    assert backends == {"piper"}
    ids = {v["id"] for v in voices}
    assert "glados" in ids
    assert "nicole" not in ids   # kokoro preset suppressed
    assert "myclone" not in ids  # chatterbox clip suppressed


def test_piper_only_kokoro_request_errors_cleanly(tmp_path):
    client, _ = _client_piper_only(tmp_path)
    with client.websocket_connect("/v1/audio/stream") as ws:
        ws.send_json({"input": "hi", "backend": "kokoro", "voice": "nicole"})
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "unavailable" in msg["detail"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-dev/bin/python -m pytest tests/test_server.py::test_piper_only_voices_lists_only_piper tests/test_server.py::test_piper_only_kokoro_request_errors_cleanly -v`
Expected: FAIL — voices test sees `nicole`/`myclone` (kokoro/chatterbox emitted unconditionally); kokoro-request test crashes or hangs (resolves a `None` engine, then `_pcm_frames` calls `None.stream`).

- [ ] **Step 3: Guard the `/v1/voices` builder**

In `hosaka/server/app.py`, wrap the kokoro-preset block and the chatterbox-library block in `if registry.<engine> is not None:`. The result:

```python
    @app.get("/v1/voices")
    def voices():
        out = []
        if registry.kokoro is not None:
            out += [
                VoiceInfo(
                    id=p, backend="kokoro", source="preset", description=KOKORO_DESC.get(p, "")
                ).model_dump()
                for p in KOKORO_PRESETS
            ]
        # Library clips used only as an RVC source (e.g. Charlie's Chatterbox
        # clone) are not standalone voices -- hide them from the listing.
        rvc_sources = {
            s["source"] for s in RVC_VOICES.values() if s.get("source_backend") == "chatterbox"
        }
        if registry.chatterbox is not None:
            out += [
                VoiceInfo(
                    id=e.id,
                    backend="chatterbox",
                    source=e.source,
                    description=e.params.get("description", ""),
                    cb=True,
                ).model_dump()
                for e in library.list()
                if e.id not in rvc_sources
            ]
        if registry.piper is not None:
            # ... unchanged piper block ...
```

(Leave the existing `piper` and `rvc` blocks exactly as they are.)

- [ ] **Step 4: Guard `_resolve` against a None engine**

In `hosaka/server/app.py`, in `_resolve`, after `engine = registry.get(backend)` (and its `except KeyError`), add a None-engine guard before the per-backend voice checks:

```python
    try:
        engine = registry.get(backend)
    except KeyError:
        return None, f"unknown backend: {backend}"
    if engine is None:
        return None, f"backend unavailable: {backend}"
    if backend == "kokoro":
        ...
```

(`EngineRegistry.get` returns `self.kokoro` / `self.chatterbox` without a None-check, so `get("kokoro")` returns `None` rather than raising — this guard is what catches it.)

- [ ] **Step 5: Make the registry fields honestly Optional**

In `hosaka/server/engines/base.py`, change the `EngineRegistry` dataclass so kokoro/chatterbox are typed Optional (runtime already accepts None; this makes intent explicit):

```python
@dataclass
class EngineRegistry:
    kokoro: Engine | None = None
    chatterbox: Engine | None = None
    piper: Engine | None = None  # optional CPU sidecar (character voices)
    rvc: Engine | None = None  # optional GPU sidecar (converted character voices)
```

- [ ] **Step 6: Run the new tests + full non-GPU suite**

Run: `.venv-dev/bin/python -m pytest tests/test_server.py -v && .venv-dev/bin/python -m pytest -m "not gpu" -q`
Expected: PASS — new tests pass; every pre-existing test (full registry `/v1/voices` etc.) still passes (guards are benign).

- [ ] **Step 7: Lint + commit + push**

```bash
cd ~/src/hosaka
.venv-dev/bin/ruff check --fix && .venv-dev/bin/ruff format
git add hosaka/server/app.py hosaka/server/engines/base.py tests/test_server.py
git commit -m "fix(server): guard voices+resolve for absent kokoro/chatterbox"
git push origin main
```

---

### Task 2: hosaka piper-only composition root

A new module that builds the app with only Piper, importing no torch. It cannot reuse `main.py`'s `_make_piper` verbatim because that gates on `PIPER_PYTHON.exists()` (the `.venv-piper`), which is absent in the container — the container's own interpreter is both server and sidecar.

**Files:**
- Create: `hosaka/server/main_piper.py`
- Test: `tests/test_main_piper.py`

**Interfaces:**
- Consumes: `create_app` (from Task 1's guarded `app.py`), `EngineRegistry`, `PiperEngine`, `VoiceLibrary`, config `PIPER_VOICES` / `PIPER_SIDECAR` / `VOICE_DIR`.
- Produces: module-level `app` (a `FastAPI`) and `build_piper_registry() -> EngineRegistry`. `build_piper_registry` builds a `PiperEngine` whose sidecar command is `[sys.executable, str(PIPER_SIDECAR), "--voice", f"{vid}={model}"...]` for every `PIPER_VOICES` entry whose model file exists; if none exist, `piper=None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_main_piper.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv-dev/bin/python -m pytest tests/test_main_piper.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hosaka.server.main_piper'`.

- [ ] **Step 3: Write `main_piper.py`**

Create `hosaka/server/main_piper.py`:

```python
"""Piper-only composition root for the always-on droplet container.

Builds the hosaka server with kokoro/chatterbox/rvc absent, so this module
imports NO torch / Kokoro / Chatterbox / RVC -- only the CPU Piper engine.
The container has no .venv-piper: its own interpreter (sys.executable) runs
both the server and the Piper sidecar, talking over the piper_proto pipe.

Run with:  uvicorn hosaka.server.main_piper:app --host 0.0.0.0 --port 8123
"""

import sys

from hosaka.config import PIPER_SIDECAR, PIPER_VOICES, VOICE_DIR
from hosaka.library import VoiceLibrary
from hosaka.server.app import create_app
from hosaka.server.engines.base import EngineRegistry
from hosaka.server.engines.piper_engine import PiperEngine


def build_piper_registry() -> EngineRegistry:
    """Build a registry with only Piper. Missing every model file -> piper=None
    (server still boots, just advertises no voices). The sidecar runs under the
    current interpreter; there is no separate .venv-piper in the container."""
    available = {vid: spec for vid, spec in PIPER_VOICES.items() if spec["model"].exists()}
    piper = None
    if available:
        cmd = [sys.executable, str(PIPER_SIDECAR)]
        for vid, spec in available.items():
            cmd += ["--voice", f"{vid}={spec['model']}"]
        piper = PiperEngine(cmd, voices=list(available))
    return EngineRegistry(kokoro=None, chatterbox=None, piper=piper, rvc=None)


_library = VoiceLibrary(VOICE_DIR)
app = create_app(build_piper_registry(), _library, do_warmup=False)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv-dev/bin/python -m pytest tests/test_main_piper.py -v`
Expected: PASS. (If `test_main_piper_imports_without_torch_and_serves_only_piper` reports `torch` in `sys.modules`, an import leaked — trace it; `create_app`/`PiperEngine`/`VoiceLibrary` must all be torch-free.)

Note: `VoiceLibrary(VOICE_DIR)` runs against a possibly-missing dir. If this test errors on a missing voices dir, `VoiceLibrary` does not tolerate absence — in that case the container must `mkdir -p` the dir (handled in Task 3); confirm the test still constructs the app (point `VOICE_DIR` via a tmp HOME if needed) and note it for Task 3.

- [ ] **Step 5: Commit + push**

```bash
cd ~/src/hosaka
.venv-dev/bin/ruff check --fix && .venv-dev/bin/ruff format
git add hosaka/server/main_piper.py tests/test_main_piper.py
git commit -m "feat(server): piper-only composition root for droplet"
git push origin main
```

---

### Task 3: hosaka Dockerfile.piper (no torch, baked glados)

Package the piper-only server as an image: CPU deps only, glados `.onnx` baked in, `HOME` set so `config.DATA_DIR` resolves to the baked path.

**Files:**
- Create: `Dockerfile.piper` (hosaka repo root)
- Create: `.dockerignore` (hosaka repo root) — keep the venvs and data out of the build context
- Modify: `CLAUDE.md` / `ARCHITECTURE.md` (note the droplet piper-only deploy)

**Interfaces:**
- Consumes: `hosaka/server/main_piper.py` (Task 2), `scripts/fetch_glados_model.sh`.
- Produces: an image tagged `hosaka-piper:latest` that serves `/v1/voices` (lists `glados`) and `/v1/audio/stream` on port 8123.

- [ ] **Step 1: Write `.dockerignore`**

Create `.dockerignore` in the hosaka repo root:

```
.venv
.venv-*
.git
__pycache__
**/__pycache__
*.pyc
graphify-out
docs
tests
.pytest_cache
.ruff_cache
```

- [ ] **Step 2: Determine the piper-only dependency pins**

Run, to read the versions the working `.venv-piper` uses:

```bash
cd ~/src/hosaka
.venv-piper/bin/python -m pip freeze | grep -iE "piper-tts|onnxruntime|scipy|numpy|piper-phonemize"
```

Plus the pure server deps (no torch): `fastapi uvicorn pydantic`. Use these exact versions in the next step (record them inline, no "TBD").

- [ ] **Step 3: Write `Dockerfile.piper`**

Create `Dockerfile.piper` in the hosaka repo root (fill the pins from Step 2):

```dockerfile
FROM python:3.12-slim

# Piper needs espeak-ng data + libstdc++ at runtime; curl to fetch the model.
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# config.DATA_DIR = Path.home()/.local/share/hosaka. Set HOME so the baked
# model and the (empty) voices/lexicon dirs resolve inside the image.
ENV HOME=/opt/hosaka
WORKDIR /app

# CPU-only deps. NO torch, NO cu128 index. Pin to match the working .venv-piper.
RUN pip install --no-cache-dir \
        fastapi uvicorn pydantic \
        piper-tts==<PIN> onnxruntime==<PIN> scipy==<PIN> numpy==<PIN>

# Bake the GLaDOS model (the fetch script writes to $HOME/.local/share/hosaka/piper).
COPY scripts/fetch_glados_model.sh /tmp/fetch_glados_model.sh
RUN bash /tmp/fetch_glados_model.sh

# App source.
COPY hosaka /app/hosaka

# Empty dirs VoiceLibrary / Lexicon resolve against (no personal data on droplet).
RUN mkdir -p $HOME/.local/share/hosaka/voices

EXPOSE 8123
CMD ["uvicorn", "hosaka.server.main_piper:app", "--host", "0.0.0.0", "--port", "8123"]
```

- [ ] **Step 4: Build the image**

Run: `cd ~/src/hosaka && docker build -f Dockerfile.piper -t hosaka-piper:latest .`
Expected: build succeeds; the `fetch_glados_model.sh` layer prints `fetch glados/glados_piper_medium.onnx` (and `.onnx.json`).

- [ ] **Step 5: Smoke-test the container**

Run:
```bash
docker run --rm -d --name hp -p 8124:8123 hosaka-piper:latest
sleep 4
curl -s localhost:8124/v1/voices
docker rm -f hp
```
Expected: JSON listing exactly one voice, `{"id":"glados","backend":"piper",...}`. If `[]`, the model path didn't resolve — check `HOME` vs `fetch_glados_model.sh`'s `ROOT` and `config.PIPER_DIR`.

- [ ] **Step 6: Document + commit + push**

Add a short note to `ARCHITECTURE.md` (Remote/web access section) and `CLAUDE.md`: glados is also served by a piper-only `hosaka-piper` container on the droplet (built from `Dockerfile.piper`, no torch, glados baked in); the home box is unchanged.

```bash
cd ~/src/hosaka
git add Dockerfile.piper .dockerignore ARCHITECTURE.md CLAUDE.md
git commit -m "build(piper): droplet hosaka-piper image, glados baked, no torch"
git push origin main
```

---

### Task 4: exec-fn route-by-voice proxy

Make `routes_tts.py` hold two upstreams and route each utterance by backend: glados -> `hosaka-piper`, everything else -> the home tunnel. Merge voices so glados is always listed; report health as either-up.

**Files:**
- Modify: `~/src/exec-fn/api/routes_tts.py`
- Test: `~/src/exec-fn/tests/test_tts_routing.py` (new; pure unit tests, no running app)

**Interfaces:**
- Consumes: env `TTS_UPSTREAM` (home, default `172.17.0.1:8123`), new env `TTS_PIPER_UPSTREAM` (default `hosaka-piper:8123`).
- Produces: pure helper `_upstream_for(req: dict) -> str` returning the piper upstream when `req.get("backend") == "piper"`, else the home upstream; `/api/hosaka/voices` merge (piper voices always, home voices when reachable); `/api/hosaka/health` returning `{"ok": bool, "home": bool, "piper": bool}`.

- [ ] **Step 1: Write the failing unit tests**

Create `~/src/exec-fn/tests/test_tts_routing.py`:

```python
import importlib
import os

os.environ.setdefault("TTS_UPSTREAM", "home:8123")
os.environ.setdefault("TTS_PIPER_UPSTREAM", "piper:8123")

routes_tts = importlib.import_module("routes_tts")


def test_upstream_for_piper_goes_to_piper():
    assert routes_tts._upstream_for({"backend": "piper", "voice": "glados"}) == "piper:8123"


def test_upstream_for_kokoro_goes_home():
    assert routes_tts._upstream_for({"backend": "kokoro", "voice": "nicole"}) == "home:8123"


def test_upstream_for_missing_backend_defaults_home():
    assert routes_tts._upstream_for({"voice": "nicole"}) == "home:8123"
```

(`tests/conftest.py` already puts `api/` on the path for `import routes_tts`; if not, add `sys.path.insert(0, ".../api")` in the test.)

- [ ] **Step 2: Run to verify failure**

Run: `cd ~/src/exec-fn && python -m pytest tests/test_tts_routing.py -v`
Expected: FAIL — `_upstream_for` does not exist yet.

- [ ] **Step 3: Add the dual upstream + helper**

In `~/src/exec-fn/api/routes_tts.py`, below the existing `_UPSTREAM`:

```python
_UPSTREAM = os.environ.get("TTS_UPSTREAM", "172.17.0.1:8123")
# Always-on droplet-local piper (glados). Separate from the home GPU tunnel.
_PIPER_UPSTREAM = os.environ.get("TTS_PIPER_UPSTREAM", "hosaka-piper:8123")


def _upstream_for(req: dict) -> str:
    """Route an utterance to its backend's upstream. Glados (piper) is served
    by the always-on droplet container; every other backend goes to the home
    GPU box over the SSH tunnel."""
    return _PIPER_UPSTREAM if req.get("backend") == "piper" else _UPSTREAM
```

- [ ] **Step 4: Run to verify the helper tests pass**

Run: `cd ~/src/exec-fn && python -m pytest tests/test_tts_routing.py -v`
Expected: PASS.

- [ ] **Step 5: Merge voices (glados always present)**

Replace `tts_voices` in `routes_tts.py`:

```python
async def _get_voices(upstream: str):
    async with httpx.AsyncClient(timeout=5) as client:
        r = await client.get(f"http://{upstream}/v1/voices")
        return r.json()


@guest_protected.get("/api/hosaka/voices")
async def tts_voices():
    out = []
    # Piper upstream is always-on; keep only its piper voices (glados).
    try:
        out += [v for v in await _get_voices(_PIPER_UPSTREAM) if v.get("backend") == "piper"]
    except Exception:
        pass
    # Home box: contribute its (non-piper) voices when reachable.
    try:
        out += [v for v in await _get_voices(_UPSTREAM) if v.get("backend") != "piper"]
    except Exception:
        pass
    return JSONResponse(out)
```

- [ ] **Step 6: Per-upstream health**

Replace `tts_health` in `routes_tts.py`:

```python
@guest_protected.get("/api/hosaka/health")
async def tts_health():
    """ok if EITHER upstream answers. Glados alone (home box down) is still ok;
    the UI greys out GPU voices but keeps glados live. A bound tunnel port is
    not liveness -- only an actual /v1/voices response counts."""
    async def live(upstream: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(f"http://{upstream}/v1/voices")
                r.raise_for_status()
                return True
        except Exception:
            return False

    home, piper = await asyncio.gather(live(_UPSTREAM), live(_PIPER_UPSTREAM))
    ok = home or piper
    return JSONResponse({"ok": ok, "home": home, "piper": piper}, status_code=200 if ok else 503)
```

- [ ] **Step 7: Route the WS per utterance**

Rewrite `ws_tts` so it lazily opens (and caches) one upstream connection per backend and dispatches each utterance. Keep the cookie gate unchanged. Replace the body after `await ws.accept()`:

```python
@public.websocket("/ws/hosaka")
async def ws_tts(ws: WebSocket):
    if (
        ws.cookies.get("session") != SESSION_TOKEN
        and ws.cookies.get("guest_session") != GUEST_SESSION_TOKEN
    ):
        await ws.close(code=1008)
        return
    await ws.accept()

    conns: dict[str, object] = {}   # upstream url -> open websocket
    pumps: list = []                # upstream->client pump tasks

    async def upstream_for_url(url):
        if url not in conns:
            up = await websockets.connect(f"ws://{url}/v1/audio/stream", max_size=None)
            conns[url] = up
            pumps.append(asyncio.create_task(_pump_to_client(ws, up)))
        return conns[url]

    try:
        while True:
            m = await ws.receive()
            if m["type"] == "websocket.disconnect":
                break
            if m.get("text") is None:
                continue  # the audio protocol is client->server JSON utterances only
            try:
                req = json.loads(m["text"])
            except Exception:
                await ws.send_text(json.dumps({"type": "error", "detail": "bad request json"}))
                continue
            url = _upstream_for(req if isinstance(req, dict) else {})
            try:
                up = await upstream_for_url(url)
            except Exception:
                await ws.send_text(json.dumps({"type": "error", "detail": "tts upstream unreachable"}))
                continue
            await up.send(m["text"])
    except Exception:
        pass
    finally:
        for t in pumps:
            t.cancel()
        for up in conns.values():
            try:
                await up.close()
            except Exception:
                pass
        try:
            await ws.close()
        except Exception:
            pass
```

Keep `_pump_to_client` as-is. `_pump_to_upstream` is no longer used (the loop forwards text directly so it can route per utterance) — delete it.

- [ ] **Step 8: Run the routing unit tests + exec-fn suite**

Run: `cd ~/src/exec-fn && python -m pytest tests/test_tts_routing.py -v && python -m pytest -q`
Expected: routing tests PASS; the rest of the suite is unaffected (integration tests that need a running app skip as before).

- [ ] **Step 9: Lint + commit + push (exec-fn)**

```bash
cd ~/src/exec-fn
ruff check --fix . && ruff format .
git add api/routes_tts.py tests/test_tts_routing.py
git commit -m "feat(tts): route glados to always-on piper, gpu to home tunnel"
git push
```

---

### Task 5: exec-fn UI gate (grey GPU voices when home down)

The UI must keep glados selectable while the home box is down and grey out the GPU voices, driven by the per-upstream health.

**Files:**
- Modify: `~/src/exec-fn/web/tts.js` (`checkHealth` ~132-142, `loadVoices` ~144-185)
- Verify: `~/src/exec-fn/api/templates/tts.html` (no change expected; confirm the voice `<select>` is `#tts-voice`)

**Interfaces:**
- Consumes: `/api/hosaka/health` -> `{ok, home, piper}` (Task 4); `/api/hosaka/voices` (merged).
- Produces: GPU-backend `<option>`s disabled when `home` is false; glados always enabled; the Speak button enabled whenever the *selected* voice's upstream is live.

- [ ] **Step 1: Track per-upstream health**

In `web/tts.js`, add a module-level state near the other `let` declarations and update `checkHealth`:

```javascript
let health = { ok: false, home: false, piper: false };

async function checkHealth() {
  if (speaking) return;
  try {
    health = await (await fetch("/api/hosaka/health")).json();
  } catch {
    health = { ok: false, home: false, piper: false };
  }
  applyHealth();
}
```

- [ ] **Step 2: Gate voices + Speak button on health**

Add `applyHealth` and call it from the voice-select change handler. GPU backends are `kokoro/chatterbox/rvc`; `piper` follows the piper upstream:

```javascript
const PIPER_BACKENDS = new Set(["piper"]);

function backendLive(backend) {
  return PIPER_BACKENDS.has(backend) ? health.piper : health.home;
}

function applyHealth() {
  const sel = $("tts-voice");
  for (const o of sel.options) {
    o.disabled = !backendLive(o.dataset.backend);
  }
  // If the selected voice's upstream just went down, move to a live one.
  const cur = sel.selectedOptions[0];
  if (cur && cur.disabled) {
    const live = [...sel.options].find((o) => !o.disabled);
    if (live) live.selected = true;
  }
  const liveSel = sel.selectedOptions[0];
  if (liveSel && backendLive(liveSel.dataset.backend)) {
    setBtn("Speak", true);
    setStatus(health.home ? "" : "home box offline -- glados only");
  } else {
    setBtn("offline", false);
    setStatus(OFFLINE);
  }
  if (typeof reflectBackend === "function") reflectBackend();
}
```

- [ ] **Step 3: Re-apply health after voices (re)load**

At the end of `loadVoices`, after `reflectBackend();`, add `applyHealth();` so a fresh voice list is gated immediately. Also call `applyHealth` from the `sel.addEventListener("change", ...)` path (wrap the existing `reflectBackend` listener):

```javascript
  sel.addEventListener("change", () => { reflectBackend(); applyHealth(); });
  reflectBackend();
  applyHealth();
```

- [ ] **Step 4: Manual verification (no JS test harness in exec-fn)**

With `hosaka-piper` up and the home box reachable: load `/hosaka`, confirm all backends selectable, Speak enabled. Stop the home tunnel (or point `TTS_UPSTREAM` at a dead port): refresh, confirm kokoro/chatterbox/rvc options are greyed/disabled, glados stays selectable, Speak stays enabled, status reads "home box offline -- glados only", and speaking glados produces audio. Restore home, confirm all re-enable within one health poll.

- [ ] **Step 5: Lint + commit + push (exec-fn)**

```bash
cd ~/src/exec-fn
ruff format . ; npx --yes prettier -w web/tts.js 2>/dev/null || true
git add web/tts.js api/templates/tts.html
git commit -m "feat(tts-ui): keep glados live + grey gpu voices when home down"
git push
```

---

### Task 6: Deploy hosaka-piper on the droplet + end-to-end verify

Wire the image into the droplet's compose and prove glados works with the home box off.

**Files:**
- Modify: `~/src/exec-fn/docker-compose.yml` (add the `hosaka-piper` service)

**Interfaces:**
- Consumes: the `hosaka-piper:latest` image (Task 3), built on the droplet from a clone of the hosaka repo.
- Produces: a running `hosaka-piper` service on the compose network, reachable by the `api` container as `hosaka-piper:8123`; `wai-lau.net/hosaka` serves glados with the home box down.

- [ ] **Step 1: Add the compose service**

In `~/src/exec-fn/docker-compose.yml`, add under `services:` (image-only — the image is built separately from the hosaka repo, since hosaka is not in exec-fn's build context):

```yaml
  hosaka-piper:
    image: hosaka-piper:latest
    restart: unless-stopped
    # No ports: -- only the api container reaches it over the compose network.
```

The `api` service reaches it by service name; `_PIPER_UPSTREAM` defaults to `hosaka-piper:8123`. (Default compose network resolves service names; no extra config needed. If `api` runs on a non-default network, attach `hosaka-piper` to the same one.)

- [ ] **Step 2: Build the image on the droplet**

On the droplet (where exec-fn is deployed), clone/pull the hosaka repo and build:

```bash
git -C ~/src/hosaka pull   # or clone it first
docker build -f ~/src/hosaka/Dockerfile.piper -t hosaka-piper:latest ~/src/hosaka
```

- [ ] **Step 3: Bring it up**

Run: `cd ~/src/exec-fn && docker compose up -d hosaka-piper && docker compose up -d api`
Expected: both services running (`docker compose ps`). From the api container: `docker compose exec api curl -s hosaka-piper:8123/v1/voices` lists glados.

- [ ] **Step 4: End-to-end, home box OFF**

With the home box / SSH tunnel down: open `https://wai-lau.net/hosaka`. Expected: page loads, glados selectable (GPU voices greyed), status "home box offline -- glados only", SPEAK on glados plays GLaDOS audio. This is the whole point of the feature — verify it explicitly.

- [ ] **Step 5: End-to-end, home box ON**

Bring the home box up. Within one health poll: all four backends selectable; kokoro/chatterbox/rvc still route to the home box (unchanged), glados still routes to the droplet container. Confirm one GPU voice and glados both speak.

- [ ] **Step 6: Commit + push (exec-fn)**

```bash
cd ~/src/exec-fn
git add docker-compose.yml
git commit -m "deploy(tts): run always-on hosaka-piper service for glados"
git push
```

- [ ] **Step 7: Rebuild the hosaka knowledge graph**

Per repo convention, a shipped structural change rebuilds the graph:

```bash
cd ~/src/hosaka   # invoke /graphify (AST/code-only, git-tracked files)
```

---

## Self-Review

**Spec coverage:**
- Decision 1 (remote glados always droplet) -> Task 4 `_upstream_for` + Task 6 deploy. Covered.
- Decision 2 (local REPL/home unchanged) -> enforced by Global Constraints (home box out of scope); no task touches it. Covered.
- Decision 3 (full hosaka-server piper-only) -> Tasks 1-2. Covered.
- Decision 4 (container) -> Task 3 + Task 6. Covered.
- Decision 5 (Dockerfile.piper in hosaka) -> Task 3. Covered.
- Decision 6 (glados baked) -> Task 3 Step 3/5. Covered.
- 3 app.py guards -> Task 1 (voices kokoro/chatterbox guards + `_resolve` None guard). Covered.
- `_make_piper` uses sys.executable, not `.venv-piper` -> Task 2 `build_piper_registry`. Covered.
- exec-fn dual-upstream route + voices merge + either-up health -> Task 4. Covered.
- UI gate -> Task 5. Covered.
- Tests (hosaka piper-only registry; exec-fn routing; e2e home-off) -> Tasks 1,2,4,6. Covered.

**Placeholder scan:** The only intentional placeholders are `<PIN>` in `Dockerfile.piper`, resolved by Task 3 Step 2 (read from the live `.venv-piper`) before writing Step 3 — flagged, not silent.

**Type consistency:** `build_piper_registry` (Task 2) returns `EngineRegistry`; `_upstream_for(req: dict) -> str` used identically in Task 4 tests and impl; health shape `{ok, home, piper}` produced in Task 4 Step 6 and consumed in Task 5 Step 1/2; `PiperEngine._cmd[0]` asserted in Task 2 matches `piper_engine.py`'s `self._cmd = list(sidecar_cmd)`.
