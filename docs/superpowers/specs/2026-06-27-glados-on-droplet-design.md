# GLaDOS on the droplet — always-on CPU voice

**Date:** 2026-06-27
**Status:** approved design, pre-plan
**Repos touched:** `hosaka` (server side), `exec-fn` (proxy + UI)

## Problem

The whole hosaka stack — including the CPU-only GLaDOS (Piper) voice — lives
on the home WSL GPU box. When the laptop is off, the home box is down, and the
exec-fn proxy (`wai-lau.net/hosaka`, always up on the droplet) can serve
nothing: GPU voices *and* GLaDOS go dark together.

GLaDOS is Piper: CPU-only, no torch, pretrained `.onnx`. It has no reason to
share the GPU box's uptime. Moving it onto the always-on droplet gives GLaDOS
100% uptime, independent of the laptop.

## Decisions (locked during brainstorming)

1. **Remote/web GLaDOS is always served from the droplet** — never tunnels to
   the home box. (One source of truth for the remote path.)
2. **Local REPL on the home box is unchanged** — the home server keeps building
   its own Piper; the local REPL gets GLaDOS locally as today. Two Pipers, both
   cheap (CPU, pretrained onnx). The droplet copy is the remote/uptime one.
3. **Droplet runs the full hosaka-server, piper-only**, not an embedded
   reimplementation. Reuses the `/v1/audio/stream` wire protocol, chunking,
   text normalization, and `/v1/voices` verbatim — zero protocol
   reimplementation in exec-fn.
4. **Run it as a container** — a new `hosaka-piper` service on the droplet,
   joined to exec-fn's docker network, not port-published.
5. **`Dockerfile.piper` lives in the hosaka repo** (hosaka stays
   self-deploying); the Docker build is the code-transit (no submodule / vendor
   / pip-from-git).
6. **GLaDOS `.onnx` baked into the image** (not host-mounted) — reproducible,
   droplet has no `~/.local/share/hosaka`.

## Architecture

```
            exec-fn api container (FastAPI, always up on droplet)
              routes_tts.py  --- route by voice ---+
                   |                                |
   glados (piper) -+                                +- kokoro/chatterbox/rvc
       |                                                     |
       v                                                     v
  hosaka-piper container              SSH reverse tunnel -> home GPU box hosaka
  (new; piper-only; NO torch)           (172.17.0.1:8123; down when laptop off)
  same docker network, port 8123
```

- `hosaka-piper` is the same hosaka codebase with a piper-only composition root.
  No torch, no Kokoro/Chatterbox/RVC. Always up with the droplet.
- exec-fn `routes_tts.py` stops being a blind byte-pump and **routes by voice**:
  glados -> `hosaka-piper`, every other backend -> the home tunnel (unchanged).
- Home box untouched.

## hosaka side

### New composition root: `hosaka/server/main_piper.py`

Mirror of `main.py` but builds only Piper — so the module never imports torch /
Kokoro / Chatterbox / RVC:

```python
_library = VoiceLibrary(VOICE_DIR)
_registry = EngineRegistry(
    kokoro=None, chatterbox=None,
    piper=_make_piper(), rvc=None,
)
app = create_app(_registry, _library, do_warmup=False)
```

- Reuses `_make_piper()` from `main.py` (lift it into a shared place, or import
  it — it has no torch dependency). `_make_piper` already degrades gracefully.
- `do_warmup=False`: `registry.warmup_all()` + `stop_llm()` are GPU/server
  concerns; the Piper sidecar warms on first use. (Confirm `warmup_all`
  tolerates None engines; guard if not.)
- On the droplet the `.venv-piper` isolation is moot (no torch to shield):
  `PIPER_PYTHON` / the sidecar interpreter is just the container's own python.
  `_make_piper` builds `cmd = [PIPER_PYTHON, PIPER_SIDECAR, --voice ...]`; in the
  container set `PIPER_PYTHON` to `sys.executable` (same interpreter runs server
  + sidecar). The pipe protocol is unchanged.

### Three guards in `app.py` (piper-only correctness)

The current server always has Kokoro + Chatterbox, so these paths assume they
exist. With `kokoro=None` / `chatterbox=None` they misbehave:

1. **`/v1/voices` (`app.py:243-264`)** lists `KOKORO_PRESETS` and
   `library.list()` chatterbox clips **unconditionally** — it reads config
   constants, not the engine, so a piper-only server would falsely advertise
   kokoro/chatterbox voices. Guard both blocks: only emit kokoro voices when
   `registry.kokoro is not None`, chatterbox voices when
   `registry.chatterbox is not None` (symmetry with the existing piper/rvc
   guards).
2. **`_resolve` (`app.py:74`)** does `engine = registry.get(backend)` and can
   return a `None` engine with no error (e.g. backend=kokoro, valid preset
   name, but engine is None) -> `_pcm_frames` then crashes on
   `None.stream(...)`. Add: if the resolved engine is `None`, return
   `(None, f"backend unavailable: {backend}")`.
3. These guards are **benign on the full home server** (engines are never None
   there), so they ship in one place and both deploys use them.

### `Dockerfile.piper` (in hosaka repo)

- base `python:3.12-slim`
- install piper-only deps only — NO cu128 index, NO torch:
  `piper-tts onnxruntime scipy fastapi uvicorn pydantic numpy` (final list
  derived from what `create_app` + `_make_piper` + the sidecar import; pin to
  match `.venv-piper`).
- `COPY` the hosaka source into the image (this build *is* the code transit).
- bake the GLaDOS model via `scripts/fetch_glados_model.sh` into the image, at a
  path `_make_piper` / `PIPER_VOICES` resolves (set the model dir via env so it
  points inside the image, not `~/.local/share/hosaka`).
- `CMD uvicorn hosaka.server.main_piper:app --host 0.0.0.0 --port 8123`

## exec-fn side

### `routes_tts.py` — dual upstream + route by voice

- Add `_PIPER_UPSTREAM = os.environ.get("TTS_PIPER_UPSTREAM",
  "hosaka-piper:8123")` alongside `_UPSTREAM` (home tunnel).
- **`/api/hosaka/voices`** — merge:
  - GET `hosaka-piper`'s `/v1/voices`, keep only `backend == "piper"` entries
    (belt-and-suspenders even with the app.py guards).
  - GET the home `/v1/voices` when reachable; on failure contribute nothing.
  - Always return at least the piper (glados) voices. Today this returns `[]`
    on home-down; the new floor is glados.
- **`/ws/hosaka`** — no longer a blind pump. The socket is persistent (one JSON
  utterance per message), so route **per utterance**:
  - Receive the first text frame, parse enough to read `backend` / `voice`.
  - `backend == "piper"` (glados) -> proxy that utterance to `_PIPER_UPSTREAM`.
  - else -> proxy to `_UPSTREAM` (home tunnel), exactly as today.
  - Implementation: lazily hold up to two upstream connections per client
    socket, dispatch each utterance to the matching one; a mixed-voice session
    still works. Keep the existing pump helpers, just parameterized by upstream.
  - If the chosen upstream is unreachable, send the existing
    `{"type":"error","detail":"tts upstream unreachable"}` and keep the client
    socket open for the next utterance.
- **`/api/hosaka/health`** — ok if **either** upstream answers. Glados alone =
  ok (the UI must not show fully-offline when only the home box is down). Report
  per-upstream liveness in the body so the UI can gate (see below).

### UI gate (`web/tts.js` + `api/templates/tts.html`)

First-class deliverable. The UI must reflect that glados is always available
while GPU voices follow the home box:

- Drive the voice picker off the merged `/api/hosaka/voices` (glados always
  present) and the per-upstream health from `/api/hosaka/health`.
- When the home box is down: **grey out / disable the GPU voices** (kokoro,
  chatterbox, rvc) with a clear "home offline" affordance; **leave glados
  enabled and selectable**.
- When the home box is up: all voices enabled, as today.
- The pre-SPEAK "offline" indicator becomes per-capability rather than global:
  glados never shows offline.

## Testing

- **hosaka (`.venv-dev`, `-m "not gpu"`):**
  - `main_piper` builds an app with `kokoro=None`/`chatterbox=None`/`rvc=None`
    and a real-or-fake Piper; `/v1/voices` returns ONLY piper voices; kokoro /
    chatterbox / rvc requests return a clean `backend unavailable` error, not a
    crash. Drive the piper sidecar via the existing fake-sidecar-over-a-pipe
    harness.
  - Regression: the full registry (all engines present) `/v1/voices` output is
    unchanged by the new guards.
- **exec-fn:** unit-test the route-by-voice split (glados -> piper upstream,
  others -> home upstream) against fake upstream WS servers; voices-merge with
  home up vs down; health = either-up. UI gate: verify GPU voices disabled when
  health reports home down, glados stays enabled.
- **End-to-end:** build `hosaka-piper`, bring it up on the docker network, hit
  `wai-lau.net/hosaka` with the home box **off**, confirm glados speaks; with
  the home box on, confirm all four still route correctly.

## Out of scope

- Any change to the home box server runtime, the GPU queue, or the local REPL.
- Putting any GPU voice on the droplet.
- Auth changes — the existing cookie/session gate on `/ws/hosaka` is reused
  unchanged for the glados path.

## Open items folded into the plan

- Confirm `EngineRegistry` dataclass accepts `None` for `kokoro`/`chatterbox`
  (it's a plain dataclass; type hints aren't enforced, but make the Optional
  explicit).
- Confirm `registry.warmup_all()` is not reached with `do_warmup=False` (it
  isn't) and otherwise tolerates None engines.
- Final piper-only dependency pin list for `Dockerfile.piper`.
