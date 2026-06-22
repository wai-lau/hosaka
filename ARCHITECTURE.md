# hosaka Architecture

A local, near-real-time text-to-speech tool for a single user on a Blackwell
RTX 5070 Ti under WSL2. Type a line, hear it. Choose and tune the voice;
clone voices; design new voices from a text description.

## Shape

```
                    HTTP (raw PCM, 24kHz mono, float32 LE)
  REPL client  ───────────────────────────────────────────►  FastAPI server
  (hosaka.cli.repl)                                            (hosaka.server)
       │  ▲                                                      │
       │  │ pacat (WSLg PulseAudio)                              ├─ KokoroEngine   (presets, realtime)
       ▼  │                                                      └─ ChatterboxEngine (cloning, quality)
   speakers                                                          ▲
                                                                     │ reference WAV
  bake CLI (hosaka.cli.bake, isolated venv) ── Parler ── seed.wav ──►│
                                                          voice library
                                                   (~/.local/share/hosaka/voices)
```

Three deployable units, each with one clear responsibility:

1. **Server** (`hosaka/server/`) — holds both models resident in VRAM, exposes
   an OpenAI-shaped HTTP API, streams raw PCM.
2. **REPL client** (`hosaka/cli/repl.py`) — auto-spawns the server, reads lines,
   pipes streamed PCM into `pacat`.
3. **Bake CLI** (`hosaka/cli/bake.py`) — offline, isolated venv; turns a text
   voice description into a clone-able seed WAV.

## Two engines, one decision

No single open model does presets + cloning + tuning + style-prompt AND hits
<1s on this card. So the work is split:

| Engine | Role | Why | Latency on this box |
|--------|------|-----|---------------------|
| **Kokoro-82M** | presets + the realtime path | tiny, fast, ~28 English voices | ~60ms to first audio |
| **Chatterbox** (original, `davidbrowne17/chatterbox-streaming`) | cloning + tuning | best open zero-shot clone, keeps `exaggeration`/`cfg_weight`/`temperature` | ~2.3s (see below) |
| **Parler-TTS Mini** (offline) | style-prompt voice authoring | only model that designs a voice from words | irrelevant (offline) |

Both Kokoro and Chatterbox stay pinned in VRAM (~5-6 GB of 16).

### The bake-once idea

Text style-prompt models generate autoregressively — multi-second per utterance,
every time. So they cannot serve the live path. Instead the slow step runs
**once, offline**: `bake` synthesizes a seed WAV matching a description, which
joins the voice library and is then cloned live by Chatterbox. Per-utterance,
the path is always a fast engine.

### Realtime vs quality (the hard constraint)

Benchmarked on the 5070 Ti + WSL2:

- **Kokoro = realtime.** RTF ~0.04, ~60ms first chunk. The live path.
- **Chatterbox = quality mode, NOT realtime.** The model runs at RTF ~1.0 with a
  ~2s fixed per-call overhead, so per-chunk streaming underruns and stutters.
  `ChatterboxEngine.stream()` therefore generates each fragment **in full**, then
  hands back the whole waveform. The server's fragment loop overlaps a fragment's
  generation with the previous fragment's playback, so audio stays smooth at the
  cost of ~2-3s before the first cloned line is heard. Full knob set retained.

Fast realtime cloning (Chatterbox Turbo or XTTS-v2) is a deferred follow-up —
Turbo is not in the streaming fork, so it needs separate integration.

## Request path

`POST /v1/audio/speech` →

1. Resolve the backend (`400` if unknown).
2. Acquire the single-GPU slot. The busy check and acquire are atomic (no `await`
   between them), so a second concurrent request gets `503` rather than racing.
3. Split `input` into sentence fragments (`hosaka/chunking.py`). The first
   fragment is kept short — this is the real low-latency lever, more than any
   model's internal "streaming" flag.
4. For each fragment: run the blocking `engine.stream()` in a worker thread
   (`run_in_executor`), pushing PCM bytes (or an exception) into an
   `asyncio.Queue`; an async generator drains the queue into a `StreamingResponse`.
5. The GPU slot is held until the worker thread has actually finished (the future
   is `shield`-awaited), even on client disconnect — preserving the single-GPU
   serialization invariant and surfacing engine errors instead of returning a
   silently truncated `200`.

Other endpoints: `GET /health` (ready when both models are warmed),
`GET /v1/voices` (presets + library clips), `POST /shutdown` (clean `:quit --stop`).

## Data + audio

- **Audio contract:** raw PCM, 24 kHz, mono, float32 LE, everywhere. Engines
  resample to 24 kHz if a model differs.
- **Voice library** (`hosaka/library.py`): a directory of seed WAVs +
  `manifest.json` mapping voice-id → {path, source, params, created}. Lives in
  `~/.local/share/hosaka/voices` — outside the repo, so personal recordings never
  enter git. The repo ships a couple of Kokoro-rendered sample seeds.
- **Audio out** (`hosaka/audio.py`): `pacat --raw` from `pulseaudio-utils`,
  spawned once per REPL session. Uses the WSLg PulseAudio server directly,
  avoiding the fragile PortAudio/ALSA shim.

## VRAM lifecycle

On server start the lifespan hook best-effort `ollama stop gpt-oss:20b` to free
VRAM, then loads and warms both models (a tiny synth each) so the first real
request is not a cold start. Models stay pinned for the session.

## Why two venvs

`hosaka/server/` (`.venv-server`) and the bake CLI (`.venv-bake`) are separate
Python environments. Parler hard-pins an old `transformers`, and Chatterbox needs
`transformers==4.46.3`; keeping Parler isolated means its dependency constraints
can never poison the live server. The bake CLI writes a WAV to disk; the server
never imports Parler. See `scripts/setup_server_venv.sh` and
`scripts/setup_bake_venv.sh` for the exact, Blackwell-verified recipes.

## Module map

| Path | Responsibility |
|------|----------------|
| `hosaka/config.py` | constants: sample rate, ports, paths, defaults |
| `hosaka/chunking.py` | sentence-fragment splitter (low-latency lever) |
| `hosaka/library.py` | voice library + JSON manifest |
| `hosaka/schemas.py` | request/response models, param clamping |
| `hosaka/audio.py` | `pacat` streaming player |
| `hosaka/server/app.py` | FastAPI app: lifespan, routes, GPU serialization |
| `hosaka/server/main.py` | wires engines + library into the app (uvicorn entry) |
| `hosaka/server/engines/base.py` | `Engine` protocol + `EngineRegistry` |
| `hosaka/server/engines/kokoro_engine.py` | Kokoro presets engine |
| `hosaka/server/engines/chatterbox_engine.py` | Chatterbox cloning (quality mode) |
| `hosaka/cli/replcmd.py` | REPL colon-command parser |
| `hosaka/cli/repl.py` | REPL client (auto-spawn, stream, play) |
| `hosaka/cli/bake.py` | offline Parler voice-bake CLI |
| `scripts/` | venv setup, GPU verify, latency benchmark, e2e smoke |
