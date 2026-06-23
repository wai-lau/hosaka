# CLAUDE.md

Guidance for Claude Code working in the hosaka repo. See `ARCHITECTURE.md` for
the full design and `README.md` for install + usage.

## What this is

A local near-real-time TTS tool: a FastAPI server holding two models resident in
VRAM (Kokoro for realtime presets, Chatterbox for cloning), a REPL client that
streams PCM to `pacat`, and an isolated offline Parler "bake" CLI that designs
voices from text. Single user, English only, runs on a Blackwell RTX 5070 Ti
under WSL2.

## Environments — there are THREE venvs, use the right one

| Venv | Holds | Use for |
|------|-------|---------|
| `.venv-dev` | pytest, pydantic, numpy, fastapi, httpx (NO torch) | the fast non-GPU test suite |
| `.venv-server` | torch cu128 + Kokoro + Chatterbox + server deps | running the server, GPU tests |
| `.venv-bake` | torch cu128 + Parler (pinned old transformers) | the bake CLI only |

Run the non-GPU suite: `.venv-dev/bin/python -m pytest -m "not gpu"`
Run GPU tests:        `PYTHONPATH=$PWD .venv-server/bin/python -m pytest -m gpu`

GPU-touching code can only be imported in `.venv-server` (engines import
torch/kokoro/chatterbox). Pure-logic modules (chunking, library, schemas,
replcmd, audio, server/app with a FakeEngine) run in `.venv-dev`.

## Hard rules (these were learned the hard way — do not break them)

- **Blackwell sm_120:** install `torch==2.9.1+cu128` from the cu128 index FIRST,
  then TTS packages `--no-deps`, then re-add non-torch deps. Never trust
  `torch.cuda.is_available()` — run `scripts/verify_gpu.py` (real matmul,
  capability `(12, 0)`). Recipes: `scripts/setup_server_venv.sh`,
  `scripts/setup_bake_venv.sh`.
- **Chatterbox needs `transformers==4.46.3`.** Newer (5.x) breaks its alignment
  hook (attention weights come back `None`). Do not bump it in `.venv-server`.
- **Parler needs `protobuf>=4`** (sentencepiece wants the `builder` module);
  keep it isolated in `.venv-bake` — never let its deps near the server.
- **Audio:** install only `pulseaudio-utils` (client). NEVER `apt install
  pulseaudio` (the daemon) — it breaks WSLg audio. Never install Linux GPU
  drivers or `cuda`/`cuda-drivers` meta-packages inside WSL. WSLg's RDP bridge
  adds static to in-WSL playback, so on WSLg audio plays on the Windows side
  (`ffplay.exe` via ffmpeg, or a buffered `SoundPlayer` fallback); `pacat` is the
  native-Linux path. See `make_player` in `hosaka/audio.py`.
- **Chatterbox is quality mode, ~4s to first audio — by design.** Measured RTF
  ~0.8 (faster than realtime; the card just can't stream it sub-second). It
  delivers each fragment **whole**; do not "optimize" it into per-chunk streaming
  *within* a fragment — that reintroduces stutter. Fast first audio comes from
  the fragment-cap **ramp** (`FIRST_FRAGMENT_MAX_CHARS` / `FRAGMENT_GROWTH` in
  `config.py`, applied in `_fragments_for`), tuned to stay gapless at RTF ~0.8;
  see ARCHITECTURE.md. Realtime is the Kokoro preset path. Don't bother with bf16
  (T3 is overhead-bound, no gain; full cast crashes the s3tokenizer FFT). A lone
  RTF ~2.0 reading means the GPU is degrading toward a crash — re-measure after a
  clean restart, don't "fix" the model.

## Conventions

- Audio is float32 LE PCM, 24 kHz, mono, everywhere. Resample at the engine
  boundary if a model differs.
- No Unicode emoji anywhere (code, comments, commits, docs). Plain text only.
- The server serializes GPU work with `asyncio.Semaphore(1)` behind a bounded
  FIFO wait queue (`_GpuQueue`, cap `MAX_QUEUE`): exactly one request touches the
  GPU at a time, the rest line up, and only a full queue returns `503`. The cap
  check + reserve (`try_admit`) must stay atomic (no `await` between them) and the
  executor future must be held until done. See `hosaka/server/app.py` — do not
  reintroduce an untracked `run_in_executor` future.
- Personal voice data lives in `~/.local/share/hosaka/voices` (untracked).
  Only the couple of sample seeds under `hosaka/sample_voices/` are committed.
- The server runs persistently as the systemd user unit `hosaka-server.service`
  (execs `scripts/start_server.sh`), linger-enabled so it starts on WSL boot.
  Bound to `127.0.0.1:8123`; the REPL also auto-spawns it if down. For remote
  use it is reverse-proxied by the exec-fn app — see ARCHITECTURE.md.

## Linting

`ruff` (lint + format) is configured in `pyproject.toml` and runs at commit via
`.pre-commit-config.yaml`. Before committing: `.venv-dev/bin/ruff check --fix`
and `.venv-dev/bin/ruff format`. Keep the suite green
(`.venv-dev/bin/python -m pytest -m "not gpu"`).

## Verify a change end-to-end

`bash scripts/smoke_server.sh` starts the real server, hits both backends, plays
through `pacat`, and shuts down. `scripts/benchmark_latency.py` gates the Kokoro
realtime path (<1s first chunk) and reports Chatterbox quality-mode timing.
