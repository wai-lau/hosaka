# CLAUDE.md

Guidance for Claude Code working in the hosaka repo. See `ARCHITECTURE.md` for
the full design and `README.md` for install + usage.

## What this is

A local near-real-time TTS tool: a FastAPI server holding two models resident in
VRAM (Kokoro for realtime presets, Chatterbox for cloning) plus a CPU Piper
sidecar for fixed character voices (GLaDOS), a GPU RVC sidecar for
voice-converted character voices (Charlie), a REPL client that streams PCM to
`pacat`, and an isolated offline Parler "bake" CLI that designs voices from text.
Single user, English only, runs on a Blackwell RTX 5070 Ti under WSL2.

## Environments — there are FIVE venvs, use the right one

| Venv | Holds | Use for |
|------|-------|---------|
| `.venv-dev` | pytest, pydantic, numpy, fastapi, httpx (NO torch) | the fast non-GPU test suite |
| `.venv-server` | torch cu128 + Kokoro + Chatterbox + server deps | running the server, GPU tests |
| `.venv-bake` | torch cu128 + Parler (pinned old transformers) | the bake CLI only |
| `.venv-piper` | piper-tts + onnxruntime + scipy (CPU, NO torch) | the Piper character-voice sidecar only |
| `.venv-rvc` | python3.10 (uv); torch cu128 + rvc-python + fairseq-git | the RVC voice-conversion sidecar only |

Run the non-GPU suite: `.venv-dev/bin/python -m pytest -m "not gpu"`
Run GPU tests:        `PYTHONPATH=$PWD .venv-server/bin/python -m pytest -m gpu`

GPU-touching code can only be imported in `.venv-server` (engines import
torch/kokoro/chatterbox). Pure-logic modules (chunking, library, schemas,
replcmd, audio, server/app with a FakeEngine) run in `.venv-dev`. The Piper
sidecar (`piper_sidecar.py`) imports piper/onnxruntime and runs ONLY in
`.venv-piper` — never import it from the server. Its client (`piper_engine.py`,
numpy + subprocess only) and the wire protocol (`piper_proto.py`) are pure and
tested in `.venv-dev` by driving a fake sidecar over a real pipe. The RVC
sidecar (`rvc_sidecar.py`) imports rvc-python/fairseq and runs ONLY in
`.venv-rvc`; its client (`rvc_engine.py`) and protocol (`rvc_proto.py`) are pure
and tested in `.venv-dev` the same way.

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
- **Piper runs CPU-only in `.venv-piper`, isolated by choice.** Its deps
  (onnxruntime + numpy) happen to match the server today, but the server venv
  must never import piper: it spawns `piper_sidecar.py` under `.venv-piper` and
  talks over the `piper_proto` pipe. Character voices are pretrained `.onnx`
  downloads (NO training here) — add one with a `PIPER_VOICES` entry +
  `scripts/fetch_glados_model.sh`. Setup: `scripts/setup_piper_venv.sh`.
  The GLaDOS Piper path is unchanged by the addition of RVC.
- **RVC requires Python 3.10 (not 3.12) in `.venv-rvc`.** fairseq 0.12.2 and its
  pinned omegaconf 2.0.6 / hydra-core 1.0.7 use the pre-3.11 mutable-default
  dataclass pattern that Python 3.11+ rejects. Python 3.10 is provided by `uv`
  as a prebuilt standalone (no sudo, no compile; headers included). fairseq must
  come from git (`git+https://github.com/facebookresearch/fairseq.git`), NOT the
  PyPI 0.12.2 wheel (it imports `torch._six`, removed in torch 2.0), built with
  `--no-build-isolation`; also requires `setuptools<81` (setuptools 81 removed
  `pkg_resources`) and `pip<24.1` for the legacy metadata. Full recipe:
  `scripts/setup_rvc_venv.sh`; gated by `scripts/verify_rvc.py` (real
  conversion, capability (12,0), warm RTF).
- **The server never imports rvc-python.** It spawns `rvc_sidecar.py` under
  `.venv-rvc` and talks over the `rvc_proto` pipe. rvc-python self-manages its
  HuBERT and rmvpe checkpoints in its own package directory; `setup_rvc_venv.sh`
  pre-seeds them via symlinks from `~/.local/share/hosaka/rvc/` so the
  auto-download never runs. The sidecar takes no `--hubert`/`--rmvpe` args.
- **RVC source is per-voice pluggable; emotion comes from the source.** RVC keeps
  the source's prosody/delivery and swaps only timbre, so a flat source yields a
  flat character. Kokoro is flat (realtime), so an expressive character sources
  from **Chatterbox cloning a real reference clip**: Charlie = Chatterbox clone of
  the `charlie_cb` library voice (exaggeration 0.4 / cfg_weight 0.5 / temperature
  0.3) -> RVC erika, plus per-voice `gate` + `passes` + `speed` (output tempo
  stretch via ffmpeg `atempo`/WSOLA -- librosa's phase vocoder colored/smeared
  the voice, swapped out by A/B; ffmpeg is a sidecar runtime dep). `speed` is
  request-overridable (live `:speed`): the request value wins over the voice's
  configured default (Charlie 1.3), which ships in `/v1/voices` as the voice's
  `speed` and the REPL preloads on `:voice` so it round-trips until tuned.
  A voice sets
  `source_backend` (`kokoro`|`chatterbox`) + `source` (+ `source_params`); the
  `sources` dict (in `_make_rvc`) maps backend -> engine. The `source_params` cb
  knobs are *defaults*, not fixed: `_source_pcm` merges a request's cb knobs
  (exaggeration/cfg_weight/temperature -- NOT speed, which is the RVC output
  stretch) over them, so `:exag` etc. tune Charlie live. `/v1/voices` ships each
  cb voice's defaults as `cb_params`; the REPL preloads them on `:voice` so the
  knobs round-trip the character until tuned. Kokoro sources still
  follow match-the-character -- `resolve_source` validates the
  `af_`/`am_`/`bf_`/`bm_` prefix (American/British only); a Chatterbox-clone
  source is just a library voice id. This is the quality path (~4-5s first audio),
  not realtime.
- **RVC silence gate.** RVC hallucinates phonemes in the source's silent gaps.
  Mitigated globally by `protect` 0.5 + `index_rate` 0.3, plus the sidecar's
  silence gate (the per-voice `gate` flag) which mutes the output wherever the
  source is silent.
- **RVC sidecar redirects stdout to stderr at startup.** rvc-python and fairseq
  print model-load messages to stdout, but the sidecar's stdout is the binary
  frame pipe. fd 1 is redirected to stderr before any imports so library chatter
  cannot corrupt the protocol. Do not remove this redirect.
- **Charlie's erika model is native 32 kHz; the sidecar resamples to 24k.** Do
  not change the model's synthesis sample rate. Adding an RVC character = drop
  model + index + one `RVC_VOICES` entry in `config.py` (data-only, like Piper).
- **Offered voices are curated to three:** `nicole` (kokoro -- a display alias
  for the `af_nicole` embedding via `KOKORO_ALIASES`; `KokoroEngine` maps it
  before KPipeline), `glados` (piper), `charlie` (rvc hybrid). `/v1/voices`
  carries a `cb` flag -- true when generation runs through Chatterbox (so the
  exaggeration/cfg_weight/temperature knobs apply) -- and the REPL `:voices`
  shows two columns: name + `cb`/`non-cb`. Library clips used only as an RVC
  source (e.g. `charlie_cb`) are hidden from the listing.
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
  reintroduce an untracked `run_in_executor` future. A per-generation watchdog
  (`GEN_TIMEOUT_S`) bounds each queue read: a generation that makes no progress
  for that long is presumed wedged on an uncancellable GPU call (it would hold
  the slot forever) and triggers `_do_shutdown()` so systemd respawns clean.
  Do not remove it — the silent slot wedge under sustained load is what it cures.
- Personal voice data lives in `~/.local/share/hosaka/voices` (untracked).
  Only the couple of sample seeds under `hosaka/sample_voices/` are committed.
- Piper character voices are pretrained `.onnx` models under
  `~/.local/share/hosaka/piper/<voice>/` (untracked); the sidecar resamples them
  22.05k->24k. Missing model/venv degrades gracefully (engine simply not built).
  Piper is CPU but for now still routes through the GPU admission queue.
- The server runs persistently as the systemd user unit `hosaka-server.service`
  (execs `scripts/start_server.sh`), linger-enabled so it starts on WSL boot.
  Bound to `127.0.0.1:8123`. The REPL defers to this unit when installed —
  waiting for it, or `systemctl --user start`ing it if down — rather than
  spawning a competing process on the port; it spawns its own server only as a
  fallback when no unit exists (see `_startup_action` in `repl.py`). A spawn
  that races the unit for the port orphans and squats it, breaking the unit.
  For remote use it is reverse-proxied by the exec-fn app — see ARCHITECTURE.md.

## Linting

`ruff` (lint + format) is configured in `pyproject.toml` and runs at commit via
`.pre-commit-config.yaml`. Before committing: `.venv-dev/bin/ruff check --fix`
and `.venv-dev/bin/ruff format`. Keep the suite green
(`.venv-dev/bin/python -m pytest -m "not gpu"`).

## Commits

This repo is prod: **always `git push origin main` immediately after every
commit**, so origin never lags the running/deployed code. (Overrides the default
"commit/push only when asked".)

## Verify a change end-to-end

`bash scripts/smoke_server.sh` starts the real server, hits all four backends
(Kokoro, Chatterbox, Piper/GLaDOS, RVC/Charlie), plays through `pacat`, and
shuts down.
`scripts/benchmark_latency.py` gates the Kokoro realtime path (<1s first chunk)
and reports Chatterbox quality-mode timing.
