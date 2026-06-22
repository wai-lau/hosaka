# hosaka

Local near-real-time TTS. Kokoro presets + Chatterbox cloning live (<1s to first
audio target), Parler offline voice-baking. English only, single user, runs on a
Blackwell RTX 5070 Ti under WSL2.

See `docs/API.md` for the HTTP API. The design and build plan live in the
`local-llm-learning` repo under `docs/superpowers/`.

## Voice control

- **Presets** — pick a Kokoro voice (`:voice af_heart`). Fast path.
- **Cloning** — clone any reference clip (`:clone <id|path>`) via Chatterbox.
- **Tuning** — `:exag`, `:cfg`, `:temp` (Chatterbox), `:speed` (Kokoro). No pitch.
- **Style-prompt** — describe a voice in words and bake it once into a seed clip
  (`hosaka-bake`), then clone it live.

## Install (WSL2, Blackwell sm_120)

Hard rules:
- Install `torch` from the cu128 index FIRST, then TTS packages `--no-deps`.
  Never trust `torch.cuda.is_available()` alone — run `scripts/verify_gpu.py`.
- Install only `pulseaudio-utils` (client). NEVER `apt install pulseaudio`
  (the daemon) — it breaks WSLg audio.
- Never install Linux GPU drivers or `cuda`/`cuda-drivers` meta-packages in WSL.

### Server venv (Kokoro + Chatterbox)

```
cd ~/src/hosaka
python3.12 -m venv .venv-server
.venv-server/bin/pip install \
  torch==2.9.1+cu128 torchaudio==2.9.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
.venv-server/bin/pip install torchcodec
.venv-server/bin/pip install \
  "git+https://github.com/davidbrowne17/chatterbox-streaming.git" --no-deps
.venv-server/bin/pip install transformers accelerate scipy numpy peft soundfile
.venv-server/bin/pip install kokoro --no-deps
.venv-server/bin/pip install soundfile numpy fastapi uvicorn httpx pydantic
sudo apt install -y espeak-ng pulseaudio-utils
.venv-server/bin/python scripts/verify_gpu.py   # must print capability (12, 0) + matmul ok
```

### Bake venv (Parler, isolated)

```
cd ~/src/hosaka
python3.12 -m venv .venv-bake
.venv-bake/bin/pip install \
  torch==2.9.1+cu128 torchaudio==2.9.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
.venv-bake/bin/pip install parler-tts --no-deps
.venv-bake/bin/pip install transformers==4.46.1 descript-audio-codec soundfile numpy scipy sentencepiece
.venv-bake/bin/pip install "protobuf>=4.0.0,<5"   # sentencepiece needs >=3.20; default drags in 3.19.6
```

(Or just run `bash scripts/setup_bake_venv.sh`.)

## Use

```
.venv-server/bin/python -m hosaka.cli.repl
```

Type a line and press enter to hear it. The REPL auto-spawns the server (which
stops `gpt-oss:20b` first to free VRAM, then loads + warms both models).

Commands: `:voice <name>`, `:clone <id|path>`, `:backend kokoro|chatterbox`,
`:exag/:cfg/:temp/:speed <number>`, `:voices`, `:help`, `:quit` (or `:quit --stop`).

Bake a described voice (runs offline, in the bake venv):

```
.venv-bake/bin/python -m hosaka.cli.bake \
  "A calm, deep British male voice, very clear audio, moderate pace." \
  --out calm_brit
```

Then in the REPL: `:clone calm_brit`, speak a line.

## Realtime vs quality

Benchmarked on the RTX 5070 Ti (WSL2):

- **Kokoro presets = realtime.** RTF ~0.04, first audio well under 1s. This is
  the live path (`:voice <name>`).
- **Chatterbox cloning = quality mode (NOT realtime).** The model runs at
  RTF ~1.0 with a ~2s fixed per-call overhead, so it cannot stream smoothly
  under 1s on this card. Instead each fragment is generated in full and then
  played (smooth, no stutter), at the cost of ~2-3s before you hear a cloned
  line. It keeps the full knob set (`:exag`, `:cfg`, `:temp`). Fast realtime
  cloning (Chatterbox Turbo or XTTS-v2) is a deferred follow-up — Turbo is not
  in the `davidbrowne17` streaming fork, so it needs separate integration.

## Accepted constraints

- Cloning is ~2-3s, not realtime, on this hardware (see above).
- No native pitch control on any engine.
- Chatterbox output carries a non-removable Perth watermark. Local use only.
- Parler is the least-maintained piece; isolated in its own venv, offline only.
