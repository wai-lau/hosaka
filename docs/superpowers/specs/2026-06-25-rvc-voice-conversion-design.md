# RVC Voice Conversion — Design

Date: 2026-06-25
Status: approved, pre-implementation

## Goal

Add **RVC (Retrieval-based Voice Conversion)** as a fifth engine: character
voices produced by converting a clean, neutral Kokoro base into a trained
target speaker. First target: **Charlie Morningstar** (Hazbin Hotel), RVC V2.

This is **additive**. The existing Piper GLaDOS path is not touched — it is
"basically perfect" and stays exactly as-is. RVC is a parallel character-voice
backend, not a replacement.

## Scope / non-goals

In scope:

- A new `rvc` backend selectable via the existing `SpeechRequest.backend`
  field, target in `voice` (exactly parallel to Piper).
- An isolated `.venv-rvc` GPU sidecar mirroring the Piper sidecar trio.
- Source-voice selection that matches the character's gender and accent.
- Fetch + setup + verify scripts following the project's established recipes.
- One working target end-to-end (Charlie), with a registry that makes adding
  more a data-only change.

Explicit non-goals (YAGNI / hard rules):

- **No RVC training.** Targets are pretrained downloads (same stance as Piper).
- **No output-filter-over-any-backend.** RVC is a backend, not a post-process
  layer over Kokoro/Chatterbox/Piper. (Considered and rejected: more API
  surface, touches every engine.)
- **No new REPL knobs / no `SpeechParams` change** for MVP. RVC tuning is fixed
  per-voice in config.
- **Do not modify** the Piper engine, the Piper GLaDOS voice, or `glados_fx.py`.
- **No new accents beyond Kokoro's.** Kokoro English is American (`a`) or
  British (`b`) only.

## Architecture

RVC becomes the 5th engine and the 2nd character-voice path. It runs as an
out-of-process **GPU sidecar** under a new `.venv-rvc`, mirroring Piper's
client/protocol/sidecar split. The server venv never imports RVC code.

Why a separate venv (consistent with the project's whole philosophy): RVC's
deps (an RVC inference lib, `fairseq` for the HuBERT/ContentVec encoder,
`faiss`, `pyworld`/`torchcrepe`) are a dependency minefield on Python 3.12 +
`torch==2.9.1+cu128`. Isolating them keeps the delicate
Kokoro/Chatterbox/`transformers==4.46.3` stack untouchable — the same reason
`.venv-piper` and `.venv-bake` exist.

### Data flow (one RVC request)

```
text
  -> lexicon.apply
  -> sentence split (plain split_fragments; the Chatterbox ramp does NOT apply)
  -> per fragment, inside ONE held GPU slot:
       Kokoro generates neutral SOURCE pcm   (server process, GPU)
         (source preset chosen by character gender+accent; see rule below)
       accumulate the whole fragment to one float32 24k array
       send {JSON header + length-prefixed source PCM} over the pipe
         -> rvc_sidecar (.venv-rvc process, GPU):
              HuBERT/ContentVec encode (internally @16k)
              F0 via rmvpe (+ per-voice transpose)
              optional faiss index retrieval blend (index_rate)
              net_g synth -> audio at the model's native SR (32k for Charlie)
              resample 32k -> 24k  (engine-boundary contract, like Piper 22.05->24)
              frame back: A/X/E tagged float32 24k PCM
       yield converted PCM
  [release slot]
```

The key property: `RvcEngine` wraps the existing Kokoro engine as its source
generator, so **`app.py`'s request path is unchanged** — it calls
`engine.stream()` like any other backend.

### Serialization + VRAM

Both GPU operations (Kokoro source gen in the server process, RVC conversion in
the sidecar process) run **sequentially inside one held GPU slot** per request
stream — `_pcm_frames` already holds the single slot for the whole stream and
releases it in `finally`. The two CUDA contexts coexist in VRAM but never fire
simultaneously, so the "exactly one request touches the GPU at a time"
invariant is preserved with zero changes to `_GpuQueue` / `_pcm_frames`.

VRAM budget: server (Kokoro + Chatterbox) ~5-6 GB + sidecar (HuBERT ~190 MB +
rmvpe ~180 MB + net_g ~100 MB) ~0.5-1 GB = ~7 GB of 16. Comfortable.

Disk: `.venv-rvc` is another cu128 torch env (~5-8 GB) on the WSL ext4 vhdx
(SATA SSD, 894 GB free). Models ~0.5-1 GB under the data dir. No concern.

## Components

New files (mirror the Piper trio's names and responsibilities):

| Path | Responsibility | Runs in |
|------|----------------|---------|
| `scripts/setup_rvc_venv.sh` | build `.venv-rvc`: Blackwell cu128 dance + RVC deps + verify | — |
| `scripts/verify_rvc.py` | real conversion smoke; assert capability `(12,0)`, non-silent output, print RTF | `.venv-rvc` |
| `scripts/fetch_rvc_model.sh` | fetch HuBERT base + rmvpe + Charlie `.zip` -> data dir; idempotent | — |
| `hosaka/server/engines/rvc_proto.py` | pipe wire protocol (request: JSON header + PCM block; response: `A`/`X`/`E` frames). Pure stdlib | every venv |
| `hosaka/server/engines/rvc_sidecar.py` | RVC synth worker. Imports torch/RVC/faiss. **Server never imports it** | `.venv-rvc` (GPU) |
| `hosaka/server/engines/rvc_engine.py` | client: Kokoro source + sidecar convert. numpy + subprocess only | server/dev venv |
| `tests/rvc_proto` + `rvc_engine` + `fake_rvc_sidecar` | proto unit tests; engine driven against a fake sidecar over a real pipe | `.venv-dev` |

Edited files:

| Path | Change |
|------|--------|
| `hosaka/config.py` | `RVC_PYTHON` / `RVC_SIDECAR` / `RVC_DIR` / `RVC_HUBERT` / `RVC_RMVPE`; `SOURCE_PRESETS`; `RVC_VOICES` (Charlie); fixed knob defaults |
| `hosaka/server/engines/base.py` | `EngineRegistry.rvc: Engine \| None`; `get()` + `warmup_all()` branches |
| `hosaka/server/main.py` | `_make_rvc()` graceful-degrade builder; construct Kokoro once, pass into both registry and `RvcEngine` |
| `hosaka/server/app.py` | `_resolve` rvc branch; `/v1/voices` rvc block |
| `ARCHITECTURE.md` | 5th engine row; RVC character-voice + sidecar section; FIVE venvs; module map; Shape diagram nodes |
| `CLAUDE.md` | FOUR -> FIVE venvs table; RVC hard rules (Blackwell pin, sidecar isolation, source-match rule, 32k->24k) |
| `README.md` | install: `setup_rvc_venv.sh` + `fetch_rvc_model.sh` |

## Source-voice selection rule

RVC converts *timbre*; it needs source audio. The source's **gender** and
**accent** must match the character (a male base through a female model, or a
British base through an American character, sounds wrong). The source must also
be **neutral-toned** — clean prosody, since the model supplies the character.

Each `RVC_VOICES` entry declares the character's `gender` + `accent`. A
`SOURCE_PRESETS` map resolves the neutral Kokoro base per tuple:

```python
SOURCE_PRESETS = {
    ("female", "american"): "af_sarah",   # clear + neutral
    ("male",   "american"): "am_michael",
    ("female", "british"):  "bf_emma",
    ("male",   "british"):  "bm_george",
}
```

The engine resolves `source = SOURCE_PRESETS[(gender, accent)]`. A configured
`source` that disagrees with the tuple is a config error — fail loud at build.

**"Download if missing" mechanism:** Kokoro voice embeddings auto-fetch from
`hexgrad/Kokoro-82M` via `KPipeline` on first use — there is no manual download
step, just reference the id. So an unseen `(gender, accent)` tuple is handled by
naming a Kokoro voice for it; it caches on first synth. The full Kokoro roster
(~54 voices) is available beyond the curated 8 presets. A source voice used only
internally by RVC does **not** need to be added to the public `KOKORO_PRESETS`
menu.

**Hard limit:** Kokoro English = American or British only. A character needing
an accent Kokoro lacks (Australian, Irish, ...) has no matching Kokoro base —
a real gap, not fixable by "download more presets." Flag it; do not fake it.

For MVP, only `(female, american) -> af_sarah` is verified by ear. The other
tuples ship as documented defaults, confirmed when a second character needs them.

## Registry shape (Charlie)

```python
RVC_DIR    = DATA_DIR / "rvc"
RVC_HUBERT = RVC_DIR / "hubert_base.pt"
RVC_RMVPE  = RVC_DIR / "rmvpe.pt"

RVC_VOICES = {
    "charlie": {
        "model":       RVC_DIR / "charlie" / "charlie.pth",
        "index":       RVC_DIR / "charlie" / "charlie.index",
        "model_sr":    32000,        # native; sidecar resamples -> 24000
        "gender":      "female",
        "accent":      "american",
        "source":      "af_sarah",   # must equal SOURCE_PRESETS[(female, american)]
        "transpose":   0,            # semitones; tune by ear (demo showed -16..+16)
        "description": "Charlie Morningstar (Hazbin Hotel), RVC V2",
    },
}
```

Fixed conversion knobs for MVP (module-level defaults, not per-request):
`f0_method="rmvpe"`, `index_rate=0.5`, `protect=0.33`, `rms_mix_rate=0.25`.
Adding a character = drop a model + one `RVC_VOICES` entry, no code change
(same ergonomics as `PIPER_VOICES`).

## Wire protocol (`rvc_proto.py`)

Pure stdlib so it imports in every venv. Reuses Piper's response framing; the
request additionally carries binary PCM.

- **Request** (server -> sidecar): one JSON header line (newline-terminated)
  `{voice, transpose, index_rate, f0_method, protect, rms_mix_rate}`, then a
  length-prefixed source PCM block `uint32be N + N bytes float32 LE @ 24k`.
- **Response** (sidecar -> server): tagged length-prefixed frames, identical
  contract to `piper_proto`:
  - `b'A' + uint32be N + N bytes` — converted PCM chunk (float32 LE @ 24k)
  - `b'X' + uint32be N + N bytes` — synthesis error (utf-8) -> raises, sidecar
    stays alive
  - `b'E'` — end of utterance (success)
  - stream closing before `E` -> `RvcProtocolError` (never a silent truncation)

## Sidecar (`rvc_sidecar.py`, `.venv-rvc`, GPU)

CLI: `--voice id=<pth>:<index> --hubert <path> --rmvpe <path>` (`--voice`
repeatable). Loads HuBERT/ContentVec, rmvpe, and each voice's net_g + faiss
index once, resident. Per request: parse header + read source PCM, run the RVC
inference pipeline, resample model SR -> 24k, frame back. A per-request failure
becomes an `X` frame; the sidecar survives for the next request (same contract
as the Piper sidecar). Never imported by the server.

Inference lib: primary candidate `rvc-python` (programmatic `RVCInference`-style
API). The exact import/call surface is pinned during setup; the sidecar is the
only module that touches it.

## Client engine (`rvc_engine.py`)

`RvcEngine(source_engine, sidecar_cmd, *, voices, hubert, rmvpe, ...)` —
numpy + subprocess only, so it is testable in `.venv-dev` against a fake
sidecar over a real pipe (exactly like `PiperEngine`).

- `voice_ids` — voices this sidecar serves.
- `stream(text, voice, params)`: resolve voice -> `(source_preset, transpose)`;
  run `source_engine.stream(text, source_preset, {"speed": params["speed"]})`
  to completion and concatenate to one float32 24k array; send header + PCM;
  yield converted arrays from the framed response.
- `warmup()`: touch each voice so its model is resident and the source path is
  warm.
- Spawn-once / respawn-on-broken-pipe / `atexit` close, mirroring `PiperEngine`.

`main.py` constructs `KokoroEngine()` once and passes the instance to both the
registry's `kokoro` slot and `RvcEngine` (RVC's source). `_make_rvc()` returns
`None` — graceful degrade — if `.venv-rvc`, HuBERT, rmvpe, or every voice's
model/index is missing; the server then runs the other four engines.

## Venv setup (the main risk)

`setup_rvc_venv.sh` follows the Blackwell hard-rule recipe:

1. Create `.venv-rvc` (python3.12).
2. Install `torch==2.9.1+cu128` from the cu128 index **first**.
3. Install the RVC inference lib + `faiss-cpu` + F0 deps, controlling torch so
   the lib cannot drag in a CPU/older torch (`--no-deps` then re-add non-torch
   deps, as the other venvs do).
4. Verify with `verify_rvc.py`: a real conversion of a short clip, asserting CUDA
   capability `(12,0)` via a real op (never trust `torch.cuda.is_available()`),
   non-silent output, and an RTF print.

This dependency pinning is the build's primary unknown and is nailed
**empirically on the box** — the same way every other venv here was built. The
architecture does not depend on the outcome.

`fetch_rvc_model.sh` (idempotent, mirrors `fetch_glados_model.sh`):

- HuBERT base + rmvpe from the canonical RVC assets repo
  (`lj1995/VoiceConversionWebUI` on HF) -> `RVC_DIR`.
- Charlie:
  `huggingface.co/ScruffyRVC/RFCRVCV2/resolve/main/charlie-morningstar-v3.zip`
  -> unzip into `RVC_DIR/charlie/`, place the `.pth` as `charlie.pth` and the
  `.index` as `charlie.index`.

## Tests (`.venv-dev`, no GPU)

- `rvc_proto`: frame pack/unpack, header+PCM request encode/parse, error frame
  raises, end marker, truncation -> `RvcProtocolError`.
- `rvc_engine`: drive `RvcEngine` against `fake_rvc_sidecar.py` over a real pipe
  with a fake source engine; assert the source is fully accumulated before send,
  the conversion round-trips, errors surface, and a broken pipe respawns.
- `fake_rvc_sidecar.py`: deterministic PCM transform, no real models.
- GPU-touching paths (real sidecar) are covered by `verify_rvc.py` +
  `smoke_server.sh`, not the dev suite.

## End-to-end verification

Extend `scripts/smoke_server.sh` to add a Charlie RVC request alongside the
existing Kokoro/Chatterbox/Piper checks: real server, convert a line, play
through the normal output path, confirm non-silent audio and clean shutdown.

## Risks

- **RVC dep pinning on py3.12 + cu128** (fairseq especially) — the one real
  unknown; contained in `setup_rvc_venv.sh`, gated by `verify_rvc.py`.
- Two CUDA contexts on one card — fine; sequential within a request, ~7 GB total.
- Source/character mismatch — prevented by the gender+accent rule + loud config
  validation.
```
