# Charlie RVC voice — handoff

Status as of 2026-06-25. All work is committed to `main` and pushed; the voice is
live on the persistent server. This doc is the pick-up point for the Charlie /
RVC voice-conversion work.

## TL;DR

- Added **RVC voice conversion** as a 5th engine (mirrors the Piper sidecar:
  `rvc_proto` wire protocol + `rvc_engine` client + `rvc_sidecar` under a new
  `.venv-rvc`). Branch was merged to `main` (squash-free, ~20 commits) and the
  feature is live.
- The shipped voice **`charlie`** is a **hybrid**, not plain RVC:
  `Chatterbox clone (emotion) -> RVC erika (timbre) -> silence gate -> tempo
  stretch`. Plain Kokoro->RVC sounded emotionless (Kokoro is flat); cloning a
  real Charlie clip with Chatterbox supplies the emotion, RVC nails the timbre.
- Final offered voice list (curated by request): **`nicole`** (kokoro),
  **`glados`** (piper), **`charlie`** (rvc). Everything else removed/hidden.

## The charlie pipeline (per request)

1. `RvcEngine` generates the **source** with **Chatterbox**, cloning the
   `charlie_cb` library clip (NOT Kokoro — that's the realtime/flat path).
2. The whole fragment's PCM is piped to the `.venv-rvc` GPU sidecar.
3. Sidecar: HuBERT/ContentVec -> rmvpe F0 -> faiss retrieval -> net_g (erika, 32k)
   -> resample 32k->24k -> **silence gate** (mute where the source is silent;
   RVC hallucinates phonemes in gaps) -> **tempo time-stretch** (speed, keeps
   pitch) -> framed PCM back.
4. Engine yields converted PCM. GPU serialization unchanged (one held slot,
   both GPU ops sequential).

Quality path: ~4-5s to first audio, ~10s for a long line. NOT realtime (correct
for a character voice).

## Where to tune charlie — `hosaka/config.py` `RVC_VOICES["charlie"]`

```python
"source_backend": "chatterbox",
"source": "charlie_cb",       # the library reference clip
"source_params": {"exaggeration": 0.4, "cfg_weight": 0.5, "temperature": 0.3},
"transpose": 0,               # clone is already at Charlie's pitch
"passes": 1,                  # 2+ flattens emotion
"gate": True,                 # mute RVC's silent-gap hallucinations
"speed": 1.1,                 # tempo stretch on the output (Chatterbox has no speed knob)
"model_sr": 32000,            # erika native; sidecar resamples to 24k
```
Global RVC knobs (`RVC_KNOBS`): `index_rate 0.3`, `protect 0.5` (both low/high to
cut gap hallucination), `f0_method rmvpe`, `rms_mix_rate 0.25`.

After editing config, **restart the unit** to load it:
`systemctl --user restart hosaka-server.service` (warmup ~100s).

## The Chatterbox knobs (what they do)

- `exaggeration` — emotional intensity (0.5 default; up = more dramatic).
- `cfg_weight` — guidance/pacing (lower = looser, slower, more expressive).
- `temperature` — randomness (lower = more consistent).
Recipe for emotion: exaggeration up + cfg_weight down. Charlie ended at a
restrained 0.4 / 0.5 / 0.3 by ear.

## Models + data (untracked, in `~/.local/share/hosaka/`)

- `rvc/charlie/charlie.pth` + `charlie.index` — the **erika** RVC model
  (Darkynauta/HazbinHotel "Charlie Morningstar - Erika Henningsen", the English
  VA). Fetched by `scripts/fetch_rvc_model.sh`.
- `rvc/hubert_base.pt` + `rvc/rmvpe.pt` — RVC base assets (lj1995).
- `voices/charlie_cb.wav` — the Chatterbox reference clip: a ~20s clean Charlie
  snippet (YouTube rip -> Audacity snip -> resampled 24k -> `library.add`). This
  is charlie's source; hidden from `/v1/voices` (it's an internal RVC source).

## Build / setup

`scripts/setup_rvc_venv.sh` builds `.venv-rvc`. The hard-won recipe:
- **python3.10 via uv** (NOT system 3.12 — fairseq 0.12.2 + its omegaconf/hydra
  pins break on 3.11+).
- torch 2.9.1+cu128 first (Blackwell sm_120).
- rvc-python `--no-deps`.
- legacy deps under `pip<24.1`, `setuptools<81` (pkg_resources, for pyworld).
- **fairseq from git** (`facebookresearch/fairseq.git`), `--no-build-isolation`
  (PyPI's 0.12.2 wheel imports the removed `torch._six`). Needs `python3.10-dev`
  headers — uv's standalone python ships them.
- pre-seed rvc-python's `base_model/` with symlinks to our hubert/rmvpe.
Gated by `scripts/verify_rvc.py` (real conversion, capability (12,0), warm RTF
~0.18). See `[[rvc-model-sourcing]]` memory for where good models live.

## Key files

| File | Role |
|------|------|
| `hosaka/server/engines/rvc_proto.py` | pipe protocol (JSON hdr + PCM; A/X/E frames) |
| `hosaka/server/engines/rvc_engine.py` | client; per-voice source engine (`sources` dict) |
| `hosaka/server/engines/rvc_sidecar.py` | GPU worker (.venv-rvc only); passes, gate, speed |
| `hosaka/config.py` | `RVC_VOICES`, `RVC_KNOBS`, `resolve_source`, `KOKORO_ALIASES` |
| `hosaka/server/main.py` | `_make_rvc(sources)` wiring (kokoro + chatterbox) |
| `hosaka/server/app.py` | `_resolve` rvc branch, `/v1/voices` (+ `cb` flag, hide rvc sources) |
| `scripts/{setup_rvc_venv,fetch_rvc_model,verify_rvc}` | venv / models / gate |

## Voice list / REPL details

- `/v1/voices` exposes a `cb` flag: true when generation runs through Chatterbox
  (chatterbox backend, or an rvc voice with a chatterbox source) -> the cb knobs
  apply. The REPL `:voices` prints 2 columns: name + `cb`/`non-cb`.
- `nicole` is a display alias for the `af_nicole` Kokoro embedding
  (`KOKORO_ALIASES`; `KokoroEngine` maps it before KPipeline).

## Known issues / follow-ups

1. **GPU slot wedge under sustained load.** A hung generation holds the
   `Semaphore(1)` and never releases, so later requests queue forever (the REPL
   uses `timeout=None` -> hangs; health/`:voices` still work). Seen after this
   session's heavy A/B marathon. Fix: `systemctl --user restart
   hosaka-server.service`. **Open follow-up:** add a per-generation timeout in
   `app.py` that kills + releases the slot and surfaces an error instead of a
   silent hang. (Offered, not yet done.)
2. **Charlie's cb knobs are fixed in config, not live-tunable.** Request params
   (REPL knobs) do NOT flow to the chatterbox source — `source_params` is fixed.
   So `:exaggeration` etc. don't change charlie despite the `cb` tag. To make
   them live: merge request cb-knobs over the config defaults in
   `RvcEngine._source_pcm`, and have the REPL load per-voice defaults on `:voice`.
3. **Non-deterministic.** Chatterbox has temperature -> each charlie generation
   differs slightly. Expected.

## Operate

- Live server: systemd user unit `hosaka-server.service`, `127.0.0.1:8123`,
  runs from this working tree (so a branch checkout changes the deployed code).
- REPL: `voice` alias (or `python -m hosaka.cli.repl`). `:voice charlie`, type to
  speak. `:voices` lists the three. `:help` for commands.
- Tests: `.venv-dev/bin/python -m pytest -m "not gpu"` (182 passing). GPU paths
  via `scripts/verify_rvc.py` + `scripts/smoke_server.sh`.
- Docs: `ARCHITECTURE.md` (5th engine, FIVE venvs, the hybrid flow), `CLAUDE.md`
  (hard rules incl. "always push after commit" for this prod repo).
