# RVC Voice Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `rvc` backend that speaks a trained character voice (Charlie Morningstar) by converting a neutral Kokoro base through a Retrieval-based Voice Conversion model running in an isolated GPU sidecar.

**Architecture:** Mirror the Piper trio exactly — a pure-stdlib wire protocol (`rvc_proto`), an in-server client engine (`rvc_engine`) that wraps the existing Kokoro engine as its audio source, and an out-of-process GPU sidecar (`rvc_sidecar`) under a new `.venv-rvc`. The server venv never imports RVC code. The client accumulates a whole fragment of Kokoro PCM, ships it to the sidecar, and yields back the converted PCM — so the server's request path, GPU queue, and serialization are untouched.

**Tech Stack:** Python 3.12 (sidecar may fall back to 3.11), `torch==2.9.1+cu128`, `rvc-python` (HuBERT/ContentVec + rmvpe F0 + faiss retrieval), numpy, scipy, FastAPI. Source generation: Kokoro-82M.

## Global Constraints

Every task implicitly includes these (verbatim from the spec / CLAUDE.md):

- **Blackwell sm_120:** install `torch==2.9.1+cu128` from the cu128 index FIRST, then RVC packages, then re-add non-torch deps. Never trust `torch.cuda.is_available()` — verify with a real matmul and capability `(12, 0)`.
- **Server venv never imports RVC.** The sidecar runs only under `.venv-rvc`; the server spawns it and talks the `rvc_proto` pipe. `rvc_proto.py` and `rvc_engine.py` are pure (stdlib + numpy + subprocess) and test in `.venv-dev`.
- **Do not touch** the Piper engine, the Piper GLaDOS voice, `glados_fx.py`, or `transformers==4.46.3` in `.venv-server`.
- **Audio contract:** float32 LE PCM, 24 kHz, mono, everywhere. Charlie's model is native 32 kHz; the sidecar resamples 32k -> 24k at the boundary.
- **GPU serialization invariant unchanged:** both GPU ops (Kokoro source + RVC convert) run sequentially inside the one held slot per request. Do not add an untracked `run_in_executor` future or a second queue.
- **Source rule:** the source Kokoro preset must match the character's gender + accent; a mismatch is a loud config error. Kokoro English = American or British only.
- **No emoji** anywhere (code, comments, commits, docs). No per-request RVC knobs for MVP.
- **Before each commit:** `.venv-dev/bin/ruff check --fix` and `.venv-dev/bin/ruff format`; keep `.venv-dev/bin/python -m pytest -m "not gpu"` green. Commit messages get the standard `Co-Authored-By` / `Claude-Session` trailers (per CLAUDE.md); subjects shown below.

---

## File Structure

New:

| Path | Responsibility |
|------|----------------|
| `hosaka/server/engines/rvc_proto.py` | pipe wire protocol: request = JSON header + length-prefixed source PCM; response = `A`/`X`/`E` frames. Pure stdlib |
| `hosaka/server/engines/rvc_engine.py` | in-server client: accumulate Kokoro source, drive sidecar, yield converted PCM. numpy + subprocess only |
| `hosaka/server/engines/rvc_sidecar.py` | RVC synth worker (runs ONLY in `.venv-rvc`, GPU). Imports torch/rvc-python/faiss |
| `tests/test_rvc_proto.py` | protocol unit tests (`.venv-dev`) |
| `tests/test_rvc_engine.py` | engine tests driven against the fake sidecar over a real pipe (`.venv-dev`) |
| `tests/fake_rvc_sidecar.py` | fake sidecar: speaks the protocol, echoes source PCM, no models |
| `scripts/setup_rvc_venv.sh` | build `.venv-rvc` (Blackwell recipe) + verify |
| `scripts/verify_rvc.py` | real-conversion GPU smoke (capability `(12,0)`, non-silent, RTF) |
| `scripts/fetch_rvc_model.sh` | fetch HuBERT + rmvpe + Charlie zip into the data dir |

Modified:

| Path | Change |
|------|--------|
| `hosaka/config.py` | RVC paths, `SOURCE_PRESETS`, `RVC_KNOBS`, `RVC_VOICES`, `resolve_source()` |
| `hosaka/server/engines/base.py` | `EngineRegistry.rvc` field + `get()` / `warmup_all()` branches |
| `hosaka/server/main.py` | `_make_rvc()` builder; construct Kokoro once, share with RVC |
| `hosaka/server/app.py` | `_resolve` rvc branch; `/v1/voices` rvc block |
| `tests/test_server.py` | RVC fake-engine server tests |
| `scripts/smoke_server.sh` | add a Charlie RVC end-to-end check |
| `ARCHITECTURE.md`, `CLAUDE.md`, `README.md` | 5th engine + 5th venv docs |

---

## Task 1: Wire protocol (`rvc_proto.py`)

**Files:**
- Create: `hosaka/server/engines/rvc_proto.py`
- Test: `tests/test_rvc_proto.py`

**Interfaces:**
- Produces:
  - `RvcSidecarError(RuntimeError)`, `RvcProtocolError(RuntimeError)`
  - `pack_audio(pcm: bytes) -> bytes`, `pack_error(msg: str) -> bytes`, `pack_end() -> bytes`
  - `read_response(reader) -> Iterator[bytes]` (yields PCM payloads; raises on error/truncation)
  - `encode_request(pcm: bytes, *, voice: str, transpose: int, index_rate: float, f0_method: str, protect: float, rms_mix_rate: float) -> bytes`
  - `read_request(reader) -> dict | None` (one request: params + `pcm` bytes; `None` at EOF)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rvc_proto.py
import io

import pytest

from hosaka.server.engines.rvc_proto import (
    RvcProtocolError,
    RvcSidecarError,
    encode_request,
    pack_audio,
    pack_end,
    pack_error,
    read_request,
    read_response,
)


def _stream(*frames):
    return io.BytesIO(b"".join(frames))


# --- response framing (sidecar -> server), same contract as piper_proto ---
def test_single_audio_chunk_then_end():
    pcm = b"\x00\x01\x02\x03"
    assert list(read_response(_stream(pack_audio(pcm), pack_end()))) == [pcm]


def test_multiple_chunks_preserve_order():
    a, b = b"\xaa\xbb", b"\xcc\xdd\xee\xff"
    assert list(read_response(_stream(pack_audio(a), pack_audio(b), pack_end()))) == [a, b]


def test_end_only_yields_nothing():
    assert list(read_response(_stream(pack_end()))) == []


def test_error_frame_raises_with_message():
    with pytest.raises(RvcSidecarError, match="kaboom"):
        list(read_response(_stream(pack_error("kaboom"))))


def test_truncated_stream_raises_not_silent():
    with pytest.raises(RvcProtocolError):
        list(read_response(_stream(pack_audio(b"\x00\x01\x02\x03"))))


def test_partial_header_raises():
    with pytest.raises(RvcProtocolError):
        list(read_response(_stream(b"A\x00\x00")))


# --- request framing (server -> sidecar): JSON header + PCM block ---
def test_request_roundtrips_params_and_pcm():
    pcm = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    raw = encode_request(
        pcm, voice="charlie", transpose=2, index_rate=0.5, f0_method="rmvpe",
        protect=0.33, rms_mix_rate=0.25,
    )
    d = read_request(io.BytesIO(raw))
    assert d["voice"] == "charlie"
    assert d["transpose"] == 2
    assert d["index_rate"] == 0.5
    assert d["f0_method"] == "rmvpe"
    assert d["pcm"] == pcm


def test_request_pcm_may_contain_newline_bytes():
    # The PCM block is length-prefixed, so embedded 0x0a bytes must not be
    # mistaken for the header's line terminator.
    pcm = b"\x0a\x0a\x0a\x0a"
    d = read_request(io.BytesIO(encode_request(
        pcm, voice="charlie", transpose=0, index_rate=0.5, f0_method="rmvpe",
        protect=0.33, rms_mix_rate=0.25,
    )))
    assert d["pcm"] == pcm


def test_read_request_returns_none_at_eof():
    assert read_request(io.BytesIO(b"")) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-dev/bin/python -m pytest tests/test_rvc_proto.py -v`
Expected: FAIL — `ModuleNotFoundError: hosaka.server.engines.rvc_proto`

- [ ] **Step 3: Write the implementation**

```python
# hosaka/server/engines/rvc_proto.py
"""Wire protocol between the server-side RvcEngine client and the isolated
.venv-rvc sidecar. Pure stdlib so it imports cleanly in every venv.

Request  (server -> sidecar): one JSON header line, newline-terminated, then a
length-prefixed source-PCM block:

    {json params}\n
    uint32be N + <N bytes float32 LE PCM @ 24k>

Response (sidecar -> server): tagged length-prefixed frames (same contract as
piper_proto):

    b'A' + uint32be N + <N bytes float32 LE PCM>   converted audio chunk
    b'X' + uint32be N + <N bytes utf-8 message>    synthesis error -> raises
    b'E'                                           end of utterance (success)

A stream that closes before an 'E' marker is a dead sidecar -> RvcProtocolError,
never a silently truncated success.
"""

import json
import struct

_AUDIO = b"A"
_ERROR = b"X"
_END = b"E"
_LEN = struct.Struct(">I")


class RvcSidecarError(RuntimeError):
    """The sidecar reported a synthesis failure (error frame)."""


class RvcProtocolError(RuntimeError):
    """The frame stream was malformed or ended before the end marker."""


def pack_audio(pcm: bytes) -> bytes:
    return _AUDIO + _LEN.pack(len(pcm)) + pcm


def pack_error(message: str) -> bytes:
    m = message.encode("utf-8")
    return _ERROR + _LEN.pack(len(m)) + m


def pack_end() -> bytes:
    return _END


def _read_exact(reader, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining:
        b = reader.read(remaining)
        if not b:
            raise RvcProtocolError(f"stream ended: wanted {n} bytes, short by {remaining}")
        chunks.append(b)
        remaining -= len(b)
    return b"".join(chunks)


def read_response(reader):
    """Yield converted PCM payloads frame by frame until the end marker.
    Raises RvcSidecarError on an error frame, RvcProtocolError on malformed /
    truncated streams."""
    while True:
        tag = reader.read(1)
        if tag == _END:
            return
        if not tag:
            raise RvcProtocolError("stream closed before end marker")
        if tag == _AUDIO:
            (n,) = _LEN.unpack(_read_exact(reader, 4))
            yield _read_exact(reader, n)
        elif tag == _ERROR:
            (n,) = _LEN.unpack(_read_exact(reader, 4))
            raise RvcSidecarError(_read_exact(reader, n).decode("utf-8", "replace"))
        else:
            raise RvcProtocolError(f"bad frame tag: {tag!r}")


def encode_request(
    pcm: bytes,
    *,
    voice: str,
    transpose: int,
    index_rate: float,
    f0_method: str,
    protect: float,
    rms_mix_rate: float,
) -> bytes:
    header = json.dumps(
        {
            "voice": voice,
            "transpose": transpose,
            "index_rate": index_rate,
            "f0_method": f0_method,
            "protect": protect,
            "rms_mix_rate": rms_mix_rate,
        }
    ).encode("utf-8") + b"\n"
    return header + _LEN.pack(len(pcm)) + pcm


def read_request(reader):
    """Read one request (JSON header line + length-prefixed PCM block) from a
    binary reader. Returns a dict with the params plus a `pcm` bytes field, or
    None at clean EOF. Uses the SAME reader for the line and the block, so a
    BufferedReader's read-ahead stays consistent."""
    line = reader.readline()
    if not line:
        return None
    d = json.loads(line)
    (n,) = _LEN.unpack(_read_exact(reader, 4))
    d["pcm"] = _read_exact(reader, n)
    return d
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-dev/bin/python -m pytest tests/test_rvc_proto.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Lint + commit**

```bash
.venv-dev/bin/ruff check --fix hosaka/server/engines/rvc_proto.py tests/test_rvc_proto.py
.venv-dev/bin/ruff format hosaka/server/engines/rvc_proto.py tests/test_rvc_proto.py
git add hosaka/server/engines/rvc_proto.py tests/test_rvc_proto.py
git commit -m "feat(rvc): add sidecar wire protocol"
```

---

## Task 2: Client engine + fake sidecar (`rvc_engine.py`)

**Files:**
- Create: `hosaka/server/engines/rvc_engine.py`
- Create: `tests/fake_rvc_sidecar.py`
- Test: `tests/test_rvc_engine.py`

**Interfaces:**
- Consumes: `rvc_proto.encode_request`, `read_response`, `RvcSidecarError`, `RvcProtocolError`; a `source_engine` with `stream(text, voice, params) -> Iterator[np.ndarray]` (the Kokoro engine).
- Produces:
  - `class RvcEngine` with `__init__(self, source_engine, sidecar_cmd, *, voices: dict, knobs: dict, cwd=None, stderr=None)` where `voices` maps `vid -> {"source": str, "transpose": int}` and `knobs` is `{"index_rate", "f0_method", "protect", "rms_mix_rate"}`.
  - `voice_ids: list[str]`
  - `stream(text, voice, params) -> Iterator[np.ndarray]`, `warmup() -> None`, `close() -> None`

- [ ] **Step 1: Write the fake sidecar**

```python
# tests/fake_rvc_sidecar.py
#!/usr/bin/env python
"""Fake RVC sidecar for RvcEngine tests: speaks the wire protocol with no
torch / rvc-python / models. Behavior is keyed off the request `voice` so tests
can drive edge cases over a real subprocess pipe.

  voice == "boom" -> one error frame (sidecar stays alive)
  voice == "die"  -> exit mid-utterance (no end marker)
  voice == "echo" -> error frame carrying the received params (routing proof)
  otherwise       -> echo the received source PCM back as one audio frame, end
                     (proves source accumulation + PCM round-trips both ways)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from hosaka.server.engines.rvc_proto import (  # noqa: E402
    pack_audio,
    pack_end,
    pack_error,
    read_request,
)


def main():
    out = sys.stdout.buffer
    while True:
        req = read_request(sys.stdin.buffer)
        if req is None:
            return
        voice = req.get("voice", "")
        if voice == "echo":
            params = {k: v for k, v in req.items() if k != "pcm"}
            out.write(pack_error(json.dumps(params, sort_keys=True)))
            out.flush()
            continue
        if voice == "boom":
            out.write(pack_error("boom"))
            out.flush()
            continue
        if voice == "die":
            sys.exit(1)
        out.write(pack_audio(req["pcm"]))  # echo the source straight back
        out.write(pack_end())
        out.flush()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_rvc_engine.py
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from hosaka.server.engines.rvc_engine import RvcEngine
from hosaka.server.engines.rvc_proto import RvcProtocolError, RvcSidecarError

FAKE = [sys.executable, str(Path(__file__).parent / "fake_rvc_sidecar.py")]

KNOBS = {"index_rate": 0.5, "f0_method": "rmvpe", "protect": 0.33, "rms_mix_rate": 0.25}


class FakeSource:
    """Stand-in Kokoro: yields two deterministic chunks so a test can check the
    engine concatenates the whole fragment before sending it to the sidecar."""

    def __init__(self):
        self.calls = []

    def stream(self, text, voice, params):
        self.calls.append((text, voice, params))
        yield np.full(50, 0.1, dtype=np.float32)
        yield np.full(30, 0.2, dtype=np.float32)

    def warmup(self):
        pass


def _voices():
    return {
        "charlie": {"source": "af_sarah", "transpose": 2},
        "boom": {"source": "af_sarah", "transpose": 0},
        "die": {"source": "af_sarah", "transpose": 0},
        "echo": {"source": "af_sarah", "transpose": 7},
    }


def _engine(src=None):
    return RvcEngine(src or FakeSource(), FAKE, voices=_voices(), knobs=dict(KNOBS))


def test_voice_ids_lists_configured_voices():
    assert set(_engine().voice_ids) == {"charlie", "boom", "die", "echo"}


def test_stream_converts_and_round_trips_source():
    # The fake echoes the source PCM back; the engine must have accumulated BOTH
    # source chunks (80 float32 samples) before sending.
    eng = _engine()
    out = np.concatenate(list(eng.stream("Hi there.", "charlie", {"speed": 1.0})))
    assert out.dtype == np.float32
    assert len(out) == 80
    assert np.allclose(out[:50], 0.1) and np.allclose(out[50:], 0.2)
    eng.close()


def test_stream_uses_voices_source_preset():
    src = FakeSource()
    eng = _engine(src)
    list(eng.stream("Hi.", "charlie", {"speed": 1.0}))
    # Kokoro was asked for the configured source preset, not the rvc voice id.
    assert src.calls[0][1] == "af_sarah"
    eng.close()


def test_stream_sends_transpose_and_knobs_to_sidecar():
    eng = _engine()
    with pytest.raises(RvcSidecarError) as exc:
        list(eng.stream("Hi.", "echo", {"speed": 1.0}))
    params = json.loads(str(exc.value))
    assert params["voice"] == "echo"
    assert params["transpose"] == 7
    assert params["index_rate"] == 0.5
    assert params["f0_method"] == "rmvpe"
    eng.close()


def test_stream_surfaces_sidecar_error_and_stays_alive():
    eng = _engine()
    with pytest.raises(RvcSidecarError, match="boom"):
        list(eng.stream("Hi.", "boom", {"speed": 1.0}))
    # A normal synth error leaves the sidecar alive: the next call still works.
    assert len(np.concatenate(list(eng.stream("Hi.", "charlie", {"speed": 1.0})))) == 80
    eng.close()


def test_stream_respawns_after_sidecar_death():
    eng = _engine()
    with pytest.raises(RvcProtocolError):
        list(eng.stream("Hi.", "die", {"speed": 1.0}))
    assert len(np.concatenate(list(eng.stream("Hi.", "charlie", {"speed": 1.0})))) == 80
    eng.close()


def test_warmup_does_not_raise():
    eng = _engine()
    eng.warmup()
    eng.close()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv-dev/bin/python -m pytest tests/test_rvc_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: hosaka.server.engines.rvc_engine`

- [ ] **Step 4: Write the implementation**

```python
# hosaka/server/engines/rvc_engine.py
import atexit
import subprocess

import numpy as np

from hosaka.server.engines.rvc_proto import (
    RvcProtocolError,
    RvcSidecarError,
    encode_request,
    read_response,
)


class RvcEngine:
    """Client for the out-of-process RVC sidecar (.venv-rvc).

    RVC converts timbre, so it needs source audio. This wraps the Kokoro engine
    as the source: it generates a neutral base for the configured preset,
    accumulates the WHOLE fragment (F0 needs the full phrase), ships it to the
    sidecar, and yields back the converted float32 24 kHz PCM -- matching the
    Engine protocol. The server venv never imports rvc-python; that lives only
    in the sidecar this drives over the rvc_proto pipe.

    voices: {vid: {"source": kokoro_preset, "transpose": semitones}}.
    knobs:  {"index_rate", "f0_method", "protect", "rms_mix_rate"} (fixed).
    sidecar_cmd is injected (prod points .venv-rvc python at rvc_sidecar.py +
    models; tests point at a fake), so the wire path is exercised without GPU.
    """

    def __init__(self, source_engine, sidecar_cmd, *, voices, knobs, cwd=None, stderr=None):
        self._source = source_engine
        self._cmd = list(sidecar_cmd)
        self._voices = dict(voices)
        self._knobs = dict(knobs)
        self.voice_ids = list(voices)
        self._cwd = cwd
        self._stderr = stderr
        self._proc = None
        atexit.register(self.close)

    def _ensure_proc(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                self._cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                cwd=self._cwd,
                bufsize=0,
            )
        return self._proc

    def _source_pcm(self, text, voice, params) -> bytes:
        cfg = self._voices[voice]
        speed = float(params.get("speed", 1.0))
        chunks = list(self._source.stream(text, cfg["source"], {"speed": speed}))
        if not chunks:
            return b""
        arr = np.concatenate([np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks])
        return np.ascontiguousarray(arr, dtype="<f4").tobytes()

    def stream(self, text, voice, params):
        cfg = self._voices[voice]
        pcm = self._source_pcm(text, voice, params)
        req = encode_request(
            pcm,
            voice=voice,
            transpose=int(cfg["transpose"]),
            index_rate=float(self._knobs["index_rate"]),
            f0_method=str(self._knobs["f0_method"]),
            protect=float(self._knobs["protect"]),
            rms_mix_rate=float(self._knobs["rms_mix_rate"]),
        )
        proc = self._ensure_proc()
        try:
            proc.stdin.write(req)
            proc.stdin.flush()
            for out_pcm in read_response(proc.stdout):
                yield np.frombuffer(out_pcm, dtype="<f4")
        except RvcSidecarError:
            raise  # per-utterance failure; the sidecar is still healthy
        except (RvcProtocolError, OSError):
            self.close()  # broken pipe / dead sidecar: respawn clean next call
            raise

    def warmup(self) -> None:
        for v in self.voice_ids:
            for _ in self.stream("Warm up.", v, {"speed": 1.0}):
                pass

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv-dev/bin/python -m pytest tests/test_rvc_engine.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Lint + commit**

```bash
.venv-dev/bin/ruff check --fix hosaka/server/engines/rvc_engine.py tests/test_rvc_engine.py tests/fake_rvc_sidecar.py
.venv-dev/bin/ruff format hosaka/server/engines/rvc_engine.py tests/test_rvc_engine.py tests/fake_rvc_sidecar.py
git add hosaka/server/engines/rvc_engine.py tests/test_rvc_engine.py tests/fake_rvc_sidecar.py
git commit -m "feat(rvc): add sidecar client engine"
```

---

## Task 3: Config — registry + source resolution (`config.py`)

**Files:**
- Modify: `hosaka/config.py` (append after the `PIPER_VOICES` block, ~line 111)
- Test: `tests/test_config_rvc.py`

**Interfaces:**
- Produces: `RVC_PYTHON`, `RVC_SIDECAR`, `RVC_DIR`, `RVC_HUBERT`, `RVC_RMVPE` (Paths); `SOURCE_PRESETS: dict[tuple[str,str], str]`; `RVC_KNOBS: dict`; `RVC_VOICES: dict`; `resolve_source(voice_cfg: dict) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config_rvc.py
import pytest

from hosaka.config import RVC_KNOBS, RVC_VOICES, SOURCE_PRESETS, resolve_source


def test_charlie_registered_female_american():
    c = RVC_VOICES["charlie"]
    assert c["gender"] == "female"
    assert c["accent"] == "american"
    assert c["model_sr"] == 32000


def test_resolve_source_matches_tuple():
    assert resolve_source(RVC_VOICES["charlie"]) == "af_sarah"
    assert SOURCE_PRESETS[("female", "american")] == "af_sarah"


def test_resolve_source_rejects_mismatch():
    bad = dict(RVC_VOICES["charlie"], source="am_michael")
    with pytest.raises(ValueError, match="source"):
        resolve_source(bad)


def test_resolve_source_unknown_tuple_raises():
    # Kokoro has no Australian base -> no matching source (hard stop).
    bad = {"gender": "female", "accent": "australian", "source": "x"}
    with pytest.raises(KeyError):
        resolve_source(bad)


def test_knobs_are_fixed_rmvpe():
    assert RVC_KNOBS["f0_method"] == "rmvpe"
    assert set(RVC_KNOBS) == {"index_rate", "f0_method", "protect", "rms_mix_rate"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-dev/bin/python -m pytest tests/test_config_rvc.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_source'`

- [ ] **Step 3: Add the config block**

Append to `hosaka/config.py` (after `PIPER_VOICES`, before `BAKE_SEED_TEXT`):

```python
# RVC character voices (e.g. Charlie Morningstar). Run GPU in an isolated venv
# (.venv-rvc) as an out-of-process sidecar; the server venv never imports
# rvc-python. RVC converts timbre, so each voice declares the character's
# gender + accent -- the neutral Kokoro SOURCE preset must match (see
# resolve_source). Model weights live untracked under the data dir -- fetch
# them with scripts/fetch_rvc_model.sh. Adding a character = drop a model + one
# RVC_VOICES entry, no code change.
RVC_PYTHON = _REPO_ROOT / ".venv-rvc" / "bin" / "python"
RVC_SIDECAR = _REPO_ROOT / "hosaka" / "server" / "engines" / "rvc_sidecar.py"
RVC_DIR = DATA_DIR / "rvc"
RVC_HUBERT = RVC_DIR / "hubert_base.pt"  # ContentVec/HuBERT feature encoder
RVC_RMVPE = RVC_DIR / "rmvpe.pt"  # F0 (pitch) estimator

# (gender, accent) -> neutral Kokoro base preset. Kokoro English is American or
# British only; a tuple absent here has no matching base (hard stop, not a
# "download more presets" case).
SOURCE_PRESETS = {
    ("female", "american"): "af_sarah",  # clear + neutral
    ("male", "american"): "am_michael",
    ("female", "british"): "bf_emma",
    ("male", "british"): "bm_george",
}

# Fixed conversion knobs for MVP (no per-request RVC knobs).
RVC_KNOBS = {
    "index_rate": 0.5,
    "f0_method": "rmvpe",
    "protect": 0.33,
    "rms_mix_rate": 0.25,
}

RVC_VOICES = {
    "charlie": {
        "model": RVC_DIR / "charlie" / "charlie.pth",
        "index": RVC_DIR / "charlie" / "charlie.index",
        "model_sr": 32000,  # native; the sidecar resamples 32k -> 24k
        "gender": "female",
        "accent": "american",
        "source": "af_sarah",  # must equal SOURCE_PRESETS[(female, american)]
        "transpose": 0,  # semitones; tune by ear (demo showed -16..+16)
        "description": "Charlie Morningstar (Hazbin Hotel), RVC V2",
    },
}


def resolve_source(voice_cfg: dict) -> str:
    """The neutral Kokoro source preset for an RVC voice, validated against its
    gender + accent. A configured `source` that disagrees with the tuple is a
    config error (fail loud). KeyError if Kokoro has no base for the tuple."""
    want = SOURCE_PRESETS[(voice_cfg["gender"], voice_cfg["accent"])]
    if voice_cfg["source"] != want:
        raise ValueError(
            f"RVC source {voice_cfg['source']!r} != {want!r} for "
            f"{voice_cfg['gender']}/{voice_cfg['accent']}"
        )
    return want
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-dev/bin/python -m pytest tests/test_config_rvc.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Lint + commit**

```bash
.venv-dev/bin/ruff check --fix hosaka/config.py tests/test_config_rvc.py
.venv-dev/bin/ruff format hosaka/config.py tests/test_config_rvc.py
git add hosaka/config.py tests/test_config_rvc.py
git commit -m "feat(rvc): register Charlie voice + source rule"
```

---

## Task 4: Registry slot (`base.py`)

**Files:**
- Modify: `hosaka/server/engines/base.py`
- Test: `tests/test_engine_registry.py`

**Interfaces:**
- Consumes: existing `EngineRegistry`.
- Produces: `EngineRegistry(..., rvc: Engine | None = None)`; `get("rvc")` returns it or raises `KeyError` when `None`; `warmup_all()` warms it when present.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_engine_registry.py
import pytest

from hosaka.server.engines.base import EngineRegistry


class Spy:
    def __init__(self):
        self.warmed = False

    def stream(self, text, voice, params):
        yield None

    def warmup(self):
        self.warmed = True


def _reg(**kw):
    return EngineRegistry(kokoro=Spy(), chatterbox=Spy(), **kw)


def test_get_rvc_returns_engine_when_present():
    rvc = Spy()
    assert _reg(rvc=rvc).get("rvc") is rvc


def test_get_rvc_raises_when_absent():
    with pytest.raises(KeyError, match="rvc"):
        _reg().get("rvc")


def test_warmup_all_warms_rvc_when_present():
    rvc = Spy()
    _reg(rvc=rvc).warmup_all()
    assert rvc.warmed


def test_warmup_all_skips_rvc_when_absent():
    _reg().warmup_all()  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-dev/bin/python -m pytest tests/test_engine_registry.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'rvc'`

- [ ] **Step 3: Edit `base.py`**

Replace the dataclass + `get` + `warmup_all` (lines 14-35) with:

```python
@dataclass
class EngineRegistry:
    kokoro: Engine
    chatterbox: Engine
    piper: Engine | None = None  # optional CPU sidecar (character voices)
    rvc: Engine | None = None  # optional GPU sidecar (converted character voices)

    def get(self, backend: str) -> Engine:
        if backend == "kokoro":
            return self.kokoro
        if backend == "chatterbox":
            return self.chatterbox
        if backend == "piper":
            if self.piper is None:
                raise KeyError("piper backend not available")
            return self.piper
        if backend == "rvc":
            if self.rvc is None:
                raise KeyError("rvc backend not available")
            return self.rvc
        raise KeyError(f"unknown backend: {backend}")

    def warmup_all(self) -> None:
        self.kokoro.warmup()
        self.chatterbox.warmup()
        if self.piper is not None:
            self.piper.warmup()
        if self.rvc is not None:
            self.rvc.warmup()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-dev/bin/python -m pytest tests/test_engine_registry.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Lint + commit**

```bash
.venv-dev/bin/ruff check --fix hosaka/server/engines/base.py tests/test_engine_registry.py
.venv-dev/bin/ruff format hosaka/server/engines/base.py tests/test_engine_registry.py
git add hosaka/server/engines/base.py tests/test_engine_registry.py
git commit -m "feat(rvc): add rvc slot to engine registry"
```

---

## Task 5: Server resolution + voice listing (`app.py`)

**Files:**
- Modify: `hosaka/server/app.py` (`_resolve` ~lines 87-97; `/v1/voices` ~lines 242-252)
- Test: `tests/test_server.py` (append)

**Interfaces:**
- Consumes: `EngineRegistry.rvc`, `RVC_VOICES`, an engine exposing `voice_ids`.
- Produces: backend `"rvc"` resolves like Piper (voice must be in `engine.voice_ids`); `/v1/voices` lists rvc voices tagged `backend="rvc"` when present, omits them when absent.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server.py`:

```python
class RvcFakeEngine(FakeEngine):
    voice_ids = ["charlie", "charlie2"]


def _client_with_rvc(tmp_path):
    reg = EngineRegistry(kokoro=FakeEngine(), chatterbox=FakeEngine(), rvc=RvcFakeEngine())
    lib = VoiceLibrary(tmp_path / "voices")
    return TestClient(create_app(reg, lib, do_warmup=False)), lib


def test_voices_list_rvc_when_available(tmp_path):
    from hosaka.config import RVC_VOICES

    client, _ = _client_with_rvc(tmp_path)
    voices = {v["id"]: v for v in client.get("/v1/voices").json()}
    assert voices["charlie"]["backend"] == "rvc"
    assert voices["charlie2"]["backend"] == "rvc"
    assert voices["charlie"]["description"] == RVC_VOICES["charlie"]["description"]


def test_voices_omit_rvc_when_unavailable(tmp_path):
    client, _ = _client(tmp_path)
    backends = {v["backend"] for v in client.get("/v1/voices").json()}
    assert "rvc" not in backends


def test_speech_rvc_voice_streams(tmp_path):
    client, _ = _client_with_rvc(tmp_path)
    r = client.post("/v1/audio/speech", json={"input": "hi", "backend": "rvc", "voice": "charlie"})
    assert r.status_code == 200
    assert len(r.content) > 0 and len(r.content) % 4 == 0


def test_speech_unknown_rvc_voice_is_400(tmp_path):
    client, _ = _client_with_rvc(tmp_path)
    r = client.post("/v1/audio/speech", json={"input": "hi", "backend": "rvc", "voice": "bogus"})
    assert r.status_code == 400
    assert "bogus" in r.json()["detail"]


def test_speech_rvc_unavailable_is_400(tmp_path):
    client, _ = _client(tmp_path)
    r = client.post("/v1/audio/speech", json={"input": "hi", "backend": "rvc", "voice": "charlie"})
    assert r.status_code == 400
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-dev/bin/python -m pytest tests/test_server.py -k rvc -v`
Expected: FAIL — rvc voices not listed / backend rejected differently than asserted.

- [ ] **Step 3: Edit `app.py`**

Add the import (extend the `hosaka.config` import block, ~lines 14-22): add `RVC_VOICES` next to `PIPER_VOICES`.

In `_resolve`, after the `piper` branch (line 92), add:

```python
        elif backend == "rvc":
            if voice not in engine.voice_ids:
                return None, f"unknown rvc voice: {voice}"
```

In the `voices()` route, after the piper block (line 252), add:

```python
        if registry.rvc is not None:
            out += [
                VoiceInfo(
                    id=vid,
                    backend="rvc",
                    source="rvc",
                    description=RVC_VOICES.get(vid, {}).get("description", ""),
                ).model_dump()
                for vid in registry.rvc.voice_ids
            ]
```

(No change to `_fragments_for`: rvc is not `chatterbox`, so it already takes the plain sentence split — correct, since the sidecar converts each whole fragment.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-dev/bin/python -m pytest tests/test_server.py -v`
Expected: PASS (all server tests, including the 5 new rvc ones)

- [ ] **Step 5: Lint + commit**

```bash
.venv-dev/bin/ruff check --fix hosaka/server/app.py tests/test_server.py
.venv-dev/bin/ruff format hosaka/server/app.py tests/test_server.py
git add hosaka/server/app.py tests/test_server.py
git commit -m "feat(rvc): resolve rvc backend + list voices"
```

---

## Task 6: Wire the engine into the server (`main.py`)

**Files:**
- Modify: `hosaka/server/main.py`

**Interfaces:**
- Consumes: `RvcEngine`, `RVC_*` config, `resolve_source`, the constructed `KokoroEngine`.
- Produces: `_make_rvc(source_engine) -> RvcEngine | None`; registry built with `rvc=_make_rvc(_kokoro)`.

Note: `main.py` imports torch-bound engines, so it loads only in `.venv-server`. There is no dev unit test; correctness is verified by the smoke run in Task 11 (graceful degrade before models exist; Charlie present after).

- [ ] **Step 1: Rewrite `main.py`**

```python
from hosaka.config import (
    PIPER_PYTHON,
    PIPER_SIDECAR,
    PIPER_VOICES,
    RVC_HUBERT,
    RVC_KNOBS,
    RVC_PYTHON,
    RVC_RMVPE,
    RVC_SIDECAR,
    RVC_VOICES,
    VOICE_DIR,
    resolve_source,
)
from hosaka.library import VoiceLibrary
from hosaka.server.app import create_app
from hosaka.server.engines.base import EngineRegistry
from hosaka.server.engines.chatterbox_engine import ChatterboxEngine
from hosaka.server.engines.kokoro_engine import KokoroEngine
from hosaka.server.engines.piper_engine import PiperEngine
from hosaka.server.engines.rvc_engine import RvcEngine


def _make_piper():
    """Build the Piper sidecar engine, or None if it isn't set up. Missing the
    .venv-piper interpreter or every model file degrades gracefully."""
    available = {vid: spec for vid, spec in PIPER_VOICES.items() if spec["model"].exists()}
    if not PIPER_PYTHON.exists() or not available:
        return None
    cmd = [str(PIPER_PYTHON), str(PIPER_SIDECAR)]
    for vid, spec in available.items():
        cmd += ["--voice", f"{vid}={spec['model']}"]
    return PiperEngine(cmd, voices=list(available))


def _make_rvc(source_engine):
    """Build the RVC sidecar engine, or None if it isn't set up. Missing the
    .venv-rvc interpreter, the HuBERT/rmvpe assets, or every voice's model+index
    degrades gracefully: the server runs the other engines. RVC uses Kokoro as
    its neutral audio source, so the engine is shared in."""
    available = {
        vid: spec
        for vid, spec in RVC_VOICES.items()
        if spec["model"].exists() and spec["index"].exists()
    }
    if (
        not RVC_PYTHON.exists()
        or not RVC_HUBERT.exists()
        or not RVC_RMVPE.exists()
        or not available
    ):
        return None
    cmd = [str(RVC_PYTHON), str(RVC_SIDECAR), "--hubert", str(RVC_HUBERT), "--rmvpe", str(RVC_RMVPE)]
    voices = {}
    for vid, spec in available.items():
        cmd += ["--voice", f"{vid}={spec['model']}:{spec['index']}"]
        voices[vid] = {"source": resolve_source(spec), "transpose": spec["transpose"]}
    return RvcEngine(source_engine, cmd, voices=voices, knobs=dict(RVC_KNOBS))


_library = VoiceLibrary(VOICE_DIR)
_kokoro = KokoroEngine()
_registry = EngineRegistry(
    kokoro=_kokoro,
    chatterbox=ChatterboxEngine(_library),
    piper=_make_piper(),
    rvc=_make_rvc(_kokoro),
)
app = create_app(_registry, _library, do_warmup=True)
```

- [ ] **Step 2: Verify the non-GPU suite still imports/passes**

Run: `.venv-dev/bin/python -m pytest -m "not gpu"`
Expected: PASS (main.py is not imported by the dev suite; this confirms nothing else broke)

- [ ] **Step 3: Commit**

```bash
.venv-dev/bin/ruff check --fix hosaka/server/main.py
.venv-dev/bin/ruff format hosaka/server/main.py
git add hosaka/server/main.py
git commit -m "feat(rvc): wire rvc engine into the server"
```

---

## Task 7: Build the `.venv-rvc` environment (`setup_rvc_venv.sh`)

**Files:**
- Create: `scripts/setup_rvc_venv.sh`

This is the build's primary unknown (the spec's flagged risk). The script is the place to nail RVC's deps empirically; it ends by running `verify_rvc.py` (Task 8). `fairseq` is the high-risk dep — if it has no Python 3.12 wheel and won't build, fall back to `PY=python3.11` for this venv (a documented contingency in the script).

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Build the isolated RVC venv: GPU voice conversion for converted character
# voices (Charlie and any other RVC .pth). Kept separate from the server venv so
# rvc-python / fairseq / faiss pins can never perturb the Kokoro/Chatterbox
# stack. The server never imports rvc-python; it talks to this venv only
# out-of-process, via the sidecar (hosaka/server/engines/rvc_sidecar.py).
#
# Blackwell sm_120: torch 2.9.1+cu128 goes in FIRST, then rvc-python WITHOUT its
# torch, then its non-torch deps. The exact dep set is pinned by iterating here
# until scripts/verify_rvc.py passes -- that is expected, not a failure.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

PY=python3.12             # CONTINGENCY: if fairseq has no 3.12 wheel, use python3.11
VENV=.venv-rvc
CU128=https://download.pytorch.org/whl/cu128

echo "=== [1/5] create venv ==="
$PY -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel

echo "=== [2/5] torch FIRST from cu128 (Blackwell sm_120) ==="
"$VENV/bin/pip" install --index-url "$CU128" torch==2.9.1+cu128 torchaudio==2.9.1+cu128

echo "=== [3/5] rvc-python WITHOUT its torch (keep ours) ==="
"$VENV/bin/pip" install --no-deps rvc-python

echo "=== [4/5] rvc-python's non-torch deps (pinned; iterate until verify passes) ==="
# Known-needed set; adjust here if verify_rvc.py reports a missing import.
"$VENV/bin/pip" install \
  "faiss-cpu" "librosa" "scipy" "soundfile" "numpy" \
  "praat-parselmouth" "pyworld" "torchcrepe" "fairseq" "omegaconf" "ffmpeg-python"

echo "=== [5/5] verify (real conversion, capability (12,0), RTF) ==="
PYTHONPATH="$PWD" "$VENV/bin/python" scripts/verify_rvc.py

echo "SETUP_RVC_DONE"
```

- [ ] **Step 2: Make executable + commit (do not run yet — needs models from Task 9)**

```bash
chmod +x scripts/setup_rvc_venv.sh
git add scripts/setup_rvc_venv.sh
git commit -m "build(rvc): add .venv-rvc setup recipe"
```

---

## Task 8: GPU verification harness (`verify_rvc.py`)

**Files:**
- Create: `scripts/verify_rvc.py`

**Interfaces:**
- Consumes: `RVC_DIR`, `RVC_HUBERT`, `RVC_RMVPE`, `RVC_VOICES`, `SAMPLE_RATE`, rvc-python.

This is the executable spec for the GPU conversion path. It is the failing "test" that Task 10's sidecar and the venv must satisfy. It drives rvc-python directly (independent of the sidecar framing); the full sidecar+engine path is proven in Task 11.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python
"""Verify the .venv-rvc GPU conversion path end to end (no server).

Real matmul capability check (never trust torch.cuda.is_available()), then load
Charlie and convert ~2s of a generated source tone, asserting non-silent finite
output and printing RTF. Run: PYTHONPATH=$PWD .venv-rvc/bin/python scripts/verify_rvc.py
"""

import time

import numpy as np
import soundfile as sf
import torch

from hosaka.config import RVC_HUBERT, RVC_RMVPE, RVC_VOICES, SAMPLE_RATE


def _check_gpu():
    cap = torch.cuda.get_device_capability()
    assert cap == (12, 0), f"expected Blackwell sm_120 (12, 0), got {cap}"
    x = torch.randn(256, 256, device="cuda")
    y = x @ x
    assert torch.isfinite(y).all(), "GPU matmul produced non-finite values"
    print(f"GPU ok: capability {cap}, real matmul finite")


def _source_wav(path, seconds=2.0):
    # A simple voiced-ish tone so F0 has something to track.
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    wav = (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    sf.write(path, wav, SAMPLE_RATE, subtype="FLOAT")
    return seconds


def main():
    import tempfile
    from pathlib import Path

    from rvc_python.infer import RVCInference  # noqa: F401  (verify the import)

    _check_gpu()
    assert RVC_HUBERT.exists(), f"missing {RVC_HUBERT}"
    assert RVC_RMVPE.exists(), f"missing {RVC_RMVPE}"
    spec = RVC_VOICES["charlie"]
    assert spec["model"].exists() and spec["index"].exists(), "fetch Charlie first"

    rvc = RVCInference(device="cuda:0")
    rvc.load_model(str(spec["model"]), index_path=str(spec["index"]))
    rvc.set_params(
        f0method="rmvpe", f0up_key=spec["transpose"], index_rate=0.5,
        protect=0.33, rms_mix_rate=0.25,
    )

    with tempfile.TemporaryDirectory() as d:
        src = str(Path(d) / "src.wav")
        out = str(Path(d) / "out.wav")
        seconds = _source_wav(src)
        t0 = time.perf_counter()
        rvc.infer_file(src, out)
        dt = time.perf_counter() - t0
        wav, sr = sf.read(out, dtype="float32")

    assert np.isfinite(wav).all() and np.abs(wav).max() > 1e-3, "silent / non-finite output"
    print(f"convert ok: {seconds:.1f}s @ model {sr} Hz, {dt:.2f}s wall, RTF {dt / seconds:.2f}")
    print("VERIFY_RVC_DONE")


if __name__ == "__main__":
    main()
```

Note: `RVCInference` / `load_model` / `set_params` / `infer_file` is the rvc-python high-level API. **Confirm these signatures against the version pinned in Task 7** and adjust the calls if the installed lib differs (e.g. exposes `infer_array`). This file and the sidecar (Task 10) are the only two places that touch rvc-python, so they change together.

- [ ] **Step 2: Commit (run it in Task 11 after setup + fetch)**

```bash
git add scripts/verify_rvc.py
git commit -m "test(rvc): add GPU conversion verify harness"
```

---

## Task 9: Fetch models (`fetch_rvc_model.sh`)

**Files:**
- Create: `scripts/fetch_rvc_model.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Fetch RVC assets + the Charlie Morningstar voice into the hosaka data dir.
#
#   hubert_base.pt, rmvpe.pt   canonical RVC assets (lj1995/VoiceConversionWebUI)
#   charlie                    ScruffyRVC/RFCRVCV2 charlie-morningstar-v3.zip
#                              (RVC V2, 32k; contains the .pth + .index)
#
# Only the trained weights are fetched. Idempotent: skips files already present.
set -euo pipefail

ROOT="${HOME}/.local/share/hosaka/rvc"
HF="https://huggingface.co"
mkdir -p "$ROOT"

fetch() {  # url dest
  if [ -f "$2" ]; then echo "have    ${2#$ROOT/}"; return; fi
  echo "fetch   ${2#$ROOT/}"
  curl -fL "$1" -o "$2"
}

# --- shared assets ---
fetch "${HF}/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt" "${ROOT}/hubert_base.pt"
fetch "${HF}/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt" "${ROOT}/rmvpe.pt"

# --- Charlie ---
CDIR="${ROOT}/charlie"
mkdir -p "$CDIR"
if [ -f "${CDIR}/charlie.pth" ] && [ -f "${CDIR}/charlie.index" ]; then
  echo "have    charlie/charlie.{pth,index}"
else
  ZIP="${CDIR}/charlie-morningstar-v3.zip"
  fetch "${HF}/ScruffyRVC/RFCRVCV2/resolve/main/charlie-morningstar-v3.zip?download=true" "$ZIP"
  echo "unzip   charlie"
  unzip -o -j "$ZIP" -d "$CDIR" >/dev/null
  # Normalize whatever the archive named them to charlie.{pth,index}.
  pth="$(find "$CDIR" -maxdepth 1 -name '*.pth' | head -1)"
  idx="$(find "$CDIR" -maxdepth 1 -name '*.index' | head -1)"
  [ -n "$pth" ] || { echo "ERROR: no .pth in zip"; exit 1; }
  [ -n "$idx" ] || { echo "ERROR: no .index in zip"; exit 1; }
  [ "$pth" = "${CDIR}/charlie.pth" ] || mv -f "$pth" "${CDIR}/charlie.pth"
  [ "$idx" = "${CDIR}/charlie.index" ] || mv -f "$idx" "${CDIR}/charlie.index"
  rm -f "$ZIP"
fi

echo "FETCH_RVC_DONE -> ${ROOT}"
```

- [ ] **Step 2: Make executable + commit**

```bash
chmod +x scripts/fetch_rvc_model.sh
git add scripts/fetch_rvc_model.sh
git commit -m "build(rvc): add model fetch script (Charlie)"
```

---

## Task 10: The RVC sidecar (`rvc_sidecar.py`)

**Files:**
- Create: `hosaka/server/engines/rvc_sidecar.py`

**Interfaces:**
- Consumes: `rvc_proto.read_request/pack_audio/pack_end/pack_error`, `SAMPLE_RATE`, rvc-python.
- Runs ONLY under `.venv-rvc`. The server never imports it.

There is no dev unit test (needs the venv + models). It is verified by Task 11's smoke run. Keep the rvc-python interaction in one function (`_convert`) — the same seam as `verify_rvc.py`.

- [ ] **Step 1: Write the sidecar**

```python
#!/usr/bin/env python
"""RVC voice-conversion sidecar -- runs ONLY under .venv-rvc (it imports torch /
rvc-python / faiss). The server never imports this module; RvcEngine spawns it
and speaks the rvc_proto pipe protocol to it.

Loads the HuBERT/ContentVec encoder, the rmvpe F0 model, and each --voice
(.pth + .index) once, resident on the GPU. For every request it reads the
neutral source PCM (float32 24k), converts it to the target speaker, resamples
the model's native rate (e.g. 32k) down to the hosaka 24k, and streams framed
float32 LE PCM back. A per-request failure becomes an error frame; the sidecar
stays alive for the next request.

  usage: rvc_sidecar.py --hubert hubert_base.pt --rmvpe rmvpe.pt \
                        --voice charlie=/path/charlie.pth:/path/charlie.index
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from hosaka.config import SAMPLE_RATE  # noqa: E402
from hosaka.server.engines.rvc_proto import (  # noqa: E402
    pack_audio,
    pack_end,
    pack_error,
    read_request,
)


def load_voices(specs, hubert, rmvpe):
    """specs: ["id=pth:index", ...] -> {id: (RVCInference, model_sr)}.

    One RVCInference per voice keeps each model resident. rvc-python locates the
    HuBERT/rmvpe assets via its asset dir; point it at ours.
    """
    from rvc_python.infer import RVCInference

    os.environ.setdefault("RVC_MODELDIR", str(Path(hubert).parent))  # assets live here
    voices = {}
    for spec in specs:
        vid, paths = spec.split("=", 1)
        pth, index = paths.split(":", 1)
        rvc = RVCInference(device="cuda:0")
        rvc.load_model(pth, index_path=index)
        voices[vid] = rvc
    return voices


def convert(rvc, req, out):
    """Convert one request's source PCM to the target and stream it back."""
    rvc.set_params(
        f0method=req["f0_method"],
        f0up_key=req["transpose"],
        index_rate=req["index_rate"],
        protect=req["protect"],
        rms_mix_rate=req["rms_mix_rate"],
    )
    src = np.frombuffer(req["pcm"], dtype="<f4")
    with tempfile.TemporaryDirectory() as d:
        sp, op = str(Path(d) / "s.wav"), str(Path(d) / "o.wav")
        sf.write(sp, src, SAMPLE_RATE, subtype="FLOAT")
        rvc.infer_file(sp, op)  # writes at the model's native SR
        wav, sr = sf.read(op, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        wav = resample_poly(wav, SAMPLE_RATE, sr)
    f32 = np.ascontiguousarray(wav.astype(np.float32), dtype="<f4")
    out.write(pack_audio(f32.tobytes()))
    out.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hubert", required=True)
    ap.add_argument("--rmvpe", required=True)
    ap.add_argument("--voice", action="append", default=[], metavar="id=pth:index")
    args = ap.parse_args()
    voices = load_voices(args.voice, args.hubert, args.rmvpe)

    out = sys.stdout.buffer
    while True:
        req = read_request(sys.stdin.buffer)
        if req is None:
            return
        try:
            vid = req.get("voice", "")
            if vid not in voices:
                if len(voices) == 1:
                    vid = next(iter(voices))
                else:
                    raise KeyError(f"unknown rvc voice: {vid!r}")
            convert(voices[vid], req, out)
            out.write(pack_end())
            out.flush()
        except Exception as exc:  # stay alive; report THIS request's failure
            out.write(pack_error(f"{type(exc).__name__}: {exc}"))
            out.flush()


if __name__ == "__main__":
    main()
```

Note: `set_params` / `infer_file` / asset-dir resolution must match the rvc-python version pinned in Task 7. If the lib offers in-memory array inference, swap the temp-wav round-trip in `convert()` for it (same inputs/outputs). This is the second of the two rvc-python seams (with `verify_rvc.py`).

- [ ] **Step 2: Commit (no dev test; exercised in Task 11)**

```bash
git add hosaka/server/engines/rvc_sidecar.py
git commit -m "feat(rvc): add GPU conversion sidecar"
```

---

## Task 11: End-to-end on the box (setup, fetch, verify, smoke)

**Files:**
- Modify: `scripts/smoke_server.sh` (add a Charlie RVC check)

This task runs the GPU/venv/model steps in order and proves the full path. Run on the Blackwell box. If `verify_rvc` fails on a missing import, add it to Task 7's dep list and re-run setup (the expected iteration).

- [ ] **Step 1: Fetch the models** (before setup — setup's verify needs Charlie)

Run: `bash scripts/fetch_rvc_model.sh`
Expected: `FETCH_RVC_DONE`; `~/.local/share/hosaka/rvc/` has `hubert_base.pt`, `rmvpe.pt`, `charlie/charlie.pth`, `charlie/charlie.index`.

- [ ] **Step 2: Build the venv**

Run: `bash scripts/setup_rvc_venv.sh`
Expected: ends with `VERIFY_RVC_DONE`. If it fails on a missing import, add the package to Task 7 Step 1's dep list and re-run (the expected iteration). If `fairseq` won't build on 3.12, set `PY=python3.11` in the script and re-run.

- [ ] **Step 3: Re-verify the GPU conversion path (standalone)**

Run: `PYTHONPATH=$PWD .venv-rvc/bin/python scripts/verify_rvc.py`
Expected: `GPU ok ...`, `convert ok ... RTF <n>`, `VERIFY_RVC_DONE`. RTF well under 1 (spec estimate ~0.1-0.2). If `RVCInference` API differs, fix `verify_rvc.py` + `rvc_sidecar.py` together, re-run.

- [ ] **Step 4: Add a Charlie check to the smoke script**

`scripts/smoke_server.sh` already has a `play backend voice text` helper. Add one
line after the `play piper ...` line (currently line 58):

```bash
play rvc charlie "Oh. It's you. I'm afraid the testing never ends."
```

The existing `play()` function streams from `/v1/audio/speech` and prints the
byte/sample count, so no other change is needed.

- [ ] **Step 5: Run the full smoke**

Run: `bash scripts/smoke_server.sh`
Expected: all four backends (Kokoro, Chatterbox, Piper/GLaDOS, RVC/Charlie) synthesize and play; clean shutdown. Confirm `/v1/voices` lists `charlie` tagged `rvc`.

- [ ] **Step 6: Commit**

```bash
git add scripts/smoke_server.sh
git commit -m "test(rvc): add Charlie end-to-end smoke check"
```

---

## Task 12: Documentation

**Files:**
- Modify: `ARCHITECTURE.md`, `CLAUDE.md`, `README.md`

- [ ] **Step 1: `ARCHITECTURE.md`**

- Add an `rvc` row to the engine table (§"Three engines, one decision"): RVC V2, role "converted character voices", GPU sidecar, RTF ~0.1-0.2 + Kokoro source.
- Add a "Character voices 2: RVC" subsection after the Piper one: Kokoro neutral source -> `.venv-rvc` GPU sidecar -> HuBERT + rmvpe + faiss -> 32k->24k; source must match character gender+accent (`SOURCE_PRESETS`); both GPU ops in one held slot; graceful degrade.
- Add the four `rvc_*` files + scripts to the module map.
- Update the Shape mermaid diagram: `rvc` engine node, `rvc_sidecar` node (`.venv-rvc`, GPU), `rmodels` data node, the Kokoro->RVC source edge.
- Update "Why three venvs" -> FIVE; note `.venv-rvc` isolates rvc-python/fairseq from the Kokoro/Chatterbox stack.

- [ ] **Step 2: `CLAUDE.md`**

- "FOUR venvs" -> FIVE; add the `.venv-rvc` row (torch cu128 + rvc-python, GPU, the RVC sidecar only).
- Add an RVC hard-rules bullet: Blackwell pin (torch first); server never imports rvc-python (sidecar over `rvc_proto`); source preset must match character gender+accent (loud config error otherwise); Charlie native 32k->24k; rvc-python/fairseq pinned empirically in `setup_rvc_venv.sh`, gated by `verify_rvc.py`.
- Note the Piper GLaDOS path is unchanged.

- [ ] **Step 3: `README.md`**

- Install: add `bash scripts/setup_rvc_venv.sh` and `bash scripts/fetch_rvc_model.sh`.
- Usage: a Charlie example (`backend=rvc voice=charlie`).

- [ ] **Step 4: Verify suite still green + commit**

```bash
.venv-dev/bin/python -m pytest -m "not gpu"
.venv-dev/bin/ruff check hosaka tests
git add ARCHITECTURE.md CLAUDE.md README.md
git commit -m "docs(rvc): document 5th engine + venv"
```

---

## Done criteria

- `.venv-dev/bin/python -m pytest -m "not gpu"` green (proto, engine, config, registry, server rvc tests).
- `verify_rvc.py` prints `VERIFY_RVC_DONE` with RTF < 1 on the box.
- `smoke_server.sh` synthesizes + plays Charlie alongside the other three backends.
- `/v1/voices` lists `charlie` as `rvc`; missing venv/models degrade gracefully (rvc omitted, other engines serve).
- Piper GLaDOS, the Chatterbox `transformers` pin, and the GPU serialization invariant are all untouched.
