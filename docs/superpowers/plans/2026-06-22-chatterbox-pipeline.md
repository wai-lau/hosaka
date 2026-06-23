# Chatterbox Playback Pipeline (ffplay streaming) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream the server's sequential fragment output into a persistent `ffplay.exe` on the Windows host so fragment N plays while fragment N+1 synthesizes — seamless playback that starts after ~one fragment instead of the whole utterance.

**Architecture:** Client-side only. Unify the three players on a `write(chunk)` / `end_utterance()` streaming interface; `_speak` streams bytes instead of buffering. A new `FfplayPlayer` pipes gain-applied float32 PCM into a long-lived `ffplay.exe` (native Windows audio, clean), with a time-based lead buffer as the RTF~1 cushion. `make_player` selects it on WSLg when ffmpeg is present.

**Tech Stack:** Python 3.12, numpy, httpx, subprocess; ffplay.exe (Gyan.FFmpeg 8.1.1) on the Windows host; pytest in `.venv-dev`.

## Global Constraints

- Audio is float32 LE PCM, 24 kHz, mono everywhere (`SAMPLE_RATE = 24000`).
- No Unicode emoji anywhere (code, comments, commits). Plain text only.
- Non-GPU tests run in `.venv-dev`: `.venv-dev/bin/python -m pytest -m "not gpu"`. This plan adds no GPU-touching code; everything is testable in `.venv-dev`.
- ffplay raw-input flags are exactly `-f f32le -ar 24000 -ch_layout mono -i pipe:0` (ffplay 8.1 rejects `-ac`).
- Lint/format before each commit: `.venv-dev/bin/ruff check --fix` and `.venv-dev/bin/ruff format`.
- No silent suppression of errors; broken-pipe handling must report.

---

### Task 1: Config + gain/align helper

**Files:**
- Modify: `hosaka/config.py`
- Modify: `hosaka/audio.py`
- Test: `tests/test_audio.py`

**Interfaces:**
- Produces: `config.PIPELINE_LEAD_MS: int` (= 1500); `audio._gain_align(chunk: bytes, tail: bytes, gain: float) -> tuple[bytes, bytes]` returning `(gained_aligned_bytes, leftover_tail)`.

- [ ] **Step 1: Write the failing test**

In `tests/test_audio.py`, add:

```python
def test_gain_align_scales_and_holds_partial_float():
    from hosaka.audio import _gain_align

    src = np.full(4, 0.2, dtype=np.float32).astype("<f4").tobytes()  # 16 bytes
    # Feed 10 bytes: 2 whole floats + 2 leftover; gain 2.0
    out, tail = _gain_align(src[:10], b"", 2.0)
    assert len(out) == 8 and len(tail) == 2
    assert np.allclose(np.frombuffer(out, dtype="<f4"), 0.4, atol=1e-6)
    # Next call prepends the tail and completes the floats
    out2, tail2 = _gain_align(src[10:], tail, 2.0)
    assert tail2 == b""
    assert np.allclose(np.frombuffer(out2, dtype="<f4"), 0.4, atol=1e-6)


def test_pipeline_lead_ms_default():
    from hosaka.config import PIPELINE_LEAD_MS

    assert PIPELINE_LEAD_MS == 1500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-dev/bin/python -m pytest tests/test_audio.py::test_gain_align_scales_and_holds_partial_float tests/test_audio.py::test_pipeline_lead_ms_default -v`
Expected: FAIL (`ImportError: cannot import name '_gain_align'` / `PIPELINE_LEAD_MS`).

- [ ] **Step 3: Write minimal implementation**

In `hosaka/config.py`, below `PLAYBACK_LEAD_SILENCE_MS`:

```python
# Lead buffer for the streaming (ffplay) player: bytes are withheld until this
# much audio is buffered, then released, giving the player a cushion to ride
# over per-fragment synth jitter at Chatterbox's RTF ~1.
PIPELINE_LEAD_MS = 1500
```

In `hosaka/audio.py`, add a module-level helper (after the imports, before `PacatPlayer`):

```python
def _gain_align(chunk: bytes, tail: bytes, gain: float) -> tuple[bytes, bytes]:
    """Apply gain to a byte chunk on 4-byte (float32) boundaries.

    Network chunks may split a float; combine with the held tail, process only
    the 4-aligned prefix, and return the new leftover tail.
    """
    buf = tail + chunk
    n = len(buf) - (len(buf) % 4)
    aligned, new_tail = buf[:n], buf[n:]
    if not aligned:
        return b"", new_tail
    arr = np.frombuffer(aligned, dtype="<f4")
    if gain != 1.0:
        arr = np.clip(arr * gain, -1.0, 1.0)
    return arr.astype("<f4").tobytes(), new_tail
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-dev/bin/python -m pytest tests/test_audio.py::test_gain_align_scales_and_holds_partial_float tests/test_audio.py::test_pipeline_lead_ms_default -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
.venv-dev/bin/ruff check --fix hosaka/config.py hosaka/audio.py tests/test_audio.py
.venv-dev/bin/ruff format hosaka/config.py hosaka/audio.py tests/test_audio.py
git add hosaka/config.py hosaka/audio.py tests/test_audio.py
git commit -m "feat(audio): add PIPELINE_LEAD_MS and gain/align helper"
```

---

### Task 2: ffplay discovery

**Files:**
- Modify: `hosaka/audio.py`
- Test: `tests/test_audio.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `audio._find_ffplay(probe=..., glob_fn=glob.glob) -> str | None`. Returns the ffplay command/path to run, or None if absent. `probe(name) -> bool` checks whether a command is runnable (default tries `ffplay.exe -version`). `glob_fn(pattern) -> list[str]` defaults to `glob.glob`.

- [ ] **Step 1: Write the failing test**

```python
def test_find_ffplay_on_path():
    from hosaka.audio import _find_ffplay

    found = _find_ffplay(probe=lambda name: name == "ffplay.exe", glob_fn=lambda p: [])
    assert found == "ffplay.exe"


def test_find_ffplay_via_winget_glob():
    from hosaka.audio import _find_ffplay

    fake = "/mnt/c/Users/x/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_1/ffmpeg-8/bin/ffplay.exe"
    found = _find_ffplay(probe=lambda name: False, glob_fn=lambda p: [fake])
    assert found == fake


def test_find_ffplay_absent():
    from hosaka.audio import _find_ffplay

    assert _find_ffplay(probe=lambda name: False, glob_fn=lambda p: []) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-dev/bin/python -m pytest tests/test_audio.py -k find_ffplay -v`
Expected: FAIL (`cannot import name '_find_ffplay'`).

- [ ] **Step 3: Write minimal implementation**

In `hosaka/audio.py`, add `import glob` and `import os` to the imports if missing, then:

```python
_FFPLAY_GLOB = (
    "/mnt/c/Users/*/AppData/Local/Microsoft/WinGet/Packages/"
    "Gyan.FFmpeg*/ffmpeg*/bin/ffplay.exe"
)


def _probe_ffplay(name: str) -> bool:
    try:
        subprocess.run(
            [name, "-version"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _find_ffplay(probe=_probe_ffplay, glob_fn=glob.glob) -> str | None:
    """Locate ffplay: prefer it on PATH, else the winget install location."""
    if probe("ffplay.exe"):
        return "ffplay.exe"
    matches = sorted(glob_fn(_FFPLAY_GLOB))
    return matches[0] if matches else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-dev/bin/python -m pytest tests/test_audio.py -k find_ffplay -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
.venv-dev/bin/ruff check --fix hosaka/audio.py tests/test_audio.py
.venv-dev/bin/ruff format hosaka/audio.py tests/test_audio.py
git add hosaka/audio.py tests/test_audio.py
git commit -m "feat(audio): locate ffplay via PATH or winget install"
```

---

### Task 3: Unify player interface (end_utterance)

**Files:**
- Modify: `hosaka/audio.py` (`PacatPlayer`, `WinSoundPlayer`)
- Test: `tests/test_audio.py`

**Interfaces:**
- Produces: every player gains `write(chunk: bytes)`, `end_utterance() -> None`, and a `play(pcm_bytes: bytes)` convenience (`write` then `end_utterance`). `PacatPlayer.end_utterance` is a no-op (continuous stream). `WinSoundPlayer.write` accumulates; `WinSoundPlayer.end_utterance` plays the accumulated buffer as one WAV (today's `play` body, including lead-in silence).

- [ ] **Step 1: Write the failing test**

```python
def test_winsound_accumulates_then_plays_on_end(tmp_path):
    src = np.full(50, 0.3, dtype=np.float32).astype("<f4").tobytes()
    p = WinSoundPlayer(
        gain=1.0, tmp_dir=tmp_path, runner=lambda a: None,
        to_winpath=lambda s: s, lead_silence_ms=0,
    )
    with p:
        p.write(src[:80])
        p.write(src[80:])
        assert list(tmp_path.glob("*.wav")) == []  # nothing played until end
        p.end_utterance()
    with wave.open(str(next(tmp_path.glob("*.wav"))), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    assert np.allclose(pcm / 32767, 0.3, atol=1e-3)


def test_pacat_end_utterance_is_noop(tmp_path):
    out = tmp_path / "p.raw"
    p = PacatPlayer(cmd=["sh", "-c", f"cat > {out}"], gain=1.0)
    src = np.full(10, 0.5, dtype=np.float32)
    with p:
        p.write(src.astype("<f4").tobytes())
        p.end_utterance()  # must not raise, must not duplicate output
    assert out.stat().st_size == src.nbytes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-dev/bin/python -m pytest tests/test_audio.py -k "accumulates_then_plays or end_utterance_is_noop" -v`
Expected: FAIL (`WinSoundPlayer` plays immediately / no `end_utterance`).

- [ ] **Step 3: Write minimal implementation**

In `hosaka/audio.py`, add to `PacatPlayer` (after `play`):

```python
    def end_utterance(self) -> None:
        pass  # continuous stream; nothing to flush per utterance
```

Refactor `WinSoundPlayer`: rename the current body of `play` to `_play_buffer`, add a `_buf` accumulator, and make `write` accumulate / `end_utterance` flush. Replace the class's `play`/`write` with:

```python
    def __enter__(self):
        self._buf = bytearray()
        return self

    def write(self, pcm_bytes: bytes) -> None:
        self._buf.extend(pcm_bytes)

    def end_utterance(self) -> None:
        if self._buf:
            self._play_buffer(bytes(self._buf))
            self._buf = bytearray()

    def play(self, pcm_bytes: bytes) -> None:
        self.write(pcm_bytes)
        self.end_utterance()

    def _play_buffer(self, pcm_bytes: bytes) -> None:
        pcm16 = np.concatenate([self._lead, _gained_pcm16(pcm_bytes, self.gain)])
        path = self._tmp / f"hosaka_play_{self._n % 3}.wav"
        self._n += 1
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm16.tobytes())
        winpath = self._to_winpath(str(path))
        self._run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                f"(New-Object Media.SoundPlayer '{winpath}').PlaySync()",
            ]
        )
```

Also initialize `self._buf = bytearray()` in `WinSoundPlayer.__init__` (so `play` works without entering the context manager, as existing tests do).

- [ ] **Step 4: Run the full audio suite to verify nothing regressed**

Run: `.venv-dev/bin/python -m pytest tests/test_audio.py -v`
Expected: PASS (including the existing `test_winsound_writes_gained_wav_and_invokes_player`, `test_winsound_prepends_lead_silence` which call `play`).

- [ ] **Step 5: Commit**

```bash
.venv-dev/bin/ruff check --fix hosaka/audio.py tests/test_audio.py
.venv-dev/bin/ruff format hosaka/audio.py tests/test_audio.py
git add hosaka/audio.py tests/test_audio.py
git commit -m "refactor(audio): unify players on write/end_utterance"
```

---

### Task 4: FfplayPlayer

**Files:**
- Modify: `hosaka/audio.py`
- Test: `tests/test_audio.py`

**Interfaces:**
- Consumes: `_gain_align`, `config.PIPELINE_LEAD_MS`, `config.OUTPUT_GAIN`, `config.SAMPLE_RATE`.
- Produces: `audio.FfplayPlayer(ffplay_path: str, gain: float = OUTPUT_GAIN, lead_ms: int = PIPELINE_LEAD_MS, popen=subprocess.Popen)`. Methods: `write(chunk: bytes)`, `end_utterance()`, `play(pcm_bytes)`, `close()`, context manager, attribute `gain`. Withholds gained bytes until `lead_ms` of audio is buffered, then releases and streams continuously; `end_utterance` flushes any held remainder and re-primes for the next utterance. `BrokenPipeError` on write is caught (process marked dead, relaunched on next write).

- [ ] **Step 1: Write the failing test**

```python
class _FakeProc:
    def __init__(self):
        self.stdin = io.BytesIO()
        self.stdin.close = lambda: None  # keep bytes readable after close()
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def wait(self, timeout=None):
        self._alive = False


def test_ffplay_holds_lead_then_streams():
    import io as _io
    procs = []

    def fake_popen(cmd, **kw):
        p = _FakeProc()
        procs.append((cmd, p))
        return p

    # lead_ms 10 at 24kHz f32 = 24000*0.01*4 = 960 bytes
    p = FfplayPlayer("ffplay.exe", gain=1.0, lead_ms=10, popen=fake_popen)
    with p:
        small = np.full(100, 0.5, dtype=np.float32).astype("<f4").tobytes()  # 400 B
        p.write(small)
        assert procs[0][1].stdin.getvalue() == b""  # below lead, withheld
        p.write(small)
        p.write(small)  # now 1200 B >= 960 -> flush
        assert len(procs[0][1].stdin.getvalue()) == 1200
    # launched with the verified raw flags
    assert procs[0][0][:1] == ["ffplay.exe"]
    assert "-ch_layout" in procs[0][0] and "f32le" in procs[0][0]


def test_ffplay_end_utterance_flushes_short_utterance():
    def fake_popen(cmd, **kw):
        return _FakeProc()

    p = FfplayPlayer("ffplay.exe", gain=1.0, lead_ms=10000, popen=fake_popen)
    with p:
        data = np.full(100, 0.5, dtype=np.float32).astype("<f4").tobytes()
        p.write(data)  # below the huge lead -> withheld
        assert p._proc.stdin.getvalue() == b""
        p.end_utterance()  # must flush the remainder
        assert len(p._proc.stdin.getvalue()) == 400


def test_ffplay_broken_pipe_is_handled(capsys):
    class _BrokenProc(_FakeProc):
        def __init__(self):
            super().__init__()
            def boom(_):
                raise BrokenPipeError()
            self.stdin.write = boom

    launched = []

    def fake_popen(cmd, **kw):
        p = _BrokenProc()
        launched.append(p)
        return p

    p = FfplayPlayer("ffplay.exe", gain=1.0, lead_ms=0, popen=fake_popen)
    with p:
        p.write(np.full(8, 0.5, dtype=np.float32).astype("<f4").tobytes())
    assert "ffplay" in capsys.readouterr().out.lower()  # reported, no raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-dev/bin/python -m pytest tests/test_audio.py -k ffplay -v`
Expected: FAIL (`cannot import name 'FfplayPlayer'`). Add `import io` to the test file imports.

- [ ] **Step 3: Write minimal implementation**

In `hosaka/audio.py`:

```python
class FfplayPlayer:
    """Streams float32 PCM into a persistent ffplay.exe on the Windows host.

    ffplay plays natively on Windows (clean, bypassing WSLg's RDP audio) and
    gaplessly, so fragment N plays while the server synthesizes N+1. A
    time-based lead buffer is withheld then released to give ffplay a cushion
    against per-fragment synth jitter at Chatterbox's RTF ~1.
    """

    def __init__(self, ffplay_path, gain=OUTPUT_GAIN, lead_ms=PIPELINE_LEAD_MS,
                 popen=subprocess.Popen):
        self._cmd = [
            ffplay_path, "-hide_banner", "-loglevel", "error", "-nodisp",
            "-autoexit", "-f", "f32le", "-ar", str(SAMPLE_RATE),
            "-ch_layout", "mono", "-i", "pipe:0",
        ]
        self.gain = float(gain)
        self._lead_bytes = SAMPLE_RATE * lead_ms // 1000 * 4
        self._popen = popen
        self._proc = None
        self._tail = b""      # partial-float carry for alignment
        self._hold = bytearray()
        self._primed = False

    def __enter__(self):
        self._launch()
        return self

    def _launch(self):
        self._proc = self._popen(self._cmd, stdin=subprocess.PIPE)
        self._tail = b""
        self._hold = bytearray()
        self._primed = False

    def _feed(self, data):
        if not data:
            return
        if self._proc is None or self._proc.poll() is not None:
            self._launch()
        try:
            self._proc.stdin.write(data)
        except (BrokenPipeError, OSError) as exc:
            print(f"[ffplay closed: {exc}; will relaunch]")
            self._proc = None

    def write(self, chunk):
        out, self._tail = _gain_align(chunk, self._tail, self.gain)
        if not out:
            return
        if self._primed:
            self._feed(out)
            return
        self._hold.extend(out)
        if len(self._hold) >= self._lead_bytes:
            self._feed(bytes(self._hold))
            self._hold = bytearray()
            self._primed = True

    def end_utterance(self):
        if self._hold:
            self._feed(bytes(self._hold))
        self._hold = bytearray()
        self._primed = False

    def play(self, pcm_bytes):
        self.write(pcm_bytes)
        self.end_utterance()

    def close(self):
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            self._proc.wait(timeout=10)
            self._proc = None

    def __exit__(self, *exc):
        self.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-dev/bin/python -m pytest tests/test_audio.py -k ffplay -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
.venv-dev/bin/ruff check --fix hosaka/audio.py tests/test_audio.py
.venv-dev/bin/ruff format hosaka/audio.py tests/test_audio.py
git add hosaka/audio.py tests/test_audio.py
git commit -m "feat(audio): FfplayPlayer streaming player with lead buffer"
```

---

### Task 5: make_player selection

**Files:**
- Modify: `hosaka/audio.py` (`make_player`)
- Test: `tests/test_audio.py`

**Interfaces:**
- Consumes: `on_wslg`, `_find_ffplay`, `FfplayPlayer`, `WinSoundPlayer`, `PacatPlayer`.
- Produces: updated `make_player(gain=OUTPUT_GAIN)` -> on WSLg returns `FfplayPlayer` when ffplay is found, else prints a one-line install hint and returns `WinSoundPlayer`; off WSLg returns `PacatPlayer`.

- [ ] **Step 1: Write the failing test**

```python
def test_make_player_ffplay_when_present(monkeypatch):
    monkeypatch.setattr(audio, "on_wslg", lambda: True)
    monkeypatch.setattr(audio, "_find_ffplay", lambda: "ffplay.exe")
    monkeypatch.setattr(audio.FfplayPlayer, "_launch", lambda self: None)
    assert isinstance(audio.make_player(), audio.FfplayPlayer)


def test_make_player_winsound_when_ffplay_absent(monkeypatch, capsys):
    monkeypatch.setattr(audio, "on_wslg", lambda: True)
    monkeypatch.setattr(audio, "_find_ffplay", lambda: None)
    monkeypatch.setattr(audio, "_win_temp_dir", lambda: __import__("pathlib").Path("/tmp"))
    player = audio.make_player()
    assert isinstance(player, audio.WinSoundPlayer)
    assert "ffmpeg" in capsys.readouterr().out.lower()  # install hint


def test_make_player_pacat_off_wslg(monkeypatch):
    monkeypatch.setattr(audio, "on_wslg", lambda: False)
    assert isinstance(audio.make_player(), audio.PacatPlayer)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-dev/bin/python -m pytest tests/test_audio.py -k make_player -v`
Expected: FAIL (current `make_player` returns `WinSoundPlayer` on WSLg, no ffplay branch).

- [ ] **Step 3: Write minimal implementation**

Replace `make_player` in `hosaka/audio.py`:

```python
def make_player(gain: float = OUTPUT_GAIN):
    """Pick the playback path: ffplay (streaming) on WSLg when present, else the
    buffered Windows player; pacat on native Linux."""
    if not on_wslg():
        return PacatPlayer(gain=gain)
    ffplay = _find_ffplay()
    if ffplay:
        return FfplayPlayer(ffplay, gain=gain)
    print("[no ffmpeg on Windows; using buffered playback. Install ffmpeg "
          "(winget install ffmpeg) for seamless streaming.]")
    return WinSoundPlayer(gain=gain)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-dev/bin/python -m pytest tests/test_audio.py -k make_player -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
.venv-dev/bin/ruff check --fix hosaka/audio.py tests/test_audio.py
.venv-dev/bin/ruff format hosaka/audio.py tests/test_audio.py
git add hosaka/audio.py tests/test_audio.py
git commit -m "feat(audio): select ffplay streaming player on WSLg"
```

---

### Task 6: Stream in _speak

**Files:**
- Modify: `hosaka/cli/repl.py` (`_speak`)
- Test: `tests/test_replcmd.py`

**Interfaces:**
- Consumes: any player with `write(chunk)` / `end_utterance()`.
- Produces: `_speak` writes each received chunk immediately and calls `end_utterance()` exactly once (after the stream completes or errors). No behavior change for callers.

- [ ] **Step 1: Write the failing test**

In `tests/test_replcmd.py`, add (uses a fake player + monkeypatched httpx stream):

```python
def test_speak_streams_chunks_then_ends(monkeypatch):
    import hosaka.cli.repl as repl

    events = []

    class FakePlayer:
        def write(self, chunk):
            events.append(("write", bytes(chunk)))

        def end_utterance(self):
            events.append(("end", None))

    class FakeResp:
        status_code = 200

        def iter_bytes(self):
            yield b"\x00\x00\x00\x00"
            yield b"\x01\x01\x01\x01"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(repl.httpx, "stream", lambda *a, **k: FakeResp())
    repl._speak(FakePlayer(), "kokoro", "af_heart", {}, "hi")

    assert events == [
        ("write", b"\x00\x00\x00\x00"),
        ("write", b"\x01\x01\x01\x01"),
        ("end", None),
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-dev/bin/python -m pytest tests/test_replcmd.py::test_speak_streams_chunks_then_ends -v`
Expected: FAIL (`_speak` buffers and calls `player.play`, not `write`/`end_utterance`).

- [ ] **Step 3: Write minimal implementation**

Replace the body of `_speak` in `hosaka/cli/repl.py`:

```python
def _speak(player, backend, voice, params, text):
    body = {"input": text, "backend": backend, "voice": voice, "params": params, "stream": True}
    try:
        with httpx.stream("POST", f"{SERVER_URL}/v1/audio/speech", json=body, timeout=None) as r:
            if r.status_code != 200:
                print(f"[server {r.status_code}] {r.read().decode(errors='ignore')}")
                return
            for raw in r.iter_bytes():
                if raw:
                    player.write(raw)
    except httpx.HTTPError as exc:
        # A failure inside the engine closes the stream mid-body (status was
        # already 200). Report it and keep the REPL alive instead of crashing.
        print(f"[stream error] {exc}; see /tmp/hosaka-server.log")
    finally:
        player.end_utterance()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-dev/bin/python -m pytest tests/test_replcmd.py::test_speak_streams_chunks_then_ends -v`
Expected: PASS.

- [ ] **Step 5: Run the full non-GPU suite**

Run: `.venv-dev/bin/python -m pytest --ignore=tests/test_engines_gpu.py -q`
Expected: PASS (all tests).

- [ ] **Step 6: Commit**

```bash
.venv-dev/bin/ruff check --fix hosaka/cli/repl.py tests/test_replcmd.py
.venv-dev/bin/ruff format hosaka/cli/repl.py tests/test_replcmd.py
git add hosaka/cli/repl.py tests/test_replcmd.py
git commit -m "feat(repl): stream audio chunks to player, end per utterance"
```

---

### Task 7: End-to-end manual verification (GPU)

**Files:** none (manual check).

- [ ] **Step 1: Restart the server** so it has the latest chunking + code.

```bash
pkill -f "[u]vicorn hosaka" || true
PYTHONPATH=$PWD .venv-server/bin/python -m uvicorn hosaka.server.main:app \
  --host 127.0.0.1 --port 8123 > /tmp/hosaka-server.log 2>&1 &
```

Wait for health: `until curl -sf http://127.0.0.1:8123/health; do sleep 1; done`

- [ ] **Step 2: Launch the REPL and confirm ffplay is selected** (no install-hint line printed):

```bash
.venv-server/bin/python -m hosaka.cli.repl
```

- [ ] **Step 3: Long Chatterbox utterance** — set an angry Chatterbox voice and paste a multi-sentence line; confirm audio starts after ~one fragment and plays without gaps between fragments:

```
:backend c
:voice calm_brit
:exag 87
You were told repeatedly not to touch that. Now look what happened. Get out, and do not come back tonight.
```

Expected: first audio after ~one lead's worth (~1.5-3s), continuous through all sentences. If gaps appear, raise `PIPELINE_LEAD_MS` and retry.

- [ ] **Step 4: Kokoro still clean** — `:voice af_heart` then a sentence; confirm low-latency, clean playback through the same ffplay path.

---

## Self-Review

**Spec coverage:**
- ffplay streaming player + raw flags -> Task 4 (flags asserted in test).
- write/end_utterance unification -> Tasks 3, 6.
- lead buffer (PIPELINE_LEAD_MS) -> Tasks 1, 4.
- gain + 4-byte realignment -> Task 1 (`_gain_align`), used in Task 4.
- discovery (PATH + winget glob) -> Task 2.
- make_player selection (WSLg+ffplay / fallback / native) -> Task 5.
- error handling (BrokenPipe, not-found fallback, stream error) -> Tasks 4, 5, 6.
- Kokoro regains streaming via same player -> Tasks 5, 7.
- testing all in `.venv-dev` -> Tasks 1-6.

**Placeholder scan:** none — every code step has complete code.

**Type consistency:** `write(chunk: bytes)`, `end_utterance()`, `play(pcm_bytes)`, `make_player(gain)`, `_gain_align(chunk, tail, gain) -> (bytes, bytes)`, `_find_ffplay(probe, glob_fn) -> str | None`, `FfplayPlayer(ffplay_path, gain, lead_ms, popen)` consistent across tasks.

## Out of Scope
- Server-side changes (fragment-boundary protocol, batching).
- True parallel GPU synthesis.
