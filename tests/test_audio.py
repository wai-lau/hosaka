import io
import wave

import numpy as np

import hosaka.audio as audio
from hosaka.audio import (
    FfplayPlayer,
    PacatPlayer,
    WinSoundPlayer,
)


def test_player_writes_bytes_to_subprocess(tmp_path):
    out = tmp_path / "out.raw"
    # Fake player: shell that copies stdin to a file.
    cmd = ["sh", "-c", f"cat > {out}"]
    data = np.ones(100, dtype=np.float32)
    with PacatPlayer(cmd=cmd) as p:
        p.write(data)
    assert out.exists()
    assert out.stat().st_size == data.nbytes


def test_default_cmd_targets_pacat():
    p = PacatPlayer()
    assert p.cmd[0] == "pacat"
    assert "--rate=24000" in p.cmd
    assert "--format=float32le" in p.cmd


def _roundtrip(tmp_path, cmd_out, writes, gain):
    p = PacatPlayer(cmd=["sh", "-c", f"cat > {cmd_out}"], gain=gain)
    with p:
        for w in writes:
            p.write(w)
    return np.frombuffer(cmd_out.read_bytes(), dtype="<f4")


def test_gain_scales_numpy_output(tmp_path):
    out = tmp_path / "g.raw"
    data = np.full(50, 0.2, dtype=np.float32)
    res = _roundtrip(tmp_path, out, [data], gain=2.5)
    assert np.allclose(res, 0.5, atol=1e-6)


def test_gain_clips_to_unit_range(tmp_path):
    out = tmp_path / "c.raw"
    data = np.array([0.8, -0.9, 0.1], dtype=np.float32)
    res = _roundtrip(tmp_path, out, [data], gain=3.0)
    assert res.max() <= 1.0 and res.min() >= -1.0
    assert res[0] == 1.0 and res[1] == -1.0


def test_byte_writes_realign_across_chunk_boundaries(tmp_path):
    out = tmp_path / "a.raw"
    src = np.linspace(-0.3, 0.3, 16, dtype=np.float32)
    raw = src.astype("<f4").tobytes()  # 64 bytes
    # Split mid-float so each write is not 4-aligned (10 + 54 bytes).
    res = _roundtrip(tmp_path, out, [raw[:10], raw[10:]], gain=1.0)
    assert np.allclose(res, src, atol=1e-6)


def test_winsound_writes_gained_wav_and_invokes_player(tmp_path):
    calls = []
    src = np.full(100, 0.2, dtype=np.float32)
    p = WinSoundPlayer(
        gain=2.0,
        tmp_dir=tmp_path,
        runner=calls.append,
        to_winpath=lambda s: s,
        lead_silence_ms=0,
    )
    with p:
        p.play(src.astype("<f4").tobytes())
    wavs = list(tmp_path.glob("hosaka_play_*.wav"))
    assert len(wavs) == 1
    with wave.open(str(wavs[0]), "rb") as w:
        assert w.getframerate() == 24000
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    assert np.allclose(pcm / 32767, 0.4, atol=1e-3)  # 0.2 * gain 2.0
    assert calls and "SoundPlayer" in calls[0][-1]


def test_winsound_prepends_lead_silence(tmp_path):
    src = np.full(100, 0.5, dtype=np.float32)
    p = WinSoundPlayer(
        gain=1.0,
        tmp_dir=tmp_path,
        runner=lambda a: None,
        to_winpath=lambda s: s,
        lead_silence_ms=100,
    )
    with p:
        p.play(src.astype("<f4").tobytes())
    with wave.open(str(next(tmp_path.glob("*.wav"))), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    lead = 24000 * 100 // 1000
    assert pcm.size == lead + 100
    assert (pcm[:lead] == 0).all()  # silence first
    assert np.allclose(pcm[lead:] / 32767, 0.5, atol=1e-3)


def test_make_player_selects_by_environment(monkeypatch):
    monkeypatch.setattr(audio, "on_wslg", lambda: True)
    monkeypatch.setattr(audio, "_find_ffplay", lambda: None)
    monkeypatch.setattr(audio, "_win_temp_dir", lambda: __import__("pathlib").Path("/tmp"))
    assert isinstance(audio.make_player(), WinSoundPlayer)
    monkeypatch.setattr(audio, "on_wslg", lambda: False)
    assert isinstance(audio.make_player(), PacatPlayer)


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


def test_find_ffplay_on_path():
    from hosaka.audio import _find_ffplay

    found = _find_ffplay(probe=lambda name: name == "ffplay.exe", glob_fn=lambda p: [])
    assert found == "ffplay.exe"


def test_find_ffplay_via_winget_glob():
    from hosaka.audio import _find_ffplay

    fake = (
        "/mnt/c/Users/x/AppData/Local/Microsoft/WinGet/Packages/"
        "Gyan.FFmpeg_1/ffmpeg-8/bin/ffplay.exe"
    )
    found = _find_ffplay(probe=lambda name: False, glob_fn=lambda p: [fake])
    assert found == fake


def test_find_ffplay_absent():
    from hosaka.audio import _find_ffplay

    assert _find_ffplay(probe=lambda name: False, glob_fn=lambda p: []) is None


def test_winsound_accumulates_then_plays_on_end(tmp_path):
    src = np.full(50, 0.3, dtype=np.float32).astype("<f4").tobytes()
    p = WinSoundPlayer(
        gain=1.0,
        tmp_dir=tmp_path,
        runner=lambda a: None,
        to_winpath=lambda s: s,
        lead_silence_ms=0,
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


def test_ffplay_relaunches_after_broken_pipe():
    # First process's stdin raises once (broken pipe); a later write must
    # relaunch a fresh process and deliver bytes to it.
    class _OneShotBrokenProc(_FakeProc):
        def __init__(self):
            super().__init__()

            def boom(_):
                raise BrokenPipeError()

            self.stdin.write = boom

    procs = []

    def fake_popen(cmd, **kw):
        # first call -> broken proc, subsequent calls -> healthy procs
        p = _OneShotBrokenProc() if not procs else _FakeProc()
        procs.append(p)
        return p

    data = np.full(8, 0.5, dtype=np.float32).astype("<f4").tobytes()
    p = FfplayPlayer("ffplay.exe", gain=1.0, lead_ms=0, popen=fake_popen)
    with p:
        p.write(data)  # broken pipe -> _proc set to None
        p.write(data)  # must relaunch and deliver
    assert len(procs) == 2  # a second process was launched
    assert procs[1].stdin.getvalue() == data  # bytes delivered to the new proc


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
