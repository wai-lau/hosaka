import wave

import numpy as np

import hosaka.audio as audio
from hosaka.audio import PacatPlayer, WinSoundPlayer


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
    monkeypatch.setattr(audio, "_win_temp_dir", lambda: __import__("pathlib").Path("/tmp"))
    assert isinstance(audio.make_player(), WinSoundPlayer)
    monkeypatch.setattr(audio, "on_wslg", lambda: False)
    assert isinstance(audio.make_player(), PacatPlayer)
