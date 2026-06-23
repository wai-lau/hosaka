import glob
import subprocess
import wave
from pathlib import Path

import numpy as np

from hosaka.config import (
    OUTPUT_GAIN,
    PIPELINE_LEAD_MS,
    PLAYBACK_LATENCY_MSEC,
    PLAYBACK_LEAD_SILENCE_MS,
    SAMPLE_RATE,
)


def on_wslg() -> bool:
    """True under WSLg, whose RDP audio bridge can't play cleanly (see below)."""
    return Path("/mnt/wslg").exists()


def _gained_pcm16(pcm_bytes: bytes, gain: float) -> np.ndarray:
    arr = np.frombuffer(pcm_bytes, dtype="<f4")
    if gain != 1.0:
        arr = arr * gain
    return (np.clip(arr, -1.0, 1.0) * 32767).astype("<i2")


def _win_temp_dir() -> Path:
    wt = (
        subprocess.check_output(["cmd.exe", "/c", "echo %TEMP%"], stderr=subprocess.DEVNULL)
        .decode()
        .strip()
    )
    return Path(subprocess.check_output(["wslpath", "-u", wt]).decode().strip())


def _wslpath_w(p: str) -> str:
    return subprocess.check_output(["wslpath", "-w", p]).decode().strip()


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


_FFPLAY_GLOB = (
    "/mnt/c/Users/*/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg*/ffmpeg*/bin/ffplay.exe"
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


class PacatPlayer:
    """Streams raw PCM to PulseAudio via pacat. Correct for native Linux audio."""

    def __init__(
        self,
        cmd: list[str] | None = None,
        latency_msec: int = PLAYBACK_LATENCY_MSEC,
        gain: float = OUTPUT_GAIN,
    ):
        self.cmd = cmd or [
            "pacat",
            "--raw",
            f"--rate={SAMPLE_RATE}",
            "--channels=1",
            "--format=float32le",
            f"--latency-msec={latency_msec}",
        ]
        self.gain = float(gain)
        self._proc: subprocess.Popen | None = None
        self._tail = b""  # carries a partial float across chunk boundaries

    def __enter__(self):
        self._proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, bufsize=0)
        return self

    def _apply_gain(self, arr: np.ndarray) -> np.ndarray:
        if self.gain == 1.0:
            return arr
        return np.clip(arr * self.gain, -1.0, 1.0)

    def write(self, chunk) -> None:
        if isinstance(chunk, np.ndarray):
            data = self._apply_gain(chunk.astype(np.float32))
            self._proc.stdin.write(data.astype("<f4").tobytes())
            return
        # Raw bytes off the network may split a 4-byte float; only process a
        # 4-aligned prefix and hold the remainder for the next write.
        buf = self._tail + chunk
        n = len(buf) - (len(buf) % 4)
        aligned, self._tail = buf[:n], buf[n:]
        if aligned:
            arr = np.frombuffer(aligned, dtype="<f4")
            self._proc.stdin.write(self._apply_gain(arr).astype("<f4").tobytes())

    def play(self, pcm_bytes: bytes) -> None:
        """Play one complete utterance (uniform interface with WinSoundPlayer)."""
        self.write(pcm_bytes)

    def end_utterance(self) -> None:
        pass  # continuous stream; nothing to flush per utterance

    def close(self) -> None:
        if self._proc and self._proc.stdin:
            if self._tail:  # flush any trailing partial bytes unmodified
                self._proc.stdin.write(self._tail)
                self._tail = b""
            self._proc.stdin.close()
            self._proc.wait(timeout=10)
            self._proc = None

    def __exit__(self, *exc):
        self.close()


class WinSoundPlayer:
    """Plays full utterances on the Windows host, bypassing WSLg/PulseAudio.

    WSLg's RDP audio bridge intermittently adds static to ALL WSL playback
    (pacat and paplay alike, any format) and degrades over a session. Writing a
    WAV to a Windows-visible temp dir and playing it with the native
    SoundPlayer is reliably clean -- and tolerates output gain, which the RDP
    path does not. No streaming: each utterance plays whole after synthesis.
    """

    def __init__(
        self,
        gain: float = OUTPUT_GAIN,
        tmp_dir=None,
        runner=None,
        to_winpath=None,
        lead_silence_ms: int = PLAYBACK_LEAD_SILENCE_MS,
    ):
        self.gain = float(gain)
        self._tmp = Path(tmp_dir) if tmp_dir else _win_temp_dir()
        self._run = runner or (lambda args: subprocess.run(args, check=False))
        self._to_winpath = to_winpath or _wslpath_w
        self._lead = np.zeros(SAMPLE_RATE * lead_silence_ms // 1000, dtype="<i2")
        self._n = 0
        self._buf = bytearray()

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

    def close(self) -> None:
        pass

    def __exit__(self, *exc):
        self.close()


class FfplayPlayer:
    """Streams float32 PCM into a persistent ffplay.exe on the Windows host.

    ffplay plays natively on Windows (clean, bypassing WSLg's RDP audio) and
    gaplessly, so fragment N plays while the server synthesizes N+1. A
    time-based lead buffer is withheld then released to give ffplay a cushion
    against per-fragment synth jitter at Chatterbox's RTF ~1.
    """

    def __init__(
        self,
        ffplay_path,
        gain=OUTPUT_GAIN,
        lead_ms=PIPELINE_LEAD_MS,
        popen=subprocess.Popen,
    ):
        self._cmd = [
            ffplay_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nodisp",
            "-autoexit",
            "-f",
            "f32le",
            "-ar",
            str(SAMPLE_RATE),
            "-ch_layout",
            "mono",
            "-i",
            "pipe:0",
        ]
        self.gain = float(gain)
        self._lead_bytes = SAMPLE_RATE * lead_ms // 1000 * 4
        self._popen = popen
        self._proc = None
        self._tail = b""  # partial-float carry for alignment
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
        self._tail = b""  # don't bleed a partial float into the next utterance
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


def make_player(gain: float = OUTPUT_GAIN):
    """Pick the playback path: ffplay (streaming) on WSLg when present, else the
    buffered Windows player; pacat on native Linux."""
    if not on_wslg():
        return PacatPlayer(gain=gain)
    ffplay = _find_ffplay()
    if ffplay:
        return FfplayPlayer(ffplay, gain=gain)
    print(
        "[no ffmpeg on Windows; using buffered playback. Install ffmpeg "
        "(winget install ffmpeg) for seamless streaming.]"
    )
    return WinSoundPlayer(gain=gain)
