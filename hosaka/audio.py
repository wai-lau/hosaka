import subprocess
import wave
from pathlib import Path

import numpy as np

from hosaka.config import (
    OUTPUT_GAIN,
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

    def __enter__(self):
        return self

    def play(self, pcm_bytes: bytes) -> None:
        pcm16 = np.concatenate([self._lead, _gained_pcm16(pcm_bytes, self.gain)])
        path = self._tmp / f"hosaka_play_{self._n % 3}.wav"  # rotate a few files
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

    # Accept incremental writes too, so it can stand in for PacatPlayer; here a
    # write IS a full utterance (the caller buffers before calling).
    def write(self, pcm_bytes: bytes) -> None:
        self.play(pcm_bytes)

    def close(self) -> None:
        pass

    def __exit__(self, *exc):
        self.close()


def make_player(gain: float = OUTPUT_GAIN):
    """Pick the playback path: Windows host under WSLg, else pacat."""
    return WinSoundPlayer(gain=gain) if on_wslg() else PacatPlayer(gain=gain)
