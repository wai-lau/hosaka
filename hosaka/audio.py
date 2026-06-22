import subprocess
import numpy as np
from hosaka.config import SAMPLE_RATE


class PacatPlayer:
    def __init__(self, cmd: list[str] | None = None, latency_msec: int = 50):
        self.cmd = cmd or [
            "pacat", "--raw",
            f"--rate={SAMPLE_RATE}", "--channels=1",
            "--format=float32le", f"--latency-msec={latency_msec}",
        ]
        self._proc: subprocess.Popen | None = None

    def __enter__(self):
        self._proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, bufsize=0)
        return self

    def write(self, chunk) -> None:
        if isinstance(chunk, np.ndarray):
            chunk = chunk.astype("<f4").tobytes()
        self._proc.stdin.write(chunk)

    def close(self) -> None:
        if self._proc and self._proc.stdin:
            self._proc.stdin.close()
            self._proc.wait(timeout=10)
            self._proc = None

    def __exit__(self, *exc):
        self.close()
