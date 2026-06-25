import atexit
import subprocess

import numpy as np

from hosaka.server.engines.piper_proto import (
    PiperProtocolError,
    PiperSidecarError,
    encode_request,
    read_response,
)

# Piper model defaults (glados_piper_medium.onnx.json "inference" block).
DEFAULT_NOISE_SCALE = 0.667
DEFAULT_NOISE_W = 0.8


def speed_to_length_scale(speed: float) -> float:
    """Piper's length_scale is a duration multiplier (higher = slower). Map the
    REPL/schema `speed` knob (higher = faster) to its inverse, clamped to the
    same [0.5, 2.0] range the schema clamps speed to."""
    speed = max(0.5, min(2.0, float(speed)))
    return 1.0 / speed


class PiperEngine:
    """Client for the out-of-process Piper sidecar (.venv-piper).

    The server venv never imports piper; this drives the sidecar over a pipe
    and yields float32 24 kHz PCM, matching the Engine protocol. The sidecar
    keeps the model resident, so first-audio stays ~40-80 ms (ARCHITECTURE.md).
    Piper is CPU-only, so this never touches the GPU; it still routes through
    the server's GPU queue for now (serialized with the GPU backends).

    sidecar_cmd is injected (the prod command points .venv-piper python at
    piper_sidecar.py + the model; tests point at a fake), so the wire path is
    exercised end-to-end without a real model.
    """

    def __init__(
        self,
        sidecar_cmd,
        *,
        voices=None,
        noise_scale: float = DEFAULT_NOISE_SCALE,
        noise_w: float = DEFAULT_NOISE_W,
        cwd=None,
        stderr=None,
    ):
        self._cmd = list(sidecar_cmd)
        self.voice_ids = list(voices) if voices else []  # voices this sidecar serves
        self._noise_scale = noise_scale
        self._noise_w = noise_w
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

    def stream(self, text, voice, params):
        length_scale = speed_to_length_scale(params.get("speed", 1.0))
        req = encode_request(
            text,
            voice=voice,
            length_scale=length_scale,
            noise_scale=self._noise_scale,
            noise_w=self._noise_w,
        )
        proc = self._ensure_proc()
        try:
            proc.stdin.write(req)
            proc.stdin.flush()
            for pcm in read_response(proc.stdout):
                yield np.frombuffer(pcm, dtype="<f4")
        except PiperSidecarError:
            raise  # a per-utterance synth failure; the sidecar is still healthy
        except (PiperProtocolError, OSError):
            # Broken pipe / dead sidecar: drop it so the next call respawns clean.
            self.close()
            raise

    def warmup(self) -> None:
        # Touch every configured voice so each model is resident (loaded + the
        # onnx session warm) before the first real request.
        for v in self.voice_ids or [""]:
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
