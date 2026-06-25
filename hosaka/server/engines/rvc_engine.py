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

    def __init__(
        self,
        sources,
        sidecar_cmd,
        *,
        voices,
        knobs,
        cwd=None,
        stderr=None,
    ):
        # sources: {backend -> Engine} (e.g. {"kokoro": ..., "chatterbox": ...}).
        # Each RVC voice picks which engine generates its source audio.
        self._sources = dict(sources)
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
        engine = self._sources[cfg.get("source_backend", "kokoro")]
        # A voice may pin its own source params (e.g. Chatterbox exaggeration for
        # an expressive clone); otherwise pass through the request speed (Kokoro).
        src_params = cfg.get("source_params") or {"speed": float(params.get("speed", 1.0))}
        chunks = list(engine.stream(text, cfg["source"], src_params))
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
            passes=int(cfg.get("passes", 1)),
            gate=bool(cfg.get("gate", False)),
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
