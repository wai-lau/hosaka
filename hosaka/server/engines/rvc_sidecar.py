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

  usage: rvc_sidecar.py --voice charlie=/path/charlie.pth:/path/charlie.index

rvc-python self-manages the hubert/rmvpe assets in its own base_model dir
(pre-seeded by scripts/setup_rvc_venv.sh from the fetched models), so no asset
paths are passed here.
"""

import argparse
import os
import subprocess
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


def load_voices(specs):
    """specs: ["id=pth:index", ...] -> {id: RVCInference}.

    One RVCInference per voice keeps each model resident on the GPU. rvc-python
    locates the hubert/rmvpe assets in its own base_model dir (pre-seeded by
    setup_rvc_venv.sh from the fetched models), so no asset paths are needed here.
    """
    from rvc_python.infer import RVCInference

    voices = {}
    for spec in specs:
        vid, paths = spec.split("=", 1)
        pth, index = paths.split(":", 1)
        rvc = RVCInference(device="cuda:0")
        rvc.load_model(pth, index_path=index)
        voices[vid] = rvc
    return voices


def _silence_gate(source, converted):
    """Mute the converted audio wherever the SOURCE is silent. RVC hallucinates
    phonemes in silent gaps (HuBERT/F0 have nothing real there), but the source's
    silence marks where there should be none. Both are float32 @ SAMPLE_RATE and
    RVC preserves timing, so they align frame-for-frame."""
    n = min(len(source), len(converted))
    if n == 0:
        return converted
    src, conv = source[:n], converted[:n]
    fl = int(0.025 * SAMPLE_RATE)
    hop = max(1, fl // 2)
    nf = 1 + max(0, (n - fl) // hop)
    env = np.array(
        [np.sqrt(np.mean(src[i * hop : i * hop + fl] ** 2)) for i in range(nf)],
        dtype=np.float32,
    )
    if env.max() <= 0:
        return conv
    mask_f = (env > 0.03 * env.max()).astype(np.float32)
    mask_f = np.clip(np.convolve(mask_f, np.ones(5) / 5, mode="same"), 0, 1)  # smooth -> no clicks
    mask = np.clip(np.interp(np.arange(n), np.arange(nf) * hop, mask_f), 0, 1).astype(np.float32)
    return conv * mask


def _time_stretch(wav, speed):
    """Tempo change only (keeps pitch -- Chatterbox has no speed knob). Uses
    ffmpeg's atempo (WSOLA), which is far cleaner on voice than a phase vocoder
    (librosa.effects.time_stretch smeared/colored the sound -- chosen by A/B).
    atempo spans 0.5-2.0 in a single pass, covering the speed knob's range.
    Pipes raw float32 LE in and out; ffmpeg is already on the box, so the sidecar
    needs no python time-stretch dependency."""
    raw = np.ascontiguousarray(wav, dtype="<f4").tobytes()
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-af",
            f"atempo={speed:.6f}",
            "-f",
            "f32le",
            "pipe:1",
        ],
        input=raw,
        stdout=subprocess.PIPE,
        check=True,
    )
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)


def convert(rvc, req, out):
    """Convert one request's source PCM to the target and stream it back.

    Runs `passes` RVC passes in series (each pass's output feeds the next) -- a
    2nd pass locks the timbre harder onto the target. The transpose is applied on
    the FIRST pass ONLY; later passes use f0up_key 0, since re-shifting every pass
    would compound the pitch.
    """
    passes = max(1, int(req.get("passes", 1)))
    src = np.frombuffer(req["pcm"], dtype="<f4")
    with tempfile.TemporaryDirectory() as d:
        cur = str(Path(d) / "p_in.wav")
        sf.write(cur, src, SAMPLE_RATE, subtype="FLOAT")
        for i in range(passes):
            rvc.set_params(
                f0method=req["f0_method"],
                f0up_key=req["transpose"] if i == 0 else 0,
                index_rate=req["index_rate"],
                protect=req["protect"],
                rms_mix_rate=req["rms_mix_rate"],
            )
            nxt = str(Path(d) / f"p{i}.wav")
            rvc.infer_file(cur, nxt)  # writes at the model's native SR
            cur = nxt
        wav, sr = sf.read(cur, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        wav = resample_poly(wav, SAMPLE_RATE, sr)
    wav = wav.astype(np.float32)
    if req.get("gate"):
        wav = _silence_gate(src, wav)  # src is the source PCM @ 24k, read above
    speed = float(req.get("speed", 1.0))
    if abs(speed - 1.0) > 1e-3:
        wav = _time_stretch(wav, speed)
    f32 = np.ascontiguousarray(wav, dtype="<f4")
    out.write(pack_audio(f32.tobytes()))
    out.flush()


def main():
    # rvc-python and fairseq are chatty on stdout (print() + INFO logging), but
    # fd 1 is our binary protocol pipe to the engine. Save the real stdout for
    # the frames, then redirect fd 1 -> stderr so nothing a library prints can
    # corrupt the frame stream. (Piper's sidecar was silent, so it never needed
    # this.) All library noise then flows to stderr.
    protocol_out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", action="append", default=[], metavar="id=pth:index")
    args = ap.parse_args()
    voices = load_voices(args.voice)

    out = protocol_out
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
