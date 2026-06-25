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
