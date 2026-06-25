#!/usr/bin/env python
"""Piper TTS sidecar -- runs ONLY under .venv-piper (it imports piper / onnx /
scipy). The server never imports this module; PiperEngine spawns it and speaks
the piper_proto pipe protocol to it.

Loads each --voice model once and keeps it resident (CPU). For every JSON
request it synthesizes per sentence, resamples each chunk to the hosaka sample
rate, and streams framed float32 LE PCM back. A per-request failure becomes an
error frame and the sidecar stays alive for the next request.

  usage: piper_sidecar.py --voice glados=/path/medium.onnx \
                          --voice glados_high=/path/high.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from piper import PiperVoice
from piper.config import SynthesisConfig
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from hosaka.config import SAMPLE_RATE  # noqa: E402
from hosaka.server.engines.piper_proto import (  # noqa: E402
    pack_audio,
    pack_end,
    pack_error,
    parse_request,
)


def load_voices(specs):
    """specs: ["id=path", ...] -> ({id: PiperVoice}, {id: sample_rate})."""
    voices, rates = {}, {}
    for spec in specs:
        vid, path = spec.split("=", 1)
        v = PiperVoice.load(path, use_cuda=False)  # finds the .json beside it
        voices[vid] = v
        rates[vid] = int(v.config.sample_rate)
    return voices, rates


def synthesize(voice, rate, req, out):
    cfg = SynthesisConfig(
        length_scale=req["length_scale"],
        noise_scale=req["noise_scale"],
        noise_w_scale=req["noise_w"],
    )
    for chunk in voice.synthesize(req["text"], cfg):
        f32 = chunk.audio_int16_array.astype(np.float32) / 32768.0
        if rate != SAMPLE_RATE:
            f32 = resample_poly(f32, SAMPLE_RATE, rate).astype(np.float32)
        out.write(pack_audio(np.ascontiguousarray(f32, dtype="<f4").tobytes()))
        out.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--voice",
        action="append",
        default=[],
        metavar="id=path",
        help="voice id and onnx model path; repeatable",
    )
    voices, rates = load_voices(ap.parse_args().voice)

    out = sys.stdout.buffer
    for line in sys.stdin.buffer:
        if not line.strip():
            continue
        try:
            req = parse_request(line)
            vid = req.get("voice", "")
            if vid not in voices:
                if len(voices) == 1:  # single-voice config: ignore the label
                    vid = next(iter(voices))
                else:
                    raise KeyError(f"unknown piper voice: {vid!r}")
            synthesize(voices[vid], rates[vid], req, out)
            out.write(pack_end())
            out.flush()
        except Exception as exc:  # stay alive; report THIS request's failure
            out.write(pack_error(f"{type(exc).__name__}: {exc}"))
            out.flush()


if __name__ == "__main__":
    main()
