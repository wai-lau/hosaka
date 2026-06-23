#!/usr/bin/env python
"""GLaDOS-character post-processing effect (prototype).

The GLaDOS sound is not a voice clone -- it is an effects chain over a
clean voice (pitch + formant manipulation, per the Melodyne-automation
recipe at github.com/EtiennePerot/gladosvoicegen). This applies a modern,
headless equivalent: formant shift + slight pitch + light ring modulation
+ comb/metallic resonance + bandpass + a touch of reverb.

Iterate by ear. Run on any WAV, tweak the knobs, re-listen. Once a preset
sounds right, lift the chain into hosaka/audio.py as an optional output
filter.

    .venv-server/bin/python scripts/glados_fx.py in.wav out.wav \\
        --formant 1.10 --pitch -1.5 --ring-hz 0 --comb 0.4 --bandpass 1

All stages are individually dialable; set a knob to 0 (or its identity
value) to bypass that stage.
"""

import argparse

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from hosaka.config import SAMPLE_RATE


def _load_mono(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.ascontiguousarray(wav.astype(np.float32))


def formant_shift(wav, ratio):
    """Warp the spectral envelope along frequency to move formants while
    keeping pitch roughly put. ratio > 1 pushes formants up (smaller/
    brighter throat), < 1 down. Identity = 1.0."""
    if abs(ratio - 1.0) < 1e-3:
        return wav
    n_fft = 1024
    hop = n_fft // 4
    stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(stft), np.angle(stft)
    bins = mag.shape[0]
    src = np.arange(bins)
    # Resample the magnitude envelope of each frame along the freq axis.
    warped = np.empty_like(mag)
    tgt = np.clip(src / ratio, 0, bins - 1)
    for i in range(mag.shape[1]):
        warped[:, i] = np.interp(src, tgt, mag[:, i])
    out = librosa.istft(warped * np.exp(1j * phase), hop_length=hop, length=len(wav))
    return out.astype(np.float32)


def pitch_shift(wav, semitones):
    if abs(semitones) < 1e-3:
        return wav
    return librosa.effects.pitch_shift(wav, sr=SAMPLE_RATE, n_steps=semitones).astype(np.float32)


def ring_mod(wav, carrier_hz, mix):
    """Multiply by a sine carrier -- the classic metallic/robotic edge.
    Subtle is GLaDOS; heavy is Dalek. carrier_hz=0 bypasses."""
    if carrier_hz <= 0 or mix <= 0:
        return wav
    t = np.arange(len(wav)) / SAMPLE_RATE
    carrier = np.sin(2 * np.pi * carrier_hz * t).astype(np.float32)
    return ((1 - mix) * wav + mix * wav * carrier).astype(np.float32)


def comb(wav, depth, delay_ms=6.0, feedback=0.5):
    """Short feedback comb -> metallic resonance / robotic 'tube' tone."""
    if depth <= 0:
        return wav
    d = max(1, int(SAMPLE_RATE * delay_ms / 1000.0))
    out = wav.copy()
    for i in range(d, len(out)):
        out[i] += feedback * out[i - d]
    out = out / (np.max(np.abs(out)) + 1e-9)
    return ((1 - depth) * wav + depth * out).astype(np.float32)


def bandpass(wav, lo=300.0, hi=3400.0):
    """Telephone/intercom band -> thins the voice, very GLaDOS-PA."""
    stft = librosa.stft(wav, n_fft=1024, hop_length=256)
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=1024)
    mask = ((freqs >= lo) & (freqs <= hi)).astype(np.float32)[:, None]
    out = librosa.istft(stft * mask, hop_length=256, length=len(wav))
    return out.astype(np.float32)


def reverb(wav, amount, decay=0.3):
    """Tiny exponential-decay reverb tail for the chamber feel."""
    if amount <= 0:
        return wav
    ir_len = int(SAMPLE_RATE * decay)
    ir = np.exp(-np.linspace(0, 6, ir_len)).astype(np.float32)
    ir[0] = 1.0
    wet = fftconvolve(wav, ir)[: len(wav)].astype(np.float32)
    wet /= np.max(np.abs(wet)) + 1e-9
    return ((1 - amount) * wav + amount * wet).astype(np.float32)


def process(wav, args):
    wav = formant_shift(wav, args.formant)
    wav = pitch_shift(wav, args.pitch)
    wav = ring_mod(wav, args.ring_hz, args.ring_mix)
    wav = comb(wav, args.comb)
    if args.bandpass:
        wav = bandpass(wav)
    wav = reverb(wav, args.reverb)
    peak = np.max(np.abs(wav))
    if peak > 0:
        wav = wav / peak * 0.97
    return wav.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="GLaDOS post-processing FX (prototype)")
    p.add_argument("infile")
    p.add_argument("outfile")
    p.add_argument("--formant", type=float, default=1.10, help="formant ratio (1=off)")
    p.add_argument("--pitch", type=float, default=-1.0, help="semitones (0=off)")
    p.add_argument("--ring-hz", type=float, default=0.0, help="ring-mod carrier Hz (0=off)")
    p.add_argument("--ring-mix", type=float, default=0.25, help="ring-mod wet 0..1")
    p.add_argument("--comb", type=float, default=0.35, help="comb depth 0..1 (0=off)")
    p.add_argument("--bandpass", type=int, default=1, help="1=300-3400Hz band, 0=off")
    p.add_argument("--reverb", type=float, default=0.15, help="reverb wet 0..1 (0=off)")
    args = p.parse_args()

    wav = _load_mono(args.infile)
    out = process(wav, args)
    sf.write(args.outfile, out, SAMPLE_RATE, subtype="FLOAT")
    print(f"wrote {args.outfile}  ({len(out) / SAMPLE_RATE:.2f}s)")


if __name__ == "__main__":
    main()
