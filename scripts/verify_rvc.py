#!/usr/bin/env python
"""Verify the .venv-rvc GPU conversion path end to end (no server).

Real matmul capability check (never trust torch.cuda.is_available()), then load
Charlie and convert ~2s of a generated source tone, asserting non-silent finite
output and printing RTF. Run: PYTHONPATH=$PWD .venv-rvc/bin/python scripts/verify_rvc.py
"""

import time

import numpy as np
import soundfile as sf
import torch

from hosaka.config import RVC_HUBERT, RVC_RMVPE, RVC_VOICES, SAMPLE_RATE


def _check_gpu():
    cap = torch.cuda.get_device_capability()
    assert cap == (12, 0), f"expected Blackwell sm_120 (12, 0), got {cap}"
    x = torch.randn(256, 256, device="cuda")
    y = x @ x
    assert torch.isfinite(y).all(), "GPU matmul produced non-finite values"
    print(f"GPU ok: capability {cap}, real matmul finite")


def _source_wav(path, seconds=2.0):
    # A simple voiced-ish tone so F0 has something to track.
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    wav = (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    sf.write(path, wav, SAMPLE_RATE, subtype="FLOAT")
    return seconds


def main():
    import tempfile
    from pathlib import Path

    from rvc_python.infer import RVCInference  # noqa: F401  (verify the import)

    _check_gpu()
    assert RVC_HUBERT.exists(), f"missing {RVC_HUBERT}"
    assert RVC_RMVPE.exists(), f"missing {RVC_RMVPE}"
    spec = RVC_VOICES["charlie"]
    assert spec["model"].exists() and spec["index"].exists(), "fetch Charlie first"

    rvc = RVCInference(device="cuda:0")
    rvc.load_model(str(spec["model"]), index_path=str(spec["index"]))
    rvc.set_params(
        f0method="rmvpe",
        f0up_key=spec["transpose"],
        index_rate=0.5,
        protect=0.33,
        rms_mix_rate=0.25,
    )

    with tempfile.TemporaryDirectory() as d:
        src = str(Path(d) / "src.wav")
        out = str(Path(d) / "out.wav")
        seconds = _source_wav(src)
        t0 = time.perf_counter()
        rvc.infer_file(src, out)
        dt = time.perf_counter() - t0
        wav, sr = sf.read(out, dtype="float32")

    assert np.isfinite(wav).all() and np.abs(wav).max() > 1e-3, "silent / non-finite output"
    print(f"convert ok: {seconds:.1f}s @ model {sr} Hz, {dt:.2f}s wall, RTF {dt / seconds:.2f}")
    print("VERIFY_RVC_DONE")


if __name__ == "__main__":
    main()
