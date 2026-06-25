#!/usr/bin/env bash
# Build the isolated RVC venv: GPU voice conversion for converted character
# voices (Charlie and any other RVC .pth). Kept separate from the server venv so
# rvc-python / fairseq / faiss pins can never perturb the Kokoro/Chatterbox
# stack. The server never imports rvc-python; it talks to this venv only
# out-of-process, via the sidecar (hosaka/server/engines/rvc_sidecar.py).
#
# WHY python3.10 (not the system 3.12): fairseq 0.12.2 and its pinned
# omegaconf==2.0.6 / hydra-core==1.0.7 use the pre-3.11 mutable-default dataclass
# pattern, which Python 3.11+ rejects -- and patching it cascades into
# omegaconf's structured-config validation. py3.10 runs the stack as shipped.
# uv fetches a prebuilt standalone py3.10 (dev headers included, no sudo, no
# compile), so fairseq's C extension still builds.
#
# WHY fairseq from git (not the PyPI 0.12.2 wheel): the wheel imports
# torch._six, removed in torch 2.0; the Blackwell sm_120 card requires
# torch 2.9/cu128. Building from git avoids the torch._six import.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

VENV=.venv-rvc
PYBIN="$VENV/bin/python"
CU128=https://download.pytorch.org/whl/cu128
pip() { uv pip install --python "$PYBIN" "$@"; }

echo "=== [1/7] prebuilt python3.10 + venv (via uv) ==="
uv python install 3.10
rm -rf "$VENV"
uv venv --python 3.10 "$VENV"

echo "=== [2/7] torch FIRST from cu128 (Blackwell sm_120) ==="
pip --index-url "$CU128" torch==2.9.1+cu128 torchaudio==2.9.1+cu128

echo "=== [3/7] RVC audio + F0 deps ==="
# pyworld is REQUIRED: rvc-python's vc/pipeline.py imports it at module load
# (even on the rmvpe path). On py3.10 it has a wheel; no build needed.
pip numpy==1.26.4 scipy soundfile faiss-cpu librosa torchcrepe praat-parselmouth pyworld av ffmpeg-python loguru tqdm

echo "=== [4/7] rvc-python inference code only (its pins are torch-incompatible) ==="
pip --no-deps rvc-python

echo "=== [5/7] fairseq from git (hubert loader); legacy deps + no build isolation ==="
# setuptools/wheel: uv venvs are minimal; --no-build-isolation needs them to run
# fairseq's setup.py build backend. Pin <81 -- setuptools 81 removed
# pkg_resources, which the (old) pyworld imports at load.
pip "setuptools<81" wheel cython cffi regex bitarray sacrebleu "omegaconf==2.0.6" "hydra-core==1.0.7"
pip --no-deps --no-build-isolation "fairseq @ git+https://github.com/facebookresearch/fairseq.git"

echo "=== [6/7] pre-seed rvc-python asset dir from our fetched hubert/rmvpe ==="
SP="$("$PYBIN" -c 'import os, rvc_python; print(os.path.dirname(rvc_python.__file__))')"
mkdir -p "$SP/base_model"
ln -sf "$HOME/.local/share/hosaka/rvc/hubert_base.pt" "$SP/base_model/hubert_base.pt"
ln -sf "$HOME/.local/share/hosaka/rvc/rmvpe.pt" "$SP/base_model/rmvpe.pt"

echo "=== [7/7] verify (real conversion, capability (12,0), RTF) ==="
PYTHONPATH="$PWD" "$PYBIN" scripts/verify_rvc.py

echo "SETUP_RVC_DONE"
