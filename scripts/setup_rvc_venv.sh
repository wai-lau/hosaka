#!/usr/bin/env bash
# Build the isolated RVC venv: GPU voice conversion for converted character
# voices (Charlie and any other RVC .pth). Kept separate from the server venv so
# rvc-python / fairseq / faiss pins can never perturb the Kokoro/Chatterbox
# stack. The server never imports rvc-python; it talks to this venv only
# out-of-process, via the sidecar (hosaka/server/engines/rvc_sidecar.py).
#
# Blackwell sm_120: torch 2.9.1+cu128 goes in FIRST, then rvc-python WITHOUT its
# torch, then its non-torch deps. The exact dep set is pinned by iterating here
# until scripts/verify_rvc.py passes -- that is expected, not a failure.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

PY=python3.12             # CONTINGENCY: if fairseq has no 3.12 wheel, use python3.11
VENV=.venv-rvc
CU128=https://download.pytorch.org/whl/cu128

echo "=== [1/5] create venv ==="
$PY -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel

echo "=== [2/5] torch FIRST from cu128 (Blackwell sm_120) ==="
"$VENV/bin/pip" install --index-url "$CU128" torch==2.9.1+cu128 torchaudio==2.9.1+cu128

echo "=== [3/5] rvc-python WITHOUT its torch (keep ours) ==="
"$VENV/bin/pip" install --no-deps rvc-python

echo "=== [4/5] rvc-python's non-torch deps (pinned; iterate until verify passes) ==="
# Known-needed set; adjust here if verify_rvc.py reports a missing import.
"$VENV/bin/pip" install \
  "faiss-cpu" "librosa" "scipy" "soundfile" "numpy" \
  "praat-parselmouth" "pyworld" "torchcrepe" "fairseq" "omegaconf" "ffmpeg-python"

echo "=== [5/5] verify (real conversion, capability (12,0), RTF) ==="
PYTHONPATH="$PWD" "$VENV/bin/python" scripts/verify_rvc.py

echo "SETUP_RVC_DONE"
