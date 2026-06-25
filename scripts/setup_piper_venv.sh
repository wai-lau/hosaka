#!/usr/bin/env bash
# Build the isolated Piper venv: CPU-only neural TTS for character voices
# (GLaDOS and any other Piper .onnx). Kept separate from the server venv so its
# onnxruntime / numpy / protobuf pins can never perturb the Kokoro/Chatterbox
# stack -- they happen to match today, and isolation keeps it that way. The
# server never imports piper; it talks to this venv only out-of-process, via the
# sidecar (hosaka/server/engines/piper_sidecar.py). No torch, no CUDA.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

PY=python3.12
VENV=.venv-piper

echo "=== [1/3] create venv ==="
$PY -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip

echo "=== [2/3] piper-tts (CPU) + scipy (resample 22050->24000) ==="
"$VENV/bin/pip" install "piper-tts==1.4.2" scipy

echo "=== [3/3] verify import (CPU onnxruntime, no CUDA expected) ==="
"$VENV/bin/python" - <<'PY'
import onnxruntime as ort
import piper  # noqa: F401
import scipy  # noqa: F401

print("piper ok | onnxruntime", ort.__version__, "| EPs", ort.get_available_providers())
PY

echo "SETUP_PIPER_DONE"
