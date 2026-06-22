#!/usr/bin/env bash
# Build the hosaka server venv (Kokoro + Chatterbox) for Blackwell sm_120.
# torch from cu128 index FIRST, TTS packages --no-deps, then re-add deps.
# Logs everything; no sudo here (apt step is run separately).
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

PY=python3.12
VENV=.venv-server

echo "=== [1/6] create venv ==="
$PY -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip

echo "=== [2/6] torch cu128 (Blackwell) FIRST ==="
"$VENV/bin/pip" install \
  torch==2.9.1+cu128 torchaudio==2.9.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

echo "=== [3/6] torchcodec ==="
"$VENV/bin/pip" install torchcodec

echo "=== [4/6] Chatterbox streaming fork --no-deps + its deps ==="
"$VENV/bin/pip" install \
  "git+https://github.com/davidbrowne17/chatterbox-streaming.git" --no-deps
"$VENV/bin/pip" install transformers accelerate scipy numpy peft soundfile

echo "=== [5/6] Kokoro --no-deps + G2P deps (misaki) ==="
"$VENV/bin/pip" install kokoro --no-deps
"$VENV/bin/pip" install "misaki[en]" huggingface_hub

echo "=== [6/6] server deps ==="
"$VENV/bin/pip" install fastapi uvicorn httpx pydantic soundfile numpy

echo "=== verify torch sees the GPU (real matmul) ==="
"$VENV/bin/python" scripts/verify_gpu.py

echo "SETUP_SERVER_DONE"
