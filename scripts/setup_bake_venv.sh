#!/usr/bin/env bash
# Build the isolated Parler bake venv. Separate from the server venv so its
# pinned transformers==4.46.1 cannot poison Kokoro/Chatterbox. Offline use only.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

PY=python3.12
VENV=.venv-bake

echo "=== [1/4] create venv ==="
$PY -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip

echo "=== [2/4] torch cu128 FIRST (cached wheels) ==="
"$VENV/bin/pip" install \
  torch==2.9.1+cu128 torchaudio==2.9.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

echo "=== [3/4] parler-tts --no-deps ==="
"$VENV/bin/pip" install parler-tts --no-deps

echo "=== [4/4] parler deps (pinned transformers) ==="
"$VENV/bin/pip" install \
  transformers==4.46.1 descript-audio-codec soundfile numpy scipy sentencepiece
# sentencepiece's generated protobuf needs the `builder` module (protobuf>=3.20);
# the default resolve drags in 3.19.6 via descript-audiotools. parler-tts itself
# wants protobuf>=4.0.0. Force it; audiotools only uses protobuf for training
# logging, not the bake decode path.
"$VENV/bin/pip" install "protobuf>=4.0.0,<5"

echo "=== verify torch GPU in bake venv ==="
"$VENV/bin/python" scripts/verify_gpu.py

echo "=== import parler (the real Risk 3 check) ==="
"$VENV/bin/python" -c "from parler_tts import ParlerTTSForConditionalGeneration; from transformers import AutoTokenizer; import transformers; print('parler import ok, transformers', transformers.__version__)"

echo "SETUP_BAKE_DONE"
