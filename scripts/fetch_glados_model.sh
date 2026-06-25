#!/usr/bin/env bash
# Fetch the GLaDOS Piper voice into the hosaka data dir.
#
#   glados  DavesArmoury/GLaDOS_TTS (Piper/VITS, fine-tuned on Portal 1/2 lines)
#
# Only the trained weights are redistributable; the Portal training audio is
# Valve copyright and is NOT included. Idempotent: skips files already present.
# (An AIHeaven "high" tier exists but sounds less like GLaDOS -- bigger model,
# generic pipeline -- so it is intentionally not fetched.)
set -euo pipefail

ROOT="${HOME}/.local/share/hosaka/piper"
HF="https://huggingface.co"

# subdir | base url | filename stem
VOICES=(
  "glados|${HF}/DavesArmoury/GLaDOS_TTS/resolve/main|glados_piper_medium"
)

for spec in "${VOICES[@]}"; do
  IFS='|' read -r sub base stem <<<"$spec"
  dest="${ROOT}/${sub}"
  mkdir -p "$dest"
  for ext in onnx onnx.json; do
    f="${stem}.${ext}"
    if [ -f "${dest}/${f}" ]; then
      echo "have    ${sub}/${f}"
      continue
    fi
    echo "fetch   ${sub}/${f}"
    curl -fL "${base}/${f}" -o "${dest}/${f}"
  done
done

echo "FETCH_GLADOS_DONE -> ${ROOT}"
