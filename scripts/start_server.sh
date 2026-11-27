#!/usr/bin/env bash
# Launch the hosaka TTS server (Kokoro + Chatterbox on the GPU) on
# 127.0.0.1:8123. This is the canonical start command.
#
# It runs on WSL startup via the systemd *user* unit hosaka-server.service,
# which execs this script; `loginctl enable-linger` makes that unit come up at
# boot without an interactive login. Run it by hand the same way:
#
#     bash scripts/start_server.sh
#
# exec (not a child) so systemd tracks the python process directly for clean
# Restart=on-failure handling.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=qwen_lock.sh
. scripts/qwen_lock.sh
# Qwen wins the 5070 Ti, but nicole fits beside it: Kokoro takes ~0.9 GB on the
# GTX 1660 and Qwen leaves ~2.8 GB free there (measured 2026-11-27). So while
# Qwen holds the GPUs start nicole only: no Charlie, and no GLaDOS either (the
# droplet serves GLaDOS, so the home copy would only take CPU from Qwen).
if ! refuse_if_qwen_holds_gpu 2>/dev/null; then
  echo "qwen holds the GPUs: starting nicole only (no Charlie, no GLaDOS)" >&2
  export HOSAKA_CHARLIE=0 HOSAKA_GLADOS=0
fi
# Every model (Kokoro, Chatterbox) is already in the HF cache. Online, Kokoro's
# KPipeline spends ~80 s on hub round-trips before it builds (measured
# 2026-11-25: 85 s online vs 4 s offline), which pushed a nicole-only start
# past the droplet's 90 s switch window. A new HF model needs one run with
# HF_HUB_OFFLINE=0 to fetch it.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
exec .venv-server/bin/python -m uvicorn hosaka.server.main:app \
  --host 127.0.0.1 --port 8123
