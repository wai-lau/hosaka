#!/usr/bin/env bash
# Launch the always-on gpu-mode service on 127.0.0.1:8124 under .venv-dev
# (GPU-free). Reads GPU_MODE_TOKEN from the environment (set by the systemd unit
# via EnvironmentFile). Run by hand the same way: bash scripts/start_gpu_mode.sh
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv-dev/bin/python -m uvicorn hosaka.server.main_gpu_mode:app \
  --host 127.0.0.1 --port 8124
