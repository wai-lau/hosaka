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
exec .venv-server/bin/python -m uvicorn hosaka.server.main:app \
  --host 127.0.0.1 --port 8123
