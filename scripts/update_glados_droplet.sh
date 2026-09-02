#!/usr/bin/env bash
# Redeploy the always-on droplet glados container after a hosaka CODE change
# reaches origin/main (e.g. normalize.py, main_piper.py, Dockerfile.piper).
#
# Pulls /hosaka, rebuilds the hosaka-piper image, recreates the service. Unlike
# the pronunciation lexicon (DATA -- synced live by scripts/sync_lexicon.sh and
# the REPL :pron command), code is baked into the image, so it only reaches the
# droplet on a rebuild. Run this after pushing such a change. Idempotent.
#
# Usage: scripts/update_glados_droplet.sh   (HOSAKA_DROPLET overrides the host)
set -euo pipefail

DEST_HOST="${HOSAKA_DROPLET:-wai-root@wai-lau.net}"

# wai-root owns /hosaka (plain git pull) but is not in the docker group: docker
# goes through its passwordless sudo.
ssh "$DEST_HOST" 'set -e
  cd /hosaka && git pull --ff-only
  sudo docker build -f Dockerfile.piper -t hosaka-piper:latest .
  cd /exec-fn && sudo docker compose up -d --force-recreate hosaka-piper
  printf "hosaka-piper redeployed: %s\n" "$(sudo docker inspect -f "{{.State.Status}}" exec-fn-hosaka-piper-1)"'
echo "done"
