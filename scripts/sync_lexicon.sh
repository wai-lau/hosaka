#!/usr/bin/env bash
# Push the home-box pronunciation lexicon to the droplet so the always-on
# hosaka-piper (glados) applies the same custom pronunciations as the home box.
#
# rsync --inplace overwrites the destination IN PLACE (preserves its inode), so
# the container's single-file bind mount sees the new mtime and hosaka's Lexicon
# hot-reloads it -- no container restart. A default rsync (or any rename-based
# copy) swaps the inode and the bind mount would keep serving the stale file.
# Idempotent; safe to run on every lexicon change.
set -euo pipefail

SRC="${HOME}/.local/share/hosaka/lexicon.json"
DEST_HOST="${HOSAKA_DROPLET:-wai-root@wai-lau.net}"
DEST_PATH="/srv/hosaka-piper/lexicon.json"

[ -f "$SRC" ] || { echo "no lexicon at $SRC; nothing to sync"; exit 0; }

ssh "$DEST_HOST" "mkdir -p $(dirname "$DEST_PATH")"
rsync --inplace -z "$SRC" "${DEST_HOST}:${DEST_PATH}"
echo "synced lexicon -> ${DEST_HOST}:${DEST_PATH}"
