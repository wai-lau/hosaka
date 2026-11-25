#!/usr/bin/env bash
# gpu_mode.sh -- arbitrate the home-box GPUs between Charlie (hosaka's quality
# voice) and ollama. Single source of truth for the systemctl logic; ~/bin/homo,
# ~/bin/emo and the gpu-mode service all call this. Idempotent.
#
# hosaka-server runs in EVERY mode with nicole (Kokoro, GTX 1660) + GLaDOS
# (Piper, CPU), so the reader voice is always on. The mode only decides who gets
# the 5070 Ti:
#
#   homo  -> Charlie loaded (Chatterbox + RVC), ollama off
#   emo   -> ollama on, Charlie not loaded
#   idle  -> ollama off, Charlie not loaded
#   status-> print one of: homo | emo | idle | mixed
#
# Charlie is chosen at server start from $XDG_STATE_HOME/hosaka/charlie
# (hosaka/charlie_state.py), so a switch writes that file and restarts
# hosaka-server (nicole blips for the restart). While Qwen holds the GPUs
# (scripts/qwen_lock.sh) hosaka-server stays down: homo is refused, and emo/idle
# only write the file -- `qwen stop` starts hosaka with it.
set -euo pipefail
here=$(dirname "$0")
# shellcheck source=qwen_lock.sh
. "$here/qwen_lock.sh"

state=${XDG_STATE_HOME:-$HOME/.local/state}/hosaka/charlie

ollama_active()  { [ "$(systemctl is-active ollama 2>/dev/null)" = active ]; }
hosaka_active()  { [ "$(systemctl --user is-active hosaka-server 2>/dev/null)" = active ]; }
charlie_flag()   { [ "$(cat "$state" 2>/dev/null)" = on ]; }
charlie_loaded() { hosaka_active && charlie_flag; }

set_charlie() { mkdir -p "$(dirname "$state")"; echo "$1" > "$state"; }

status() {
  local o=0 c=0
  ollama_active && o=1
  charlie_loaded && c=1
  if   [ "$c" = 1 ] && [ "$o" = 0 ]; then echo homo
  elif [ "$o" = 1 ] && [ "$c" = 0 ]; then echo emo
  elif [ "$o" = 0 ] && [ "$c" = 0 ]; then echo idle
  else echo mixed   # invariant violation: Charlie and ollama both up
  fi
}

# Drop Charlie (if loaded) and make sure nicole is serving. A restart frees the
# 5070 Ti; an already nicole-only server is left alone (no blip).
nicole_only() {
  local was_loaded=0
  charlie_loaded && was_loaded=1
  set_charlie off
  refuse_if_qwen_holds_gpu 2>/dev/null || return 0
  if [ "$was_loaded" = 1 ] || ! hosaka_active; then
    systemctl --user restart hosaka-server
  fi
}

case "${1:-status}" in
  homo)
    [ "$(status)" = homo ] && { echo "already homo"; exit 0; }
    refuse_if_qwen_holds_gpu || exit 1
    sudo systemctl stop ollama
    set_charlie on
    systemctl --user restart hosaka-server
    echo "-> homo (ollama off, Charlie + nicole up)"
    ;;
  emo)
    [ "$(status)" = emo ] && { echo "already emo"; exit 0; }
    nicole_only
    sudo systemctl start ollama
    echo "-> emo (ollama up, nicole up, Charlie off)"
    ;;
  idle)
    [ "$(status)" = idle ] && hosaka_active && { echo "already idle"; exit 0; }
    nicole_only
    sudo systemctl stop ollama
    echo "-> idle (ollama off, nicole up, Charlie off)"
    ;;
  status) status ;;
  *) echo "usage: gpu_mode.sh {homo|emo|idle|status}" >&2; exit 2 ;;
esac
