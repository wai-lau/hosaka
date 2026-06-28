#!/usr/bin/env bash
# gpu_mode.sh -- arbitrate the home-box GPU between hosaka TTS and ollama.
# Single source of truth for the systemctl logic; ~/bin/homo, ~/bin/emo and the
# gpu-mode service all call this. Idempotent.
#
#   homo  -> hosaka holds the GPU  (ollama off, hosaka-server on)
#   emo   -> ollama holds the GPU  (hosaka-server off, ollama on)
#   idle  -> nothing on the GPU    (both off)
#   status-> print one of: homo | emo | idle | mixed
set -euo pipefail

ollama_active()  { [ "$(systemctl is-active ollama 2>/dev/null)" = active ]; }
hosaka_active()  { [ "$(systemctl --user is-active hosaka-server 2>/dev/null)" = active ]; }

status() {
  local o=0 h=0
  ollama_active && o=1
  hosaka_active && h=1
  if   [ "$h" = 1 ] && [ "$o" = 0 ]; then echo homo
  elif [ "$o" = 1 ] && [ "$h" = 0 ]; then echo emo
  elif [ "$o" = 0 ] && [ "$h" = 0 ]; then echo idle
  else echo mixed   # invariant violation: both up
  fi
}

case "${1:-status}" in
  homo)
    [ "$(status)" = homo ] && { echo "already homo"; exit 0; }
    sudo systemctl stop ollama
    systemctl --user start hosaka-server
    echo "-> homo (ollama off, hosaka up)"
    ;;
  emo)
    [ "$(status)" = emo ] && { echo "already emo"; exit 0; }
    systemctl --user stop hosaka-server
    sudo systemctl start ollama
    echo "-> emo (hosaka off, ollama up)"
    ;;
  idle)
    [ "$(status)" = idle ] && { echo "already idle"; exit 0; }
    systemctl --user stop hosaka-server
    sudo systemctl stop ollama
    echo "-> idle (gpu free)"
    ;;
  status) status ;;
  *) echo "usage: gpu_mode.sh {homo|emo|idle|status}" >&2; exit 2 ;;
esac
