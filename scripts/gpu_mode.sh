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
#   lomo  -> Charlie not loaded, ollama left as it is (the box is busy with
#            other work; nicole + GLaDOS only)
#   emo   -> ollama on, Charlie not loaded
#   idle  -> ollama off, Charlie not loaded
#   status-> print one of: homo | lomo | emo | idle | mixed
#
# The last verb asked for is kept in $XDG_STATE_HOME/hosaka/gpu-mode so status
# can tell lomo from emo/idle (all three serve nicole without Charlie): status
# says lomo while hosaka-server is up, Charlie is off and lomo was the last ask.
#
# Charlie is chosen at server start from $XDG_STATE_HOME/hosaka/charlie
# (hosaka/charlie_state.py), so a switch writes that file and restarts
# hosaka-server (nicole blips for the restart). While Qwen holds the GPUs
# (scripts/qwen_lock.sh) Charlie cannot load: homo is refused, and the other
# modes run nicole-only (start_server.sh forces Charlie and GLaDOS off then;
# nicole fits on the GTX 1660 beside Qwen).
set -euo pipefail
here=$(dirname "$0")
# shellcheck source=qwen_lock.sh
. "$here/qwen_lock.sh"

state=${XDG_STATE_HOME:-$HOME/.local/state}/hosaka/charlie
asked=${XDG_STATE_HOME:-$HOME/.local/state}/hosaka/gpu-mode

ollama_active()  { [ "$(systemctl is-active ollama 2>/dev/null)" = active ]; }
hosaka_active()  { [ "$(systemctl --user is-active hosaka-server 2>/dev/null)" = active ]; }
charlie_flag()   { [ "$(cat "$state" 2>/dev/null)" = on ]; }
charlie_loaded() { hosaka_active && charlie_flag; }

set_charlie() { mkdir -p "$(dirname "$state")"; echo "$1" > "$state"; }
set_asked()   { mkdir -p "$(dirname "$asked")"; echo "$1" > "$asked"; }
asked_lomo()  { [ "$(cat "$asked" 2>/dev/null)" = lomo ]; }

status() {
  local o=0 c=0
  ollama_active && o=1
  charlie_loaded && c=1
  if   [ "$c" = 1 ] && [ "$o" = 0 ]; then echo homo
  elif [ "$c" = 1 ]; then echo mixed   # invariant violation: Charlie and ollama both up
  elif hosaka_active && asked_lomo; then echo lomo
  elif [ "$o" = 1 ]; then echo emo
  else echo idle
  fi
}

# Drop Charlie (if loaded) and make sure nicole is serving. A restart frees the
# 5070 Ti; an already nicole-only server is left alone (no blip).
nicole_only() {
  local was_loaded=0
  charlie_loaded && was_loaded=1
  set_charlie off
  if [ "$was_loaded" = 1 ] || ! hosaka_active; then
    systemctl --user restart hosaka-server
  fi
}

case "${1:-status}" in
  homo)
    [ "$(status)" = homo ] && { echo "already homo"; exit 0; }
    refuse_if_qwen_holds_gpu || exit 1
    sudo systemctl stop ollama
    set_asked homo
    set_charlie on
    systemctl --user restart hosaka-server
    echo "-> homo (ollama off, Charlie + nicole up)"
    ;;
  lomo)
    [ "$(status)" = lomo ] && { echo "already lomo"; exit 0; }
    set_asked lomo
    nicole_only
    echo "-> lomo (nicole up, Charlie off, ollama untouched)"
    ;;
  emo)
    [ "$(status)" = emo ] && { echo "already emo"; exit 0; }
    set_asked emo
    nicole_only
    sudo systemctl start ollama
    echo "-> emo (ollama up, nicole up, Charlie off)"
    ;;
  idle)
    [ "$(status)" = idle ] && hosaka_active && { echo "already idle"; exit 0; }
    set_asked idle
    nicole_only
    sudo systemctl stop ollama
    echo "-> idle (ollama off, nicole up, Charlie off)"
    ;;
  status) status ;;
  *) echo "usage: gpu_mode.sh {homo|lomo|emo|idle|status}" >&2; exit 2 ;;
esac
