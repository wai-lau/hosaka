#!/usr/bin/env bash
# Qwen wins vs TTS: `qwen` (OpenCode) writes ${XDG_STATE_HOME:-~/.local/state}/qwen/locks/<pid>
# while it holds the GPUs and removes it on exit. Source this and call refuse_if_qwen_holds_gpu;
# it returns 1 (with a message on stderr) while any lock names a live pid. A lock whose pid is
# gone is stale: removed, and it does not block.
refuse_if_qwen_holds_gpu() {
  local dir=${XDG_STATE_HOME:-$HOME/.local/state}/qwen/locks f pid live=0
  for f in "$dir"/*; do
    [ -f "$f" ] || continue
    pid=${f##*/}
    case $pid in '' | *[!0-9]*) continue ;; esac
    if kill -0 "$pid" 2>/dev/null; then
      live=$((live + 1))
    else
      rm -f -- "$f"
    fi
  done
  [ "$live" -eq 0 ] && return 0
  echo "qwen is using the GPU ($live session(s)); close them or run \`qwen stop\`" >&2
  return 1
}
