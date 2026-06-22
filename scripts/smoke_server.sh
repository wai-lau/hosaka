#!/usr/bin/env bash
# End-to-end smoke: start the real server (lifespan stops the LLM + warms both
# models), hit /health, synthesize via both backends, play the Kokoro clip
# through pacat (audible), then shut the server down.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
PY=.venv-server/bin/python
PORT=8123
BASE="http://127.0.0.1:$PORT"

echo "=== starting server ==="
$PY -m uvicorn hosaka.server.main:app --host 127.0.0.1 --port $PORT \
  > /tmp/hosaka_server.log 2>&1 &
SRV=$!

echo "=== waiting for /health (model load + warmup) ==="
ok=0
for i in $(seq 1 120); do
  if .venv-server/bin/python -c "import httpx,sys; sys.exit(0 if httpx.get('$BASE/health',timeout=1).status_code==200 else 1)" 2>/dev/null; then
    ok=1; echo "healthy after ~$((i*1))s checks"; break
  fi
  sleep 1
done
if [ "$ok" != 1 ]; then echo "SERVER NOT HEALTHY"; tail -20 /tmp/hosaka_server.log; kill $SRV 2>/dev/null; exit 1; fi

echo "=== /v1/voices ==="
.venv-server/bin/python - <<PY
import httpx
for v in httpx.get("$BASE/v1/voices").json():
    print(" ", v["id"], v["backend"], v["source"])
PY

play() {  # backend voice text
  echo "=== speak [$1 / $2] ==="
  .venv-server/bin/python - "$1" "$2" "$3" <<'PY'
import sys, httpx
backend, voice, text = sys.argv[1], sys.argv[2], sys.argv[3]
body = {"input": text, "backend": backend, "voice": voice, "stream": True}
n = 0
import subprocess
pac = subprocess.Popen(["pacat","--raw","--rate=24000","--channels=1",
                        "--format=float32le","--latency-msec=50"],
                       stdin=subprocess.PIPE, bufsize=0)
with httpx.stream("POST", "http://127.0.0.1:8123/v1/audio/speech",
                  json=body, timeout=None) as r:
    print("  status", r.status_code)
    for chunk in r.iter_bytes():
        if chunk:
            n += len(chunk); pac.stdin.write(chunk)
pac.stdin.close(); pac.wait()
print(f"  streamed {n} bytes ({n//4} float32 samples, ~{n/4/24000:.1f}s)")
PY
}

play kokoro af_heart "Hello, this is hosaka speaking with a preset voice."
play chatterbox calm_brit "And this is a cloned voice in quality mode."

echo "=== shutdown ==="
.venv-server/bin/python -c "import httpx; httpx.post('$BASE/shutdown',timeout=2)" 2>/dev/null
sleep 2
kill $SRV 2>/dev/null
echo "SMOKE_DONE"
