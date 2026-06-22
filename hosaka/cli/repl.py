import subprocess
import sys
import tempfile
import time
from pathlib import Path
import httpx
from hosaka.config import (SERVER_URL, SERVER_PORT, DEFAULT_BACKEND,
                           DEFAULT_VOICE, VOICE_DIR)
from hosaka.audio import PacatPlayer
from hosaka.cli.replcmd import parse_line
from hosaka.library import VoiceLibrary


def _server_up() -> bool:
    try:
        return httpx.get(f"{SERVER_URL}/health", timeout=1.0).status_code == 200
    except Exception:
        return False


def _spawn_server() -> None:
    log_path = Path(tempfile.gettempdir()) / "hosaka-server.log"
    log = open(log_path, "w")     # capture startup errors instead of discarding
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "hosaka.server.main:app",
         "--host", "127.0.0.1", "--port", str(SERVER_PORT)],
        stdout=log, stderr=log)
    for _ in range(120):          # wait up to ~60s for model load + warmup
        if _server_up():
            return
        time.sleep(0.5)
    raise RuntimeError(
        f"server did not become healthy; see {log_path} for the cause")


def _speak(player, backend, voice, params, text):
    body = {"input": text, "backend": backend, "voice": voice,
            "params": params, "stream": True}
    with httpx.stream("POST", f"{SERVER_URL}/v1/audio/speech",
                      json=body, timeout=None) as r:
        if r.status_code != 200:
            print(f"[server {r.status_code}] {r.read().decode(errors='ignore')}")
            return
        for raw in r.iter_bytes():
            if raw:
                player.write(raw)


def main():
    if not _server_up():
        print("starting hosaka server (loading models)...")
        _spawn_server()

    lib = VoiceLibrary(VOICE_DIR)
    backend = DEFAULT_BACKEND
    voice = DEFAULT_VOICE
    params = {"exaggeration": 0.5, "cfg_weight": 0.4,
              "temperature": 0.8, "speed": 1.0}

    print("hosaka ready. Type to speak; :help for commands. (^D to quit)")
    with PacatPlayer() as player:
        for line in sys.stdin:
            a = parse_line(line.rstrip("\n"))
            if a.kind == "speak":
                if a.value:
                    _speak(player, backend, voice, params, a.value)
            elif a.kind == "voice":
                voice, backend = a.value, "kokoro"
            elif a.kind == "clone":
                p = Path(a.value)
                if p.exists():
                    vid = p.stem
                    lib.add(vid, p, source="recording")
                    voice = vid
                else:
                    voice = a.value
                backend = "chatterbox"
            elif a.kind == "backend":
                backend = a.value
            elif a.kind == "set_param":
                name, val = a.value
                params[name] = val
            elif a.kind == "voices":
                for v in httpx.get(f"{SERVER_URL}/v1/voices").json():
                    print(f"  {v['id']:20s} {v['backend']:11s} {v['source']}")
            elif a.kind == "help":
                print(":voice <name> | :clone <id|path> | :backend k|c | "
                      ":exag/:cfg/:temp/:speed <f> | :voices | :quit[ --stop]")
            elif a.kind == "error":
                print(f"  {a.value}")
            elif a.kind in ("quit", "quit_stop"):
                if a.kind == "quit_stop":
                    try:
                        httpx.post(f"{SERVER_URL}/shutdown", timeout=1.0)
                    except Exception:
                        pass
                break


if __name__ == "__main__":
    main()
