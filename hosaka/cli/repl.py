import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from hosaka.audio import make_player
from hosaka.cli.replcmd import parse_line
from hosaka.config import (
    DEFAULT_BACKEND,
    DEFAULT_VOICE,
    SERVER_PORT,
    SERVER_URL,
    VOICE_DIR,
    native_to_pct,
    pct_to_native,
)
from hosaka.library import VoiceLibrary


def _server_up() -> bool:
    try:
        return httpx.get(f"{SERVER_URL}/health", timeout=1.0).status_code == 200
    except Exception:
        return False


def _spawn_server() -> None:
    log_path = Path(tempfile.gettempdir()) / "hosaka-server.log"
    log = open(log_path, "w")  # capture startup errors instead of discarding
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "hosaka.server.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(SERVER_PORT),
        ],
        stdout=log,
        stderr=log,
    )
    for _ in range(120):  # wait up to ~60s for model load + warmup
        if _server_up():
            return
        time.sleep(0.5)
    raise RuntimeError(f"server did not become healthy; see {log_path} for the cause")


def _voice_backend(name, fallback):
    """Resolve a voice's real backend from the registry; fall back on miss."""
    try:
        for v in httpx.get(f"{SERVER_URL}/v1/voices", timeout=2.0).json():
            if v["id"] == name:
                return v["backend"]
    except Exception:
        pass
    return fallback


def _speak(player, backend, voice, params, text):
    body = {"input": text, "backend": backend, "voice": voice, "params": params, "stream": True}
    buf = bytearray()
    try:
        with httpx.stream("POST", f"{SERVER_URL}/v1/audio/speech", json=body, timeout=None) as r:
            if r.status_code != 200:
                print(f"[server {r.status_code}] {r.read().decode(errors='ignore')}")
                return
            for raw in r.iter_bytes():
                if raw:
                    buf.extend(raw)
    except httpx.HTTPError as exc:
        # A failure inside the engine closes the stream mid-body (status was
        # already 200). Report it and keep the REPL alive instead of crashing.
        print(f"[stream error] {exc}; see /tmp/hosaka-server.log")
        return
    if buf:
        player.play(bytes(buf))


_PASTE_START = "\x1b[200~"
_PASTE_END = "\x1b[201~"


def _logical_lines(stream):
    """Yield one logical input per line, but coalesce a bracketed paste (which
    arrives as many lines wrapped in ESC[200~ / ESC[201~) into a single input
    with its newlines flattened to spaces -- so pasted multi-line text speaks
    as one utterance instead of N choppy ones."""
    buf = None
    for raw in stream:
        line = raw.rstrip("\n")
        if buf is not None:
            if _PASTE_END in line:
                buf.append(line.replace(_PASTE_END, ""))
                yield " ".join(p for p in buf if p)
                buf = None
            else:
                buf.append(line)
            continue
        if _PASTE_START in line:
            line = line.replace(_PASTE_START, "")
            if _PASTE_END in line:  # whole paste fit on one line
                yield line.replace(_PASTE_END, "")
            else:
                buf = [line]
            continue
        yield line
    if buf:  # paste left open (no end marker before EOF)
        yield " ".join(p for p in buf if p)


def main():
    if not _server_up():
        print("starting hosaka server (loading models)...")
        _spawn_server()

    lib = VoiceLibrary(VOICE_DIR)
    backend = DEFAULT_BACKEND
    voice = DEFAULT_VOICE
    params = {"exaggeration": 0.5, "cfg_weight": 0.4, "temperature": 0.8, "speed": 1.0}

    print("hosaka ready. Type to speak; :help for commands. (^D to quit)")
    sys.stdout.write("\x1b[?2004h")  # enable bracketed paste
    sys.stdout.flush()
    try:
        with make_player() as player:
            for line in _logical_lines(sys.stdin):
                a = parse_line(line)
                if a.kind == "speak":
                    if a.value:
                        _speak(player, backend, voice, params, a.value)
                elif a.kind == "voice":
                    name, text = a.value
                    voice = name
                    backend = _voice_backend(name, backend)
                    if text:
                        _speak(player, backend, voice, params, text)
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
                    name, pct = a.value
                    params[name] = pct_to_native(name, pct)
                    print(f"  {name} {pct:.0f}/100 -> {params[name]:.3f}")
                elif a.kind == "volume":
                    player.gain = pct_to_native("gain", a.value)
                    print(f"  volume {a.value:.0f}/100 -> {player.gain:.2f}x")
                elif a.kind == "voices":
                    for v in httpx.get(f"{SERVER_URL}/v1/voices").json():
                        desc = f"  {v['id']:20s} {v['backend']:11s} {v['source']:9s}"
                        if v.get("description"):
                            desc += f"  {v['description']}"
                        print(desc)
                elif a.kind == "status":
                    print(f"  voice   {voice}  ({backend})")
                    for label, key in (
                        ("exag", "exaggeration"),
                        ("cfg", "cfg_weight"),
                        ("temp", "temperature"),
                        ("speed", "speed"),
                    ):
                        print(
                            f"  {label:7s} {native_to_pct(key, params[key]):3.0f}/100"
                            f"  ({params[key]:.3f})"
                        )
                    print(
                        f"  volume  {native_to_pct('gain', player.gain):3.0f}/100"
                        f"  ({player.gain:.2f}x)"
                    )
                elif a.kind == "help":
                    print(
                        ":voice <name> | :clone <id|path> | :backend k|c | "
                        ":exag/:cfg/:temp/:speed/:vol <0-100> | :voices | :status | "
                        ":quit[ --stop]"
                    )
                elif a.kind == "error":
                    print(f"  {a.value}")
                elif a.kind in ("quit", "quit_stop"):
                    if a.kind == "quit_stop":
                        try:
                            httpx.post(f"{SERVER_URL}/shutdown", timeout=1.0)
                        except Exception:
                            pass
                    break
    finally:
        sys.stdout.write("\x1b[?2004l")  # disable bracketed paste
        sys.stdout.flush()


if __name__ == "__main__":
    main()
