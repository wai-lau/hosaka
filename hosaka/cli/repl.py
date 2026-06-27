import readline  # noqa: F401 -- importing it gives input() line editing + history
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from hosaka.audio import make_player
from hosaka.cli.replcmd import parse_line
from hosaka.config import (
    DATA_DIR,
    DEFAULT_BACKEND,
    DEFAULT_VOICE,
    LEXICON_PATH,
    SERVER_PORT,
    SERVER_UNIT,
    SERVER_URL,
    VOICE_DIR,
    native_to_pct,
    pct_to_native,
)
from hosaka.lexicon import add_entry, load_map, remove_entry
from hosaka.library import VoiceLibrary

HISTORY_FILE = DATA_DIR / "repl_history"


def _server_up() -> bool:
    try:
        return httpx.get(f"{SERVER_URL}/health", timeout=1.0).status_code == 200
    except Exception:
        return False


# systemd ActiveState values that mean the managed unit owns the port right now
# (running, or in the middle of (re)starting). In any of these the REPL must
# wait for the unit, never spawn its own server on the same port.
_UNIT_PROVIDING = ("active", "activating", "reloading", "deactivating")


def _startup_action(server_up, load_state, active_state):
    """Decide how the REPL should obtain a server. Pure policy, no I/O.

    Returns one of:
      "attach"     -- a healthy server already answers; just use it.
      "wait"       -- the systemd unit owns the port and is up / coming up;
                      wait for it instead of spawning a competing process.
      "start_unit" -- the unit is installed but stopped/failed; ask systemd to
                      start it (still never spawn our own on its port).
      "spawn"      -- no managed unit (systemctl or the unit is absent); the
                      REPL runs its own server as a fallback.

    Spawning while the unit is merely down races systemd for port 8123: a spawn
    that outlives the health wait orphans and squats the port, sending the unit
    into an unbindable restart loop. So we only ever spawn when no unit exists.
    """
    if server_up:
        return "attach"
    if load_state != "loaded":
        return "spawn"
    if active_state in _UNIT_PROVIDING:
        return "wait"
    return "start_unit"


def _unit_info():
    """(LoadState, ActiveState) of the managed unit, or (None, None) when
    systemd or the unit is unavailable (then the REPL owns the server)."""
    if shutil.which("systemctl") is None:
        return None, None
    try:
        r = subprocess.run(
            ["systemctl", "--user", "show", SERVER_UNIT, "-p", "LoadState", "-p", "ActiveState"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    fields = {}
    for line in r.stdout.splitlines():
        key, _, val = line.partition("=")
        fields[key.strip()] = val.strip()
    return fields.get("LoadState") or None, fields.get("ActiveState") or None


def _wait_healthy(timeout_s: float = 90.0) -> bool:
    """Poll /health until it answers or the timeout elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _server_up():
            return True
        time.sleep(0.5)
    return False


def _start_unit() -> None:
    """Clear any failed state and (re)start the managed systemd unit."""
    for args in (["reset-failed", SERVER_UNIT], ["start", SERVER_UNIT]):
        subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True, timeout=10)


def _spawn_server() -> None:
    """Fallback: run our own uvicorn when no systemd unit manages the server."""
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
    if not _wait_healthy():
        raise RuntimeError(f"server did not become healthy; see {log_path} for the cause")


def _push_lexicon() -> None:
    """Push the lexicon to the droplet so the always-on glados picks up a :pron
    change immediately. :pron writes atomically (tmp+rename), which a single-file
    inotify watch can miss, so push directly here rather than rely on the systemd
    path unit (that unit stays a backup for in-place manual edits). Best-effort:
    a missing script or an unreachable droplet prints a note, never raises."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "sync_lexicon.sh"
    if not script.exists():
        return
    try:
        r = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        print("  (droplet push skipped)")
        return
    if r.returncode == 0:
        print("  (pushed to droplet)")
    else:
        print(f"  (droplet push failed: {(r.stderr or '').strip() or r.returncode})")


def _ensure_server() -> None:
    """Make a healthy server reachable, deferring to systemd when it owns the
    port and only spawning our own as a last resort. See _startup_action."""
    if _server_up():
        return
    action = _startup_action(False, *_unit_info())
    if action == "spawn":
        print("starting hosaka server (loading models)...")
        _spawn_server()
        return
    if action == "start_unit":
        print("hosaka server unit is down; starting via systemd...")
        _start_unit()
    else:  # "wait"
        print("hosaka server is starting (systemd); waiting...")
    if not _wait_healthy():
        raise RuntimeError(
            f"hosaka server did not become healthy; check: journalctl --user -u {SERVER_UNIT} -n 50"
        )


def _voice_meta(name, fallback):
    """Resolve a voice's (backend, cb_params, speed) from the registry; fall back
    on miss. cb_params is the voice's tuned cb-knob defaults (or None) and speed
    its default output tempo (or None), both used to preload the REPL knobs so
    they round-trip the character until tuned."""
    try:
        for v in httpx.get(f"{SERVER_URL}/v1/voices", timeout=2.0).json():
            if v["id"] == name:
                return v["backend"], v.get("cb_params"), v.get("speed")
    except Exception:
        pass
    return fallback, None, None


def _speak(player, backend, voice, params, text):
    body = {"input": text, "backend": backend, "voice": voice, "params": params, "stream": True}
    try:
        with httpx.stream("POST", f"{SERVER_URL}/v1/audio/speech", json=body, timeout=None) as r:
            if r.status_code != 200:
                print(f"[server {r.status_code}] {r.read().decode(errors='ignore')}")
                return
            for raw in r.iter_bytes():
                if raw:
                    player.write(raw)
    except httpx.HTTPError as exc:
        # A failure inside the engine closes the stream mid-body (status was
        # already 200). Report it and keep the REPL alive instead of crashing.
        print(f"[stream error] {exc}; see /tmp/hosaka-server.log")
    finally:
        player.end_utterance()


def _input_lines(prompt):
    """Yield logical input lines using readline (importing it above wires the
    editing in): left/right arrows, Ctrl-A/Ctrl-E, and Up/Down history all work
    inside input().

    Ctrl-C abandons the half-typed line and keeps the REPL alive (like a shell);
    Ctrl-D (EOF) ends it. readline's bracketed paste delivers a multi-line paste
    as one string -- flatten embedded newlines to spaces so it speaks as a
    single utterance instead of N choppy ones.
    """
    while True:
        try:
            line = input(prompt)
        except EOFError:
            return
        except KeyboardInterrupt:
            print()  # drop the current line, like a shell ^C, and reprompt
            continue
        yield line.replace("\n", " ")


def main():
    _ensure_server()

    lib = VoiceLibrary(VOICE_DIR)
    backend = DEFAULT_BACKEND
    voice = DEFAULT_VOICE
    params = {"exaggeration": 0.5, "cfg_weight": 0.4, "temperature": 0.8, "speed": 1.0}

    try:
        readline.read_history_file(HISTORY_FILE)
    except OSError:
        pass  # no history yet (first run) or unreadable -- start fresh

    print("hosaka ready. Type to speak; :help for commands. (^D to quit)")
    prompt = "> " if sys.stdin.isatty() else ""
    try:
        with make_player() as player:
            for line in _input_lines(prompt):
                a = parse_line(line)
                if a.kind == "speak":
                    if a.value:
                        _speak(player, backend, voice, params, a.value)
                elif a.kind == "voice":
                    name, text = a.value
                    voice = name
                    backend, cb_params, vspeed = _voice_meta(name, backend)
                    # Preload the character's tuned cb-knob defaults so :status
                    # and the request reflect them; the user can still :exag etc.
                    if cb_params:
                        for k, val in cb_params.items():
                            if k in params:
                                params[k] = float(val)
                        print(
                            f"  loaded {voice} defaults: "
                            + ", ".join(f"{k}={float(v):.2f}" for k, v in cb_params.items())
                        )
                    # Preload the voice's default output speed so :speed tunes it
                    # live and round-trips the character until the user changes it.
                    if vspeed is not None:
                        params["speed"] = float(vspeed)
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
                elif a.kind == "pron":
                    sub, val = a.value
                    if sub == "list":
                        m = load_map(LEXICON_PATH)
                        if not m:
                            print("  (no custom pronunciations)")
                        for word in sorted(m):
                            print(f"  {word:20s} -> {m[word]}")
                    elif sub == "add":
                        word, respelling = val
                        add_entry(LEXICON_PATH, word, respelling)
                        print(f"  {word} -> {respelling}")
                        _push_lexicon()
                    elif sub == "rm":
                        _, removed = remove_entry(LEXICON_PATH, val)
                        print(f"  {'removed' if removed else 'not found'}: {val}")
                        if removed:
                            _push_lexicon()
                elif a.kind == "voices":
                    for v in httpx.get(f"{SERVER_URL}/v1/voices").json():
                        print(f"  {v['id']:20s} {'cb' if v.get('cb') else 'non-cb'}")
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
                        ":voice <name> | :clone <id|path> | :backend k|c|p | "
                        ":exag/:cfg/:temp/:speed/:vol <0-100> | :voices | "
                        ":pron [list|add <word> <respelling>|rm <word>] | :status | "
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
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            readline.write_history_file(HISTORY_FILE)
        except OSError:
            pass  # best-effort -- never let a history write sink the REPL exit


if __name__ == "__main__":
    main()
