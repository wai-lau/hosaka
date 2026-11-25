# tests/test_gpu_mode_sh.py
#
# Modes (always-on nicole): hosaka-server runs in every mode with nicole (Kokoro,
# GTX 1660) + GLaDOS. The mode decides Charlie (Chatterbox + RVC, 5070 Ti) vs ollama:
#   homo -> Charlie loaded, ollama off
#   emo  -> ollama on, Charlie not loaded
#   idle -> ollama off, Charlie not loaded
# Charlie is chosen at server start from the state file
# $XDG_STATE_HOME/hosaka/charlie ("on"/"off"), so a switch rewrites the file and
# restarts hosaka-server.
import os
import subprocess
import textwrap
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gpu_mode.sh"


def _fake_bin(tmp_path: Path, ollama: str, hosaka: str, charlie: str | None = None) -> Path:
    """A bin dir with fake `systemctl` + `sudo` on PATH, plus the Charlie state file.

    `systemctl is-active ollama`            -> $ollama (from state file)
    `systemctl --user is-active hosaka...`  -> $hosaka (from state file)
    start/stop/restart mutate the state files and log the verb + unit.
    `sudo X...`                             -> exec X... (so sudo is transparent)
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (tmp_path / "ollama.state").write_text(ollama)
    (tmp_path / "hosaka.state").write_text(hosaka)
    if charlie is not None:
        (tmp_path / "xdg" / "hosaka").mkdir(parents=True)
        (tmp_path / "xdg" / "hosaka" / "charlie").write_text(charlie + "\n")
    sysctl = bindir / "systemctl"
    sysctl.write_text(
        textwrap.dedent(f"""\
        #!/usr/bin/env bash
        args=("$@"); [ "${{args[0]}}" = "--user" ] && unset 'args[0]' && args=("${{args[@]}}")
        verb="${{args[0]}}"; unit="${{args[1]}}"
        case "$unit" in
          *hosaka*) sf="{tmp_path}/hosaka.state";;
          *) sf="{tmp_path}/ollama.state";;
        esac
        case "$verb" in
          is-active) cat "$sf";;
          start|restart)
            # record which Charlie setting the server would boot with
            c=$(cat "{tmp_path}/xdg/hosaka/charlie" 2>/dev/null || echo none)
            echo active > "$sf"; echo "$verb $unit charlie=$c" >> "{tmp_path}/calls.log";;
          stop)  echo inactive > "$sf"; echo "stop $unit"  >> "{tmp_path}/calls.log";;
        esac
    """)
    )
    sudo = bindir / "sudo"
    sudo.write_text('#!/usr/bin/env bash\nexec "$@"\n')
    for f in (sysctl, sudo):
        f.chmod(0o755)
    return bindir


def _run(tmp_path, bindir, arg):
    env = {
        "PATH": f"{bindir}:/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        "XDG_STATE_HOME": str(tmp_path / "xdg"),
    }
    return subprocess.run([str(SCRIPT), arg], capture_output=True, text=True, env=env)


def _log(tmp_path) -> str:
    p = tmp_path / "calls.log"
    return p.read_text() if p.exists() else ""


def _charlie(tmp_path) -> str:
    return (tmp_path / "xdg" / "hosaka" / "charlie").read_text().strip()


def _qwen_lock(tmp_path):
    d = tmp_path / "xdg" / "qwen" / "locks"
    d.mkdir(parents=True, exist_ok=True)
    (d / str(os.getpid())).write_text("opencode\n")  # a live pid: this test process


# --- status ---------------------------------------------------------------


def test_status_homo(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="inactive", hosaka="active", charlie="on")
    assert _run(tmp_path, bindir, "status").stdout.strip() == "homo"


def test_status_nicole_only_is_idle(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="inactive", hosaka="active", charlie="off")
    assert _run(tmp_path, bindir, "status").stdout.strip() == "idle"


def test_status_no_state_file_is_idle(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="inactive", hosaka="active")
    assert _run(tmp_path, bindir, "status").stdout.strip() == "idle"


def test_status_emo_with_nicole_up(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="active", hosaka="active", charlie="off")
    assert _run(tmp_path, bindir, "status").stdout.strip() == "emo"


def test_status_emo_with_hosaka_down(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="active", hosaka="inactive", charlie="on")
    assert _run(tmp_path, bindir, "status").stdout.strip() == "emo"


def test_status_flag_on_but_server_down_is_idle(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="inactive", hosaka="inactive", charlie="on")
    assert _run(tmp_path, bindir, "status").stdout.strip() == "idle"


def test_status_mixed(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="active", hosaka="active", charlie="on")
    assert _run(tmp_path, bindir, "status").stdout.strip() == "mixed"


# --- switches -------------------------------------------------------------


def test_homo_stops_ollama_restarts_hosaka_with_charlie(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="active", hosaka="active", charlie="off")
    r = _run(tmp_path, bindir, "homo")
    assert r.returncode == 0, r.stderr
    log = _log(tmp_path)
    assert "stop ollama" in log
    assert "restart hosaka-server charlie=on" in log
    assert log.index("stop ollama") < log.index("restart hosaka-server")
    assert _charlie(tmp_path) == "on"
    assert _run(tmp_path, bindir, "status").stdout.strip() == "homo"


def test_homo_from_down_server_starts_it(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="inactive", hosaka="inactive")
    r = _run(tmp_path, bindir, "homo")
    assert r.returncode == 0, r.stderr
    assert "hosaka-server charlie=on" in _log(tmp_path)
    assert _run(tmp_path, bindir, "status").stdout.strip() == "homo"


def test_homo_refused_while_qwen_holds_gpu(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="inactive", hosaka="active", charlie="off")
    _qwen_lock(tmp_path)
    r = _run(tmp_path, bindir, "homo")
    assert r.returncode != 0
    assert "qwen" in r.stderr
    assert _log(tmp_path) == ""
    assert _charlie(tmp_path) == "off"


def test_emo_drops_charlie_keeps_nicole_then_starts_ollama(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="inactive", hosaka="active", charlie="on")
    r = _run(tmp_path, bindir, "emo")
    assert r.returncode == 0, r.stderr
    log = _log(tmp_path)
    assert "restart hosaka-server charlie=off" in log
    assert "start ollama" in log
    assert log.index("restart hosaka-server") < log.index("start ollama")  # free VRAM first
    assert "stop hosaka-server" not in log
    assert (tmp_path / "hosaka.state").read_text().strip() == "active"
    assert _run(tmp_path, bindir, "status").stdout.strip() == "emo"


def test_emo_from_idle_does_not_restart_nicole(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="inactive", hosaka="active", charlie="off")
    r = _run(tmp_path, bindir, "emo")
    assert r.returncode == 0, r.stderr
    log = _log(tmp_path)
    assert "hosaka-server" not in log  # nicole keeps serving, no blip
    assert "start ollama" in log


def test_idle_drops_charlie_and_ollama_keeps_nicole(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="active", hosaka="active", charlie="on")
    r = _run(tmp_path, bindir, "idle")
    assert r.returncode == 0, r.stderr
    log = _log(tmp_path)
    assert "stop ollama" in log
    assert "restart hosaka-server charlie=off" in log
    assert "stop hosaka-server" not in log
    assert _charlie(tmp_path) == "off"
    assert _run(tmp_path, bindir, "status").stdout.strip() == "idle"


def test_idle_brings_down_server_back_up(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="active", hosaka="inactive", charlie="off")
    r = _run(tmp_path, bindir, "idle")
    assert r.returncode == 0, r.stderr
    assert "hosaka-server charlie=off" in _log(tmp_path)
    assert (tmp_path / "hosaka.state").read_text().strip() == "active"


def test_emo_while_qwen_holds_gpu_leaves_server_alone(tmp_path):
    # start_server.sh refuses while Qwen holds the GPU, so a restart would only
    # crash-loop; `qwen stop` brings hosaka back with the flag as written.
    bindir = _fake_bin(tmp_path, ollama="inactive", hosaka="inactive", charlie="on")
    _qwen_lock(tmp_path)
    r = _run(tmp_path, bindir, "emo")
    assert r.returncode == 0, r.stderr
    log = _log(tmp_path)
    assert "hosaka-server" not in log
    assert "start ollama" in log
    assert _charlie(tmp_path) == "off"


def test_homo_already_homo_is_noop(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="inactive", hosaka="active", charlie="on")
    r = _run(tmp_path, bindir, "homo")
    assert r.returncode == 0
    assert _log(tmp_path) == ""
    assert "already" in r.stdout


def test_idle_already_idle_is_noop(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="inactive", hosaka="active", charlie="off")
    r = _run(tmp_path, bindir, "idle")
    assert r.returncode == 0
    assert _log(tmp_path) == ""


# --- lomo -------------------------------------------------------------------


def _asked(tmp_path) -> str:
    return (tmp_path / "xdg" / "hosaka" / "gpu-mode").read_text().strip()


def test_lomo_from_homo_drops_charlie_keeps_nicole(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="inactive", hosaka="active", charlie="on")
    r = _run(tmp_path, bindir, "lomo")
    assert r.returncode == 0, r.stderr
    log = _log(tmp_path)
    assert "restart hosaka-server charlie=off" in log
    assert "stop hosaka-server" not in log
    assert "ollama" not in log  # lomo leaves ollama as it is
    assert _charlie(tmp_path) == "off"
    assert _asked(tmp_path) == "lomo"
    assert _run(tmp_path, bindir, "status").stdout.strip() == "lomo"


def test_lomo_leaves_running_ollama_up(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="active", hosaka="active", charlie="off")
    r = _run(tmp_path, bindir, "lomo")
    assert r.returncode == 0, r.stderr
    assert _log(tmp_path) == ""  # nicole-only already: no blip, ollama kept
    assert _run(tmp_path, bindir, "status").stdout.strip() == "lomo"


def test_lomo_starts_down_server_kokoro_only(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="inactive", hosaka="inactive")
    r = _run(tmp_path, bindir, "lomo")
    assert r.returncode == 0, r.stderr
    assert "hosaka-server charlie=off" in _log(tmp_path)
    assert _run(tmp_path, bindir, "status").stdout.strip() == "lomo"


def test_lomo_survives_server_restart(tmp_path):
    # A crash + systemd/watchdog restart boots from the state files: still lomo,
    # still no Charlie.
    bindir = _fake_bin(tmp_path, ollama="inactive", hosaka="active", charlie="on")
    _run(tmp_path, bindir, "lomo")
    (tmp_path / "hosaka.state").write_text("inactive")
    assert _run(tmp_path, bindir, "status").stdout.strip() == "idle"  # down is not lomo
    (tmp_path / "hosaka.state").write_text("active")
    assert _charlie(tmp_path) == "off"
    assert _run(tmp_path, bindir, "status").stdout.strip() == "lomo"


def test_homo_from_lomo_loads_charlie(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="inactive", hosaka="active", charlie="on")
    _run(tmp_path, bindir, "lomo")
    (tmp_path / "calls.log").unlink()
    r = _run(tmp_path, bindir, "homo")
    assert r.returncode == 0, r.stderr
    assert "restart hosaka-server charlie=on" in _log(tmp_path)
    assert _run(tmp_path, bindir, "status").stdout.strip() == "homo"


def test_idle_from_lomo_is_not_a_noop(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="active", hosaka="active", charlie="off")
    _run(tmp_path, bindir, "lomo")
    r = _run(tmp_path, bindir, "idle")
    assert r.returncode == 0, r.stderr
    assert "stop ollama" in _log(tmp_path)
    assert _run(tmp_path, bindir, "status").stdout.strip() == "idle"


def test_already_lomo_is_noop(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="inactive", hosaka="active", charlie="off")
    _run(tmp_path, bindir, "lomo")
    r = _run(tmp_path, bindir, "lomo")
    assert "already lomo" in r.stdout
