# tests/test_gpu_mode_sh.py
import subprocess
import textwrap
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gpu_mode.sh"


def _fake_bin(tmp_path: Path, ollama: str, hosaka: str) -> Path:
    """A bin dir with fake `systemctl` + `sudo` on PATH.

    `systemctl is-active ollama`            -> $ollama (from state file)
    `systemctl --user is-active hosaka...`  -> $hosaka (from state file)
    start/stop mutate the state files and log the argv.
    `sudo X...`                             -> exec X... (so sudo is transparent)
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (tmp_path / "ollama.state").write_text(ollama)
    (tmp_path / "hosaka.state").write_text(hosaka)
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
          start) echo active > "$sf"; echo "start $unit" >> "{tmp_path}/calls.log";;
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
    env = {"PATH": f"{bindir}:/usr/bin:/bin"}
    return subprocess.run([str(SCRIPT), arg], capture_output=True, text=True, env=env)


def test_status_homo(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="inactive", hosaka="active")
    r = _run(tmp_path, bindir, "status")
    assert r.stdout.strip() == "homo"


def test_status_emo(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="active", hosaka="inactive")
    assert _run(tmp_path, bindir, "status").stdout.strip() == "emo"


def test_status_idle(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="inactive", hosaka="inactive")
    assert _run(tmp_path, bindir, "status").stdout.strip() == "idle"


def test_emo_stops_hosaka_starts_ollama(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="inactive", hosaka="active")
    r = _run(tmp_path, bindir, "emo")
    assert r.returncode == 0
    log = (tmp_path / "calls.log").read_text()
    assert "stop hosaka-server" in log
    assert "start ollama" in log
    assert _run(tmp_path, bindir, "status").stdout.strip() == "emo"


def test_status_mixed(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="active", hosaka="active")
    assert _run(tmp_path, bindir, "status").stdout.strip() == "mixed"


def test_homo_stops_ollama_starts_hosaka(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="active", hosaka="inactive")
    r = _run(tmp_path, bindir, "homo")
    assert r.returncode == 0
    log = (tmp_path / "calls.log").read_text()
    assert "stop ollama" in log
    assert "start hosaka-server" in log
    assert _run(tmp_path, bindir, "status").stdout.strip() == "homo"


def test_idle_stops_both(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="active", hosaka="active")
    r = _run(tmp_path, bindir, "idle")
    assert r.returncode == 0
    assert _run(tmp_path, bindir, "status").stdout.strip() == "idle"


def test_homo_already_homo_is_noop(tmp_path):
    bindir = _fake_bin(tmp_path, ollama="inactive", hosaka="active")
    r = _run(tmp_path, bindir, "homo")
    assert r.returncode == 0
    calls_log = tmp_path / "calls.log"
    assert not calls_log.exists() or calls_log.read_text().strip() == ""
    assert "already" in r.stdout
