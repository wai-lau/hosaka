import numpy as np
from pathlib import Path
from hosaka.audio import PacatPlayer


def test_player_writes_bytes_to_subprocess(tmp_path):
    out = tmp_path / "out.raw"
    # Fake player: shell that copies stdin to a file.
    cmd = ["sh", "-c", f"cat > {out}"]
    data = np.ones(100, dtype=np.float32)
    with PacatPlayer(cmd=cmd) as p:
        p.write(data)
    assert out.exists()
    assert out.stat().st_size == data.nbytes


def test_default_cmd_targets_pacat():
    p = PacatPlayer()
    assert p.cmd[0] == "pacat"
    assert "--rate=24000" in p.cmd
    assert "--format=float32le" in p.cmd
