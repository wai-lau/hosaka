"""Whether hosaka-server loads Charlie (Chatterbox + the RVC sidecar) at start.

nicole (Kokoro, GTX 1660) and GLaDOS (Piper, CPU) load in every GPU mode, so the
reader voice stays up. Charlie holds the 5070 Ti, so it loads only in GPU mode
homo. scripts/gpu_mode.sh writes the choice to a state file and restarts the
server; this module reads it once at startup. Pure: no torch, safe in .venv-dev.

The env var HOSAKA_CHARLIE ("1"/"0") overrides the file, for a one-off run.
A missing or unreadable file means off: a fresh box comes up nicole-only.
"""

import os
from pathlib import Path


def state_path() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "hosaka" / "charlie"


def wanted() -> bool:
    env = os.environ.get("HOSAKA_CHARLIE")
    if env is not None:
        return env.strip() == "1"
    try:
        return state_path().read_text().strip() == "on"
    except OSError:
        return False
