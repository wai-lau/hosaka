"""Always-on, GPU-free home-box service that arbitrates the GPU between hosaka
TTS and ollama by shelling out to scripts/gpu_mode.sh. Runs under .venv-dev on
127.0.0.1:8124 and is reverse-tunneled to the droplet. Imports NO torch / no
hosaka engines -- keep it that way.

Auth: every route requires `Authorization: Bearer $GPU_MODE_TOKEN`. The primary
boundary is loopback + the SSH tunnel; the token is defense-in-depth on the
tunnel hop."""

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException

from hosaka.gpu_mode import VALID_ACTIONS, parse_mode

_SCRIPT = Path(__file__).resolve().parent.parent.parent / "scripts" / "gpu_mode.sh"


def _shell_runner(action: str) -> str:
    """Default runner: run gpu_mode.sh <action>, return current mode.

    For `status` we run the script once and return directly. For verb actions
    (homo/emo/idle) we run the verb then re-read `status` so the response
    reflects settled reality, not the verb we asked for."""
    if action == "status":
        out = subprocess.run([str(_SCRIPT), "status"], check=True, capture_output=True, text=True)
        return out.stdout
    subprocess.run([str(_SCRIPT), action], check=True, capture_output=True, text=True)
    out = subprocess.run([str(_SCRIPT), "status"], check=True, capture_output=True, text=True)
    return out.stdout


def create_gpu_mode_app(
    runner: Callable[[str], str] = _shell_runner, token: str | None = None
) -> FastAPI:
    token = token if token is not None else os.environ.get("GPU_MODE_TOKEN", "")
    app = FastAPI()

    def require_token(authorization: str | None = Header(default=None)):
        expected = f"Bearer {token}"
        if not token or authorization != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

    def _mode(action: str) -> dict:
        try:
            return {"mode": parse_mode(runner(action))}
        except Exception as e:  # subprocess.CalledProcessError or anything else
            raise HTTPException(status_code=500, detail=str(e)[:200]) from e

    @app.get("/mode", dependencies=[Depends(require_token)])
    def get_mode():
        return _mode("status")

    for action in VALID_ACTIONS:
        # bind `action` per-iteration via default arg
        @app.post(f"/{action}", dependencies=[Depends(require_token)])
        def do_action(action=action):
            return _mode(action)

    return app


app = create_gpu_mode_app()
