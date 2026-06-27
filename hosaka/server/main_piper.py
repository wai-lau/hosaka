"""Piper-only composition root for the always-on droplet container.

Builds the hosaka server with kokoro/chatterbox/rvc absent, so this module
imports NO torch / Kokoro / Chatterbox / RVC -- only the CPU Piper engine.
The container has no .venv-piper: its own interpreter (sys.executable) runs
both the server and the Piper sidecar, talking over the piper_proto pipe.

Run with:  uvicorn hosaka.server.main_piper:app --host 0.0.0.0 --port 8123
"""

import sys

from hosaka.config import PIPER_SIDECAR, PIPER_VOICES, VOICE_DIR
from hosaka.library import VoiceLibrary
from hosaka.server.app import create_app
from hosaka.server.engines.base import EngineRegistry
from hosaka.server.engines.piper_engine import PiperEngine


def build_piper_registry() -> EngineRegistry:
    """Build a registry with only Piper. Missing every model file -> piper=None
    (server still boots, just advertises no voices). The sidecar runs under the
    current interpreter; there is no separate .venv-piper in the container."""
    available = {vid: spec for vid, spec in PIPER_VOICES.items() if spec["model"].exists()}
    piper = None
    if available:
        cmd = [sys.executable, str(PIPER_SIDECAR)]
        for vid, spec in available.items():
            cmd += ["--voice", f"{vid}={spec['model']}"]
        piper = PiperEngine(cmd, voices=list(available))
    return EngineRegistry(kokoro=None, chatterbox=None, piper=piper, rvc=None)


_library = VoiceLibrary(VOICE_DIR)
app = create_app(build_piper_registry(), _library, do_warmup=False)
