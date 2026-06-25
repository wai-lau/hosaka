from hosaka.config import PIPER_PYTHON, PIPER_SIDECAR, PIPER_VOICES, VOICE_DIR
from hosaka.library import VoiceLibrary
from hosaka.server.app import create_app
from hosaka.server.engines.base import EngineRegistry
from hosaka.server.engines.chatterbox_engine import ChatterboxEngine
from hosaka.server.engines.kokoro_engine import KokoroEngine
from hosaka.server.engines.piper_engine import PiperEngine


def _make_piper():
    """Build the Piper sidecar engine, or None if it isn't set up. Missing the
    .venv-piper interpreter or every model file degrades gracefully: the server
    still serves Kokoro + Chatterbox. Only voices whose model exists are loaded
    and advertised."""
    available = {vid: spec for vid, spec in PIPER_VOICES.items() if spec["model"].exists()}
    if not PIPER_PYTHON.exists() or not available:
        return None
    cmd = [str(PIPER_PYTHON), str(PIPER_SIDECAR)]
    for vid, spec in available.items():
        cmd += ["--voice", f"{vid}={spec['model']}"]
    return PiperEngine(cmd, voices=list(available))


_library = VoiceLibrary(VOICE_DIR)
_registry = EngineRegistry(
    kokoro=KokoroEngine(),
    chatterbox=ChatterboxEngine(_library),
    piper=_make_piper(),
)
app = create_app(_registry, _library, do_warmup=True)
