from hosaka.config import (
    PIPER_PYTHON,
    PIPER_SIDECAR,
    PIPER_VOICES,
    RVC_HUBERT,
    RVC_KNOBS,
    RVC_PYTHON,
    RVC_RMVPE,
    RVC_SIDECAR,
    RVC_VOICES,
    VOICE_DIR,
    resolve_source,
)
from hosaka.library import VoiceLibrary
from hosaka.server.app import create_app
from hosaka.server.engines.base import EngineRegistry
from hosaka.server.engines.chatterbox_engine import ChatterboxEngine
from hosaka.server.engines.kokoro_engine import KokoroEngine
from hosaka.server.engines.piper_engine import PiperEngine
from hosaka.server.engines.rvc_engine import RvcEngine


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


def _make_rvc(source_engine):
    """Build the RVC sidecar engine, or None if it isn't set up. Missing the
    .venv-rvc interpreter, the HuBERT/rmvpe assets, or every voice's model+index
    degrades gracefully: the server runs the other engines. RVC uses Kokoro as
    its neutral audio source, so the engine is shared in."""
    available = {
        vid: spec
        for vid, spec in RVC_VOICES.items()
        if spec["model"].exists() and spec["index"].exists()
    }
    if (
        not RVC_PYTHON.exists()
        or not RVC_HUBERT.exists()
        or not RVC_RMVPE.exists()
        or not available
    ):
        return None
    cmd = [str(RVC_PYTHON), str(RVC_SIDECAR)]
    voices = {}
    for vid, spec in available.items():
        cmd += ["--voice", f"{vid}={spec['model']}:{spec['index']}"]
        voices[vid] = {
            "source": resolve_source(spec),
            "transpose": spec["transpose"],
            "passes": spec.get("passes", 1),
        }
    return RvcEngine(source_engine, cmd, voices=voices, knobs=dict(RVC_KNOBS))


_library = VoiceLibrary(VOICE_DIR)
_kokoro = KokoroEngine()
_registry = EngineRegistry(
    kokoro=_kokoro,
    chatterbox=ChatterboxEngine(_library),
    piper=_make_piper(),
    rvc=_make_rvc(_kokoro),
)
app = create_app(_registry, _library, do_warmup=True)
