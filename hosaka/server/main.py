import os

from hosaka import charlie_state
from hosaka.cache import SourceCache
from hosaka.config import (
    ENGINE_GPU,
    PIPER_PYTHON,
    PIPER_SIDECAR,
    PIPER_VOICES,
    RVC_HUBERT,
    RVC_KNOBS,
    RVC_PYTHON,
    RVC_RMVPE,
    RVC_SIDECAR,
    RVC_VOICES,
    SOURCE_CACHE_DIR,
    SOURCE_CACHE_DISK_BYTES,
    SOURCE_CACHE_RAM_BYTES,
    SOURCE_CACHE_VERSION,
    VOICE_DIR,
    resolve_source,
)
from hosaka.gpu import cuda_device, present_uuid
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


def _make_rvc(sources):
    """Build the RVC sidecar engine, or None if it isn't set up. Missing the
    .venv-rvc interpreter, the HuBERT/rmvpe assets, or every voice's model+index
    degrades gracefully: the server runs the other engines. `sources` maps a
    backend name to the engine that generates a voice's source audio (Kokoro for
    realtime presets, Chatterbox for an expressive cloned character source)."""
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
            "source_backend": spec.get("source_backend", "kokoro"),
            "source": resolve_source(spec),
            "source_params": spec.get("source_params"),
            "transpose": spec["transpose"],
            "passes": spec.get("passes", 1),
            "gate": spec.get("gate", False),
            "speed": spec.get("speed", 1.0),
        }
    cache = SourceCache(
        SOURCE_CACHE_DIR,
        SOURCE_CACHE_RAM_BYTES,
        SOURCE_CACHE_DISK_BYTES,
        SOURCE_CACHE_VERSION,
    )
    return RvcEngine(
        sources,
        cmd,
        voices=voices,
        knobs=dict(RVC_KNOBS),
        source_cache=cache,
        gpu=present_uuid(ENGINE_GPU.get("rvc")),
    )


_library = VoiceLibrary(VOICE_DIR)
_kokoro = KokoroEngine(device=cuda_device(ENGINE_GPU.get("kokoro")))
# nicole + GLaDOS load in every GPU mode; Charlie (Chatterbox + RVC, the whole
# 5070 Ti) only when gpu_mode.sh set homo -- see hosaka/charlie_state.py.
_charlie = charlie_state.wanted()
_chatterbox = (
    ChatterboxEngine(_library, device=cuda_device(ENGINE_GPU.get("chatterbox")))
    if _charlie
    else None
)
_registry = EngineRegistry(
    kokoro=_kokoro,
    chatterbox=_chatterbox,
    # HOSAKA_GLADOS=0 skips GLaDOS: start_server.sh sets it while Qwen holds the
    # GPUs (the droplet serves GLaDOS, so home nicole is all that is needed then).
    piper=_make_piper() if os.environ.get("HOSAKA_GLADOS", "1") == "1" else None,
    rvc=_make_rvc({"kokoro": _kokoro, "chatterbox": _chatterbox}) if _charlie else None,
)
app = create_app(_registry, _library, do_warmup=True)
