from hosaka.config import VOICE_DIR
from hosaka.library import VoiceLibrary
from hosaka.server.app import create_app
from hosaka.server.engines.base import EngineRegistry
from hosaka.server.engines.chatterbox_engine import ChatterboxEngine
from hosaka.server.engines.kokoro_engine import KokoroEngine

_library = VoiceLibrary(VOICE_DIR)
_registry = EngineRegistry(
    kokoro=KokoroEngine(),
    chatterbox=ChatterboxEngine(_library),
)
app = create_app(_registry, _library, do_warmup=True)
