from pathlib import Path

SAMPLE_RATE = 24000  # Hz, mono, float32 LE everywhere
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8123
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

DEFAULT_BACKEND = "kokoro"
DEFAULT_VOICE = "af_heart"

LLM_MODEL = "gpt-oss:20b"  # stopped on server start to free VRAM

VOICE_DIR = Path.home() / ".local" / "share" / "hosaka" / "voices"
MANIFEST_PATH = VOICE_DIR / "manifest.json"

# Default sentence the bake CLI speaks to produce a clone seed clip.
BAKE_SEED_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "She sells seashells by the seashore, and the "
    "voice you hear now is the one you will keep."
)
