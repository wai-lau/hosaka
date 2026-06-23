from pathlib import Path

SAMPLE_RATE = 24000  # Hz, mono, float32 LE everywhere
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8123
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

DEFAULT_BACKEND = "kokoro"
DEFAULT_VOICE = "af_heart"

# Playback gain applied client-side before pacat, clipped to [-1, 1].
# Default 1.0 (off): WSLg's RDP audio bridge distorts HOT signals (adds static)
# because amplifying here happens BEFORE the lossy RDP hop. For more loudness on
# WSLg, raise volume on the Windows side (post-decode) instead. :vol still tunes
# this for non-RDP / native-Linux audio where pre-gain is clean.
OUTPUT_GAIN = 1.0

# pacat buffer depth (native-Linux path only; WSLg uses the Windows player).
PLAYBACK_LATENCY_MSEC = 800

# Silence prepended to each Windows-played clip so the output device finishing
# its wake-from-idle clips silence instead of the first phonemes.
PLAYBACK_LEAD_SILENCE_MS = 200

# REPL knobs are entered on a uniform 0-100 scale. 50 is the DEFAULT (neutral)
# value for each knob, 0 and 100 the extremes -- so the map is piecewise linear,
# anchored at the default rather than the range midpoint. (min, default, max);
# keep ranges in sync with schemas.clamp_params.
KNOB_RANGES = {
    "exaggeration": (0.0, 0.5, 2.0),
    "cfg_weight": (0.0, 0.4, 1.0),
    "temperature": (0.1, 0.8, 2.0),
    "speed": (0.5, 1.0, 2.0),
    "gain": (0.0, 1.0, 5.0),
}


def pct_to_native(name: str, pct: float) -> float:
    """Map a 0-100 knob value onto its native range; 50 -> the default."""
    lo, mid, hi = KNOB_RANGES[name]
    pct = max(0.0, min(100.0, pct))
    if pct <= 50.0:
        return lo + (pct / 50.0) * (mid - lo)
    return mid + ((pct - 50.0) / 50.0) * (hi - mid)


def native_to_pct(name: str, value: float) -> float:
    """Inverse of pct_to_native, for showing values on the 0-100 scale."""
    lo, mid, hi = KNOB_RANGES[name]
    if value <= mid:
        return 0.0 if mid == lo else (value - lo) / (mid - lo) * 50.0
    return 100.0 if hi == mid else 50.0 + (value - mid) / (hi - mid) * 50.0


LLM_MODEL = "gpt-oss:20b"  # stopped on server start to free VRAM

VOICE_DIR = Path.home() / ".local" / "share" / "hosaka" / "voices"
MANIFEST_PATH = VOICE_DIR / "manifest.json"

# Default sentence the bake CLI speaks to produce a clone seed clip.
BAKE_SEED_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "She sells seashells by the seashore, and the "
    "voice you hear now is the one you will keep."
)
