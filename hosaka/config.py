from pathlib import Path

SAMPLE_RATE = 24000  # Hz, mono, float32 LE everywhere
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8123
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
# systemd --user unit that owns the server (and port) when installed. The REPL
# defers to it instead of spawning a competing process on the same port.
SERVER_UNIT = "hosaka-server.service"

DEFAULT_BACKEND = "piper"
DEFAULT_VOICE = "glados"

# Voice Kokoro warms up on -- a real Kokoro preset, independent of the REPL's
# DEFAULT_VOICE (which may point at another backend, e.g. piper/glados).
KOKORO_WARMUP_VOICE = "af_heart"

# Max requests admitted at once (the one running on the GPU + those waiting in
# line). The GPU still serves strictly one at a time; this only bounds how deep
# the wait queue can grow before new requests are turned away with 503, so a
# backlog can't pile up unbounded memory / unbounded wait. Kokoro drains fast
# (RTF ~0.04) so a modest depth feels concurrent; Chatterbox (RTF ~1.0) does
# not, so a deep queue there just means long waits.
MAX_QUEUE = 16

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

# Lead buffer for the streaming (ffplay) player: bytes are withheld until this
# much audio is buffered, then released, giving the player a cushion to ride
# over per-fragment synth jitter at Chatterbox's RTF ~1.
PIPELINE_LEAD_MS = 1500

# Chatterbox fragment ramp (quality path only). The model delivers each
# fragment whole and runs at measured RTF ~0.8 (faster than realtime), so to
# get fast first-audio without a mid-utterance gap the fragment cap RAMPS:
# fragment k is capped at min(CHATTERBOX_MAX_CHARS, ceil(FIRST * GROWTH**k)).
# A small first fragment -> ~3-4s to first sound (vs the full ~10s gen of one
# long fragment); GROWTH 1.1 keeps each later fragment inside the gapless
# budget at RTF ~0.8 (see chunking.split_fragments / ARCHITECTURE.md). Raise
# FIRST for fewer seams + slower start, lower GROWTH for more gap safety under
# GPU jitter. Set FRAGMENT_GROWTH = None to disable the ramp.
FIRST_FRAGMENT_MAX_CHARS = 64
FRAGMENT_GROWTH = 1.1
CHATTERBOX_MAX_CHARS = 280  # hard per-fragment cap (token-limit safety)

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

DATA_DIR = Path.home() / ".local" / "share" / "hosaka"  # untracked personal data
VOICE_DIR = DATA_DIR / "voices"
MANIFEST_PATH = VOICE_DIR / "manifest.json"
# Custom-pronunciation map ({word: respelling}); applied to input before
# chunking on every path into the engines. See hosaka/lexicon.py.
LEXICON_PATH = DATA_DIR / "lexicon.json"

# Piper neural character voices (e.g. GLaDOS). Run CPU-only in an isolated venv
# (.venv-piper) as an out-of-process sidecar; the server venv never imports
# piper. Model weights live untracked under the data dir -- fetch them with
# scripts/fetch_glados_model.sh. Adding a character = drop a model + one entry
# here (id -> onnx path + blurb), no code change.
_REPO_ROOT = Path(__file__).resolve().parents[1]
PIPER_PYTHON = _REPO_ROOT / ".venv-piper" / "bin" / "python"
PIPER_SIDECAR = _REPO_ROOT / "hosaka" / "server" / "engines" / "piper_sidecar.py"
PIPER_DIR = DATA_DIR / "piper"
PIPER_VOICES = {
    "glados": {
        "model": PIPER_DIR / "glados" / "glados_piper_medium.onnx",
        "description": "GLaDOS (Portal), DavesArmoury",
    },
}

# RVC character voices (e.g. Charlie Morningstar). Run GPU in an isolated venv
# (.venv-rvc) as an out-of-process sidecar; the server venv never imports
# rvc-python. RVC converts timbre, so each voice declares the character's
# gender + accent -- the neutral Kokoro SOURCE preset must match (see
# resolve_source). Model weights live untracked under the data dir -- fetch
# them with scripts/fetch_rvc_model.sh. Adding a character = drop a model + one
# RVC_VOICES entry, no code change.
RVC_PYTHON = _REPO_ROOT / ".venv-rvc" / "bin" / "python"
RVC_SIDECAR = _REPO_ROOT / "hosaka" / "server" / "engines" / "rvc_sidecar.py"
RVC_DIR = DATA_DIR / "rvc"
RVC_HUBERT = RVC_DIR / "hubert_base.pt"  # ContentVec/HuBERT feature encoder
RVC_RMVPE = RVC_DIR / "rmvpe.pt"  # F0 (pitch) estimator

# (gender, accent) -> neutral Kokoro base preset. Kokoro English is American or
# British only; a tuple absent here has no matching base (hard stop, not a
# "download more presets" case).
SOURCE_PRESETS = {
    ("female", "american"): "af_sarah",  # clear + neutral
    ("male", "american"): "am_michael",
    ("female", "british"): "bf_emma",
    ("male", "british"): "bm_george",
}

# Fixed conversion knobs for MVP (no per-request RVC knobs).
RVC_KNOBS = {
    "index_rate": 0.5,
    "f0_method": "rmvpe",
    "protect": 0.33,
    "rms_mix_rate": 0.25,
}

RVC_VOICES = {
    "charlie": {
        "model": RVC_DIR / "charlie" / "charlie.pth",
        "index": RVC_DIR / "charlie" / "charlie.index",
        "model_sr": 40000,  # native (Loren85 v2); the sidecar resamples 40k -> 24k
        "gender": "female",
        "accent": "american",
        "source": "af_sarah",  # must equal SOURCE_PRESETS[(female, american)]
        "transpose": 0,  # semitones; tune by ear (demo showed -16..+16)
        "description": "Charlie Morningstar (Hazbin Hotel), RVC V2",
    },
}


def resolve_source(voice_cfg: dict) -> str:
    """The neutral Kokoro source preset for an RVC voice, validated against its
    gender + accent. A configured `source` that disagrees with the tuple is a
    config error (fail loud). KeyError if Kokoro has no base for the tuple."""
    want = SOURCE_PRESETS[(voice_cfg["gender"], voice_cfg["accent"])]
    if voice_cfg["source"] != want:
        raise ValueError(
            f"RVC source {voice_cfg['source']!r} != {want!r} for "
            f"{voice_cfg['gender']}/{voice_cfg['accent']}"
        )
    return want


# Default sentence the bake CLI speaks to produce a clone seed clip.
BAKE_SEED_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "She sells seashells by the seashore, and the "
    "voice you hear now is the one you will keep."
)
