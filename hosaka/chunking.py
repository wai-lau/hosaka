import math
import re

from hosaka.config import DASH_PAUSE_MS

_SENT_END = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_END = re.compile(r"[,;:](?=\s)")
# Dash punctuation (-- , --- , em/en dash), NOT a word hyphen. A dash is a
# deliberate pause: the text is split there into separate spoken fragments with a
# pause sentinel between them, which the server renders as real silence. Captured
# so split_fragments can size the pause from the dash run.
_DASH = re.compile(r"\s*(-{2,}|[—–])\s*")

# Sentinel fragment standing in for a pause. The server (app._pcm_frames) turns
# it into silence and never hands it to an engine; \x00 cannot occur in input
# text, so it can't collide with a real fragment.
_PAUSE_PREFIX = "\x00PAUSE:"


def _pause_marker(ms: int) -> str:
    return f"{_PAUSE_PREFIX}{ms}\x00"


def pause_ms(frag: str) -> int | None:
    """Silence duration (ms) if frag is a pause sentinel, else None. The server
    uses this to inject silence between spoken fragments where a dash was."""
    if frag.startswith(_PAUSE_PREFIX) and frag.endswith("\x00"):
        try:
            return int(frag[len(_PAUSE_PREFIX) : -1])
        except ValueError:
            return None
    return None


def _dash_ms(dash: str) -> int:
    """Pause length for a dash run: a single dash (-- / em / en) is one beat,
    --- is two, ----+ is three (capped)."""
    n = len(dash) if "-" in dash else 2  # em/en dash ~ a single "--"
    return DASH_PAUSE_MS * min(3, n - 1)


def split_fragments(
    text: str,
    first_max_chars: int | None = None,
    max_chars: int = 280,
    growth: float | None = None,
) -> list[str]:
    """Break text into fragments small enough for any engine to synth safely.

    Fragments break only at sentence boundaries, sub-split at clause/word
    boundaries when a sentence exceeds `max_chars` -- so a seam never lands
    mid-phrase. `max_chars` caps *every* fragment: Chatterbox corrupts the
    CUDA context (device-side assert) if a single fragment exceeds its token
    limit, so an uncapped long sentence must never reach the engine whole.

    A dash run (-- , --- , em/en dash) is a deliberate pause: the text is split
    there and a pause sentinel (see pause_ms) is emitted between the spoken
    fragments, which the server renders as real silence -- the dash is NOT spoken
    and never reaches an engine. A leading/trailing dash (no speech on one side)
    emits no pause.

    Two streaming-latency modes for the first fragment(s):

    - `growth` set (with `first_max_chars`): RAMP mode. The per-fragment cap
      starts at `first_max_chars` and grows geometrically by `growth` each
      fragment up to `max_chars`. The first fragment is small (fast first
      audio) and each later one stays small enough that its generation
      finishes before the previous fragment finishes playing -- gapless at
      Chatterbox's measured RTF ~0.8, since the model delivers each fragment
      whole. The ramp continues across dash pauses (a pause does not reset the
      cap). Used by the server for the Chatterbox quality path. See
      ARCHITECTURE.md (Chatterbox latency) for the derivation.
    - `growth` None: legacy. `first_max_chars` (if set) shrinks only the first
      fragment; everything else is capped at `max_chars`.
    """
    text = text.strip()
    if not text:
        return []

    out: list[str] = []
    spoken = 0  # count of real (non-pause) fragments, for ramp continuity
    parts = _DASH.split(text)  # [segment, dash, segment, dash, ...]
    for i, part in enumerate(parts):
        if i % 2 == 1:  # a captured dash run -> a pause where it stood
            if spoken:  # ignore a leading dash with nothing spoken before it
                out.append(_pause_marker(_dash_ms(part)))
            continue
        seg = part.strip()
        if not seg:
            continue
        pieces = _split_segment(seg, first_max_chars, max_chars, growth, start=spoken)
        out.extend(pieces)
        spoken += len(pieces)

    while out and pause_ms(out[-1]) is not None:  # drop a trailing dash's pause
        out.pop()
    return out


def _split_segment(
    text: str,
    first_max_chars: int | None,
    max_chars: int,
    growth: float | None,
    start: int,
) -> list[str]:
    """Split one dash-free segment into fragments. `start` is the number of
    spoken fragments already emitted before this segment, so the ramp's cap keeps
    growing across dash pauses and the first-fragment shrink only fires once."""
    sentences = [s.strip() for s in _SENT_END.split(text) if s.strip()]
    if not sentences:
        return []

    if growth is not None and first_max_chars is not None:
        return _ramp_wrap(sentences, first_max_chars, max_chars, growth, start)

    out: list[str] = []
    for i, s in enumerate(sentences):
        pieces = _wrap(s, max_chars)
        if i > 0:
            pieces[0] = " " + pieces[0]  # preserve the inter-sentence space
        out.extend(pieces)

    # Optionally shrink the very first fragment for fast first-audio (streaming).
    if start == 0 and first_max_chars is not None and len(out[0]) > first_max_chars:
        out = _wrap(out[0], first_max_chars) + out[1:]
    return out


def _wrap(s: str, limit: int) -> list[str]:
    """Split s into pieces no longer than limit, breaking on word boundaries."""
    s = s.strip()
    out: list[str] = []
    while len(s) > limit:
        cut = _boundary(s, limit)
        out.append(s[:cut].rstrip())
        s = s[cut:].lstrip()
    if s:
        out.append(s)
    return out


def _ramp_wrap(
    sentences: list[str],
    first_max_chars: int,
    max_chars: int,
    growth: float,
    start: int = 0,
) -> list[str]:
    """Emit fragments whose size cap grows geometrically from first_max_chars
    to max_chars, breaking at clause/word boundaries and never merging across
    sentence boundaries. The cap for the k-th emitted fragment is
    min(max_chars, ceil(first_max_chars * growth**k)), where k counts from
    `start` -- so the ramp continues across dash pauses instead of resetting.
    Early fragments are small for fast first-audio and stay within the gapless
    budget."""
    out: list[str] = []
    for i, sentence in enumerate(sentences):
        s = sentence.strip()
        first_piece = True
        while s:
            k = start + len(out)
            cap = min(max_chars, math.ceil(first_max_chars * growth**k))
            if len(s) <= cap:
                piece, s = s, ""
            else:
                cut = _boundary(s, cap)
                piece, s = s[:cut].rstrip(), s[cut:].lstrip()
            if i > 0 and first_piece:
                piece = " " + piece  # preserve the inter-sentence space
            out.append(piece)
            first_piece = False
    return out


def _boundary(s: str, limit: int) -> int:
    """Best cut index within limit: prefer a clause break, then a space.

    A clause break (`, ; :` followed by space) makes the fragment seam land
    on a natural pause; it is only taken past the halfway point so fragments
    stay reasonably full. Falls back to the last word boundary, then a hard
    cut for a single word longer than the limit.
    """
    cut = s.rfind(" ", 0, limit)
    if cut <= 0:  # single word longer than limit
        cut = limit
    for m in _CLAUSE_END.finditer(s[:limit]):
        if m.start() >= limit // 2:
            cut = m.start() + 1  # keep the punctuation with the fragment
    return cut
