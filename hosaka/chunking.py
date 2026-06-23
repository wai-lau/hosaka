import math
import re

_SENT_END = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_END = re.compile(r"[,;:](?=\s)")
# Dashes used as punctuation (-- , --- , em/en dash), not word hyphens. The
# models don't pause on these, so turn them into a comma to force the pause.
_DASH = re.compile(r"\s*(?:-{2,}|[—–])\s*")


def normalize_punct(text: str) -> str:
    return _DASH.sub(", ", text)


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

    Two streaming-latency modes for the first fragment(s):

    - `growth` set (with `first_max_chars`): RAMP mode. The per-fragment cap
      starts at `first_max_chars` and grows geometrically by `growth` each
      fragment up to `max_chars`. The first fragment is small (fast first
      audio) and each later one stays small enough that its generation
      finishes before the previous fragment finishes playing -- gapless at
      Chatterbox's measured RTF ~0.8, since the model delivers each fragment
      whole. Used by the server for the Chatterbox quality path. See
      ARCHITECTURE.md (Chatterbox latency) for the derivation.
    - `growth` None: legacy. `first_max_chars` (if set) shrinks only the first
      fragment; everything else is capped at `max_chars`.
    """
    text = normalize_punct(text.strip())
    if not text:
        return []

    sentences = [s.strip() for s in _SENT_END.split(text) if s.strip()]
    if not sentences:
        return []

    if growth is not None and first_max_chars is not None:
        return _ramp_wrap(sentences, first_max_chars, max_chars, growth)

    out: list[str] = []
    for i, s in enumerate(sentences):
        pieces = _wrap(s, max_chars)
        if i > 0:
            pieces[0] = " " + pieces[0]  # preserve the inter-sentence space
        out.extend(pieces)

    # Optionally shrink the first fragment for fast first-audio (streaming only).
    if first_max_chars is not None and len(out[0]) > first_max_chars:
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
    sentences: list[str], first_max_chars: int, max_chars: int, growth: float
) -> list[str]:
    """Emit fragments whose size cap grows geometrically from first_max_chars
    to max_chars, breaking at clause/word boundaries and never merging across
    sentence boundaries. The cap for the k-th emitted fragment (0-based) is
    min(max_chars, ceil(first_max_chars * growth**k)), so early fragments are
    small for fast first-audio and stay within the gapless budget."""
    out: list[str] = []
    for i, sentence in enumerate(sentences):
        s = sentence.strip()
        first_piece = True
        while s:
            cap = min(max_chars, math.ceil(first_max_chars * growth ** len(out)))
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
