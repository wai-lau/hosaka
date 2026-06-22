import re

_SENT_END = re.compile(r"(?<=[.!?])\s+")


def split_fragments(text: str, first_max_chars: int = 60, max_chars: int = 280) -> list[str]:
    """Break text into fragments small enough for any engine to synth safely.

    `first_max_chars` keeps the very first fragment short so first audio is
    heard fast. `max_chars` caps *every* fragment: Chatterbox corrupts the
    CUDA context (device-side assert) if a single fragment exceeds its token
    limit, so an uncapped long sentence must never reach the engine whole.
    """
    text = text.strip()
    if not text:
        return []

    sentences = [s.strip() for s in _SENT_END.split(text) if s.strip()]
    if not sentences:
        return []

    out: list[str] = []
    for i, s in enumerate(sentences):
        pieces = _wrap(s, max_chars)
        if i > 0:
            pieces[0] = " " + pieces[0]  # preserve the inter-sentence space
        out.extend(pieces)

    # Shrink only the first emitted fragment for fast first-audio latency.
    if len(out[0]) > first_max_chars:
        out = _wrap(out[0], first_max_chars) + out[1:]
    return out


def _wrap(s: str, limit: int) -> list[str]:
    """Split s into pieces no longer than limit, breaking on word boundaries."""
    s = s.strip()
    out: list[str] = []
    while len(s) > limit:
        cut = s.rfind(" ", 0, limit)
        if cut <= 0:  # single word longer than limit: hard cut
            cut = limit
        out.append(s[:cut].rstrip())
        s = s[cut:].lstrip()
    if s:
        out.append(s)
    return out
