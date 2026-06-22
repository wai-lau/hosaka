import re

_SENT_END = re.compile(r"(?<=[.!?])\s+")


def split_fragments(text: str, first_max_chars: int = 60) -> list[str]:
    text = text.strip()
    if not text:
        return []

    # Split by sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    first = sentences[0]
    rest = sentences[1:]

    # If the first sentence is too long, hard-split it on a word boundary.
    if len(first) > first_max_chars:
        head, _, tail = _word_split(first, first_max_chars)
        out = [head]
        if tail:
            out.append(tail)
        out.extend(rest)
        return out

    # For sentence-split, add leading space to preserve spacing when joined
    rest_with_space = [" " + s for s in rest]
    return [first, *rest_with_space]


def _word_split(s: str, limit: int) -> tuple[str, str, str]:
    if len(s) <= limit:
        return s, "", ""
    cut = s.rfind(" ", 0, limit)
    if cut == -1:
        cut = limit
    return s[:cut].rstrip(), " ", s[cut:].lstrip()
