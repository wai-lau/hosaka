"""Pre-chunk text normalization for spoken forms the engines mishandle.

Engine G2P frontends read clock times inconsistently: Chatterbox strips the
colon and reads ``13:45`` as the integer 1345 ("thirteen thousand four five"),
espeak (Piper) spells digits, and only Kokoro's misaki handles them. The only
engine-agnostic lever is the text we hand them (same rationale as the lexicon),
so we rewrite ``HH:MM`` clock times to words *before* the lexicon and chunking,
covering every path (HTTP, WebSocket, REPL, web) and every backend.

Scope is deliberately narrow: a digit pair joined by a colon, with an optional
am/pm suffix, and only when the hours/minutes are in range. Anything else is
left verbatim so we never corrupt non-time colons (ratios, ``http://``, code).
"""

import re

_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty")

# A clock time: 1-2 digit hour, ':' , 2 digit minute, optional am/pm.
_TIME = re.compile(
    r"\b(\d{1,2}):([0-5]\d)\s*([AaPp])\.?[Mm]\.?(?![A-Za-z])"
    r"|\b(\d{1,2}):([0-5]\d)\b"
)


def _cardinal(n: int) -> str:
    """Spell 0-59 as words ('forty-five'). Out of range -> str(n)."""
    if n < 20:
        return _ONES[n]
    if n < 60:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")
    return str(n)


def _spoken_time(hour: int, minute: int, meridiem: str | None) -> str:
    """Render an in-range clock time as a spoken phrase."""
    parts = [_cardinal(hour)]
    if minute == 0:
        parts.append("o'clock")
    elif minute < 10:
        parts += ["oh", _cardinal(minute)]
    else:
        parts.append(_cardinal(minute))
    if meridiem:
        parts.append(f"{meridiem.upper()}M")
    return " ".join(parts)


def _sub(m: re.Match) -> str:
    if m.group(1) is not None:  # with am/pm
        hour, minute, mer = int(m.group(1)), int(m.group(2)), m.group(3)
        if not 1 <= hour <= 12:  # am/pm only valid on a 12h clock
            return m.group(0)
    else:
        hour, minute, mer = int(m.group(4)), int(m.group(5)), None
        if hour > 23:
            return m.group(0)
    return _spoken_time(hour, minute, mer)


def normalize_times(text: str) -> str:
    """Rewrite ``HH:MM`` clock times in ``text`` to spoken words."""
    return _TIME.sub(_sub, text)
