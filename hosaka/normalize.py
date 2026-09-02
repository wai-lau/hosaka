"""Pre-chunk text normalization for spoken forms the engines mishandle.

Engine G2P frontends read clock times inconsistently: Chatterbox strips the
colon and reads ``13:45`` as the integer 1345 ("thirteen thousand four five"),
espeak (Piper) spells digits, and only Kokoro's misaki handles them. The only
engine-agnostic lever is the text we hand them (same rationale as the lexicon),
so we rewrite ``HH:MM`` clock times to words *before* the lexicon and chunking,
covering every path (HTTP, WebSocket, REPL, web) and every backend.

Scope is deliberately narrow: a digit pair joined by a colon, with an optional
am/pm suffix, or a bare 12h hour with a meridiem (``1am``, ``12 P.M.``), and
only when the hours/minutes are in range. Anything else is left verbatim so we
never corrupt non-time colons (ratios, ``http://``, code) or ordinary numbers.

The meridiem is rendered ``A.M`` / ``P.M`` (measured against the G2P frontends,
not guessed): espeak reads a bare ``AM`` right after "one"/"two" as the word
"am", spaced ``A M`` becomes "a em" in misaki, and a trailing dot is a sentence
end for both espeak and the chunker. ``A.M`` without the final dot reads as the
letters on both; the writer's own trailing period is left in place.
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

# A clock time: 1-2 digit hour, ':' , 2 digit minute, optional am/pm; or a
# bare 1-2 digit hour with a meridiem. The meridiem's trailing dot is NOT
# consumed (it may be the sentence's period); the letters may not run on
# into a word ("1amp").
_MERIDIEM = r"\s*([AaPp])\.?[Mm](?![A-Za-z])"
_TIME = re.compile(
    r"\b(\d{1,2}):([0-5]\d)" + _MERIDIEM + r"|\b(\d{1,2}):([0-5]\d)\b|\b(\d{1,2})" + _MERIDIEM
)


def _cardinal(n: int) -> str:
    """Spell 0-59 as words ('forty-five'). Out of range -> str(n)."""
    if n < 20:
        return _ONES[n]
    if n < 60:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")
    return str(n)


def _spoken_meridiem(mer: str) -> str:
    """'a'/'p' (any case) -> 'A.M'/'P.M' -- the letters on every frontend."""
    return f"{mer.upper()}.M"


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
        parts.append(_spoken_meridiem(meridiem))
    return " ".join(parts)


def _sub(m: re.Match) -> str:
    if m.group(1) is not None:  # with am/pm
        hour, minute, mer = int(m.group(1)), int(m.group(2)), m.group(3)
        # am/pm is only meaningful on a 12h clock. A 24h hour (13-23) with a
        # stray meridiem is still a real time -- speak the 24h form and drop the
        # redundant suffix. Never fall through to verbatim: that lets the raw
        # digits reach the engine, which is the "thirteen thousand" bug itself.
        if not 1 <= hour <= 12:
            mer = None
    elif m.group(4) is not None:  # plain HH:MM
        hour, minute, mer = int(m.group(4)), int(m.group(5)), None
    else:  # bare hour + meridiem ("1am"): only meaningful on a 12h clock
        hour, mer = int(m.group(6)), m.group(7)
        if not 1 <= hour <= 12:
            return m.group(0)
        return f"{_cardinal(hour)} {_spoken_meridiem(mer)}"
    if hour > 23:  # not a clock (e.g. 24:00, 99:00) -- leave verbatim
        return m.group(0)
    return _spoken_time(hour, minute, mer)


def normalize_times(text: str) -> str:
    """Rewrite ``HH:MM`` clock times in ``text`` to spoken words."""
    return _TIME.sub(_sub, text)
