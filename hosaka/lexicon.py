"""Custom-pronunciation lexicon: respell words so the TTS lands them right.

A flat ``{word: respelling}`` map applied to the input text *before* chunking,
so it covers every path into the engines (HTTP, WebSocket, REPL, web client) and
both backends -- Kokoro and Chatterbox are plain text-driven, so the only
engine-agnostic lever is the spelling we hand them. "Wai" -> "Way" makes the
grapheme-to-phoneme step read the homophone instead of the literal letters.

Matching is whole-word for alphanumeric keys (a ``\\b`` boundary on each
word-character edge) and case-insensitive; the replacement is emitted verbatim
as stored. Whole-word keeps "Waitress" intact while rewriting "Wai";
case-insensitive means a sentence-initial "Wai" is caught too (the stored
replacement's own case wins, which TTS phonemization ignores). A punctuation key
like ``~`` has no word boundary to anchor to, so it matches anywhere -- even
glued to a digit (``~30`` -> ``about30``). Multi-word keys are allowed and
matched longest-first so a phrase wins over a word it contains.
"""

import json
import re
from pathlib import Path

_WORD = re.compile(r"\w")


def _anchored(key: str) -> str:
    """Escape ``key`` and add a ``\\b`` word boundary only on edges that are word
    characters. A word key ("Wai") gets a boundary on both sides (whole-word); a
    punctuation key ("~") gets none, so it matches even glued to a digit ("~30");
    a mixed key gets one only on its word-character side. A bare ``\\b`` around a
    non-word edge can never match (no word/non-word transition exists there),
    which is why an all-punctuation key was silently never substituted.
    """
    pat = re.escape(key)
    if _WORD.match(key[0]):
        pat = r"\b" + pat
    if _WORD.match(key[-1]):
        pat = pat + r"\b"
    return pat


def _compile(mapping: dict[str, str]) -> tuple[re.Pattern | None, dict[str, str]]:
    """Build one alternation regex over the keys + a lowercased lookup table.

    Keys are sorted longest-first so a multi-word key wins over a shorter key it
    contains; each is escaped (punctuation is literal) and anchored only where a
    word boundary makes sense (see _anchored). Empty keys are dropped.
    """
    keys = sorted((k for k in mapping if k), key=len, reverse=True)
    if not keys:
        return None, {}
    rx = re.compile("(?:" + "|".join(_anchored(k) for k in keys) + ")", re.IGNORECASE)
    lookup = {k.lower(): v for k, v in mapping.items()}
    return rx, lookup


def apply_lexicon(text: str, mapping: dict[str, str]) -> str:
    """Respell every whole-word, case-insensitive key occurrence in ``text``."""
    rx, lookup = _compile(mapping)
    if rx is None:
        return text
    return rx.sub(lambda m: lookup[m.group(0).lower()], text)


def load_map(path: Path) -> dict[str, str]:
    """Read the lexicon JSON, or {} if it does not exist yet."""
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def save_map(path: Path, mapping: dict[str, str]) -> None:
    """Atomically write the lexicon JSON (create the parent dir if needed)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    tmp.replace(p)


def add_entry(path: Path, word: str, respelling: str) -> dict[str, str]:
    """Add/overwrite ``word`` -> ``respelling`` and persist. Returns the new map."""
    mapping = load_map(path)
    mapping[word] = respelling
    save_map(path, mapping)
    return mapping


def remove_entry(path: Path, word: str) -> tuple[dict[str, str], bool]:
    """Remove ``word`` (case-insensitive) if present. Returns (map, removed?)."""
    mapping = load_map(path)
    hit = next((k for k in mapping if k.lower() == word.lower()), None)
    if hit is None:
        return mapping, False
    del mapping[hit]
    save_map(path, mapping)
    return mapping, True


class Lexicon:
    """Mtime-cached view of the lexicon file for the server's hot path.

    Reloads (and recompiles the regex) only when the file's mtime changes, so a
    REPL ``:pron`` edit is picked up on the next request without a server
    restart, and the steady-state cost per request is one ``stat`` + a regex sub.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._mtime: float | None = None
        self._rx: re.Pattern | None = None
        self._lookup: dict[str, str] = {}

    def _refresh(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            self._mtime, self._rx, self._lookup = None, None, {}
            return
        if mtime != self._mtime:
            self._mtime = mtime
            self._rx, self._lookup = _compile(load_map(self.path))

    def apply(self, text: str) -> str:
        self._refresh()
        if self._rx is None:
            return text
        return self._rx.sub(lambda m: self._lookup[m.group(0).lower()], text)
