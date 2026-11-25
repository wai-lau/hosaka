"""Pure mode helpers for the gpu-mode service. No I/O, no torch -- safe to
import under .venv-dev."""

VALID_ACTIONS = ("homo", "lomo", "emo", "idle")
_DISPLAY = {a: a for a in VALID_ACTIONS}


def parse_mode(raw: str) -> str:
    """Map gpu_mode.sh stdout to a display mode. `mixed` (both services up) is an
    invariant violation that must never reach the UI; collapse it to `idle` --
    the next status read after a settled action returns the real mode."""
    return _DISPLAY.get((raw or "").strip(), "idle")
