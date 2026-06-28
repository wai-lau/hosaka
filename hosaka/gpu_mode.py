"""Pure mode helpers for the gpu-mode service. No I/O, no torch -- safe to
import under .venv-dev."""

VALID_ACTIONS = ("homo", "emo", "idle")
_DISPLAY = {"homo": "homo", "emo": "emo", "idle": "idle"}


def parse_mode(raw: str) -> str:
    """Map gpu_mode.sh stdout to a display mode. `mixed` (both services up) is an
    invariant violation that must never reach the UI; collapse it to `idle` --
    the next status read after a settled action returns the real mode."""
    return _DISPLAY.get((raw or "").strip(), "idle")
