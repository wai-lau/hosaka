from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Engine(Protocol):
    def stream(self, text: str, voice: str, params: dict) -> Iterator[np.ndarray]: ...

    def warmup(self) -> None: ...


@dataclass
class EngineRegistry:
    kokoro: Engine | None = None
    chatterbox: Engine | None = None
    piper: Engine | None = None  # optional CPU sidecar (character voices)
    rvc: Engine | None = None  # optional GPU sidecar (converted character voices)

    def get(self, backend: str) -> Engine | None:
        if backend == "kokoro":
            return self.kokoro
        if backend == "chatterbox":
            return self.chatterbox
        if backend == "piper":
            if self.piper is None:
                raise KeyError("piper backend not available")
            return self.piper
        if backend == "rvc":
            if self.rvc is None:
                raise KeyError("rvc backend not available")
            return self.rvc
        raise KeyError(f"unknown backend: {backend}")

    def warmup_all(self) -> None:
        if self.kokoro is not None:
            self.kokoro.warmup()
        if self.chatterbox is not None:
            self.chatterbox.warmup()
        if self.piper is not None:
            self.piper.warmup()
        if self.rvc is not None:
            self.rvc.warmup()
