from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Engine(Protocol):
    def stream(self, text: str, voice: str, params: dict) -> Iterator[np.ndarray]: ...

    def warmup(self) -> None: ...


@dataclass
class EngineRegistry:
    kokoro: Engine
    chatterbox: Engine

    def get(self, backend: str) -> Engine:
        if backend == "kokoro":
            return self.kokoro
        if backend == "chatterbox":
            return self.chatterbox
        raise KeyError(f"unknown backend: {backend}")

    def warmup_all(self) -> None:
        self.kokoro.warmup()
        self.chatterbox.warmup()
