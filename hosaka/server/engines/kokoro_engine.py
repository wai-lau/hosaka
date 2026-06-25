import numpy as np
from kokoro import KPipeline

from hosaka.config import KOKORO_ALIASES, KOKORO_WARMUP_VOICE


class KokoroEngine:
    """Kokoro-82M. Presets + speed. 24 kHz mono output."""

    def __init__(self, lang_code: str = "a"):  # 'a' = American English
        self._pipe = KPipeline(lang_code=lang_code)

    def warmup(self) -> None:
        for _ in self.stream("warm up.", KOKORO_WARMUP_VOICE, {"speed": 1.0}):
            pass

    def stream(self, text, voice, params):
        speed = float(params.get("speed", 1.0))
        voice = KOKORO_ALIASES.get(voice, voice)  # display id -> real embedding
        for _gs, _ps, audio in self._pipe(text, voice=voice, speed=speed):
            arr = audio.detach().cpu().numpy().astype(np.float32)
            yield np.ascontiguousarray(arr.reshape(-1))
