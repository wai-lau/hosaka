import numpy as np
import torch
from chatterbox.tts import ChatterboxTTS
from hosaka.config import SAMPLE_RATE, DEFAULT_VOICE
from hosaka.library import VoiceLibrary

_XFADE = int(0.020 * SAMPLE_RATE)   # 20 ms crossfade


class ChatterboxEngine:
    """Original Chatterbox via the streaming fork. Cloning + tuning."""

    def __init__(self, library: VoiceLibrary):
        self.library = library
        self.model = ChatterboxTTS.from_pretrained(device="cuda")

    def warmup(self) -> None:
        # Warm up with a tiny utterance using the model's built-in voice
        # (no reference) so kernels/caches are allocated.
        for _ in self._raw_stream("warm up.", None,
                                  {"exaggeration": 0.5, "cfg_weight": 0.4,
                                   "temperature": 0.8}):
            pass

    def stream(self, text, voice, params):
        ref = self.library.path_for(voice)
        ref_path = str(ref) if ref else None
        prev_tail = None
        for chunk in self._raw_stream(text, ref_path, params):
            if prev_tail is not None and len(chunk) > _XFADE:
                fade = np.linspace(0.0, 1.0, _XFADE, dtype=np.float32)
                chunk[:_XFADE] = chunk[:_XFADE] * fade + prev_tail * (1.0 - fade)
            prev_tail = chunk[-_XFADE:].copy() if len(chunk) >= _XFADE else None
            yield chunk

    def _raw_stream(self, text, ref_path, params):
        kw = dict(
            exaggeration=float(params.get("exaggeration", 0.5)),
            cfg_weight=float(params.get("cfg_weight", 0.4)),
            temperature=float(params.get("temperature", 0.8)),
            chunk_size=50,
        )
        if ref_path:
            kw["audio_prompt_path"] = ref_path
        with torch.inference_mode():
            for audio_chunk, _metrics in self.model.generate_stream(text, **kw):
                arr = audio_chunk.detach().cpu().numpy().astype(np.float32)
                yield np.ascontiguousarray(arr.reshape(-1))
