import numpy as np
import torch
from chatterbox.tts import ChatterboxTTS
from scipy.signal import resample_poly

from hosaka.config import SAMPLE_RATE
from hosaka.library import VoiceLibrary


class ChatterboxEngine:
    """Original Chatterbox cloning + tuning, run as a NON-realtime QUALITY mode.

    On the RTX 5070 Ti this model runs at RTF ~1.0, so per-chunk streaming
    underruns and stutters. Instead we generate each fragment in full, then
    hand the whole waveform to the caller in one piece. The server's
    fragment loop overlaps a fragment's generation with the previous
    fragment's playback, so audio stays smooth at the cost of ~1-2s before
    the first fragment is heard. Keeps the full knob set
    (exaggeration / cfg_weight / temperature).
    """

    def __init__(self, library: VoiceLibrary):
        self.library = library
        self.model = ChatterboxTTS.from_pretrained(device="cuda")
        self.sr = int(self.model.sr)

    def warmup(self) -> None:
        for _ in self.stream("warm up.", "", {}):
            pass

    def stream(self, text, voice, params):
        ref = self.library.path_for(voice)
        kw = dict(
            exaggeration=float(params.get("exaggeration", 0.5)),
            cfg_weight=float(params.get("cfg_weight", 0.4)),
            temperature=float(params.get("temperature", 0.8)),
        )
        if ref is not None:
            kw["audio_prompt_path"] = str(ref)
        with torch.inference_mode():
            wav = self.model.generate(text, **kw)
        arr = wav.detach().cpu().numpy().astype(np.float32).reshape(-1)
        if self.sr != SAMPLE_RATE:
            arr = resample_poly(arr, SAMPLE_RATE, self.sr).astype(np.float32)
        yield np.ascontiguousarray(arr)
