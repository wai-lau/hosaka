from pydantic import BaseModel


class SpeechParams(BaseModel):
    exaggeration: float = 0.5
    cfg_weight: float = 0.4
    temperature: float = 0.8
    speed: float = 1.0


class SpeechRequest(BaseModel):
    input: str
    backend: str = "kokoro"
    voice: str = "af_heart"
    params: SpeechParams = SpeechParams()
    response_format: str = "pcm"
    stream: bool = True


class VoiceInfo(BaseModel):
    id: str
    backend: str
    source: str
    description: str = ""


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def clamp_params(params: SpeechParams) -> SpeechParams:
    return SpeechParams(
        exaggeration=_clamp(params.exaggeration, 0.0, 2.0),
        cfg_weight=_clamp(params.cfg_weight, 0.0, 1.0),
        temperature=_clamp(params.temperature, 0.1, 2.0),
        speed=_clamp(params.speed, 0.5, 2.0),
    )
