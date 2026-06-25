from pydantic import BaseModel


class SpeechParams(BaseModel):
    exaggeration: float = 0.5
    cfg_weight: float = 0.4
    temperature: float = 0.8
    speed: float = 1.0


class SpeechRequest(BaseModel):
    input: str
    backend: str = "piper"
    voice: str = "glados"
    params: SpeechParams = SpeechParams()
    response_format: str = "pcm"
    stream: bool = True


class VoiceInfo(BaseModel):
    id: str
    backend: str
    source: str
    description: str = ""
    cb: bool = False  # generation runs through Chatterbox -> the cb knobs apply
    # The voice's tuned cb-knob defaults (exaggeration/cfg_weight/temperature),
    # for an RVC voice whose source is a Chatterbox clone. A client preloads
    # these on :voice so the knobs round-trip the character's defaults until the
    # user tunes them. None for voices with no fixed source params.
    cb_params: dict | None = None


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def clamp_params(params: SpeechParams) -> SpeechParams:
    return SpeechParams(
        exaggeration=_clamp(params.exaggeration, 0.0, 2.0),
        cfg_weight=_clamp(params.cfg_weight, 0.0, 1.0),
        temperature=_clamp(params.temperature, 0.1, 2.0),
        speed=_clamp(params.speed, 0.5, 2.0),
    )
