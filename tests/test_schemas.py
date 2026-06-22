from hosaka.schemas import SpeechRequest, SpeechParams, clamp_params


def test_request_defaults():
    r = SpeechRequest(input="hi")
    assert r.backend == "kokoro"
    assert r.voice == "af_heart"
    assert r.params.exaggeration == 0.5
    assert r.stream is True


def test_clamp_bounds_values():
    p = SpeechParams(exaggeration=5.0, cfg_weight=-1.0,
                     temperature=9.0, speed=0.01)
    c = clamp_params(p)
    assert 0.0 <= c.exaggeration <= 2.0
    assert 0.0 <= c.cfg_weight <= 1.0
    assert 0.1 <= c.temperature <= 2.0
    assert 0.5 <= c.speed <= 2.0


def test_clamp_leaves_valid_values():
    p = SpeechParams(exaggeration=0.6, cfg_weight=0.4,
                     temperature=0.8, speed=1.1)
    assert clamp_params(p) == p
