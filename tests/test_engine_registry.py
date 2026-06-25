import pytest

from hosaka.server.engines.base import EngineRegistry


class Spy:
    def __init__(self):
        self.warmed = False

    def stream(self, text, voice, params):
        yield None

    def warmup(self):
        self.warmed = True


def _reg(**kw):
    return EngineRegistry(kokoro=Spy(), chatterbox=Spy(), **kw)


def test_get_rvc_returns_engine_when_present():
    rvc = Spy()
    assert _reg(rvc=rvc).get("rvc") is rvc


def test_get_rvc_raises_when_absent():
    with pytest.raises(KeyError, match="rvc"):
        _reg().get("rvc")


def test_warmup_all_warms_rvc_when_present():
    rvc = Spy()
    _reg(rvc=rvc).warmup_all()
    assert rvc.warmed


def test_warmup_all_skips_rvc_when_absent():
    _reg().warmup_all()  # must not raise
