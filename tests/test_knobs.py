import pytest

from hosaka.config import KNOB_RANGES, native_to_pct, pct_to_native


def test_fifty_maps_to_default():
    for name, (_lo, mid, _hi) in KNOB_RANGES.items():
        assert pct_to_native(name, 50) == pytest.approx(mid)


def test_endpoints_map_to_native_bounds():
    for name, (lo, _mid, hi) in KNOB_RANGES.items():
        assert pct_to_native(name, 0) == pytest.approx(lo)
        assert pct_to_native(name, 100) == pytest.approx(hi)


def test_speed_fifty_is_normal_rate():
    assert pct_to_native("speed", 50) == pytest.approx(1.0)


def test_out_of_range_pct_is_clamped():
    assert pct_to_native("speed", -20) == pytest.approx(0.5)
    assert pct_to_native("speed", 250) == pytest.approx(2.0)


def test_native_to_pct_round_trips():
    for name in KNOB_RANGES:
        for pct in (0, 25, 50, 75, 100):
            assert native_to_pct(name, pct_to_native(name, pct)) == pytest.approx(pct)
