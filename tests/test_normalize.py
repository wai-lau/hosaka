"""Clock-time normalization (engine-agnostic pre-chunk pass)."""

import pytest

from hosaka.normalize import normalize_times


@pytest.mark.parametrize(
    "raw, want",
    [
        # 24-hour: the Chatterbox "thirteen thousand" bug.
        ("13:45", "thirteen forty-five"),
        ("at 13:45 sharp", "at thirteen forty-five sharp"),
        # 12-hour with meridiem (am/pm must survive).
        ("12:34pm", "twelve thirty-four PM"),
        ("11:30AM", "eleven thirty AM"),
        ("9:05 p.m.", "nine oh five PM"),
        ("9:05am", "nine oh five AM"),
        # o'clock and oh-minute forms.
        ("13:00", "thirteen o'clock"),
        ("8:00", "eight o'clock"),
        ("6:07", "six oh seven"),
        ("00:30", "zero thirty"),
        ("23:59", "twenty-three fifty-nine"),
    ],
)
def test_spoken_times(raw, want):
    assert normalize_times(raw) == want


@pytest.mark.parametrize(
    "raw",
    [
        "http://example.com",  # not a clock
        "ratio 3:4 here",  # minute out of range -> left alone
        "24:00",  # hour out of range
        "13:99",  # minute out of range (no match)
        "plain text",
        "13:45pm",  # 24h hour + meridiem is invalid -> verbatim
    ],
)
def test_non_times_untouched(raw):
    assert normalize_times(raw) == raw


def test_multiple_in_one_string():
    out = normalize_times("from 9:00 to 17:30")
    assert out == "from nine o'clock to seventeen thirty"
