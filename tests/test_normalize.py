"""Clock-time normalization (engine-agnostic pre-chunk pass)."""

import pytest

from hosaka.normalize import normalize_times


@pytest.mark.parametrize(
    "raw, want",
    [
        # 24-hour: the Chatterbox "thirteen thousand" bug.
        ("13:45", "thirteen forty-five"),
        ("at 13:45 sharp", "at thirteen forty-five sharp"),
        # 12-hour with meridiem (am/pm must survive). Rendered "A.M"/"P.M":
        # espeak reads a bare "AM" after "one"/"two" as the word "am", and a
        # trailing dot would be a sentence end for espeak and the chunker.
        ("12:34pm", "twelve thirty-four P.M"),
        ("11:30AM", "eleven thirty A.M"),
        # 24h hour + stray meridiem: speak 24h form, drop pm (never verbatim).
        ("13:45pm", "thirteen forty-five"),
        # The writer's own trailing period stays (it may end the sentence).
        ("9:05 p.m.", "nine oh five P.M."),
        ("9:05am", "nine oh five A.M"),
        # Bare hour + meridiem, any spacing / case / dots: "1am" -> "one A.M".
        ("1am", "one A.M"),
        ("1AM", "one A.M"),
        ("1 Am", "one A.M"),
        ("1 a.m.", "one A.M."),
        ("1 A.M.", "one A.M."),
        ("12pm", "twelve P.M"),
        ("10 PM", "ten P.M"),
        ("at 7pm sharp", "at seven P.M sharp"),
        ("back at 1am.", "back at one A.M."),
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
        "13am",  # bare hour + meridiem only on a 12h clock
        "0am",
        "1amp",  # letters run on -> not a meridiem
        "x1am",  # no word boundary before the hour
        "team",  # "am" inside a word, no hour
    ],
)
def test_non_times_untouched(raw):
    assert normalize_times(raw) == raw


def test_multiple_in_one_string():
    out = normalize_times("from 9:00 to 17:30")
    assert out == "from nine o'clock to seventeen thirty"
