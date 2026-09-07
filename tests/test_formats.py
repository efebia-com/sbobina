import json

import pytest

from sbobina.formats import JSON, SRT, TXT, VTT, UnknownFormatError, render
from sbobina.transcribe import Segment


SEGMENTS = [
    Segment(index=1, start=0.0, end=2.5, text="Good morning everyone."),
    Segment(index=2, start=2.5, end=3661.125, text="Let us move on to item two."),
]


def test_srt_separates_milliseconds_with_a_comma():
    assert "00:00:00,000 --> 00:00:02,500" in render(SEGMENTS, SRT)


def test_vtt_separates_milliseconds_with_a_dot_and_opens_with_the_header():
    out = render(SEGMENTS, VTT)
    assert out.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:02.500" in out


def test_hours_are_not_truncated_past_sixty_minutes():
    assert "01:01:01,125" in render(SEGMENTS, SRT)


def test_txt_holds_the_text_alone_with_no_timestamps():
    out = render(SEGMENTS, TXT)
    assert "Good morning everyone." in out
    assert ":" not in out.replace("Good morning everyone.", "").replace("Let us move on to item two.", "")


def test_json_exposes_the_joined_text_alongside_the_segments():
    payload = json.loads(render(SEGMENTS, JSON))
    assert payload["text"] == "Good morning everyone. Let us move on to item two."
    assert payload["segments"][1]["start"] == 2.5


def test_an_unknown_format_raises_and_lists_the_valid_ones():
    with pytest.raises(UnknownFormatError, match="srt"):
        render(SEGMENTS, "docx")
