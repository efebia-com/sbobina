"""Renders segments into the output formats. Pure functions: data in, string out."""

import json
from collections.abc import Callable

from sbobina.transcribe import Segment


TXT = "txt"
SRT = "srt"
VTT = "vtt"
JSON = "json"
FORMATS = (TXT, SRT, VTT, JSON)


class UnknownFormatError(ValueError):
    """Unrecognised output format."""


def _clock(seconds: float, separator: str) -> str:
    """HH:MM:SS<sep>mmm — the millisecond separator is what tells SRT from VTT."""
    whole = int(seconds)
    milliseconds = round((seconds - whole) * 1000)
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{milliseconds:03d}"


def _render_txt(segments: list[Segment]) -> str:
    """Text only, one segment per line.

    No timestamps: this is the format you paste into a document or hand to an LLM,
    and there timestamps are noise. Anyone who wants them uses SRT.
    """
    return "\n".join(segment.text for segment in segments)


def _render_srt(segments: list[Segment]) -> str:
    blocks = [
        f"{segment.index}\n{_clock(segment.start, ',')} --> {_clock(segment.end, ',')}\n{segment.text}"
        for segment in segments
    ]
    return "\n\n".join(blocks)


def _render_vtt(segments: list[Segment]) -> str:
    blocks = [f"{_clock(segment.start, '.')} --> {_clock(segment.end, '.')}\n{segment.text}" for segment in segments]
    return "WEBVTT\n\n" + "\n\n".join(blocks)


def _render_json(segments: list[Segment]) -> str:
    """A stable structure for anyone consuming sbobina from a script.

    The keys are these and they do not change: `text` with the whole transcript
    already joined (that is the most common use, and whoever reassembles it
    downstream gets the spacing wrong), and `segments` with the timings for those
    who need them.
    """
    payload = {
        "text": " ".join(segment.text for segment in segments),
        "segments": [
            {"index": s.index, "start": round(s.start, 3), "end": round(s.end, 3), "text": s.text} for s in segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


_RENDERERS: dict[str, Callable[[list[Segment]], str]] = {
    TXT: _render_txt,
    SRT: _render_srt,
    VTT: _render_vtt,
    JSON: _render_json,
}


def render(segments: list[Segment], output_format: str) -> str:
    """Serialise the segments into the requested format."""
    renderer = _RENDERERS.get(output_format)
    if renderer is None:
        known = ", ".join(FORMATS)
        raise UnknownFormatError(f"unknown format '{output_format}'; available: {known}")
    return renderer(segments) + "\n"
