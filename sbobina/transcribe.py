"""The engine: an audio or video file in, transcription events out.

It prints nothing, and it is a generator rather than a function that returns at
the end. Whisper decodes one segment at a time and on a long recording minutes
pass before the last one: returning only the final result would leave the CLI
with nothing to show for all that time. This way the CLI receives every segment
as soon as it exists and can draw a real progress bar, anchored to the audio
duration.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from sbobina import backend as backend_module
from sbobina.config import CPU_BATCH_SIZE, DEFAULT_GPU_BATCH_SIZE, DEFAULT_MODEL, MODELS
from sbobina.log import log


@dataclass(frozen=True)
class Segment:
    """A block of speech with its timings, in seconds from the start of the file."""

    index: int
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Started:
    """Emitted as soon as Whisper knows language and duration, before decoding."""

    language: str
    language_probability: float
    duration: float


@dataclass(frozen=True)
class Decoded:
    """A freshly decoded segment."""

    segment: Segment


@dataclass(frozen=True)
class Finished:
    """End of the transcription, with every segment and the time it took."""

    segments: list[Segment]
    elapsed: float


Event: TypeAlias = Started | Decoded | Finished


@dataclass(frozen=True)
class Options:
    """Everything the user can change about *decoding*. One dataclass, not eight parameters.

    Device and compute type deliberately do not live here: they are resolved once
    by `backend.resolve()` and travel as a `Backend`. Keeping a second copy in
    Options would let a caller set `Options(device="cpu")` and be ignored without
    a word, because the engine reads the Backend and nothing else.
    """

    model: str = DEFAULT_MODEL
    language: str | None = None
    batch_size: int | None = None
    vad: bool = True
    model_dir: Path | None = None


class UnknownModelError(ValueError):
    """Unrecognised model alias."""


def resolve_model(name: str) -> str:
    """From alias ('turbo') to Hugging Face repo, letting explicit repos through.

    Anyone wanting their own checkpoint passes it as 'user/repo': the slash is the
    criterion, because none of the aliases contains one.
    """
    if "/" in name:
        return name
    if name not in MODELS:
        known = ", ".join(MODELS)
        raise UnknownModelError(f"unknown model '{name}'; available: {known}")
    return MODELS[name]


def resolve_batch_size(requested: int | None, device: str, *, vad: bool) -> int:
    """How many windows to decode together.

    `vad` is keyword-only and has no default deliberately: a default is what lets
    a new caller forget the constraint below and bring the crash back, and a
    positional boolean would trip FBT001 under `select = ["ALL"]` anyway.
    """
    # Any batch above 1 sends transcribe() down the BatchedInferencePipeline path,
    # and that pipeline cuts its windows out of the clip_timestamps the VAD
    # produces. With the VAD off there is no other source for them: faster-whisper
    # raises "No clip timestamps found" on any audio longer than its 30 s chunk,
    # which is every real recording. Below 30 s it quietly falls back to one clip
    # covering the whole file, and that is the only reason --no-vad ever looked
    # like it worked — short test samples pass, the meeting recording does not.
    # So this wins over an explicit --batch-size instead of honouring it: the 1 is
    # a hard requirement of the library, not a tuning preference, and honouring
    # the 8 buys the user nothing but a RuntimeError raised after the model has
    # finished loading (82 s on a first run, download included).
    if not vad:
        return 1
    if requested is not None:
        return max(1, requested)
    return DEFAULT_GPU_BATCH_SIZE if device == backend_module.DEVICE_CUDA else CPU_BATCH_SIZE


def transcribe(audio: Path, options: Options, resolved: backend_module.Backend) -> Iterator[Event]:
    """Transcribe *audio*, emitting one event per decoded segment.

    The backend arrives already resolved instead of being computed here: the CLI
    shows it to the user before the model download starts, which on a first run is
    the longest wait of all.
    """
    # Deferred import: WhisperModel pulls in ctranslate2, which must only be
    # loaded after cuda_libs.register() (see backend.ensure_available).
    backend_module.ensure_available()
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    repo = resolve_model(options.model)
    batch_size = resolve_batch_size(options.batch_size, resolved.device, vad=options.vad)
    log.info("loading model", repo=repo, device=resolved.device, compute_type=resolved.compute_type)

    model = WhisperModel(
        repo,
        device=resolved.device,
        compute_type=resolved.compute_type,
        download_root=str(options.model_dir) if options.model_dir else None,
    )

    started_at = time.monotonic()

    # `condition_on_previous_text=False` is not negotiable: with True, Whisper
    # uses the text it has already produced as context, and on a silence or a
    # patch of dirty audio it enters a loop, repeating the same sentence for
    # minutes. It is the most visible failure the model can produce, and it costs
    # some coherence across long-range references — a trade worth making.
    if batch_size > 1:
        segments_iter, info = BatchedInferencePipeline(model=model).transcribe(
            str(audio),
            batch_size=batch_size,
            language=options.language,
            condition_on_previous_text=False,
            vad_filter=options.vad,
        )
    else:
        # The same parameters, repeated rather than collected in a dict and
        # unpacked: faster-whisper is typed, and a heterogeneous `**kwargs` makes
        # it lose every check (ty produced 62 false positives on these two calls).
        # Two extra lines are worth the type checking.
        segments_iter, info = model.transcribe(
            str(audio),
            language=options.language,
            condition_on_previous_text=False,
            vad_filter=options.vad,
        )

    yield Started(
        language=info.language,
        language_probability=info.language_probability,
        duration=info.duration,
    )

    collected: list[Segment] = []
    for raw in segments_iter:
        text = raw.text.strip()
        if not text:
            continue
        segment = Segment(index=len(collected) + 1, start=raw.start, end=raw.end, text=text)
        collected.append(segment)
        yield Decoded(segment)

    elapsed = time.monotonic() - started_at
    log.info("transcription complete", segments=len(collected), seconds=round(elapsed, 1))
    yield Finished(segments=collected, elapsed=elapsed)
