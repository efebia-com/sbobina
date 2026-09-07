"""The only module that prints. Everything else returns data."""

import io
import sys
from dataclasses import dataclass
from pathlib import Path

import rich_click as click
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from sbobina import backend as backend_module
from sbobina import formats
from sbobina.config import CPU_BATCH_SIZE, DEFAULT_GPU_BATCH_SIZE, DEFAULT_MODEL, LOG_DIR, MODELS
from sbobina.log import log, setup_logging
from sbobina.transcribe import Decoded, Finished, Options, Segment, Started, transcribe


# On stderr, not stdout: with `-o -` the transcription goes to stdout to be piped,
# and the backend line and progress bar would end up inside the copied text. On a
# normal terminal nothing changes — everything is still visible — but
# `sbobina x.m4a -o - | pbcopy` copies the transcription alone.
console = Console(stderr=True)

# Write to stdout instead of a file: handy for `sbobina x.m4a -o - | pbcopy`.
STDOUT = "-"

# Below this, an auto-detected language is worth questioning out loud. Whisper
# transcribing Italian audio as if it were English does not fail — it returns
# fluent nonsense, which is far worse than an error, and the confidence score is
# the only warning anyone gets.
_LOW_CONFIDENCE = 0.5


@dataclass(frozen=True)
class RunResult:
    """What one transcription produced. A named triple beats a bare tuple."""

    segments: list[Segment]
    duration: float
    elapsed: float


def _fail(message: str) -> None:
    console.print(f"[bold red]error[/] {message}")
    sys.exit(1)


def _reason(exc: Exception) -> str:
    """A one-line cause the user can act on.

    PyAV's exceptions stringify to the file name alone: `str()` on the
    InvalidDataError raised for something ffmpeg cannot parse hands back the path
    that was just typed and nothing else. The actual sentence ("Invalid data
    found when processing input") lives in `strerror`, which OSError-shaped
    exceptions carry and plain ones do not.
    """
    strerror = getattr(exc, "strerror", None)
    return str(strerror) if strerror else str(exc)


def _describe(resolved: backend_module.Backend) -> None:
    """The line telling the user what is about to happen.

    It is the first thing you ask a colleague when "it's dead slow": nine times
    out of ten this line says `cpu` and the question has already been answered.
    """
    where = resolved.gpu_name or resolved.device.upper()
    console.print(f"[dim]backend[/] {where} · {resolved.compute_type}")
    if resolved.note:
        console.print(f"[yellow]note[/] {resolved.note}")


def _output_path(audio: Path, output: str | None, output_format: str) -> Path | None:
    """None means stdout."""
    if output == STDOUT:
        return None
    if output:
        return Path(output)
    return audio.with_suffix(f".{output_format}")


def _warn_if_overwriting(destination: Path | None) -> None:
    """Say it *before* decoding, not after.

    The report line at the end already names the file it wrote, but by then the
    old one is gone. On an hour-long recording the useful moment to learn that
    `meeting.txt` already exists is the moment there is still time to hit Ctrl-C.
    """
    if destination is not None and destination.exists():
        console.print(f"[yellow]note[/] {destination} exists and will be overwritten")


def _write(text: str, destination: Path | None) -> None:
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Written beside the destination and then moved onto it, because
        # `write_text` opens in "w" and truncates before the first byte. A failure
        # halfway through used to leave, in place of the previous transcript, a
        # file of arbitrary length that looks like a valid one: measured on a full
        # tmpfs, `OSError: [Errno 122] Disk quota exceeded` left a 0-byte output
        # sitting where the good transcript had been. `os.replace` is atomic on
        # the same filesystem, Windows included, so the destination ends up either
        # the old file or the whole new one — which is also why the temporary is a
        # sibling instead of something under the system temp dir: across
        # filesystems the move stops being a rename and becomes a copy.
        temporary = destination.with_name(f"{destination.name}.partial")
        try:
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(destination)
        except OSError:
            # The half-written file is worth removing on the way out: on the
            # disk-full path it is holding the very space a retry needs, and
            # otherwise it stays next to the transcript the user still has,
            # looking plausible enough to be mistaken for the output.
            temporary.unlink(missing_ok=True)
            raise
        return

    # Windows falls back to the ANSI code page for stdout as soon as it is
    # redirected, so a transcript holding anything outside cp1252 dies with
    # UnicodeEncodeError exactly when it is being piped somewhere — the one case
    # where the user cannot see the traceback scroll past. Files already go out as
    # UTF-8; this makes the pipe agree with them. Guarded by isinstance because
    # under pytest, and under any other capture, stdout is not a TextIOWrapper.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(text)
    # Flushed here rather than left to interpreter shutdown. click already handles
    # EPIPE (core.py:1571): it installs a _PacifyFlushWrapper and exits 1 — but
    # only for an error raised while the command is still running. With stdout
    # buffered, which is what it is as soon as it is a pipe, the write above
    # raises nothing, click returns satisfied, and the closing flush explodes
    # where nothing protects it any more: "Exception ignored while flushing
    # sys.stdout" and exit 120, after _report has already announced `done`. The
    # threshold is not the 8 KB stdio buffer but the kernel's 64 KB pipe buffer,
    # and a real meeting transcript sits below it — so `sbobina x.m4a -o - | head`
    # claimed success and delivered nothing at all.
    sys.stdout.flush()


def _note_language(event: Started, options: Options) -> None:
    if options.language is not None or event.language_probability >= _LOW_CONFIDENCE:
        return
    console.print(
        f"[yellow]note[/] language detected as [cyan]{event.language}[/] with low confidence "
        f"({event.language_probability:.0%}) — pass -l to force it"
    )


def _run(audio: Path, options: Options, resolved: backend_module.Backend) -> RunResult:
    """Consume the events while drawing the bar."""
    columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )
    segments: list[Segment] = []
    duration = 0.0
    elapsed = 0.0

    with Progress(*columns, console=console) as progress:
        # The bar starts indeterminate and with the wrong label on purpose: before
        # transcribing anything the model has to be loaded, and on a first run
        # that is a 1.6 GB download lasting longer than the transcription itself.
        # Saying "transcribing" there would make the tool look hung at exactly the
        # moment the user does not yet know whether it works.
        # The audio duration arrives with the first event, not before: it is
        # Whisper itself that decodes the file header.
        task = progress.add_task("loading model", total=None)
        for event in transcribe(audio, options, resolved):
            if isinstance(event, Started):
                duration = event.duration
                progress.update(task, total=event.duration, description=f"transcribing [cyan]{event.language}[/]")
                _note_language(event, options)
            elif isinstance(event, Decoded):
                progress.update(task, completed=event.segment.end)
            elif isinstance(event, Finished):
                segments = event.segments
                elapsed = event.elapsed
                progress.update(task, completed=duration or 1.0)

    return RunResult(segments=segments, duration=duration, elapsed=elapsed)


def _report(result: RunResult, destination: Path | None) -> None:
    """The closing line, and it only ever describes a transcript that exists.

    The empty case used to be handled here, which was one line too late: `_write`
    had already truncated the destination to a newline by the time this function
    said "no speech detected". It now belongs to the guard in `cli()`, which runs
    before anything is written.
    """
    fast = result.elapsed > 0 and result.duration > 0
    speed = f" · {result.duration / result.elapsed:.0f}x realtime" if fast else ""
    where = "stdout" if destination is None else str(destination)
    console.print(f"[green]done[/] {len(result.segments)} segments in {result.elapsed:.0f}s{speed} → [bold]{where}[/]")


@click.command()
@click.argument("audio", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False)
@click.option("-o", "--output", default=None, help=f"Destination file ('{STDOUT}' for stdout).")
@click.option(
    "-f",
    "--format",
    "output_format",
    type=click.Choice(formats.FORMATS),
    default=formats.TXT,
    show_default=True,
    help="Output format.",
)
@click.option("-l", "--lang", "language", default=None, help="Language code (it, en, ...). Default: auto-detect.")
@click.option(
    "-m",
    "--model",
    default=DEFAULT_MODEL,
    show_default=True,
    help=f"Alias ({', '.join(MODELS)}) or a Hugging Face repo.",
)
@click.option(
    "--device",
    type=click.Choice(backend_module.DEVICES),
    default=backend_module.DEVICE_AUTO,
    help="Force CPU or GPU. Default: GPU when there is one.",
)
@click.option(
    "--compute-type",
    default=backend_module.COMPUTE_AUTO,
    help="float16, int8, float32... Default: the best the card supports.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help=f"Windows decoded in parallel. Default: {DEFAULT_GPU_BATCH_SIZE} on GPU, {CPU_BATCH_SIZE} on CPU.",
)
@click.option("--no-vad", is_flag=True, help="Do not trim silences before transcribing.")
@click.option(
    "--model-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where to keep downloaded models. Default: the Hugging Face cache.",
)
@click.option("--check", is_flag=True, help="Show the detected device and compute type, then exit.")
# package_name spelled out: `rich_click.version_option` is its own function, not
# a re-export of click's, so click's frame-walking autodetection lands on
# `rich_click.decorators` and prints the version of rich-click. The project is
# about to be read by colleagues who will open issues, and the first line of every
# one of them would name a version that does not exist — and that moves on its own
# whenever `uv lock` updates rich-click.
@click.version_option(package_name="sbobina")
def cli(
    audio: Path | None,
    output: str | None,
    output_format: str,
    language: str | None,
    model: str,
    device: str,
    compute_type: str,
    batch_size: int | None,
    no_vad: bool,
    model_dir: Path | None,
    check: bool,
) -> None:
    """Transcribe an audio or video file with Whisper.

    Examples:
        sbobina meeting.m4a

        sbobina lecture.mp4 -l en -f srt

        sbobina interview.wav -o - | pbcopy
    """
    setup_logging(LOG_DIR)

    try:
        resolved = backend_module.resolve(device, compute_type)
    except backend_module.UnsupportedComputeTypeError as exc:
        _fail(str(exc))
        return

    _describe(resolved)
    if check:
        return
    if audio is None:
        _fail("no file to transcribe (use --check for diagnosis only)")
        return

    destination = _output_path(audio, output, output_format)
    _warn_if_overwriting(destination)

    options = Options(
        model=model,
        language=language,
        batch_size=batch_size,
        vad=not no_vad,
        model_dir=model_dir,
    )

    try:
        result = _run(audio, options, resolved)
    except Exception as exc:
        # Best-effort boundary facing the world: a non-existent model, a file
        # ffmpeg cannot open, exhausted VRAM all land here. The traceback is
        # already in the log; the user needs the reason, not the stack.
        log.exception("transcription failed", audio=str(audio))
        _fail(f"{audio.name}: {_reason(exc)}  [dim](details in {LOG_DIR / 'sbobina.log'})[/]")
        return

    if not result.segments:
        # Nothing to write, so nothing is written, and the exit code says so.
        # Rendering an empty transcript yields a single newline, and writing it
        # truncated whatever stood at the destination while still exiting 0:
        # measured, `sbobina meeting.mp4 -o minutes.txt && rm meeting.mp4` passed
        # the `&&`, left a 1-byte minutes.txt and deleted the 28 MB recording,
        # with nothing on screen to say so. The overwrite warning cannot cover
        # this — it fires on every run, long before anyone can know the result
        # will be empty. The non-zero exit is the part that matters: a user learns
        # to distrust a tool that prints an error, never one that exits 0.
        untouched = "" if destination is None else f", {destination} left untouched"
        _fail(f"{audio.name}: no speech detected{untouched}")
        return

    try:
        _write(formats.render(result.segments, output_format), destination)
    except BrokenPipeError:
        # `-o - | head` is the pipeline working, not sbobina failing: the reader
        # left early. Re-raised on purpose so it reaches click, which recognises
        # EPIPE and exits quietly; catching it here would mean printing an error
        # for a pipeline that did what it was asked. Skipping the `done` line
        # below is the point — that line was the lie. It has to sit before the
        # OSError branch, which would otherwise match first (BrokenPipeError is an
        # OSError) and blame a destination that is stdout.
        raise
    except OSError as exc:
        # The one failure that arrives with the work already done: the transcript
        # is in memory and about to be thrown away. The boundary above covers
        # `_run` and stops there, so a read-only destination, a path that is a
        # directory or a full disk used to answer an hour of transcription with a
        # raw Python traceback.
        log.exception("write failed", destination=str(destination))
        _fail(f"cannot write {destination}: {_reason(exc)}")
        return

    _report(result, destination)


if __name__ == "__main__":
    cli()
