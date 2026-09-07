import errno
import subprocess
import sys
from dataclasses import fields
from importlib.metadata import version
from pathlib import Path

import pytest
from click.testing import CliRunner

from sbobina import cli as cli_module
from sbobina.backend import Backend
from sbobina.cli import _output_path, _reason, _write
from sbobina.transcribe import Finished, Options, Segment, Started


class _FakeAvError(ValueError):
    """Shaped like av.error.InvalidDataError: the cause is in strerror, not in str()."""

    strerror = "Invalid data found when processing input"

    def __str__(self):
        return "'/audio/notes.txt'"


def test_the_default_output_sits_next_to_the_input_with_the_format_extension():
    assert _output_path(Path("/audio/meeting.m4a"), None, "srt") == Path("/audio/meeting.srt")


def test_a_name_with_dots_only_loses_its_last_suffix():
    assert _output_path(Path("/audio/meeting.v2.m4a"), None, "txt") == Path("/audio/meeting.v2.txt")


def test_an_explicit_output_wins_over_the_derived_one():
    assert _output_path(Path("/audio/meeting.m4a"), "/other/x.txt", "txt") == Path("/other/x.txt")


def test_a_lone_dash_means_stdout():
    assert _output_path(Path("/audio/meeting.m4a"), "-", "txt") is None


def test_an_undecodable_file_reports_its_cause_and_not_just_its_path():
    """PyAV stringifies its errors to the filename, which tells the user nothing.

    Without this, `sbobina notes.txt` printed back `'notes.txt'` and left the only
    useful sentence ("Invalid data found when processing input") in the log file.
    """
    assert _reason(_FakeAvError()) == "Invalid data found when processing input"


def test_an_exception_without_strerror_falls_back_to_its_message():
    assert _reason(ValueError("no such model")) == "no such model"


def test_a_transcript_is_written_as_utf8_whatever_the_locale(tmp_path):
    target = tmp_path / "out.txt"
    _write("romănește — così\n", target)
    assert target.read_bytes().decode("utf-8").startswith("romănește")


def test_options_carries_no_device_or_compute_type_because_the_backend_owns_them():
    """These two fields existed and were never read: the engine uses the Backend.

    A caller writing `Options(device="cpu")` was ignored without a word. This test
    is here so nobody re-adds them out of symmetry.
    """
    names = {f.name for f in fields(Options)}
    assert "device" not in names
    assert "compute_type" not in names


_BACKEND = Backend(device="cpu", compute_type="int8", gpu_name=None, note=None)


def _events(segments):
    """A `transcribe` stand-in emitting the events the CLI consumes, with no model."""

    def fake(audio, options, resolved):
        yield Started(language="it", language_probability=0.99, duration=1.0)
        yield Finished(segments=list(segments), elapsed=1.0)

    return fake


@pytest.fixture
def run_cli(monkeypatch):
    """The real command with only its external boundaries replaced.

    `resolve` imports ctranslate2 and interrogates the machine about its GPU,
    `transcribe` downloads 1.6 GB of model, and `setup_logging` writes into the
    operator's own state directory. Everything else — argument parsing, the
    guards, the write, the exit code — is the real `cli()`, which is the part
    these tests are about.
    """
    monkeypatch.setattr(cli_module, "setup_logging", lambda log_dir: None)
    monkeypatch.setattr(cli_module.backend_module, "resolve", lambda device, compute_type: _BACKEND)

    def run(args, segments):
        monkeypatch.setattr(cli_module, "transcribe", _events(segments))
        return CliRunner().invoke(cli_module.cli, args)

    return run


def test_an_audio_with_no_speech_leaves_the_destination_untouched_and_exits_nonzero(tmp_path, run_cli):
    """An empty transcript renders to a single newline, and writing it destroyed files.

    Measured before the fix: `sbobina meeting.mp4 -o minutes.txt && rm meeting.mp4`
    got past the `&&` because the exit code claimed success — leaving a 1-byte
    minutes.txt and no recording left to redo it from. Both halves are the
    invariant: the previous file survives, *and* the exit code stops the chain.
    """
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"")
    minutes = tmp_path / "minutes.txt"
    minutes.write_text("MINUTES 12 MARCH\nitem 1: budget\n", encoding="utf-8")

    result = run_cli([str(audio), "-o", str(minutes)], [])

    assert result.exit_code != 0
    assert minutes.read_text(encoding="utf-8") == "MINUTES 12 MARCH\nitem 1: budget\n"


def test_a_destination_that_cannot_be_written_is_reported_and_not_raised_as_a_traceback(tmp_path, run_cli):
    """The write used to sit *after* the try/except that guards the whole run.

    A destination that is a directory, a read-only file or a full disk answered an
    hour of finished transcription with a raw Python traceback — the only place in
    the program where a colleague sees a stack instead of the `error ...` line.
    """
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"")
    occupied = tmp_path / "already-a-directory"
    occupied.mkdir()

    result = run_cli([str(audio), "-o", str(occupied)], [Segment(index=1, start=0.0, end=1.0, text="ciao")])

    assert not isinstance(result.exception, OSError)
    assert result.exit_code == 1
    assert "cannot write" in result.stderr


def test_a_write_that_dies_halfway_leaves_the_transcript_already_on_disk_intact(tmp_path, monkeypatch):
    """`write_text` opens in "w", so it truncates before the first byte.

    Reproduced on a filled tmpfs: `OSError: [Errno 122] Disk quota exceeded` left a
    0-byte file where the previous transcript had been. This invariant does not
    descend from the error boundary and does not come with it — the boundary keeps
    the traceback off the screen, only the two-step write keeps the bytes that were
    already on disk.
    """
    target = tmp_path / "minutes.txt"
    target.write_text("GOOD TRANSCRIPT\n", encoding="utf-8")
    real_write_text = Path.write_text

    def disk_full(self, data, **kwargs):
        # Half of it lands, then the error: a write that fails *after* truncating
        # is the whole point, and one that never opens the file proves nothing.
        real_write_text(self, data[: len(data) // 2], **kwargs)
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(Path, "write_text", disk_full)

    with pytest.raises(OSError, match="No space left"):
        _write("A LONGER AND NEWER TRANSCRIPT\n", target)

    assert target.read_text(encoding="utf-8") == "GOOD TRANSCRIPT\n"
    assert not (tmp_path / "minutes.txt.partial").exists()


# Run in a real subprocess because the failure only exists there: CliRunner catches
# the exception and the interpreter never reaches the shutdown flush that is the
# whole bug. The transcript is deliberately ~200 bytes — far below the kernel's
# 64 KB pipe buffer, which is where the threshold actually is, so the write is
# guaranteed to succeed into the buffer and the failure to surface only at flush.
_CLOSED_PIPE_RUN = """
import sys

from sbobina import cli as cli_module
from sbobina.backend import Backend
from sbobina.transcribe import Finished, Segment, Started


def fake_transcribe(audio, options, resolved):
    yield Started(language="it", language_probability=0.99, duration=1.0)
    yield Finished(segments=[Segment(index=1, start=0.0, end=1.0, text="x" * 200)], elapsed=1.0)


cli_module.setup_logging = lambda log_dir: None
cli_module.backend_module.resolve = lambda device, compute_type: Backend(
    device="cpu", compute_type="int8", gpu_name=None, note=None
)
cli_module.transcribe = fake_transcribe
cli_module.cli([sys.argv[1], "-o", "-"])
"""


def test_a_transcript_that_never_reached_the_pipe_is_not_announced_as_done(tmp_path):
    """`sbobina x.m4a -o - | head -1` printed `done` and exited 120, having sent nothing.

    stdout is buffered as soon as it is a pipe, so the write raised nothing, click
    returned satisfied and `_report` announced success; the BrokenPipeError only
    arrived at interpreter shutdown, outside click's EPIPE handling, as
    "Exception ignored while flushing sys.stdout" and an exit code of 120 that no
    convention accounts for.
    """
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"")
    script = tmp_path / "piped_run.py"
    script.write_text(_CLOSED_PIPE_RUN, encoding="utf-8")

    process = subprocess.Popen(
        [sys.executable, str(script), str(audio)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    # The reader leaves before the child has finished importing sbobina, which is
    # `| head -1` exiting on its first line: the pipe has no reader left by the
    # time anything is written to it.
    process.stdout.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    exit_code = process.wait(timeout=60)

    assert "done" not in stderr
    assert "Exception ignored" not in stderr
    assert exit_code not in (0, 120)


def test_the_version_reported_is_sbobinas_own_and_not_rich_clicks(tmp_path):
    """A bare `@click.version_option()` printed 1.9.9, the version of rich-click.

    `rich_click.version_option` is not a re-export of click's, so click's
    frame-walking autodetection landed on `rich_click.decorators`. The project is
    about to be read by colleagues opening issues, and the first line of each one
    would have named a version that does not exist — one that moves on its own
    every time `uv lock` updates rich-click.
    """
    result = CliRunner().invoke(cli_module.cli, ["--version"])

    assert result.exit_code == 0
    assert version("sbobina") in result.output
