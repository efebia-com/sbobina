"""Paths, model aliases and defaults. No logic, only constants."""

import os
import sys
from pathlib import Path


def _state_dir() -> Path:
    """Where sbobina keeps logs and models, following each OS convention.

    ytm can hardcode ~/.config/ytm because it only ever runs on Linux. Not here:
    on Windows ~/.config is a directory that means nothing to anyone (and that no
    system backup considers), and on macOS the convention is
    ~/Library/Application Support. Picking the wrong directory breaks nothing,
    but it leaves files where the user will never find them again.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "sbobina"
        return Path.home() / "AppData" / "Local" / "sbobina"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "sbobina"
    return Path.home() / ".config" / "sbobina"


STATE_DIR = _state_dir()
LOG_DIR = STATE_DIR / "logs"

# Checkpoints already converted to the CTranslate2 format. OpenAI's official
# builds cannot be fed to faster-whisper without conversion, so the aliases point
# at the pre-converted repos hosted on Hugging Face.
#
# `turbo` is the default and it is not a compromise: large-v3-turbo has its
# decoder pruned from 32 layers to 4, so it runs ~5x faster than large-v3 with a
# WER difference nobody notices on spoken language. The smaller models stay for
# machines without a GPU, where turbo turns slow.
#
# `large-v3-turbo` and `large-v3` are duplicates on purpose: they are what the
# checkpoints are actually called on Hugging Face and in every tutorial, so they
# are the first thing a colleague types. Without them the answer was
# UnknownModelError, which sends someone off to hunt for the list of aliases
# instead of transcribing. The short names stay because they are what you want to
# type twice a day.
MODELS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "large": "Systran/faster-whisper-large-v3",
    "large-v3": "Systran/faster-whisper-large-v3",
}
DEFAULT_MODEL = "turbo"

# Default batch size on GPU. faster-whisper decodes several windows in parallel
# and on a mid-range GPU that is worth a 3-4x speedup; past 16 the gain flattens
# out and VRAM starts to matter for real, so 8 is the point where a 3060 is as
# comfortable as a 4090.
DEFAULT_GPU_BATCH_SIZE = 8

# On CPU batching does not pay off: there is no massive parallelism to saturate,
# and the extra copies cost more than they save.
CPU_BATCH_SIZE = 1
