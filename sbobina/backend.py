"""Picks device and compute type. The only module that knows what the machine is.

The whole point of this project is that the user passes no flags: the same
command line must do the right thing on a GTX 1050 Ti, on an RTX 5090, on a
MacBook and on a laptop with no GPU at all. There are two real decisions, and
both live here.
"""

import shutil
import subprocess
from dataclasses import dataclass
from types import ModuleType

from sbobina import cuda_libs
from sbobina.log import log


DEVICE_AUTO = "auto"
DEVICE_CUDA = "cuda"
DEVICE_CPU = "cpu"
DEVICES = (DEVICE_AUTO, DEVICE_CUDA, DEVICE_CPU)

COMPUTE_AUTO = "auto"

# Preference order on GPU, from fastest to most compatible.
#
# float16 first: on the cards that support it (Turing onwards) it is ~2x faster
# than float32 with a WER drift below one percentage point, which nobody sees in
# a transcript.
#
# The interesting part is what comes next. If float16 is missing, the card is
# Pascal or older, and that is also the only family still running with 4 GB of
# VRAM: there int8_float32 is not a fallback but the right call twice over,
# because it is faster than float32 *and* it fits in memory. That is why float32
# sits at the bottom rather than right after float16: the card that cannot do
# float16 is almost always the card without the VRAM for float32.
CUDA_PREFERENCE = ("float16", "int8_float16", "int8_float32", "int8", "float32")

# On CPU int8 is the only workable choice: float32 is 3-4x slower and on a laptop
# it turns a one-hour meeting into an afternoon. That holds for Apple Silicon
# Macs too, where CTranslate2 uses oneDNN and int8 performs genuinely well.
CPU_PREFERENCE = ("int8", "int8_float32", "float32")


@dataclass(frozen=True)
class Backend:
    """How the model will run, already resolved and ready to print."""

    device: str
    compute_type: str
    gpu_name: str | None
    note: str | None


class UnsupportedComputeTypeError(ValueError):
    """The compute type asked for by hand does not exist on this device."""


def _ctranslate2() -> ModuleType:
    """CTranslate2, imported only after the CUDA libraries have been registered.

    The order is mandatory, and an early import fails in a way that looks like
    anything but a missing library — differently on each platform. The whole
    story is in the cuda_libs module docstring, and it lives there only: a second
    copy of it here is a second copy to keep true.
    """
    cuda_libs.register()
    import ctranslate2

    return ctranslate2


def ensure_available() -> None:
    """Guarantee that importing ctranslate2 downstream will find its CUDA libraries.

    Separate from `_ctranslate2()` because the callers want different things.
    transcribe.py needs the *ordering guarantee* before it imports faster-whisper,
    not the module itself: handing it a `ModuleType` would be handing it a value
    with no type information, to be discarded on the next line.
    """
    _ctranslate2()


def choose_compute_type(device: str, requested: str, supported: set[str]) -> str:
    """Resolve `auto` into the best compute type *this* device actually supports.

    Deliberately a pure function: touching the hardware happens in `resolve()`, so
    the selection policy can be tested without owning that card — which is exactly
    the situation, given the tool has to run on cards we do not have.
    """
    if requested != COMPUTE_AUTO:
        if requested not in supported:
            available = ", ".join(sorted(supported)) or "none"
            message = f"compute type '{requested}' is not supported on {device}; available: {available}"
            raise UnsupportedComputeTypeError(message)
        return requested

    preference = CUDA_PREFERENCE if device == DEVICE_CUDA else CPU_PREFERENCE
    for candidate in preference:
        if candidate in supported:
            return candidate

    # This should not happen: CTranslate2 always advertises at least float32. If
    # it does, the install is broken and that needs saying, not working around.
    available = ", ".join(sorted(supported)) or "none"
    raise UnsupportedComputeTypeError(f"no usable compute type on {device}; available: {available}")


def _cuda_device_count(ctranslate2: ModuleType) -> int:
    try:
        return ctranslate2.get_cuda_device_count()
    except Exception as exc:
        # Without an NVIDIA driver the call may raise instead of answering zero.
        # "I have no GPU" is a legitimate answer, not an error to propagate.
        log.info("cuda device count failed", error=str(exc))
        return 0


def _gpu_name() -> str | None:
    """The first GPU's name according to nvidia-smi, or None.

    Only feeds the line sbobina prints and the log. Best-effort in the strict
    sense: nvidia-smi may be missing from the PATH even on a machine with a
    perfectly working GPU, and in that case we print one line less.
    """
    binary = shutil.which("nvidia-smi")
    if not binary:
        return None
    try:
        result = subprocess.run(
            [binary, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first = result.stdout.strip().splitlines()
    return first[0].strip() if first else None


def resolve(device: str = DEVICE_AUTO, compute_type: str = COMPUTE_AUTO) -> Backend:
    """Decide device and compute type by asking the machine."""
    ctranslate2 = _ctranslate2()

    resolved_device = device
    note = None
    if device == DEVICE_AUTO:
        resolved_device = DEVICE_CUDA if _cuda_device_count(ctranslate2) > 0 else DEVICE_CPU
    elif device == DEVICE_CUDA and _cuda_device_count(ctranslate2) == 0:
        # Someone who typed --device cuda expects the GPU. Falling back silently
        # would let them read a ten-times-slower transcription as if it were
        # normal, so the fallback happens but gets said out loud.
        resolved_device = DEVICE_CPU
        note = "no CUDA GPU detected, running on CPU"

    supported = set(ctranslate2.get_supported_compute_types(resolved_device))
    resolved_compute = choose_compute_type(resolved_device, compute_type, supported)

    backend = Backend(
        device=resolved_device,
        compute_type=resolved_compute,
        gpu_name=_gpu_name() if resolved_device == DEVICE_CUDA else None,
        note=note,
    )
    log.info(
        "backend resolved",
        device=backend.device,
        compute_type=backend.compute_type,
        gpu=backend.gpu_name,
        supported=sorted(supported),
    )
    return backend
