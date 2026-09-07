"""Makes the CUDA libraries installed as Python wheels visible to CTranslate2.

This module exists for one reason: the promise that `uv sync` is enough, Windows
included. CTranslate2 with the CUDA backend needs cuBLAS as a shared library, and
the standard way to get it is installing the CUDA Toolkit and fixing the PATH by
hand. We cannot ask colleagues to do that: it is the step every faster-whisper
guide strands people on.

cuDNN used to be on that list. Up to CT2 4.6.2 the CUDA backend needed it, so the
venv carried `nvidia-cudnn-cu12` and this module preloaded `libcudnn.so.9`. CT2
4.6.3 (#1949) reimplemented Conv1d in pure CUDA and made cuDNN optional; the
4.8.2 binaries we lock mention it nowhere, on either platform, which is why
neither the wheel nor the preload entry is here any more. Going back below CT2
4.6.3 brings both back together — see the cuda group in pyproject.toml.

The `nvidia-cublas-cu12` wheel contains exactly those libraries, and uv installs
them inside the project venv. The problem is that nobody looks for them there:
they land in `site-packages/nvidia/<package>/lib` (or `/bin` on Windows), which
is not a search directory for the dynamic linker. Both platforms end up needing
the same remedy — load the libraries ourselves, by absolute path, before
CTranslate2 ever asks for them — but they get there by different roads:

- **Linux**: setting LD_LIBRARY_PATH from Python achieves nothing, because the
  loader reads that variable when the process starts and never reads it again.
  What does work is preloading the .so files by absolute path with RTLD_GLOBAL:
  they stay registered under their SONAME, so when ctranslate2's module asks for
  `libcublas.so.12` the loader finds it already in memory. It is the same trick
  PyTorch uses.
- **Windows**: `os.add_dll_directory()` looks like the whole answer and is not.
  CTranslate2 does not import cuBLAS statically — it resolves it at runtime with
  a bare `LoadLibraryA("cublas64_12.dll")` (its `src/cuda/cublas_stub.cc`), and a
  flagless LoadLibraryA uses the standard search order. The directories added by
  `add_dll_directory` only reach loads that pass LOAD_LIBRARY_SEARCH_USER_DIRS,
  and CPython never turns that on process-wide: it sets the search flags per
  call, on its own loads only (`Python/dynload_win.c`). So on a machine without
  the CUDA Toolkit the DLL is simply not found — and the failure is not the
  quiet one either, because the device count comes from the driver and still
  answers 1: `--check` prints a perfect GPU line, the model downloads, and the
  first GEMM dies with "Library cublas64_12.dll is not found or cannot be
  loaded". Preloading by absolute path fixes it for the same reason it works on
  Linux: LoadLibraryA matches an already-loaded module by base name before it
  searches any directory. `add_dll_directory` is still called, because it does
  cover the static import chain of the compiled extension, but it is not the
  part that makes the GPU work.
- **macOS**: neither of the two, because CUDA on Mac does not exist. The NVIDIA
  wheels are not even installed (marker in pyproject) and this function finds
  nothing to register: it returns quietly and we run on CPU.

Hence the rule that holds across the project: **no module-level
`import ctranslate2`**. Everything goes through `backend.ensure_available()`,
which calls `register()` first. An import at the top of a file would run at
package import time, that is, before the preload, and what follows is a bad
error in both directions: on Linux CTranslate2 reports zero CUDA devices, which
reads as "I have no GPU" instead of "I could not find cuBLAS"; on Windows it
reports the GPU it can see through the driver and then fails much later, after
the model download, on the first matrix multiplication.
"""

import ctypes
import os
import sys
import sysconfig
from functools import cache
from pathlib import Path

from sbobina.log import log


# The names CTranslate2 passes to dlopen/LoadLibraryA, in dependency order: cuBLAS
# depends on cuBLASLt, and loading the dependent before its dependency makes the
# preload fail. Two spellings of the same pair because the platforms spell shared
# libraries differently, not because they need different libraries.
_PRELOAD_POSIX = ("libcublasLt.so.12", "libcublas.so.12")
_PRELOAD_WINDOWS = ("cublasLt64_12.dll", "cublas64_12.dll")

# The NVIDIA wheels put their libraries in `lib/` on Linux and in `bin/` on Windows.
_LIB_SUBDIRS = ("lib", "bin")


def nvidia_lib_dirs() -> list[Path]:
    """The directories holding the shared libraries of this venv's nvidia-* wheels."""
    root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    if not root.is_dir():
        return []

    found = []
    for package in sorted(root.iterdir()):
        for subdir in _LIB_SUBDIRS:
            candidate = package / subdir
            if candidate.is_dir():
                found.append(candidate)
    return found


def _preload(names: tuple[str, ...], dirs: list[Path]) -> None:
    for name in names:
        for directory in dirs:
            library = directory / name
            if not library.exists():
                continue
            try:
                # RTLD_GLOBAL is the whole point on POSIX: it is what leaves the
                # SONAME resolvable for the dlopen CTranslate2 does later. On
                # Windows ctypes overwrites the mode with its own
                # LOAD_LIBRARY_SEARCH_* flags, so passing it there is inert
                # rather than wrong — and an absolute path makes ctypes add
                # LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, which resolves the sibling
                # DLLs without any extra work from us.
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:
                # Best-effort: if the preload fails (library built for another
                # arch, file truncated by an interrupted download) we carry on.
                # CTranslate2 then falls back to CPU, which is slow but correct —
                # and the reason stays here in the log.
                log.warning("cuda preload failed", library=str(library), error=str(exc))
            break


def _register_windows(dirs: list[Path]) -> None:
    for directory in dirs:
        # Best-effort per directory: one malformed NVIDIA wheel must not stop the
        # others from being registered. This covers the static import chain of the
        # compiled extension; the preload below is what covers cuBLAS itself.
        try:
            os.add_dll_directory(str(directory))
        except OSError:
            log.warning("dll directory rejected", path=str(directory))

    _preload(_PRELOAD_WINDOWS, dirs)


def _register_posix(dirs: list[Path]) -> None:
    _preload(_PRELOAD_POSIX, dirs)


@cache
def register() -> None:
    """Make the CUDA libraries loadable. Idempotent, must run before ctranslate2.

    The cache is not an optimisation: registering the same directories twice on
    Windows is pointless, and preloading the same library twice is noise in the
    log. Calling it is therefore always safe, from anywhere.
    """
    dirs = nvidia_lib_dirs()
    if not dirs:
        # The normal case on macOS and for anyone who ran `uv sync --no-group cuda`.
        log.info("no cuda wheels in the venv, continuing on cpu")
        return

    if sys.platform == "win32":
        _register_windows(dirs)
    else:
        _register_posix(dirs)
    log.info("cuda libraries registered", dirs=[str(d) for d in dirs])
