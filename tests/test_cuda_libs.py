import ast
import ctypes
import os
import sys
import sysconfig
from pathlib import Path

import pytest

from sbobina import cuda_libs


@pytest.fixture(autouse=True)
def _forget_the_registration():
    """`register()` is @cache'd on purpose, so every test has to make it run again."""
    cuda_libs.register.cache_clear()
    yield
    cuda_libs.register.cache_clear()


def _fake_wheel(tmp_path, package, subdir, names):
    """A venv nvidia-* wheel, with empty files standing in for the shared libraries."""
    directory = tmp_path / "nvidia" / package / subdir
    directory.mkdir(parents=True)
    for name in names:
        (directory / name).write_bytes(b"")
    return directory


def _record_loads(monkeypatch, tmp_path):
    """Capture the ctypes.CDLL calls instead of really loading the empty files."""
    calls = []
    monkeypatch.setattr(ctypes, "CDLL", lambda path, **kwargs: calls.append((path, kwargs)))
    monkeypatch.setattr(sysconfig, "get_paths", lambda: {"purelib": str(tmp_path)})
    return calls


def test_without_the_nvidia_folder_there_is_nothing_to_register(monkeypatch, tmp_path):
    """This is the macOS case, and the case of anyone who ran `uv sync --no-group cuda`.

    It must return an empty list, not raise: that path ends on CPU, which is slow
    but correct.
    """
    monkeypatch.setattr(sysconfig, "get_paths", lambda: {"purelib": str(tmp_path)})
    assert cuda_libs.nvidia_lib_dirs() == []


def test_both_the_linux_lib_dirs_and_the_windows_bin_dirs_are_found(monkeypatch, tmp_path):
    (tmp_path / "nvidia" / "cuda_nvrtc" / "lib").mkdir(parents=True)
    (tmp_path / "nvidia" / "cublas" / "bin").mkdir(parents=True)
    (tmp_path / "nvidia" / "cublas" / "include").mkdir(parents=True)
    monkeypatch.setattr(sysconfig, "get_paths", lambda: {"purelib": str(tmp_path)})

    found = {p.parent.name + "/" + p.name for p in cuda_libs.nvidia_lib_dirs()}
    assert found == {"cuda_nvrtc/lib", "cublas/bin"}


def test_on_windows_cublas_is_preloaded_by_absolute_path_not_left_to_add_dll_directory(monkeypatch, tmp_path):
    """The whole Windows GPU path hangs on this preload.

    CTranslate2 resolves cuBLAS at runtime with a bare `LoadLibraryA("cublas64_12.dll")`,
    and a flagless LoadLibraryA never consults the directories `os.add_dll_directory`
    registers. Loading the DLLs ourselves by absolute path puts them in the process
    module list, where that call finds them by base name. Without it, GPU on Windows
    only works for whoever installed the CUDA Toolkit — which is the exact thing this
    module exists to avoid.
    """
    bin_dir = _fake_wheel(tmp_path, "cublas", "bin", ("cublasLt64_12.dll", "cublas64_12.dll"))
    calls = _record_loads(monkeypatch, tmp_path)
    monkeypatch.setattr(os, "add_dll_directory", lambda path: None, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")

    cuda_libs.register()

    assert [path for path, _ in calls] == [
        str(bin_dir / "cublasLt64_12.dll"),
        str(bin_dir / "cublas64_12.dll"),
    ]


def test_on_windows_the_dll_directories_are_registered_as_well_as_preloaded(monkeypatch, tmp_path):
    """`add_dll_directory` is not enough, but it is not useless either: it covers the
    static import chain of the compiled extension. Dropping it while adding the preload
    would trade one half-fix for another.
    """
    bin_dir = _fake_wheel(tmp_path, "cublas", "bin", ("cublasLt64_12.dll", "cublas64_12.dll"))
    _record_loads(monkeypatch, tmp_path)
    registered = []
    monkeypatch.setattr(os, "add_dll_directory", registered.append, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")

    cuda_libs.register()

    assert registered == [str(bin_dir)]


def test_on_linux_cublaslt_is_preloaded_before_cublas_and_with_rtld_global(monkeypatch, tmp_path):
    """Two invariants of the POSIX preload, both easy to break by tidying.

    Order: cuBLAS depends on cuBLASLt, so the dependency has to be in memory first.
    RTLD_GLOBAL: without it the SONAME stays private to our own handle, and the dlopen
    CTranslate2 does later finds nothing — the preload becomes a no-op that looks fine.
    """
    lib_dir = _fake_wheel(tmp_path, "cublas", "lib", ("libcublasLt.so.12", "libcublas.so.12"))
    calls = _record_loads(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "platform", "linux")

    cuda_libs.register()

    assert [path for path, _ in calls] == [
        str(lib_dir / "libcublasLt.so.12"),
        str(lib_dir / "libcublas.so.12"),
    ]
    assert all(kwargs["mode"] == ctypes.RTLD_GLOBAL for _, kwargs in calls)


def test_cudnn_is_not_preloaded_on_any_platform_because_ctranslate2_stopped_needing_it():
    """CT2 4.6.3 reimplemented Conv1d in pure CUDA and made cuDNN optional; 4.8.2, the
    version we lock, names it in no binary. The wheel left the cuda group in pyproject,
    so a preload entry here would hunt for a library the venv no longer installs — and
    the two have to move together or the preload logs a failure on every run.
    """
    preloaded = cuda_libs._PRELOAD_POSIX + cuda_libs._PRELOAD_WINDOWS
    assert [name for name in preloaded if "cudnn" in name.lower()] == []


def test_no_module_in_the_package_imports_ctranslate2_at_module_level():
    """The invariant the whole module is built around, finally pinned down.

    `register()` has to run before ctranslate2 is imported, and a module-level import
    anywhere in the package would run at package import time — too early, whatever the
    caller does. Deferred imports inside functions are the supported way, so only the
    top level of each file is checked.
    """
    package = Path(cuda_libs.__file__).parent
    offenders = []
    for source in sorted(package.glob("*.py")):
        for node in ast.parse(source.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "ctranslate2" for a in node.names):
                offenders.append(source.name)
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "ctranslate2":
                offenders.append(source.name)
    assert offenders == []
