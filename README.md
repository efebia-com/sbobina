# sbobina

Audio or video file in, transcript out. Whisper `large-v3-turbo` via
[faster-whisper](https://github.com/SYSTRAN/faster-whisper).

```bash
uv sync
uv run sbobina --check      # which device and compute type it picked
uv run sbobina meeting.m4a  # -> meeting.txt
```

`uv run sbobina --help` documents the options.

## Install notes

`uv sync` is the entire install. cuBLAS comes in as a Python wheel and
`sbobina/cuda_libs.py` preloads it by absolute path before CTranslate2 asks for
it, on Windows as well as Linux — no CUDA Toolkit, no PATH edits.

- No GPU: `uv sync --no-group cuda` skips ~1 GB of wheels.
- Intel Mac: `uv sync --python 3.13` (3.11 and 3.12 work too). onnxruntime's last
  macOS x86_64 wheels are cp313 — there is no 3.14 build.
- First run downloads the model (~1.6 GB for `turbo`) into the HF cache, then
  works offline.
- On PATH: `ln -sf "$PWD/.venv/bin/sbobina" ~/.local/bin/sbobina`.

## Device selection

No flags needed: `float16` on Turing and newer, `int8_float32` on Pascal (faster
than float32 *and* it fits in 4 GB), `int8` on CPU and macOS. Falls back to CPU
when there is no usable GPU.

## Logs

Tracebacks land in `sbobina.log` under `~/.config/sbobina/logs/`,
`~/Library/Application Support/sbobina/logs/`, or `%LOCALAPPDATA%\sbobina\logs\`.

## Dev

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run ty check
```
