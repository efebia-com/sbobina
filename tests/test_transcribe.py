import pytest

from sbobina.backend import DEVICE_CPU, DEVICE_CUDA
from sbobina.config import CPU_BATCH_SIZE, DEFAULT_GPU_BATCH_SIZE
from sbobina.transcribe import UnknownModelError, resolve_batch_size, resolve_model


def test_an_alias_becomes_the_matching_converted_repo():
    assert resolve_model("turbo") == "deepdml/faster-whisper-large-v3-turbo-ct2"


def test_an_explicit_repo_passes_through_untouched():
    assert resolve_model("my-user/my-whisper-ct2") == "my-user/my-whisper-ct2"


def test_an_unknown_alias_raises_and_lists_the_valid_ones():
    with pytest.raises(UnknownModelError, match="turbo"):
        resolve_model("enormous")


def test_the_canonical_upstream_names_resolve_to_the_same_repos_as_the_short_aliases():
    assert resolve_model("large-v3-turbo") == resolve_model("turbo")
    assert resolve_model("large-v3") == resolve_model("large")


def test_the_default_batch_is_large_on_gpu_and_one_on_cpu():
    assert resolve_batch_size(None, DEVICE_CUDA, vad=True) == DEFAULT_GPU_BATCH_SIZE
    assert resolve_batch_size(None, DEVICE_CPU, vad=True) == CPU_BATCH_SIZE


def test_a_zero_or_negative_batch_is_clamped_to_one():
    assert resolve_batch_size(0, DEVICE_CUDA, vad=True) == 1
    assert resolve_batch_size(-4, DEVICE_CUDA, vad=True) == 1


def test_disabling_the_vad_forces_a_single_window_because_batching_cannot_segment_without_it():
    # Not a preference: BatchedInferencePipeline takes its windows from the VAD's
    # clip_timestamps, so batch > 1 with the VAD off is a RuntimeError on any
    # audio past 30 s. The explicit request loses to the constraint on purpose.
    assert resolve_batch_size(None, DEVICE_CUDA, vad=False) == 1
    assert resolve_batch_size(8, DEVICE_CUDA, vad=False) == 1
