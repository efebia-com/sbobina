import pytest

from sbobina.backend import DEVICE_CPU, DEVICE_CUDA, UnsupportedComputeTypeError, choose_compute_type


TURING = {"float32", "float16", "int8", "int8_float16", "int8_float32"}
PASCAL = {"float32", "int8", "int8_float32"}
CPU = {"float32", "int8", "int8_float32"}


def test_a_card_that_supports_float16_gets_float16():
    assert choose_compute_type(DEVICE_CUDA, "auto", TURING) == "float16"


def test_pascal_gets_int8_float32_and_not_float32():
    """A card without float16 is an old card, and old cards have little VRAM.

    This test blocks the temptation to move float32 back up right after float16
    in the preference list: on a 4 GB 1050 Ti that would be slower and would risk
    running out of memory on the large models.
    """
    assert choose_compute_type(DEVICE_CUDA, "auto", PASCAL) == "int8_float32"


def test_cpu_gets_int8():
    assert choose_compute_type(DEVICE_CPU, "auto", CPU) == "int8"


def test_a_compute_type_asked_for_by_hand_is_honoured():
    assert choose_compute_type(DEVICE_CUDA, "float32", TURING) == "float32"


def test_a_compute_type_the_card_does_not_support_raises_and_lists_the_alternatives():
    with pytest.raises(UnsupportedComputeTypeError, match="int8_float32"):
        choose_compute_type(DEVICE_CUDA, "float16", PASCAL)


def test_with_no_supported_compute_type_at_all_it_raises_instead_of_guessing():
    with pytest.raises(UnsupportedComputeTypeError):
        choose_compute_type(DEVICE_CUDA, "auto", set())
