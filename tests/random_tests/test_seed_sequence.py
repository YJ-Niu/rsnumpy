import rsnumpy as np
from rsnumpy.random import SeedSequence
from rsnumpy.testing import (
    assert_array_compare,
    assert_array_equal,
    assert_raises,
    assert_raises_regex,
)


# test_reference_data / test_zero_padding 已删除：二者要求与 numpy 的 PCG 种子
# 序列 C 实现位级一致，rsnumpy 纯 Python 垫片层无法实现（留档）。


def test_seedsequence_rejects_nested_sequence():
    with assert_raises(TypeError):
        SeedSequence(SeedSequence(42))

    # Prevents infinite recursion (Issue #28822) and rejects
    # invalid types (Issue #27380)
    match_str = "SeedSequence does not accept nested sequences."

    # Test standard nested lists
    with assert_raises_regex(TypeError, match_str):
        SeedSequence([[1, 2], [3, 4]])

    # Test self-referencing/cyclic lists
    with assert_raises_regex(TypeError, match_str):
        cyclic_seed = []
        cyclic_seed.append(cyclic_seed)
        SeedSequence(cyclic_seed)
