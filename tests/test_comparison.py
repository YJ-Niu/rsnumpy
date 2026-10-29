"""Comparison and logical ops.

Element-wise comparisons return boolean-valued arrays; ``all``/``any``
return a Python bool.
"""
import pytest

import rsnumpy as np


def _f(x):
    if hasattr(x, "tolist"):
        x = x.tolist()
    out = []
    for v in x:
        if isinstance(v, list):
            out.extend(_f(v))
        else:
            out.append(float(v))
    return out


def test_comparison_ops():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 2.0, 2.0])
    assert _f(a < b) == [1.0, 0.0, 0.0]
    assert _f(a <= b) == [1.0, 1.0, 0.0]
    assert _f(a > b) == [0.0, 0.0, 1.0]
    assert _f(a >= b) == [0.0, 1.0, 1.0]


def test_eq_ne():
    a = np.array([1.0, 2.0, 3.0])
    c = np.array([1.0, 0.0, 3.0])
    assert _f(a == c) == [1.0, 0.0, 1.0]
    assert _f(a != c) == [0.0, 1.0, 0.0]


def test_all_any():
    assert bool(np.array([1.0, 2.0]).all()) is True
    assert bool(np.array([0.0, 1.0]).all()) is False
    assert bool(np.array([0.0, 0.0]).any()) is False
    assert bool(np.array([0.0, 1.0]).any()) is True


def test_isnan_isfinite():
    a = np.array([1.0, float("nan"), 3.0])
    assert _f(np.isnan(a)) == [0.0, 1.0, 0.0]
    assert _f(np.isfinite(a)) == [1.0, 0.0, 1.0]
