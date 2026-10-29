"""Array creation, shape metadata, and shape-manipulation ops."""
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


def test_array_from_nested_list():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert a.shape == (2, 2)
    assert a.ndim == 2
    assert a.size == 4
    assert _f(a) == [1.0, 2.0, 3.0, 4.0]


def test_asarray_no_copy_semantics():
    a = np.asarray([1.0, 2.0, 3.0])
    assert a.shape == (3,)
    assert _f(a) == [1.0, 2.0, 3.0]


def test_zeros_ones_full():
    assert _f(np.zeros(3)) == [0.0, 0.0, 0.0]
    assert _f(np.ones(3)) == [1.0, 1.0, 1.0]
    assert _f(np.full(3, 2.5)) == [2.5, 2.5, 2.5]
    assert np.zeros((2, 2)).shape == (2, 2)


def test_arange_linspace_eye():
    assert _f(np.arange(5)) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert _f(np.linspace(0.0, 1.0, 3)) == pytest.approx([0.0, 0.5, 1.0])
    e = np.eye(3)
    assert e.shape == (3, 3)
    assert _f(e.tolist()[0]) == [1.0, 0.0, 0.0]
    assert _f(e.tolist()[1]) == [0.0, 1.0, 0.0]


def test_shape_metadata():
    a = np.zeros((2, 3, 4))
    assert a.shape == (2, 3, 4)
    assert a.ndim == 3
    assert a.size == 24
    assert a.itemsize == 8
    assert a.nbytes == 192


def test_reshape_preserves_values():
    a = np.arange(6)
    b = a.reshape(2, 3)
    assert b.shape == (2, 3)
    assert _f(b) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert _f(b.tolist()[1]) == [3.0, 4.0, 5.0]


def test_ravel_and_flatten():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert _f(a.ravel()) == [1.0, 2.0, 3.0, 4.0]
    assert _f(a.flatten()) == [1.0, 2.0, 3.0, 4.0]


def test_astype_float_dtype():
    a = np.array([1, 2, 3]).astype("float64")
    assert a.dtype.kind == "f"
    assert _f(a) == [1.0, 2.0, 3.0]


def test_copy_is_independent():
    a = np.array([1.0, 2.0, 3.0])
    b = a.copy()
    b[0] = 99.0
    assert _f(a) == [1.0, 2.0, 3.0]  # original untouched
    assert _f(b) == [99.0, 2.0, 3.0]


def test_dtype_kind():
    assert np.array([1.0, 2.0]).dtype.kind == "f"
    assert np.array([1, 2]).dtype.kind in ("i", "l")  # signed int
