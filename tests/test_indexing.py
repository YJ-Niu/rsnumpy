"""Indexing (getitem) and assignment (setitem)."""
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


def _scalar(x):
    """Coerce a 0-dim array / scalar index result to a Python scalar."""
    return x.tolist() if hasattr(x, "tolist") else x


def test_getitem_slice():
    assert _f(np.array([10.0, 20.0, 30.0, 40.0])[1:3]) == [20.0, 30.0]


def test_getitem_negative_scalar():
    assert _scalar(np.array([10.0, 20.0, 30.0])[-1]) == 30.0
    assert _scalar(np.array([10.0, 20.0, 30.0])[-3]) == 10.0


def test_getitem_2d_row_and_element():
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert _f(m[1]) == [3.0, 4.0]
    assert _scalar(m[0, 1]) == 2.0
    assert _scalar(m[1, 0]) == 3.0


def test_len_and_iter():
    a = np.array([1.0, 2.0, 3.0])
    assert len(a) == 3
    assert [float(x) for x in a] == [1.0, 2.0, 3.0]


def test_tolist_roundtrip():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert a.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_setitem_scalar():
    x = np.zeros(3)
    x[1] = 5.0
    assert _f(x) == [0.0, 5.0, 0.0]


def test_setitem_negative_scalar():
    x = np.zeros(3)
    x[-1] = 7.0
    assert _f(x) == [0.0, 0.0, 7.0]


def test_setitem_slice_broadcast():
    y = np.zeros(4)
    y[0:2] = 3.0
    assert _f(y) == [3.0, 3.0, 0.0, 0.0]


def test_setitem_2d_element():
    m = np.zeros((2, 2))
    m[0, 1] = 9.0
    assert m.tolist() == [[0.0, 9.0], [0.0, 0.0]]
