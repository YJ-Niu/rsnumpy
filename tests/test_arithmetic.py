"""Element-wise arithmetic on rsnumpy ndarrays.

Covers the core operations the scicore port relies on: binary ops (both
operand directions), scalar ops (both directions), unary ops, pow, 2-D
broadcasting, and raw (unwrapped) / wrapped ndarray interop.  The known
integration-level gaps (stft/welch/spline/interp_decomp) do not reproduce from
these operations in isolation and are documented in test_known_gaps.py.
"""
import pytest

import rsnumpy as np


def _f(x):
    """Flatten an rsnumpy result to a flat Python list of floats."""
    if hasattr(x, "tolist"):
        x = x.tolist()
    out = []
    for v in x:
        if isinstance(v, list):
            out.extend(_f(v))
        else:
            out.append(float(v))
    return out


def test_binary_add_sub_mul_div():
    assert _f(np.array([1.0, 2.0, 3.0]) + np.array([10.0, 20.0, 30.0])) == [11.0, 22.0, 33.0]
    assert _f(np.array([5.0, 6.0]) - np.array([1.0, 2.0])) == [4.0, 4.0]
    assert _f(np.array([2.0, 3.0]) * np.array([4.0, 5.0])) == [8.0, 15.0]
    assert _f(np.array([10.0, 9.0]) / np.array([2.0, 3.0])) == [5.0, 3.0]


def test_scalar_op_both_directions():
    a = np.array([1.0, 2.0, 3.0])
    assert _f(a + 1.0) == [2.0, 3.0, 4.0]
    assert _f(1.0 + a) == [2.0, 3.0, 4.0]
    assert _f(a * 2.0) == [2.0, 4.0, 6.0]
    assert _f(2.0 * a) == [2.0, 4.0, 6.0]
    assert _f(10.0 - a) == pytest.approx([9.0, 8.0, 7.0])
    assert _f(10.0 / a) == pytest.approx([10.0, 5.0, 10.0 / 3.0])


def test_pow():
    assert _f(np.array([2.0, 3.0, 4.0]) ** 2) == [4.0, 9.0, 16.0]


def test_unary_neg_abs():
    a = np.array([1.0, -2.0, 3.0])
    assert _f(-a) == [-1.0, 2.0, -3.0]
    assert _f(abs(a)) == [1.0, 2.0, 3.0]


def test_broadcast_row_and_column():
    a = np.array([[1.0], [2.0], [3.0]])   # (3, 1)
    b = np.array([[10.0, 20.0]])          # (1, 2)
    c = a + b                             # (3, 2)
    assert c.shape == (3, 2)
    assert _f(c.tolist()[2]) == [13.0, 23.0]


def test_inplace_accumulation_via_reuse():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0, 30.0])
    c = a + b
    d = c - b  # (a + b) - b == a
    assert _f(d) == pytest.approx([1.0, 2.0, 3.0])


def test_raw_raw_interop():
    # raw + raw is correct element-wise: the Python ``__add__`` unwraps both
    # operands to raw (``._array``) before the Rust op, so this is the normal
    # internal path.
    raw = np.array([10.0, 20.0, 30.0])._array
    other = np.array([1.0, 1.0, 1.0])._array
    assert _f(raw + other) == [11.0, 21.0, 31.0]
    assert _f(raw * other) == [10.0, 20.0, 30.0]


def test_wrapper_add_sub_div_with_raw():
    # wrapped op raw is correct for +, -, /: the raw right operand falls to
    # ``self._array <op> raw`` and is handled by the ``b.cast::<NdArray>()``
    # branch in ``binary_op``.  (``w * raw`` is a separate scalar-misread bug —
    # see test_wrapper_mul_raw_scalar_misread.)
    w = np.array([1.0, 2.0, 3.0])
    raw = np.array([10.0, 20.0, 30.0])._array
    assert _f(w + raw) == [11.0, 22.0, 33.0]
    assert _f(w - raw) == [-9.0, -18.0, -27.0]
    assert _f(w / raw) == pytest.approx([0.1, 0.1, 0.1])


def test_raw_op_wrapper_elementwise():
    # Regression: ``raw op wrapper`` was misread as a scalar — ``binary_op``
    # checked ``b.extract::<f64>()`` before recognizing the wrapped rsnumpy
    # ndarray (whose ``__float__`` returns its first element), so it computed
    # ``raw op wrapper[0]``.  Now element-wise.  See test_known_gaps.py.
    w = np.array([1.0, 2.0, 3.0])
    raw = np.array([10.0, 20.0, 30.0])._array
    assert _f(raw + w) == [11.0, 22.0, 33.0]
    assert _f(raw - w) == [9.0, 18.0, 27.0]
    assert _f(raw * w) == [10.0, 40.0, 90.0]
    assert _f(raw / w) == pytest.approx([10.0, 10.0, 10.0])


def test_wrapper_mul_raw_elementwise():
    # Regression: ``wrapper * raw`` was misread as a scalar — the Python
    # ``__mul__``'s ``float(other)`` path took the raw ndarray's first element.
    # Now it delegates to the Rust element-wise op.  See test_known_gaps.py.
    w = np.array([1.0, 2.0, 3.0])
    raw = np.array([10.0, 20.0, 30.0])._array
    assert _f(w * raw) == [10.0, 40.0, 90.0]
