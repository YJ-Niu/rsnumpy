"""Reduction and accumulation ops.

Note the scalar return convention: ``sum``/``max``/``min``/``mean``/``prod``
return a Python scalar (float) for a 1-D input, ``argmax``/``argmin`` return a
Python int, and ``all``/``any`` return a Python bool.  ``cumsum``/``cumprod``
return arrays.
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


def _s(x):
    """Coerce a scalar-ish reduction result to a Python scalar."""
    return x.tolist() if hasattr(x, "tolist") else x


def test_sum():
    assert _s(np.array([1.0, 2.0, 3.0, 4.0]).sum()) == pytest.approx(10.0)


def test_max_min():
    assert _s(np.array([1.0, 5.0, 3.0]).max()) == pytest.approx(5.0)
    assert _s(np.array([1.0, 5.0, 3.0]).min()) == pytest.approx(1.0)


def test_mean():
    assert _s(np.array([1.0, 2.0, 3.0, 4.0]).mean()) == pytest.approx(2.5)


def test_prod():
    assert _s(np.array([1.0, 2.0, 3.0]).prod()) == pytest.approx(6.0)


def test_cumsum_array():
    assert _f(np.array([1.0, 2.0, 3.0]).cumsum()) == [1.0, 3.0, 6.0]


def test_cumprod_array():
    assert _f(np.array([1.0, 2.0, 3.0]).cumprod()) == [1.0, 2.0, 6.0]


def test_argmax_argmin():
    a = np.array([1.0, 5.0, 3.0])
    assert int(a.argmax()) == 1
    assert int(a.argmin()) == 0


def test_std_var():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert _s(a.std()) == pytest.approx(1.118033988749895)
    assert _s(a.var()) == pytest.approx(1.25)
