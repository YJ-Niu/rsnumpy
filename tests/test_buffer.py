"""Tests for the PEP 3118 buffer protocol of the rsnumpy ``ndarray``.

The core of these tests is that float64 ndarrays export a *writable*
zero-copy buffer (``memoryview(...).readonly is False``).  This is what lets
downstream Rust kernels (e.g. scicore ndimage) write results straight into an
output array, instead of a ``Vec -> list -> asarray -> bulk setitem`` round
trip.  Complex arrays expose no buffer (``BufferError``) — only the float64
real payload has a buffer view.

版本约束：CPython 纯 Python 类仅自 3.12 起可通过 PEP 633 ``__buffer__``
导出缓冲（<3.12 没有等价特殊方法——``__getbuffer__`` 并非 CPython 特殊
方法，那是 Cython cdef class 的特例处理）。<3.12 上 memoryview(a) 会失败，
buffer 协议消费方自动回退到 ``__array_interface__`` bytes 路径（既定安全
行为），故本文件仅对 3.12+ 生效。
"""
import sys

import pytest

import rsnumpy as np

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="buffer 协议需 PEP 633 __buffer__（仅 Python 3.12+ 支持纯 Python 类）；"
    "<3.12 消费方回退 __array_interface__ 路径",
)


def test_float64_buffer_is_writable():
    x = np.array([1.0, 2.0, 3.0])
    mv = memoryview(x)
    assert mv.readonly is False


def test_buffer_write_back_reflected_in_array():
    x = np.zeros(4)
    mv = memoryview(x)
    mv[0] = 10.0
    mv[3] = 40.0
    assert x.tolist() == [10.0, 0.0, 0.0, 40.0]


def test_buffer_1d_metadata():
    x = np.array([1.0, 2.0, 3.0])
    mv = memoryview(x)
    assert mv.shape == (3,)
    assert mv.ndim == 1
    assert mv.itemsize == 8
    assert len(mv) == 3
    assert mv.nbytes == 24
    assert mv.format == "d"  # C double


def test_buffer_2d_strides_c_contiguous():
    m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mv = memoryview(m)
    assert mv.shape == (2, 3)
    assert mv.ndim == 2
    # C-order 2x3: row stride = 3*8 bytes, column stride = 1*8 bytes.
    assert mv.strides == (24, 8)
    assert mv.readonly is False


def test_buffer_2d_write_back():
    m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mv = memoryview(m)
    mv[0, 0] = 77.0
    mv[1, 2] = 66.0
    assert m.tolist() == [[77.0, 2.0, 3.0], [4.0, 5.0, 66.0]]


def test_readonly_request_still_writable():
    # A read-only (PyBUF_FULL_RO) request on a writable array returns a
    # writable buffer, per PEP 3118 / rsnumpy's array_getbuffer behaviour.
    x = np.array([1.0, 2.0])
    mv = memoryview(x)
    assert mv.readonly is False
    mv[1] = 99.0
    assert x.tolist()[1] == 99.0


def test_buffer_nbytes_scales_with_size():
    x = np.zeros(10)
    mv = memoryview(x)
    assert mv.nbytes == 80
    assert len(mv) == 10


def test_complex_array_has_no_buffer():
    c = np.array([1.0 + 2.0j, 3.0 + 4.0j]).astype("complex128")
    with pytest.raises(BufferError):
        memoryview(c)
