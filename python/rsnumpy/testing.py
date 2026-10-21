"""rsnumpy.testing - 兼容 `numpy.testing` 的断言子集。

`numpy.testing` 的断言函数占了很大篇幅，而且多数人只用这三个：`assert_`、
`assert_equal`、`assert_allclose`。这里提供的就是这三个，刻意保持自包含：
不 import numpy，全部在 Plain Python 上比较，因此对 rsnumpy 数组和 Python
嵌套列表都成立。
"""

import math as _math

__all__ = ['assert_', 'assert_equal', 'assert_allclose']


def _plain(value):
    """把数组样对象摊成 Python 基础值（嵌套 list / 标量）。"""
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def assert_(val, msg=''):
    """确认 `val` 为真，否则抛出带 `msg` 的 AssertionError。"""
    if not val:
        raise AssertionError(msg)


def assert_equal(actual, desired, err_msg=''):
    """要求两个对象严格相等。"""
    a, d = _plain(actual), _plain(desired)
    if a != d:
        raise AssertionError(f'{err_msg}\n{type(actual)}: {a}\n'
                             f'{type(desired)}: {d}')


def assert_allclose(actual, desired, rtol=1e-7, atol=0, equal_nan=True,
                    err_msg='', verbose=True):
    """要求两个数组逐元素在容差内相等。

    判据与 `numpy.testing.assert_allclose` 一致：
    ``|actual - desired| <= atol + rtol * |desired|``。
    """
    a, d = _plain(actual), _plain(desired)
    _check_close(a, d, float(rtol), float(atol), equal_nan, err_msg)


def _check_close(a, d, rtol, atol, equal_nan, err_msg):
    if isinstance(d, (list, tuple)) or isinstance(a, (list, tuple)):
        if not isinstance(a, (list, tuple)) or not isinstance(d, (list, tuple)):
            raise AssertionError(f'{err_msg}\nshape mismatch: {a!r} vs {d!r}')
        if len(a) != len(d):
            raise AssertionError(
                f'{err_msg}\nlength mismatch: {len(a)} vs {len(d)}')
        for i, (x, y) in enumerate(zip(a, d)):
            try:
                _check_close(x, y, rtol, atol, equal_nan, err_msg)
            except AssertionError as exc:
                raise AssertionError(f'{exc} (index {i})') from None
        return

    if isinstance(d, str) or isinstance(a, str):
        if a != d:
            raise AssertionError(f'{err_msg}\n{a!r} != {d!r}')
        return

    if isinstance(d, bool) or isinstance(a, bool):
        if a is not d and a != d:
            raise AssertionError(f'{err_msg}\n{a!r} != {d!r}')
        return

    # Complex: compare component-wise, matching NumPy's behaviour.
    if isinstance(d, complex) or isinstance(a, complex):
        ac, dc = complex(a), complex(d)
        if not _tolerant(ac.real, dc.real, rtol, atol):
            raise AssertionError(
                f'{err_msg}\nreal part: {ac.real} vs {dc.real}')
        if not _tolerant(ac.imag, dc.imag, rtol, atol):
            raise AssertionError(
                f'{err_msg}\nimag part: {ac.imag} vs {dc.imag}')
        return

    ax, dx = float(a), float(d)
    if _math.isnan(ax) or _math.isnan(dx):
        if not (equal_nan and _math.isnan(ax) and _math.isnan(dx)):
            raise AssertionError(f'{err_msg}\nnan mismatch: {ax} vs {dx}')
        return
    if _math.isinf(ax) or _math.isinf(dx):
        if ax != dx:
            raise AssertionError(f'{err_msg}\ninf mismatch: {ax} vs {dx}')
        return

    if not _tolerant(ax, dx, rtol, atol):
        raise AssertionError(
            f'{err_msg}\nmax absolute difference exceeds tolerance: '
            f'{ax} vs {dx} (rtol={rtol}, atol={atol})')


def _tolerant(a, d, rtol, atol):
    return abs(a - d) <= atol + rtol * abs(d)
