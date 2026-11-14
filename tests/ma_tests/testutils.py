"""``numpy.ma.testutils`` 的本地替身。

被拷贝过来的 ma / lib / matrixlib 测试原本从 ``numpy.ma.testutils`` 导入断言
辅助函数；本仓库把 NumPy 的测试树放在 ``tests/`` 下，且 ``rsnumpy.ma`` 是单文件
模块（不是包），因此这些共享断言函数统一由 ``testing`` 重新导出，并在此补上
NumPy 里只在 ma.testutils 中出现的几个辅助函数。
"""
from testing import (  # noqa: F401
    assert_,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_array_less,
    assert_equal,
    assert_raises,
)

__all__ = [
    "assert_",
    "assert_almost_equal",
    "assert_array_almost_equal",
    "assert_array_equal",
    "assert_array_less",
    "assert_equal",
    "assert_equal_records",
    "assert_mask_equal",
    "assert_not_equal",
    "assert_raises",
    "fail_if_equal",
]


def _values_equal(a, b):
    try:
        res = a == b
    except Exception:
        return False
    try:
        return bool(res.all())
    except Exception:
        try:
            return bool(res)
        except Exception:
            return False


def assert_equal_records(a, b):
    """断言两个结构化数组记录相等（dtype 与逐字段值）。"""
    if not (hasattr(a, "dtype") and hasattr(b, "dtype")):
        raise TypeError("Both inputs should be ndarrays (subclasses of ndarray)")
    if a.dtype != b.dtype:
        raise AssertionError(f"Types conflict: {a.dtype!r} and {b.dtype!r}")
    for name in a.dtype.names or ():
        assert_array_equal(a[name], b[name])


def assert_mask_equal(m1, m2, names=None, err_msg=""):
    """断言两个掩码相等。"""
    if names is None:
        assert_array_equal(m1, m2, err_msg=err_msg)
    else:
        for name in names:
            assert_array_equal(m1[name], m2[name], err_msg=err_msg)


def assert_not_equal(actual, desired, err_msg="", verbose=True):
    """断言 ``actual != desired``（标量或数组逐元素）。"""
    if _values_equal(actual, desired):
        raise AssertionError(f"{err_msg}\n x: {actual!r}\n y: {desired!r}")


def fail_if_equal(actual, desired, err_msg="", verbose=True):
    """``assert_not_equal`` 的别名。"""
    assert_not_equal(actual, desired, err_msg=err_msg, verbose=verbose)
