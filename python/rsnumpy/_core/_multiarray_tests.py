"""numpy._core._multiarray_tests 兼容实现（C 测试辅助的 Python 版）。"""

__all__ = ['array_indexing']


def _rs():
    import rsnumpy
    return rsnumpy


def array_indexing(op, array, *args):
    """C-API 索引辅助：op=0 取 ``array[args]``；
    op=1 赋值 ``array[args[:-1]] = args[-1]``。

    0-d 数组与越界索引抛 IndexError（与 C 实现一致）；
    op=1 缺少值时抛 ValueError（删除不可能）。
    """
    rs = _rs()
    arr = array if isinstance(array, rs.ndarray) else rs.asarray(array)
    if arr.ndim == 0:
        raise IndexError("0-d arrays cannot be indexed")
    if op == 0:
        if not args:
            raise IndexError("no indices given")
        return arr[args]
    if op == 1:
        value = args[-1]
        idx = args[:-1]
        if not idx:
            raise ValueError("cannot delete from a C array")
        arr[idx] = value
        return None
    raise ValueError(f"unknown op {op}")


# ---------------------------------------------------------------------------
# C 内部测试辅助桩：rsnumpy 无对应 C 实现；相关测试由 conftest 的
# _SKIP_MARKERS 源串匹配跳过（这些函数在测试模块顶层被 import，
# 故此处必须存在可导入的名字）。
# ---------------------------------------------------------------------------

def _not_implemented(name):
    raise NotImplementedError(
        f"rsnumpy has no C helper {name!r}; test is expected to be skipped")


def create_custom_field_dtype(*args, **kwargs):
    _not_implemented("create_custom_field_dtype")


def get_c_wrapping_array(*args, **kwargs):
    _not_implemented("get_c_wrapping_array")


def incref_elide(*args, **kwargs):
    _not_implemented("incref_elide")


def incref_elide_l(*args, **kwargs):
    _not_implemented("incref_elide_l")


def IsPythonScalar(*args, **kwargs):
    _not_implemented("IsPythonScalar")


def npy_char_deprecation(*args, **kwargs):
    _not_implemented("npy_char_deprecation")


__all__ = __all__ + ['create_custom_field_dtype', 'get_c_wrapping_array',
                     'incref_elide', 'incref_elide_l', 'IsPythonScalar',
                     'npy_char_deprecation']


def run_scalar_intp_converter(x):
    """标量 → intp 转换（numpy._core._multiarray_tests 同名 C 桩）。"""
    import rsnumpy
    return rsnumpy.intp(x)


def run_scalar_intp_from_sequence(x):
    """序列 → intp 转换（numpy._core._multiarray_tests 同名 C 桩）。"""
    import rsnumpy
    return tuple(rsnumpy.intp(v) for v in x)


# ---- npymath 冒烟函数桩（test_npymath_* 用，Python math/cmath 实现） ----

def npy_cabs(z):
    """复数模（npymath npy_cabs 桩）。"""
    return abs(complex(z))


def npy_carg(z):
    """复数辐角（npymath npy_carg 桩）。"""
    import cmath
    return cmath.phase(complex(z))


def npy_cosh(x):
    """双曲余弦（npymath 桩）。"""
    import math
    return math.cosh(float(x))


def npy_log10(x):
    """常用对数（npymath 桩）。"""
    import math
    return math.log10(float(x))


def npy_sinh(x):
    """双曲正弦（npymath 桩）。"""
    import math
    return math.sinh(float(x))


def npy_tan(x):
    """正切（npymath 桩）。"""
    import math
    return math.tan(float(x))


def npy_tanh(x):
    """双曲正切（npymath 桩）。"""
    import math
    return math.tanh(float(x))
