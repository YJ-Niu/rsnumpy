# SPDX-License-Identifier: MIT
"""numpy.lib._function_base_impl 兼容实现。

rsnumpy 的 function_base 公共实现在 ``rsnumpy._core.function_base``；
本模块按其 NumPy 2.x 布局再导出，并补齐移植测试直接引用的私有名。
"""

from rsnumpy._core.function_base import (  # noqa: F401
    add_newdoc,
    add_docstring,
    _add_docstring,
)


import enum


class _QuantileMethods(str, enum.Enum):
    """Valid methods for computing quantiles（numpy._QuantileMethods 兼容）。

    ``list(_QuantileMethods)`` 返回五个方法成员；成员为 str 子类，
    可直接作为 method= 参数传给 quantile/nanquantile。
    """

    LINEAR = 'linear'
    LOWER = 'lower'
    HIGHER = 'higher'
    MIDPOINT = 'midpoint'
    NEAREST = 'nearest'

    @classmethod
    def is_valid(cls, value):
        return value in [m.value for m in cls]


__all__ = ['add_newdoc', 'add_docstring', '_add_docstring',
           '_QuantileMethods']
