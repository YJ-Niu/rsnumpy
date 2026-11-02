"""numpy._core._exceptions 兼容实现。

ufunc 无适用 loop 时的错误类型（移植测试直接从此路径导入）；
公共异常与 ``rsnumpy.exceptions`` 同源。
"""

from rsnumpy.exceptions import (  # noqa: F401
    AxisError,
    ComplexWarning,
    DTypePromotionError,
    RankWarning,
    VisibleDeprecationWarning,
)


class _UFuncNoLoopError(ValueError):
    """ufunc 找不到适用 loop 时抛出（numpy._core._exceptions 同名）。"""


class _ArrayMemoryError(MemoryError):
    """数组内存错误（numpy._core._exceptions 同名）。"""


__all__ = ['_UFuncNoLoopError', '_ArrayMemoryError', 'AxisError',
           'ComplexWarning', 'DTypePromotionError', 'RankWarning',
           'VisibleDeprecationWarning']
