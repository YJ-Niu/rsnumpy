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


class _UFuncNoLoopError(TypeError, ValueError):
    """ufunc 找不到适用 loop 时抛出（numpy._core._exceptions 同名）。

    numpy 中 UFuncNoLoopError 是 TypeError 子类；同时保留 ValueError
    基类以兼容既有按 ValueError 捕获的代码。"""


class _ArrayMemoryError(MemoryError):
    """数组内存错误（numpy._core._exceptions 同名）。

    ``_ArrayMemoryError(shape, dtype)``；``_total_size = prod(shape) * itemsize``；
    ``_size_to_string`` 按 3 有效数字格式化（≥1023.5 进位到下一单位）。
    """

    def __init__(self, shape, dtype):
        _n = 1
        for _s in shape:
            _n *= int(_s)
        _itemsize = int(getattr(dtype, 'itemsize', 1) or 1)
        self._shape = tuple(shape)
        self._dtype = dtype
        self._total_size = _n * _itemsize
        self._total_size_string = self._size_to_string(self._total_size)
        super().__init__(
            f"Unable to allocate {self._total_size_string} for an array "
            f"with shape {self._shape} and data type {dtype}")

    @staticmethod
    def _size_to_string(size):
        """numpy 语义：bytes 整数；其余单位 3 有效数字，≥1023.5 进位。"""
        Ki = 1024
        units = ('bytes', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB')
        v = float(size)
        for unit in units:
            if unit == 'bytes':
                if size < Ki:
                    return f"{int(size)} {unit}"
                v = size / Ki
                continue
            if v < 1023.5 or unit == 'EiB':
                iv = int(v)
                if iv >= 100:
                    return f"{round(v)}. {unit}"
                if iv >= 10:
                    return f"{v:.1f} {unit}"
                return f"{v:.2f} {unit}"
            v = v / Ki
        return f"{int(size)} bytes"

    def __reduce__(self):
        # 不经 dtype pickle（其可序列化性不定），直接还原关键字段
        return (_array_memory_error_rebuild,
                (self._shape, self._total_size, self.args[0]))


def _array_memory_error_rebuild(shape, total_size, msg):
    e = _ArrayMemoryError.__new__(_ArrayMemoryError)
    MemoryError.__init__(e, msg)
    e._shape = tuple(shape)
    e._total_size = total_size
    e._total_size_string = _ArrayMemoryError._size_to_string(total_size)
    return e


__all__ = ['_UFuncNoLoopError', '_ArrayMemoryError', 'AxisError',
           'ComplexWarning', 'DTypePromotionError', 'RankWarning',
           'VisibleDeprecationWarning']
