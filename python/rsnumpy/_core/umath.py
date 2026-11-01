"""numpy._core.umath 兼容门面：按名暴露 rsnumpy 的 ufunc 集合。

NumPy 中这是 C 层 ufunc 模块；rsnumpy 的 ufunc 是 Python 层可调用
对象，本模块以惰性属性访问按名导出（``import rsnumpy._core.umath``
即可，无需逐个枚举）。
"""

__all__ = []


def __getattr__(name):
    if name == 'identity':
        from rsnumpy._core import ufunc
        return ufunc.identity
    import rsnumpy
    obj = getattr(rsnumpy, name, None)
    if obj is not None and not name.startswith('_'):
        return obj
    raise AttributeError(f"module 'rsnumpy._core.umath' has no attribute {name!r}")
