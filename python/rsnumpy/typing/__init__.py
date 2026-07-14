"""rsnumpy.typing - 类型注解支持，对齐 numpy.typing。

提供 `NDArray`、`ArrayLike`、`DTypeLike`，主要用于类型注解，
支持 `NDArray[complex]`、`NDArray[float]` 等下标写法。
"""

import typing as _t

__all__ = ['NDArray', 'ArrayLike', 'DTypeLike']

_ScalarType = _t.TypeVar('_ScalarType')


class NDArray(_t.Generic[_ScalarType]):
    """numpy.typing.NDArray 的轻量替代，支持下标用于类型注解。"""

    def __class_getitem__(cls, item):
        return _t.cast(type, cls)


# 任意可转换为数组的对象；用于函数签名注解。
ArrayLike = _t.Any

# 任意可解释为 dtype 的对象。
DTypeLike = _t.Any
