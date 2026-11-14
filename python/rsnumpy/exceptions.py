# SPDX-License-Identifier: MIT
class AxisError(ValueError, IndexError):
    def __init__(self, axis, ndim=None, msg_prefix=None):
        # numpy 语义：单参数调用时 axis 就是完整消息（此时 axis/ndim 属性为 None）
        self.axis = None if (ndim is None and msg_prefix is None) else axis
        self.ndim = ndim
        if ndim is None and msg_prefix is None:
            super().__init__(axis)
        elif msg_prefix is not None:
            super().__init__(f"{msg_prefix}: axis {axis} is out of bounds for array of dimension {ndim}")
        else:
            super().__init__(f"axis {axis} is out of bounds for array of dimension {ndim}")
        self._init_args = (axis, ndim, msg_prefix)

    def __reduce__(self):
        # pickle 后 axis/ndim/args 保持一致（numpy 语义）
        return (self.__class__, self._init_args)


class ComplexWarning(Warning):
    pass


class DTypePromotionError(TypeError):
    pass


class ModuleDeprecationWarning(DeprecationWarning):
    pass


class RankWarning(Warning):
    pass


class TooHardError(Exception):
    pass


class VisibleDeprecationWarning(Warning):
    pass


__all__ = ['AxisError', 'ComplexWarning', 'DTypePromotionError', 'ModuleDeprecationWarning',
           'RankWarning', 'TooHardError', 'VisibleDeprecationWarning']
