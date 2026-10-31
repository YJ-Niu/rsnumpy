class AxisError(ValueError):
    def __init__(self, axis, ndim=None, msg_prefix=None):
        if ndim is None and msg_prefix is None:
            # numpy 语义：单参数调用时 axis 就是完整消息
            super().__init__(axis)
            return
        if msg_prefix is not None:
            super().__init__(f"{msg_prefix}: axis {axis} is out of bounds for array of dimension {ndim}")
        else:
            super().__init__(f"axis {axis} is out of bounds for array of dimension {ndim}")


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
