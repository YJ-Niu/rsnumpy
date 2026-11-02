"""numpy._core.umath 兼容门面：按名暴露 rsnumpy 的 ufunc 集合。

NumPy 中这是 C 层 ufunc 模块；rsnumpy 的 ufunc 是 Python 层可调用
对象。本模块在导入时把 rsnumpy 的 ufunc 函数标记（``_rs_ufunc`` +
``.types`` 循环签名）并按同名导出，使 ``np._core.umath.__dict__``
可直接按 ``isinstance(obj, np.ufunc)`` 迭代（test_umath_accuracy
模块级代码依赖此）。
"""

import rsnumpy as _np

# numpy ufunc 循环签名（numpy 实际含 object 循环 OO->O 的标注了 object
# 循环；其余为数值循环）
_UFUNC_TYPES = {
    'add': 'OO->O,dd->d',
    'subtract': 'OO->O,dd->d',
    'multiply': 'OO->O,dd->d',
    'divide': 'OO->O,dd->d',
    'true_divide': 'OO->O,dd->d',
    'floor_divide': 'OO->O,dd->d',
    'mod': 'OO->O,dd->d',
    'fmod': 'dd->d',
    'remainder': 'OO->O,dd->d',
    'power': 'OO->O,dd->d',
    'negative': 'dd->d',
    'positive': 'dd->d',
    'absolute': 'dd->d',
    'fabs': 'dd->d',
    'sign': 'OO->O,dd->d',
    'copysign': 'dd->d',
    'maximum': 'OO->O,dd->d',
    'minimum': 'OO->O,dd->d',
    'hypot': 'dd->d',
    'ldexp': 'dd->d',
    'log': 'dd->d',
    'log2': 'dd->d',
    'log10': 'dd->d',
    'log1p': 'dd->d',
    'logaddexp': 'dd->d',
    'exp': 'dd->d',
    'expm1': 'dd->d',
    'exp2': 'dd->d',
    'sqrt': 'dd->d',
    'square': 'dd->d',
    'sin': 'dd->d',
    'cos': 'dd->d',
    'tan': 'dd->d',
    'arcsin': 'dd->d',
    'arccos': 'dd->d',
    'arctan': 'dd->d',
    'arctan2': 'dd->d',
    'sinh': 'dd->d',
    'cosh': 'dd->d',
    'tanh': 'dd->d',
    'arcsinh': 'dd->d',
    'arccosh': 'dd->d',
    'arctanh': 'dd->d',
    'floor': 'dd->d',
    'ceil': 'dd->d',
    'trunc': 'dd->d',
    'round': 'dd->d',
    'bitwise_and': 'OO->O,bb->b',
    'bitwise_or': 'OO->O,bb->b',
    'bitwise_xor': 'OO->O,bb->b',
    'invert': 'OO->O,bb->b',
    'bitwise_count': 'OO->O,bb->b',
    'left_shift': 'OO->O,dd->d',
    'right_shift': 'OO->O,dd->d',
    'logical_and': 'OO->O,??->?',
    'logical_or': 'OO->O,??->?',
    'logical_xor': 'OO->O,??->?',
    'logical_not': 'OO->O,?->?',
    'equal': 'OO->O,dd->d',
    'not_equal': 'OO->O,dd->d',
    'less': 'OO->O,dd->d',
    'greater': 'OO->O,dd->d',
    'less_equal': 'OO->O,dd->d',
    'greater_equal': 'OO->O,dd->d',
    'isfinite': 'dd->?',
    'isinf': 'dd->?',
    'isnan': 'dd->?',
    'all': 'OO->O,?->?',
    'any': 'OO->O,?->?',
    'cumsum': 'OO->O,dd->d',
    'cumprod': 'OO->O,dd->d',
    'gcd': 'dd->d',
    'lcm': 'dd->d',
}

__all__ = []

for _name, _types in _UFUNC_TYPES.items():
    _f = getattr(_np, _name, None)
    if _f is None or not callable(_f):
        continue
    if not getattr(_f, '_rs_ufunc', False):
        _f._rs_ufunc = True
        _f.types = _types
    globals()[_name] = _f
    __all__.append(_name)
del _name, _types, _f


def __getattr__(name):
    if name == 'identity':
        from rsnumpy._core import ufunc
        return ufunc.identity
    import rsnumpy
    obj = getattr(rsnumpy, name, None)
    if obj is not None and not name.startswith('_'):
        return obj
    raise AttributeError(f"module 'rsnumpy._core.umath' has no attribute {name!r}")
