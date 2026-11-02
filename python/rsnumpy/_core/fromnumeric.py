"""numpy._core.fromnumeric 兼容层：从 rsnumpy 顶层再导出。

rsnumpy 的逐元素/构造公共 API 直接挂在顶层命名空间；本模块提供
NumPy 2.x 的 ``numpy._core.fromnumeric`` 导入路径。
"""

_NAMES = (
    'all', 'allclose', 'amax', 'amin', 'any', 'argmax', 'argmin',
    'argpartition', 'argsort', 'around', 'ceil', 'clip', 'copyto',
    'cumprod', 'cumsum', 'deg2rad', 'diff', 'ediff1d', 'empty',
    'empty_like', 'fix', 'floor', 'floor_divide', 'full', 'full_like',
    'gradient', 'inner', 'isclose', 'isfinite', 'isinf', 'isnan',
    'itemset', 'log2', 'log10', 'logaddexp', 'logaddexp2', 'maximum',
    'minimum', 'mod', 'modf', 'msort', 'nan_to_num', 'nanmax',
    'nanmean', 'nanmin', 'nansum', 'nonzero', 'ones', 'ones_like',
    'prod', 'product', 'ptp', 'rad2deg', 'remainder', 'round',
    'round_', 'searchsorted', 'sign', 'signbit', 'sqrt', 'square',
    'std', 'sum', 'true_divide', 'transpose', 'var',
)


def __getattr__(name):
    if name in _NAMES:
        import rsnumpy
        return getattr(rsnumpy, name)
    raise AttributeError(f"module 'rsnumpy._core.fromnumeric' has no "
                         f"attribute {name!r}")


__all__ = list(_NAMES)
