"""统计函数模块 - 所有实现位于 Rust，这里仅保留薄包装。"""

import rsnumpy._core as _core


def _nd():
    from .__init__ import ndarray as _n
    return _n


def _wrap(result):
    if hasattr(result, 'ndim') and result.ndim == 0:
        return result.tolist()
    return _nd()(result)


def _ensure_raw(a):
    if hasattr(a, '_array'):
        return a._array
    return _core.ndarray(a)


def sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
    """计算数组元素之和。"""
    return _wrap(_core.sum(_ensure_raw(a), axis))


def mean(a, axis=None, dtype=None, out=None, keepdims=False, where=True):
    """计算数组元素的平均值。"""
    return _wrap(_core.mean(_ensure_raw(a), axis))


def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True):
    """计算数组元素的标准差。"""
    return _wrap(_core.std(_ensure_raw(a), axis))


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True):
    """计算数组元素的方差。"""
    return _wrap(_core.var(_ensure_raw(a), axis))


def min(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    """计算数组元素的最小值。"""
    return _wrap(_core.min(_ensure_raw(a), axis))


def max(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    """计算数组元素的最大值。"""
    return _wrap(_core.max(_ensure_raw(a), axis))


def ptp(a, axis=None, out=None, keepdims=False):
    """计算数组元素的峰-峰值（最大值-最小值）。"""
    return _wrap(_core.ptp(_ensure_raw(a)))


def median(a, axis=None, out=None, keepdims=False):
    """计算数组的中位数。"""
    return _wrap(_core.median(_ensure_raw(a)))


def percentile(a, q, axis=None, out=None, keepdims=False, interpolation='linear'):
    """计算数组的百分位数。"""
    return _wrap(_core.percentile(_ensure_raw(a), q))


def quantile(a, q, axis=None, out=None, keepdims=False, interpolation='linear'):
    """计算数组的分位数。"""
    return percentile(a, q * 100)


def nanmedian(a, axis=None, out=None, keepdims=False):
    """忽略 NaN 的中位数。"""
    return median(a, axis)


def nanpercentile(a, q, axis=None, out=None, keepdims=False, interpolation='linear'):
    """忽略 NaN 的百分位数。"""
    return percentile(a, q)


def argmax(a, axis=None, out=None, keepdims=False):
    """返回数组沿指定轴的最大值索引。"""
    return _wrap(_core.argmax_axis(_ensure_raw(a), axis))


def argmin(a, axis=None, out=None, keepdims=False):
    """返回数组沿指定轴的最小值索引。"""
    return _wrap(_core.argmin_axis(_ensure_raw(a), axis))


def argsort(a, axis=-1, kind=None, order=None):
    """返回数组排序后的索引。"""
    return _wrap(_core.argsort(_ensure_raw(a), axis))


def sort(a, axis=-1, kind=None, order=None):
    """对数组进行排序。"""
    return _wrap(_core.sort(_ensure_raw(a), axis))


def searchsorted(a, v, side='left', sorter=None):
    """查找元素在有序数组中的插入位置。"""
    arr = a if hasattr(a, '_array') else _wrap(a)
    return _core.searchsorted(_ensure_raw(arr), v, side)


def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, *, dtype=None):
    """计算协方差矩阵。"""
    arr = m if hasattr(m, '_array') else _wrap(m)
    return _wrap(_core.cov(_ensure_raw(arr), rowvar))


def corrcoef(x, y=None, rowvar=True, bias=False, ddof=None, *, dtype=None):
    """计算相关系数矩阵。"""
    arr = x if hasattr(x, '_array') else _wrap(x)
    return _wrap(_core.corrcoef_rs(_ensure_raw(arr)))


def histogram(a, bins=10, range=None, weights=None, density=False):
    """计算直方图。"""
    if range is not None:
        result = _core.histogram(_ensure_raw(a), bins, tuple(range))
    else:
        result = _core.histogram(_ensure_raw(a), bins, None)
    return _wrap(result[0]), _wrap(result[1])


def histogram2d(x, y, bins=10, range=None, density=False, weights=None):
    """计算二维直方图。"""
    x_arr = x if hasattr(x, '_array') else _wrap(x)
    y_arr = y if hasattr(y, '_array') else _wrap(y)
    counts, (edges_x, edges_y) = _core.histogram2d_rs(_ensure_raw(x_arr), _ensure_raw(y_arr), bins)
    return _wrap(counts), [_wrap(edges_x), _wrap(edges_y)]


def histogramdd(sample, bins=10, range=None, density=False, weights=None):
    """计算多维直方图（回退到 2D）。"""
    return histogram2d(sample[:, 0], sample[:, 1], bins)


def digitize(x, bins, right=False):
    """返回输入数组中每个值所属的 bins 的索引。"""
    return _wrap(_core.digitize(_ensure_raw(x), _ensure_raw(bins)))
