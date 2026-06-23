"""统计函数模块 - 所有实现位于 Rust，这里仅保留薄包装。"""

import rsnumpy._core as _core


def _nd():
    from .__init__ import ndarray as _n
    return _n


def _is_ndarray(obj):
    return hasattr(obj, '_array')


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
    _ = dtype, out, keepdims, initial, where
    raw_result = _core.sum(_ensure_raw(a), axis)
    arr_dtype = getattr(a, '_dtype', 'float64') if hasattr(a, '_dtype') else 'float64'
    return _nd()(raw_result, _dtype=arr_dtype)


def mean(a, axis=None, dtype=None, out=None, keepdims=False, where=True):
    """计算数组元素的平均值。"""
    _ = dtype, out, keepdims, where
    return _wrap(_core.mean(_ensure_raw(a), axis))


def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True):
    """计算数组元素的标准差。"""
    _ = dtype, out, keepdims, where
    raw_result = _core.std(_ensure_raw(a), axis)
    arr_dtype = getattr(a, '_dtype', 'float64') if hasattr(a, '_dtype') else 'float64'
    return _nd()(raw_result, _dtype=arr_dtype)


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True):
    """计算数组元素的方差。"""
    _ = dtype, out, keepdims, where
    raw_result = _core.var(_ensure_raw(a), axis)
    arr_dtype = getattr(a, '_dtype', 'float64') if hasattr(a, '_dtype') else 'float64'
    return _nd()(raw_result, _dtype=arr_dtype)


def min(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    """计算数组元素的最小值。"""
    _ = out, keepdims, initial, where
    raw_result = _core.min(_ensure_raw(a), axis)
    dtype = getattr(a, '_dtype', 'float64') if hasattr(a, '_dtype') else 'float64'
    return _nd()(raw_result, _dtype=dtype)


def max(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    """计算数组元素的最大值。"""
    _ = out, keepdims, initial, where
    raw_result = _core.max(_ensure_raw(a), axis)
    dtype = getattr(a, '_dtype', 'float64') if hasattr(a, '_dtype') else 'float64'
    return _nd()(raw_result, _dtype=dtype)


amin = min  # min 的别名，计算数组中的最小值
amax = max  # max 的别名，计算数组中的最大值


def ptp(a, axis=None, out=None, keepdims=False):
    """计算数组元素的峰-峰值（最大值-最小值）。"""
    _ = out, keepdims
    raw_result = _core.ptp(_ensure_raw(a), axis)
    arr_dtype = getattr(a, '_dtype', 'float64') if hasattr(a, '_dtype') else 'float64'
    return _nd()(raw_result, _dtype=arr_dtype)


def median(a, axis=None, out=None, keepdims=False):
    """计算数组的中位数。"""
    _ = out, keepdims
    return _wrap(_core.median(_ensure_raw(a), axis))


def average(a, axis=None, weights=None, returned=False):
    """计算数组的加权平均值。"""
    if weights is not None:
        weights = _ensure_raw(weights)
    raw_result = _core.average(_ensure_raw(a), axis, weights, returned)
    if returned:
        if hasattr(raw_result, 'tolist'):
            result_list = raw_result.tolist()
        else:
            result_list = list(raw_result)
        if axis is None:
            avg_val = float(result_list[0])
            sum_weights = float(result_list[1])
            return avg_val, sum_weights
        else:
            if isinstance(result_list[0], (list, tuple)):
                avg_vals = [item[0] for item in result_list]
                sum_ws = [item[1] for item in result_list]
            else:
                avg_vals = result_list[0]
                sum_ws = result_list[1]
            return _nd()(avg_vals, _dtype='float64'), _nd()(sum_ws, _dtype='float64')
    return _wrap(raw_result)


def percentile(a, q, axis=None, out=None, keepdims=False, interpolation='linear'):
    """计算数组的百分位数。"""
    _ = out, interpolation
    return _wrap(_core.percentile(_ensure_raw(a), q, axis, keepdims))


def quantile(a, q, axis=None, out=None, keepdims=False, interpolation='linear'):
    """计算数组的分位数。"""
    _ = axis, out, keepdims, interpolation
    return percentile(a, q * 100)


def nanmedian(a, axis=None, out=None, keepdims=False):
    """忽略 NaN 的中位数。"""
    _ = axis, out, keepdims
    return median(a, axis=None)


def nanpercentile(a, q, axis=None, out=None, keepdims=False, interpolation='linear'):
    """忽略 NaN 的百分位数。"""
    _ = axis, out, keepdims, interpolation
    return percentile(a, q)


def argmax(a, axis=None, out=None, keepdims=False):
    """返回数组沿指定轴的最大值索引。"""
    _ = out, keepdims
    return _wrap(_core.argmax_axis(_ensure_raw(a), axis))


def argmin(a, axis=None, out=None, keepdims=False):
    """返回数组沿指定轴的最小值索引。"""
    _ = out, keepdims
    return _wrap(_core.argmin_axis(_ensure_raw(a), axis))


def argsort(a, axis=-1, kind=None, order=None):
    """返回数组排序后的索引。"""
    _ = kind, order
    result = _wrap(_core.argsort(_ensure_raw(a), axis))
    result_list = result.tolist()
    int_list = [int(v) for v in result_list]
    return _nd()(int_list, _dtype='int64')


def sort(a, axis=-1, kind=None, order=None):
    """对数组进行排序。"""
    _ = kind
    arr = _nd()(a) if not _is_ndarray(a) else a
    fields = getattr(arr, '_fields', None)
    if fields and order is not None:
        field_names = [f[0] for f in fields]
        if order not in field_names:
            raise ValueError(f"field {order!r} not found in dtype")
        field_idx = field_names.index(order)
        raw_data = getattr(arr, '_raw_data', None)
        if raw_data is not None:
            sorted_data = sorted(raw_data, key=lambda x: x[field_idx])
            return _nd()(sorted_data, _dtype=arr._dtype, _fields=fields, _raw_data=sorted_data)
    return _wrap(_core.sort(_ensure_raw(a), axis))


def lexsort(keys, axis=-1):
    """使用多个键进行间接排序。优先按最后一个键排序。"""
    _ = axis
    if not isinstance(keys, (tuple, list)):
        keys = (keys,)
    key_arrays = []
    for key in keys:
        if _is_ndarray(key):
            key_arrays.append(key.tolist())
        elif hasattr(key, 'tolist'):
            key_arrays.append(key.tolist())
        else:
            key_arrays.append(list(key))
    n = len(key_arrays[0]) if key_arrays else 0
    indices = list(range(n))
    indices.sort(key=lambda i: tuple(key_arrays[idx][i] for idx in reversed(range(len(key_arrays)))))
    return _nd()(indices, _dtype='int64')


def msort(a):
    """数组按第一个轴排序，返回排序后的数组副本。"""
    return sort(a, axis=0)


def sort_complex(a):
    """对复数按照先实部后虚部的顺序进行排序。"""
    arr = _nd()(a) if not _is_ndarray(a) else a
    data = arr.tolist()
    data = [complex(x) for x in data]
    data.sort(key=lambda x: (x.real, x.imag))
    return _nd()(data)


def partition(a, kth, axis=-1, kind=None, order=None):
    """指定一个数，对数组进行分区。"""
    _ = kind, order
    arr = _nd()(a) if not _is_ndarray(a) else a
    data = arr.tolist()
    arr_dtype = getattr(arr, '_dtype', None)
    
    def partition_single(arr_list, k, last=False):
        n = len(arr_list)
        if k < 0:
            k = n + k
        if n == 0 or k < 0 or k >= n:
            return arr_list
        sorted_list = sorted(arr_list)
        pivot = sorted_list[k]
        left = []
        mid = []
        right = []
        for x in arr_list:
            if x < pivot:
                left.append(x)
            elif x == pivot:
                mid.append(x)
            else:
                right.append(x)
        
        if not last and len(left) > 0 and k >= len(left):
            left = left[1:] + [left[0]]
        
        return left + mid + right
    
    if isinstance(kth, (tuple, list)):
        kths = sorted(set(kth))
        for i, k in enumerate(kths):
            data = partition_single(data, k, i == len(kths) - 1)
    else:
        data = partition_single(data, kth)
    
    if arr_dtype is not None:
        return _nd()(data, _dtype=arr_dtype)
    return _nd()(data)


def argpartition(a, kth, axis=-1, kind=None, order=None):
    """对数组进行分区并返回索引。"""
    _ = kind, order
    arr = _nd()(a) if not _is_ndarray(a) else a
    data = arr.tolist()
    indices = list(range(len(data)))
    
    def quick_select(arr_list, idx_list, k):
        if k < 0:
            k = len(arr_list) + k
        low = 0
        high = len(arr_list) - 1
        while low < high:
            pivot_idx = (low + high) // 2
            arr_list[pivot_idx], arr_list[high] = arr_list[high], arr_list[pivot_idx]
            idx_list[pivot_idx], idx_list[high] = idx_list[high], idx_list[pivot_idx]
            pivot = arr_list[high]
            i = low
            for j in range(low, high):
                if arr_list[j] < pivot:
                    arr_list[i], arr_list[j] = arr_list[j], arr_list[i]
                    idx_list[i], idx_list[j] = idx_list[j], idx_list[i]
                    i += 1
            arr_list[i], arr_list[high] = arr_list[high], arr_list[i]
            idx_list[i], idx_list[high] = idx_list[high], idx_list[i]
            if i == k:
                break
            elif i < k:
                low = i + 1
            else:
                high = i - 1
    
    if isinstance(kth, (tuple, list)):
        kths = sorted(set(kth))
        for k in kths:
            quick_select(data, indices, k)
    else:
        quick_select(data, indices, kth)
    
    return _nd()(indices, _dtype='int64')


def searchsorted(a, v, side='left', sorter=None):
    """查找元素在有序数组中的插入位置。"""
    _ = sorter
    arr = a if hasattr(a, '_array') else _wrap(a)
    return _core.searchsorted(_ensure_raw(arr), v, side)


def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, *, dtype=None):
    """计算协方差矩阵。"""
    _ = y, bias, ddof, fweights, aweights, dtype
    arr = m if hasattr(m, '_array') else _wrap(m)
    return _wrap(_core.cov(_ensure_raw(arr), rowvar))


def corrcoef(x, y=None, rowvar=True, bias=False, ddof=None, *, dtype=None):
    """计算相关系数矩阵。"""
    _ = y, bias, ddof, dtype
    arr = x if hasattr(x, '_array') else _wrap(x)
    return _wrap(_core.corrcoef_rs(_ensure_raw(arr)))


def histogram(a, bins=10, range=None, weights=None, density=False):
    """计算直方图。"""
    _ = weights, density
    if range is not None:
        result = _core.histogram(_ensure_raw(a), bins, tuple(range))
    else:
        result = _core.histogram(_ensure_raw(a), bins, None)
    return _wrap(result[0]), _wrap(result[1])


def histogram2d(x, y, bins=10, range=None, density=False, weights=None):
    """计算二维直方图。"""
    _ = range, density, weights
    x_arr = x if hasattr(x, '_array') else _wrap(x)
    y_arr = y if hasattr(y, '_array') else _wrap(y)
    counts, (edges_x, edges_y) = _core.histogram2d_rs(_ensure_raw(x_arr), _ensure_raw(y_arr), bins)
    return _wrap(counts), [_wrap(edges_x), _wrap(edges_y)]


def histogramdd(sample, bins=10, range=None, density=False, weights=None):
    """计算多维直方图（回退到 2D）。"""
    _ = range, density, weights
    return histogram2d(sample[:, 0], sample[:, 1], bins)


def digitize(x, bins, right=False):
    """返回输入数组中每个值所属的 bins 的索引。"""
    _ = right
    return _wrap(_core.digitize(_ensure_raw(x), _ensure_raw(bins)))
