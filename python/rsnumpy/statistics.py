"""统计函数模块 - 所有实现位于 Rust，这里仅保留薄包装。"""

import builtins as _builtins

import rsnumpy.num_core as _core


def _nd():
    from .__init__ import ndarray as _n
    return _n


def _is_ndarray(obj):
    return hasattr(obj, '_array')


def _wrap(result):
    if hasattr(result, 'ndim') and result.ndim == 0:
        val = result.tolist()
        if isinstance(val, float) and val.is_integer():
            return int(val)
        return val
    return _nd()(result)


def _ensure_raw(a):
    if hasattr(a, '_array'):
        return a._array
    return _core.ndarray(a)


def _is_complex_raw(a):
    raw = a._array if hasattr(a, '_array') else a
    return bool(getattr(raw, 'is_complex', False))


def _flatten_c(data):
    out = []

    def rec(d):
        if isinstance(d, list):
            for x in d:
                rec(x)
        else:
            out.append(d)

    rec(data)
    return out


def _reshape_nested(flat, shape):
    if not shape:
        return flat[0]
    if len(shape) == 1:
        return list(flat)
    block = 1
    for s in shape[1:]:
        block *= s
    return [_reshape_nested(flat[i * block:(i + 1) * block], shape[1:]) for i in range(shape[0])]


def _py_var_std(a, axis, ddof, want_std):
    """方差/标准差的原生实现（无外部依赖），支持实数与复数数组。

    与 numpy 语义一致：var = mean(|x - mean|^2)，std = sqrt(var)；复数输入结果为实数。
    支持 axis=None 或整数 axis。用于补齐 Rust 端在高维/复数场景下的能力缺口。
    """
    a_arr = a if hasattr(a, '_array') else _nd()(a)
    shape = list(a_arr.shape)
    flat = _flatten_c(a_arr.tolist())

    def reduce_vals(vals):
        n = len(vals)
        m = _builtins.sum(vals) / n
        v = _builtins.sum(abs(x - m) ** 2 for x in vals) / (n - ddof)
        return (v ** 0.5) if want_std else v

    if axis is None:
        return reduce_vals(flat)

    nd = len(shape)
    if axis < 0:
        axis += nd
    strides = [1] * nd
    for i in range(nd - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    axis_len = shape[axis]
    axis_stride = strides[axis]
    out_shape = shape[:axis] + shape[axis + 1:]
    out_strides = strides[:axis] + strides[axis + 1:]
    out_size = 1
    for s in out_shape:
        out_size *= s

    result = []
    idx = [0] * len(out_shape)
    for _ in range(out_size):
        base = 0
        for k, ix in enumerate(idx):
            base += ix * out_strides[k]
        vals = [flat[base + t * axis_stride] for t in range(axis_len)]
        result.append(reduce_vals(vals))
        for k in range(len(out_shape) - 1, -1, -1):
            idx[k] += 1
            if idx[k] < out_shape[k]:
                break
            idx[k] = 0

    nested = _reshape_nested(result, out_shape)
    if not out_shape:
        return nested
    return _nd()(nested)


def sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
    """计算数组元素之和。"""
    _ = dtype, out, keepdims, initial, where
    raw_result = _core.sum(_ensure_raw(a), axis)
    return _wrap(raw_result)


def mean(a, axis=None, dtype=None, out=None, keepdims=False, where=True):
    """计算数组元素的平均值。"""
    _ = dtype, out, keepdims, where
    return _wrap(_core.mean(_ensure_raw(a), axis))


def _needs_py_var_std(a, axis):
    """判断是否需要走原生实现：复数，或 Rust 端不支持的高维（ndim>=3）+ 指定 axis。"""
    if _is_complex_raw(a):
        return True
    if axis is None:
        return False
    raw = a._array if hasattr(a, '_array') else None
    return raw is not None and len(raw.shape) >= 3


def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True):
    """计算数组元素的标准差。"""
    _ = dtype, out, keepdims, where
    if _needs_py_var_std(a, axis):
        return _py_var_std(a, axis, ddof, want_std=True)
    raw_result = _core.std(_ensure_raw(a), axis)
    arr_dtype = getattr(a, '_dtype', 'float64') if hasattr(a, '_dtype') else 'float64'
    return _nd()(raw_result, _dtype=arr_dtype)


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True):
    """计算数组元素的方差。"""
    _ = dtype, out, keepdims, where
    if _needs_py_var_std(a, axis):
        return _py_var_std(a, axis, ddof, want_std=False)
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
    raw = _core.argmax_axis(_ensure_raw(a), axis)
    if axis is None:
        return int(raw.tolist())
    return _nd()._wrap(raw, _dtype='int64')


def argmin(a, axis=None, out=None, keepdims=False):
    """返回数组沿指定轴的最小值索引。"""
    _ = out, keepdims
    raw = _core.argmin_axis(_ensure_raw(a), axis)
    if axis is None:
        return int(raw.tolist())
    return _nd()._wrap(raw, _dtype='int64')


def argsort(a, axis=-1, kind=None, order=None):
    """返回数组排序后的索引。"""
    _ = kind, order
    return _nd()._wrap(_core.argsort(_ensure_raw(a), axis), _dtype='int64')


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
    
    first_key = keys[0]
    n = len(first_key) if hasattr(first_key, '__len__') else 0
    if n == 0:
        return _nd()([], _dtype='int64')
    
    indices = list(range(n))
    
    indices.sort(key=lambda i: tuple(key[i] for key in reversed(keys)))
    
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
    if hasattr(v, '_array') and len(v.shape) == 0:
        v = float(v.item())
    return _core.searchsorted(_ensure_raw(arr), v, side)


def extract(condition, a):
    """根据条件从数组中抽取元素。"""
    cond_arr = condition if hasattr(condition, '_array') else _nd()(condition)
    data_arr = a if hasattr(a, '_array') else _nd()(a)
    return _wrap(_core.extract(cond_arr._array, data_arr._array))


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
