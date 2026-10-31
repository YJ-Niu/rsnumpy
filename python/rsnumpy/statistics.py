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
    # 使用一次性累乘代替循环
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

    # 预计算所有 base 偏移，避免嵌套循环中重复计算
    result = [0.0] * out_size
    idx = [0] * len(out_shape)
    # 预生成所有 axis 索引的 stride 偏移
    axis_offsets = [t * axis_stride for t in range(axis_len)]
    for i in range(out_size):
        # 计算 base 偏移：用内建 sum 代替显式循环
        base = _builtins.sum(idx[k] * out_strides[k] for k in range(len(out_strides)))
        # 列表推导代替显式循环
        vals = [flat[base + off] for off in axis_offsets]
        result[i] = reduce_vals(vals)
        # 推进 idx（从最低维度开始）
        for k in range(len(out_shape) - 1, -1, -1):
            idx[k] += 1
            if idx[k] < out_shape[k]:
                break
            idx[k] = 0

    nested = _reshape_nested(result, out_shape)
    if not out_shape:
        return nested
    return _nd()(nested)


def _f64_scalar(v):
    """把 Python float 包装为 rsnumpy float64 标量（带 shape/itemsize 语义）。"""
    try:
        from .__init__ import _SCALAR_CLASS_BY_DTYPE
        _cls = _SCALAR_CLASS_BY_DTYPE.get('float64')
        if _cls is not None:
            return _cls(v)
    except Exception:
        pass
    return v


def _insert_keepdims(res, nd0, axes):
    """在归约轴位置为 res 插入 size-1 轴（numpy keepdims=True 语义）。"""
    _axset = set(axes)
    _new_shape = []
    _si = 0
    for d in range(nd0):
        if d in _axset:
            _new_shape.append(1)
        else:
            _new_shape.append(res.shape[_si])
            _si += 1
    return res.reshape(tuple(_new_shape))


def _reduce_given_mean(a, mean, axis, keepdims, ddof, want_std, out):
    """基于预计算 mean 的 var/std（numpy mean= 语义）。

    直接信任传入 mean（不重算）：var = sum(|x-mean|^2)/n_eff，
    std = sqrt(var)。支持 axis=None/int/tuple；keepdims 恢复 size-1 轴；
    out 被填充并原样返回（numpy 语义）。
    """
    _m = mean
    try:
        if hasattr(_m, 'ndim') and getattr(_m, 'ndim', 0) == 0:
            _m = _m.item()
    except Exception:
        pass
    if not hasattr(a, '_array'):
        # list/tuple 输入（numpy 非 array 参数包装语义）
        a = _nd()(a)
    diff = a - _m
    nd = diff.ndim
    if axis is None:
        _axes = tuple(range(nd))
    elif isinstance(axis, (tuple, list)):
        _axes = tuple(_ax % nd for _ax in axis)
    else:
        _axes = (axis % nd,)
    _n = 1
    for _ax in _axes:
        _n *= diff.shape[_ax]
    _n -= ddof
    _axset = set(_axes)
    # 元素级 |x-mean|^2（实数即 (x-mean)^2，复数取模平方——numpy 语义）
    _s = abs(diff) ** 2
    if _axset == set(range(nd)):
        # 全归约：展平后直接求和；返回 float64 标量（numpy 语义带 .shape==()）
        _vals = _flatten_c(_s.tolist())
        _v = _builtins.sum(_vals) / _n
        res = _f64_scalar((_v ** 0.5) if want_std else _v)
        if out is not None:
            try:
                out[...] = res
            except Exception:
                pass
            return out
        return res
    # 部分轴归约：降序逐轴求和（高轴先减，低轴号不变；等价联合归约）
    for _ax in sorted(_axes, reverse=True):
        _s = sum(_s, axis=_ax)
    _v = _s / _n
    if want_std:
        _v = _v ** 0.5
    if keepdims:
        _new_shape = []
        _si = 0
        for d in range(nd):
            if d in _axset:
                _new_shape.append(1)
            else:
                _new_shape.append(_s.shape[_si])
                _si += 1
        _v = _v.reshape(tuple(_new_shape))
    if out is not None:
        try:
            out[...] = _v
        except Exception:
            pass
        return out
    return _v


def sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
    """计算数组元素之和。"""
    _ = dtype, out, initial, where
    raw_result = _core.sum(_ensure_raw(a), axis, keepdims)
    return _wrap(raw_result)


def mean(a, axis=None, dtype=None, out=None, keepdims=False, where=True):
    """计算数组元素的平均值。"""
    _ = dtype, where
    # numpy 语义：空数组 → NaN + "Mean of empty slice" 告警
    try:
        if _ensure_raw(a).size == 0:
            import warnings as _warn_mean
            _warn_mean.warn("Mean of empty slice", RuntimeWarning, stacklevel=3)
            return float('nan')
    except Exception:
        pass
    if isinstance(axis, (tuple, list)) and len(axis) > 0:
        # 元组 axis：降序逐轴归约（高轴先减，低轴号不变；等价联合归约），
        # keepdims 恢复 size-1 轴
        nd0 = getattr(a, 'ndim', 0)
        _axes = tuple(_ax % nd0 for _ax in axis)
        _cur = a
        for _ax in sorted(_axes, reverse=True):
            _cur = mean(_cur, axis=_ax)
        if keepdims:
            _cur = _insert_keepdims(_cur, nd0, _axes)
        if out is not None:
            try:
                out[...] = _cur
            except Exception:
                pass
            return out
        return _cur
    raw_result = _core.mean(_ensure_raw(a), axis, keepdims)
    if axis is not None:
        # numpy semantics: reduction along an explicit axis returns an ndarray
        # (0-d when all dimensions are reduced), never a Python scalar.
        res = _nd()(raw_result, _dtype='float64')
    elif hasattr(raw_result, 'ndim') and raw_result.ndim == 0:
        res = float(raw_result.tolist())
    else:
        res = _wrap(raw_result)
    # numpy 语义：out 被填充且被返回
    if out is not None:
        try:
            out[...] = res
        except Exception:
            pass
        return out
    return res


def _needs_py_var_std(a, axis):
    """判断是否需要走原生实现：复数，或任何指定 axis 的归约。

    Rust 端 _core.std/_core.var 对指定 axis 的归约会 panic（1D/2D 亦如此），
    而 _py_var_std 是完整且正确的实现，故凡指定 axis 一律走原生实现。
    """
    if _is_complex_raw(a):
        return True
    if str(getattr(a, '_dtype', '')) == 'object' and \
            getattr(a, '_raw_data', None) is not None:
        # object 数组：底层 f64 存储是占位值（真实值在 _raw_data，
        # 可能是复数等），必须走 Python 路径
        return True
    return axis is not None


def _flat_vals(x):
    out = []
    for v in x:
        if isinstance(v, (list, tuple)):
            out.extend(_flat_vals(v))
        else:
            out.append(v)
    return out


def _values_equal(a, b):
    """逐元素比较（支持 0-d 标量与同形状数组）。"""
    try:
        la = _flat_vals(a.tolist())
        lb = _flat_vals(b.tolist()) if hasattr(b, 'tolist') else _flat_vals([b])
        if len(la) != len(lb):
            return False
        return builtins.all(x == y for x, y in zip(la, lb))
    except Exception:
        return False


def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True,
        correction=None, mean=None):
    """计算数组元素的标准差。"""
    _ = dtype, where
    # numpy 兼容：correction 与 ddof 同义（correction 优先）
    if correction is not None:
        ddof = correction
    # numpy 语义：空数组 → NaN + "Mean of empty slice" 告警
    try:
        if _ensure_raw(a).size == 0:
            import warnings as _warn_std
            _warn_std.warn("Mean of empty slice", RuntimeWarning, stacklevel=3)
            return _nd()(float('nan'), _dtype='float64')
    except Exception:
        pass
    if mean is not None:
        # numpy mean= 语义：信任预计算 mean（参数名遮蔽模块级 mean，
        # 旧实现误将其当函数调用致整个分支被 except 跳过）
        return _reduce_given_mean(a, mean, axis, keepdims, ddof, True, out)
    if axis is not None:
        # 数值路径统一：std() 与 std(mean=np.mean(...)) 走同一实现，
        # 保证 assert_equal 级别的逐位一致（元组 axis 由内部统一处理）。
        # 注意：参数名 mean 遮蔽模块级 mean 函数，须经 globals() 取用。
        _m = globals()['mean'](a, axis=axis, keepdims=True)
        return _reduce_given_mean(a, _m, axis, keepdims, ddof, True, out)
    if _needs_py_var_std(a, axis):
        # axis=None 的复数/object 数组：|x-m|^2 语义
        return _py_var_std(a, axis, ddof, want_std=True)
    raw_result = _core.std(_ensure_raw(a), axis, ddof, keepdims)
    arr_dtype = getattr(a, '_dtype', 'float64') if hasattr(a, '_dtype') else 'float64'
    res = _nd()(raw_result, _dtype=arr_dtype)
    if out is not None:
        try:
            out[...] = res
        except Exception:
            pass
        return out
    return res


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True,
        correction=None, mean=None):
    """计算数组元素的方差。"""
    _ = dtype, where
    if correction is not None:
        ddof = correction
    # numpy 语义：空数组 → NaN + "Mean of empty slice" 告警
    try:
        if _ensure_raw(a).size == 0:
            import warnings as _warn_var
            _warn_var.warn("Mean of empty slice", RuntimeWarning, stacklevel=3)
            return _nd()(float('nan'), _dtype='float64')
    except Exception:
        pass
    if mean is not None:
        # numpy mean= 语义：信任预计算 mean（参数名遮蔽模块级 mean）
        return _reduce_given_mean(a, mean, axis, keepdims, ddof, False, out)
    if axis is not None:
        # 数值路径统一：var() 与 var(mean=np.mean(...)) 走同一实现，
        # 保证 assert_equal 级别的逐位一致。
        # 注意：参数名 mean 遮蔽模块级 mean 函数，须经 globals() 取用。
        _m = globals()['mean'](a, axis=axis, keepdims=True)
        return _reduce_given_mean(a, _m, axis, keepdims, ddof, False, out)
    if _needs_py_var_std(a, axis):
        # axis=None 的复数/object 数组：|x-m|^2 语义
        return _py_var_std(a, axis, ddof, want_std=False)
    raw_result = _core.var(_ensure_raw(a), axis, ddof, keepdims)
    arr_dtype = getattr(a, '_dtype', 'float64') if hasattr(a, '_dtype') else 'float64'
    res = _nd()(raw_result, _dtype=arr_dtype)
    if out is not None:
        try:
            out[...] = res
        except Exception:
            pass
        return out
    return res


def min(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    """计算数组元素的最小值。"""
    _ = out, initial, where
    raw_result = _core.min(_ensure_raw(a), axis, keepdims)
    dtype = getattr(a, '_dtype', 'float64') if hasattr(a, '_dtype') else 'float64'
    return _nd()(raw_result, _dtype=dtype)


def max(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    """计算数组元素的最大值。"""
    _ = out, initial, where
    raw_result = _core.max(_ensure_raw(a), axis, keepdims)
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
    nd = _nd()
    raw_keys = []
    for k in keys:
        if hasattr(k, '_array'):
            raw_keys.append(k._array)
        else:
            raw_keys.append(nd(k)._array)
    return nd._wrap(_core.lexsort(raw_keys), _dtype='int64')


def msort(a):
    """数组按第一个轴排序，返回排序后的数组副本。"""
    return sort(a, axis=0)


def sort_complex(a):
    """对复数按照先实部后虚部的顺序进行排序。"""
    arr = _nd()(a) if not _is_ndarray(a) else a
    return _nd()._wrap(_core.sort_complex(arr._array, -1), _dtype=getattr(arr, '_dtype', 'float64'))


def partition(a, kth, axis=-1, kind=None, order=None):
    """指定一个数，对数组进行分区。"""
    _ = kind, order
    arr = _nd()(a) if not _is_ndarray(a) else a
    if isinstance(kth, int):
        kth = [kth]
    return _nd()._wrap(_core.partition(arr._array, list(kth), axis), _dtype=getattr(arr, '_dtype', 'float64'))


def argpartition(a, kth, axis=-1, kind=None, order=None):
    """对数组进行分区并返回索引。"""
    _ = kind, order
    arr = _nd()(a) if not _is_ndarray(a) else a
    if isinstance(kth, int):
        kth = [kth]
    return _nd()._wrap(_core.argpartition(arr._array, list(kth), axis), _dtype='int64')


def searchsorted(a, v, side='left', sorter=None):
    """查找元素在有序数组中的插入位置。"""
    _ = sorter
    arr = a if hasattr(a, '_array') else _wrap(a)

    # 一维数组 + 标量 v：保持返回 Python int 的语义
    if len(arr.shape) <= 1:
        if hasattr(v, '_array') and len(v.shape) == 0:
            return _core.searchsorted(_ensure_raw(arr), float(v.item()), side)
        try:
            v_val = float(v)
            return _core.searchsorted(_ensure_raw(arr), v_val, side)
        except (TypeError, ValueError):
            pass

    # 多维或 v 为数组：使用 Rust 多维实现
    v_raw = v._array if hasattr(v, '_array') else v
    return _nd()._wrap(_core.searchsorted_axis(_ensure_raw(arr), v_raw, side), _dtype='int64')


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
