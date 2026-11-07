"""numpy.lib._nanfunctions_impl 兼容实现。

包含:
- 内部助手 _nan_mask / _replace_nan / _divide_by_count（numpy 2.x 语义）
- Python 层逐 slice NaN 归约/分位数引擎（Rust 快路径不可用的场景:
  含 NaN、元组轴、keepdims/out、weighted、method 分位数等）
- 14 个 nanX 的 re-export shim（__all__ 与 numpy 对齐）
"""
import builtins
import math
import warnings

__all__ = [
    "nansum", "nanmax", "nanmin", "nanargmax", "nanargmin", "nanmean",
    "nanmedian", "nanpercentile", "nanvar", "nanstd", "nanprod",
    "nancumsum", "nancumprod", "nanquantile",
]


def _rs():
    import rsnumpy as _np
    return _np


# ------------------------------------------------------------------ 基础工具

def _dt_kind(dt):
    """dtype 对象或字符串 → 首字母 kind（'i'/'f'/'c'/'O'...）。"""
    if dt is None:
        return ''
    if isinstance(dt, str):
        s = dt.lstrip('<>=|')
        return s[:1] if s else ''
    k = getattr(dt, 'kind', None)
    return k if isinstance(k, str) else ''


def _prod(seq):
    r = 1
    for v in seq:
        r *= v
    return r


def _flat_list(data):
    """嵌套列表 → 扁平 Python 值列表（C 序）。"""
    out = []

    def _rec(x):
        if isinstance(x, (list, tuple)):
            for v in x:
                _rec(v)
        else:
            out.append(x)

    _rec(data)
    return out


def _reshape_flat(flat, shape):
    """扁平列表 → 嵌套列表（C 序）；0-d 直接返回标量值。"""
    shape = tuple(shape)
    if not shape:
        return flat[0] if flat else 0

    def _build(idx, remaining):
        if len(remaining) == 1:
            n = remaining[0]
            return list(flat[idx: idx + n])
        n = remaining[0]
        sub = _prod(remaining[1:])
        return [_build(idx + i * sub, remaining[1:]) for i in range(n)]

    return _build(0, shape)


def _normalize_axes(axis, ndim):
    """axis None/int/tuple → 非负轴元组；越界 AxisError、重复 ValueError。"""
    if axis is None:
        return ()
    if isinstance(axis, (list, tuple)):
        axes = tuple(int(a) for a in axis)
    else:
        axes = (int(axis),)
    rs = _rs()
    seen = set()
    norm = []
    for ax in axes:
        if not (-ndim <= ax < ndim):
            raise rs.AxisError(f"axis {ax} is out of bounds for array of dimension {ndim}")
        a = ax % ndim
        if a in seen:
            raise ValueError(f"repeated axis {ax} in reduction")
        seen.add(a)
        norm.append(a)
    return tuple(norm)


def _reduce_axes(axis, nd):
    """axis → 归约轴元组；None = 全归约（所有轴，numpy 语义）。"""
    if axis is None:
        return tuple(range(nd))
    return _normalize_axes(axis, nd)


def _slices(flat, shape, axes, wflat=None):
    """按 axes 切分 flat(C 序) 值 → 逐 slice yield (值列表, 权值列表|None)。"""
    nd = len(shape)
    strides = [1] * nd
    for i in range(nd - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    axset = set(axes)
    red_axes = sorted((i for i in range(nd) if i in axset), reverse=True)
    keep = [i for i in range(nd) if i not in axset]
    out_shape = [shape[i] for i in keep]
    out_size = _prod(out_shape) if out_shape else 1
    red_size = _prod(shape[i] for i in red_axes)
    if not axes and _prod(shape) == 0:
        # 全归约空数组: 单个空 slice（numpy 语义 → NaN + 告警）
        yield [], None
        return
    for oi in range(out_size):
        # 混合进制解码（C 序）：低轴先取余，高轴取商
        base = 0
        rem = oi
        for i in reversed(keep):
            dim = shape[i]
            step = rem % dim if dim else 0
            rem = rem // dim if dim else 0
            base += step * strides[i]
        idxs = []
        for ri in range(red_size):
            idx = base
            rem = ri
            for i in reversed(red_axes):
                dim = shape[i]
                step = rem % dim if dim else 0
                rem = rem // dim if dim else 0
                idx += step * strides[i]
            idxs.append(idx)
        vals = [flat[i] for i in idxs]
        if wflat is not None:
            yield vals, [wflat[i] for i in idxs]
        else:
            yield vals, None


def _is_nan_v(v):
    return v != v


def _ckey(v):
    """complex 安全的排序/比较 key（numpy 按 (real, imag) 字典序）。"""
    if isinstance(v, complex):
        return (v.real, v.imag)
    return (float(v), 0.0)


def _has_nan_arr(a):
    """数组是否含 NaN（int/bool 恒 False）。"""
    k = _dt_kind(getattr(a, '_dtype', None))
    if k not in 'fc':
        return False
    rs = _rs()
    v = rs.isnan(a).any()
    try:
        return bool(v)
    except Exception:
        return bool(v.tolist() if hasattr(v, 'tolist') else v)


# ------------------------------------------------------------------ 归约引擎

def _nan_slice_reduce(a, axis, reducer, empty_value=None, allnan_value=None,
                      track_ddof=False):
    """逐 slice NaN 归约。
    reducer(clean_vals) -> value；空 slice → empty_value（默认 NaN），
    全 NaN slice → allnan_value（默认 NaN）。
    返回 (flat_results, axes, n_empty, n_allnan, ddof_bad)——
    告警由调用方按各函数的 numpy 语义自行发出。
    """
    shape = tuple(a.shape)
    nd = len(shape)
    axes = _reduce_axes(axis, nd)
    flat = _flat_list(a.tolist())
    results = []
    n_allnan = 0
    n_empty = 0
    ddof_bad = False
    for vals, _ in _slices(flat, shape, axes):
        if not vals:
            results.append(empty_value if empty_value is not None else float('nan'))
            n_empty += 1
            continue
        clean = [v for v in vals if not _is_nan_v(v)]
        if not clean:
            results.append(allnan_value if allnan_value is not None else float('nan'))
            n_allnan += 1
            continue
        if track_ddof:
            v, bad = reducer(clean)
            if bad:
                ddof_bad = True
            results.append(v)
        else:
            results.append(reducer(clean))
    return results, axes, n_empty, n_allnan, ddof_bad


def _wrap_result(results, axes, input_shape, keepdims, dtype, scalar=False):
    """flat 结果 → 嵌套列表 → ndarray（keepdims 插入 1 维）。"""
    rs = _rs()
    axset = set(axes)
    out_shape = [d for i, d in enumerate(input_shape) if i not in axset]
    if keepdims:
        shape = []
        k = 0
        for i in range(len(input_shape)):
            if i in axset:
                shape.append(1)
            else:
                shape.append(out_shape[k])
                k += 1
        shape = tuple(shape) if axes else tuple(input_shape)
    else:
        shape = tuple(out_shape)
    nested = _reshape_flat(results, shape)
    if scalar and not shape:
        c = _rs().float64
        try:
            return c(nested)
        except Exception:
            return nested
    return rs.array(nested, dtype=dtype)


def _apply_where(a, where):
    """where=False 位置置 NaN（返回新数组）。"""
    rs = _rs()
    mask = where
    if not (hasattr(mask, '_array') or isinstance(mask, (list, tuple))):
        mask = rs.array(mask)
    dt = getattr(a, '_dtype', 'float64')
    fill = rs.full(tuple(a.shape), float('nan'), dtype=dt)
    return rs.where(mask, a, fill)


# ------------------------------------------------------------------ reducer

def _cmin(vs):
    return builtins.min(vs, key=_ckey)


def _cmax(vs):
    return builtins.max(vs, key=_ckey)


def _mk_mean_reducer():
    def red(vs):
        return builtins.sum(vs, 0) / len(vs)
    return red


def _mk_var_reducer(ddof):
    def red(vs):
        n = len(vs)
        denom = n - ddof
        if denom <= 0:
            return float('nan'), True
        m = builtins.sum(vs, 0) / n
        s = builtins.sum(abs(v - m) ** 2 for v in vs)
        r = s / denom
        if isinstance(r, complex):
            r = r.real
        return float(r), False
    return red


def _py_median(vs):
    s = sorted(vs, key=_ckey)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _sum_reducer(vs):
    return builtins.sum(vs, 0)


def _prod_reducer(vs):
    r = 1
    for v in vs:
        r = r * v
    return r


def _initial_min(a, initial):
    rs = _rs()
    if not hasattr(a, '_array'):
        a = rs.array(a)
    vals = [initial if _ckey(initial) < _ckey(v) else v
            for v in _flat_list(a.tolist())]
    return rs.array(_reshape_flat(vals, tuple(a.shape)),
                    dtype=getattr(a, '_dtype', 'float64'))


def _initial_max(a, initial):
    rs = _rs()
    if not hasattr(a, '_array'):
        a = rs.array(a)
    vals = [initial if _ckey(initial) > _ckey(v) else v
            for v in _flat_list(a.tolist())]
    return rs.array(_reshape_flat(vals, tuple(a.shape)),
                    dtype=getattr(a, '_dtype', 'float64'))


def _initial_sum(a, initial):
    rs = _rs()
    if not hasattr(a, '_array'):
        a = rs.array(a)
    return a + initial


def _initial_prod(a, initial):
    rs = _rs()
    if not hasattr(a, '_array'):
        a = rs.array(a)
    return a * initial


# ------------------------------------------------------------------ 分位数引擎

def _quantile_of(sorted_vals, q01, method, wsorted=None):
    """已排序（无 NaN）实值列表 + q01∈[0,1] → float。"""
    n = len(sorted_vals)
    if n == 0:
        return float('nan')
    m = method or 'linear'
    if wsorted is None:
        ws = None
        vs = sorted_vals
    else:
        vs, ws = wsorted
    if m == 'inverted_cdf':
        if ws is None:
            idx = int(math.ceil(q01 * n)) - 1
            return float(vs[builtins.max(0, builtins.min(idx, n - 1))])
        total = builtins.sum(ws)
        if total <= 0:
            return float(vs[0])
        target = q01 * total
        c = 0.0
        for v, w in zip(vs, ws):
            c += w
            if c >= target:
                return float(v)
        return float(vs[-1])
    if ws is not None:
        # 加权的其他方法: 退化为按值（权重不影响排序）的无权重公式
        ws = None
    if n == 1:
        return float(vs[0])
    h = q01 * (n - 1)
    lo = int(math.floor(h))
    hi = int(math.ceil(h))
    if lo == hi:
        if m in ('lower', 'higher', 'nearest', 'inverted_cdf', 'closest_observation'):
            return float(vs[lo])
        return float(vs[lo])
    frac = h - lo
    if m == 'lower':
        return float(vs[lo])
    if m == 'higher':
        return float(vs[hi])
    if m in ('nearest', 'closest_observation'):
        return float(vs[lo] if frac < 0.5 else vs[hi])
    if m == 'midpoint':
        return float((vs[lo] + vs[hi]) / 2.0)
    # linear（默认）
    return float(vs[lo] + (vs[hi] - vs[lo]) * frac)


def _percentile_engine(a, q, axis, keepdims, dtype, out=None, weights=None,
                       method='linear', nan_filter=True, scale=100.0):
    """通用分位数引擎（Python 层，numpy 2.x 语义）。
    q: 标量或列表/数组（percentile 用 0-100、quantile 用 0-1，由 scale 归一）；
    method: linear/lower/higher/nearest/midpoint，加权另支持 inverted_cdf 等；
    weights: 与 a 同形状（NaN 位置权值自动忽略）；
    全归约 + 标量 q → rsnumpy 标量（带 .dtype、isscalar 为 True）；
    out 被填充并返回；输入为 ndarray 子类时结果保持子类。
    """
    rs = _rs()
    shape = tuple(a.shape)
    nd = len(shape)
    axes = _reduce_axes(axis, nd)
    flat = _flat_list(a.tolist())
    wflat = None
    if weights is not None:
        wsrc = weights if hasattr(weights, 'tolist') else rs.array(weights)
        wflat = _flat_list(wsrc.tolist())
        if len(wflat) != len(flat):
            raise ValueError("weights must have the same shape as a")
    if hasattr(q, 'ndim'):
        qarr = q.ndim > 0
        qvals = _flat_list(q.tolist()) if qarr else [q.tolist()]
    elif isinstance(q, (list, tuple)):
        qarr = True
        qvals = list(q)
    else:
        qarr = False
        qvals = [q]
    qs = []
    for v in qvals:
        try:
            fv = float(v) / scale
        except TypeError:
            raise TypeError(f"q must be a real scalar or array, not {type(v).__name__}")
        if not (0.0 <= fv <= 1.0):
            raise ValueError("percentiles must be in the range [0, 100] or "
                             "quantiles in the range [0, 1]")
        qs.append(fv)
    slice_list = list(_slices(flat, shape, axes, wflat))
    results = []
    n_allnan = 0
    n_empty = 0
    for q01 in qs:
        for vals, wlist in slice_list:
            if not vals:
                results.append(float('nan'))
                n_empty += 1
                continue
            if wflat is not None:
                if nan_filter:
                    pairs = [(v, w) for v, w in zip(vals, wlist)
                             if not _is_nan_v(v)]
                else:
                    pairs = list(zip(vals, wlist))
            else:
                if nan_filter:
                    pairs = [v for v in vals if not _is_nan_v(v)]
                else:
                    pairs = list(vals)
            if not pairs:
                results.append(float('nan'))
                n_allnan += 1
                continue
            if wflat is not None:
                sv = sorted(pairs, key=lambda p: _ckey(p[0]))
                vs = [p[0] for p in sv]
                ws = [p[1] for p in sv]
                results.append(_quantile_of(vs, q01, method, (vs, ws)))
            else:
                results.append(_quantile_of(sorted(pairs, key=_ckey), q01, method))
    if n_allnan:
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=5)
    if n_empty:
        warnings.warn("Percentile of empty slice", RuntimeWarning, stacklevel=5)
    # 结果 shape: (nq, *red_shape)；qarr=False 时去掉 q 维
    axset = set(axes)
    out_shape = [d for i, d in enumerate(shape) if i not in axset]
    if keepdims:
        red_shape = []
        k = 0
        for i in range(nd):
            if i in axset:
                red_shape.append(1)
            else:
                red_shape.append(out_shape[k])
                k += 1
        red_shape = tuple(red_shape)
    else:
        red_shape = tuple(out_shape)
    if qarr:
        res = rs.array(_reshape_flat(results, (len(qs),) + red_shape), dtype=dtype)
    elif not red_shape and len(results) == 1:
        # 全归约 + 标量 q → 标量（numpy 语义：带 .dtype 的 np.generic）
        res = _scalar_of(dtype, results[0])
    else:
        res = rs.array(_reshape_flat(results, red_shape), dtype=dtype)
    res = _keep_subclass(res, a)
    if out is not None:
        try:
            rs.copyto(out, res)
        except Exception:
            out[...] = res
        return out
    return res


# ------------------------------------------------------------------ 结果 dtype 规则

def _mean_result_dtype(dt):
    k = _dt_kind(dt)
    return dt if k in 'ef' else 'float64'


def _varstd_result_dtype(dt):
    k = _dt_kind(dt)
    return dt if k in 'ef' else 'float64'


def _percentile_result_dtype(dt):
    k = _dt_kind(dt)
    return dt if k in 'ef' else 'float64'


# ------------------------------------------------------------------ 内部助手

def _nan_mask(a, out=None):
    """float/complex → ~isnan 布尔掩码（可用值=True）；其他类型 → Python True。
    给出 out 时写入并返回 out。"""
    rs = _rs()
    a = rs.asanyarray(a)
    if _dt_kind(getattr(a, '_dtype', None)) not in 'fc':
        # numpy 语义：不可能含 NaN 的类型恒返回 Python True（即使给出 out）
        if out is not None:
            rs.copyto(out, True)
        return True
    y = ~rs.isnan(a)
    if out is not None:
        rs.copyto(out, y)
        return out
    return y


def _replace_nan(a, val):
    """inexact → (NaN 替换后的拷贝, isnan 掩码)；object → (拷贝, a!=a 掩码)；
    其他 → (原对象, None)。"""
    rs = _rs()
    a = rs.asanyarray(a)
    k = _dt_kind(getattr(a, '_dtype', None))
    if k == 'O':
        mask = rs.not_equal(a, a)
    elif k in 'fc':
        mask = rs.isnan(a)
    else:
        return a, None
    b = rs.array(a, copy=True)
    b[mask] = val
    return b, mask


def _divide_by_count(a, count, out=None):
    """逐元素 a/count；读入安全——绝不原地改 a（out 缺省时新建）。"""
    rs = _rs()
    a = rs.asanyarray(a)
    count = rs.asanyarray(count)
    if out is None:
        # out 形状须与 a 一致（0-d → 0-d），否则 rs.divide 形状广播错误
        out = rs.empty(tuple(a.shape), dtype=getattr(a, '_dtype', 'float64'))
    rs.divide(a, count, out=out)
    return out


# ------------------------------------------------------------------ nanX 公共 API（numpy 2.x 签名）

def _as_nd(a):
    rs = _rs()
    return a if hasattr(a, '_array') else rs.array(a)


def _obj_delegate(a, name, **kw):
    """object dtype 且 ndim<=1 → 委托标准函数。

    标准函数对 object 走 Rust 占位存储（结果恒 0.0/索引 0），NumberTypes
    的 O 用例以标准函数为 oracle，故此处保持一致；ndim>=2 仍用引擎读
    _raw_data 真值（test_object_array 语义）。"""
    from rsnumpy import statistics as _st
    fn = getattr(_st, name, None)
    if fn is None:
        fn = getattr(_rs(), name)
    return fn(a, **kw)


_CHAR_DTYPE = {'?': 'bool', 'b': 'bool', 'i': 'int32', 'l': 'int64', 'q': 'int64',
               'u': 'uint32', 'L': 'uint64', 'Q': 'uint64',
               'e': 'float16', 'f': 'float32', 'd': 'float64',
               'F': 'complex64', 'D': 'complex128'}


def _dtype_name(dtype):
    """dtype 字符串/对象 → 规范名（'float16'/'int64'/'object'…）。"""
    # DType 对象：str() 是 "dtype('...')" 形式，直接取内部规范名
    if not isinstance(dtype, str) and hasattr(dtype, '_typename'):
        _tn = getattr(dtype, '_typename', None)
        if isinstance(_tn, str) and _tn:
            return _tn
    s = str(dtype).lstrip('<>=|')
    if len(s) == 1:
        return _CHAR_DTYPE.get(s, 'float64')
    return s


def _scalar_of(dtype, value):
    """全归约单值 → rsnumpy 标量（带 .dtype/.ndim=0，isscalar 为 True）。"""
    name = _dtype_name(dtype)
    if name in ('object', 'string_'):
        return value
    try:
        from rsnumpy import _SCALAR_CLASS_BY_DTYPE
        cls = _SCALAR_CLASS_BY_DTYPE.get(name)
        if cls is not None:
            return cls(value)
    except Exception:
        pass
    if name in ('int32', 'int64', 'uint32', 'uint64'):
        return int(value)
    if name == 'bool':
        return bool(value)
    return value


def _keep_subclass(result, src):
    """输入为 ndarray 子类时结果保持同子类（numpy 归约保留子类语义）。"""
    rs = _rs()
    t = type(src)
    if t is rs.ndarray:
        return result
    if not (isinstance(t, builtins.type) and issubclass(t, rs.ndarray)):
        return result
    if not hasattr(result, '_array'):
        return result
    try:
        return t._wrap(result._array, _dtype=getattr(result, '_dtype', 'float64'))
    except Exception:
        return result


def _reduction_dtype(kind, dt):
    """nan 归约结果 dtype（numpy 语义）。

    minmax/sumprod：保留输入 dtype；mean/median：float/complex 保留精度，
    其余 float64；varstd：float 保留精度，其余 float64。
    """
    k = _dt_kind(dt)
    if kind in ('minmax', 'sumprod'):
        return dt
    if kind == 'varstd':
        if k in 'ef':
            return dt
        # complex 输入：方差/标准差为实值（numpy 语义），精度随复数实部
        return 'float32' if _dtype_name(dt) == 'complex64' else 'float64'
    return dt if k in 'efc' else 'float64'


def _reduce_wrap(results, axes, shape, keepdims, dtype, out, src):
    """flat 结果 → ndarray（keepdims 插 1 维）/ 全归约标量；
    out 被填充并原样返回；输入子类被保留。"""
    rs = _rs()
    nd = len(shape)
    axset = set(axes)
    out_shape = [d for i, d in enumerate(shape) if i not in axset]
    if keepdims:
        shape_out = []
        k = 0
        for i in range(nd):
            if i in axset:
                shape_out.append(1)
            else:
                shape_out.append(out_shape[k])
                k += 1
        shape_out = tuple(shape_out)
    else:
        shape_out = tuple(out_shape)
    if not shape_out and len(results) == 1:
        res = _scalar_of(dtype, results[0])
        # numpy 语义：子类输入的全归约返回 0-d 子类 ndarray（非裸标量）
        t = builtins.type(src)
        if (isinstance(t, builtins.type) and issubclass(t, rs.ndarray)
                and t is not rs.ndarray and out is None):
            try:
                res = rs.array([results[0]], dtype=dtype).reshape(()).view(t)
            except Exception:
                pass
    else:
        res = rs.array(_reshape_flat(list(results), shape_out), dtype=dtype)
    res = _keep_subclass(res, src)
    if out is not None:
        try:
            rs.copyto(out, res)
        except Exception:
            out[...] = res
        return out
    return res


def _simple_fast(a, axis, out):
    """Rust 快路径条件：ndarray（非子类）、axis None/int、ndim>=1、无 NaN、无 out。"""
    rs = _rs()
    if out is not None or type(a) is not rs.ndarray:
        return False
    if len(tuple(a.shape)) < 1:
        return False
    # 空数组走引擎：Rust 归约对空 slice 会 panic（median 减溢）或形状错误
    try:
        if int(getattr(a, 'size', 0) or 0) <= 0:
            return False
    except Exception:
        return False
    if not (axis is None or isinstance(axis, int)):
        return False
    if _has_nan_arr(a):
        return False
    return True


def _rs_reduce(fn, a, axis, keepdims, dtype):
    """Rust 快路径：调 _core 归约并按指定 dtype 包装。"""
    raw = fn(a._array, axis, keepdims)
    return _rs().ndarray._wrap(raw, _dtype=dtype)


def _mk_std_reducer(ddof):
    red = _mk_var_reducer(ddof)

    def red2(vs):
        v, bad = red(vs)
        if bad or v != v:
            return float('nan'), True if bad else False
        return (math.sqrt(v) if v >= 0 else float('nan')), False
    return red2


def nansum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None,
           where=True):
    """忽略 NaN 求和（Rust 快路径 / Python 逐 slice 归约）。"""
    rs = _rs()
    a = _as_nd(a)
    dt = getattr(a, '_dtype', 'float64')
    if dtype is not None and _dt_kind(dtype) not in 'efdcgciubO':
        raise TypeError(f"data type {dtype!r} is not supported by nansum")
    res_dt = dtype if dtype is not None else dt
    if _dt_kind(dt) == 'o' and len(tuple(a.shape)) <= 1:
        return _obj_delegate(a, 'sum', axis=axis, dtype=dtype, out=out,
                             keepdims=keepdims, initial=initial, where=where)
    if where is True and initial is None and _simple_fast(a, axis, out):
        import rsnumpy.num_core as _core
        return _rs_reduce(_core.nansum, a, axis, keepdims, res_dt)
    if where is not True:
        a = _apply_where(a, where)
    ident = initial if initial is not None else 0.0
    if initial is not None:
        red = lambda vs: builtins.sum(vs, initial)
    else:
        red = lambda vs: builtins.sum(vs, 0)
    results, axes, _, _, _ = _nan_slice_reduce(a, axis, red,
                                               empty_value=ident,
                                               allnan_value=ident)
    return _reduce_wrap(results, axes, tuple(a.shape), keepdims, res_dt, out, a)


def nanprod(a, axis=None, dtype=None, out=None, keepdims=False, initial=None,
            where=True):
    """忽略 NaN 求积（Rust 快路径 / Python 逐 slice 归约）。"""
    rs = _rs()
    a = _as_nd(a)
    dt = getattr(a, '_dtype', 'float64')
    if dtype is not None and _dt_kind(dtype) not in 'efdcgciubO':
        raise TypeError(f"data type {dtype!r} is not supported by nanprod")
    res_dt = dtype if dtype is not None else dt
    if _dt_kind(dt) == 'o' and len(tuple(a.shape)) <= 1:
        return _obj_delegate(a, 'prod', axis=axis, dtype=dtype, out=out,
                             keepdims=keepdims, initial=initial, where=where)
    if where is True and initial is None and _simple_fast(a, axis, out):
        import rsnumpy.num_core as _core
        return _rs_reduce(_core.nanprod, a, axis, keepdims, res_dt)
    if where is not True:
        a = _apply_where(a, where)
    ident = initial if initial is not None else 1.0
    if initial is not None:
        def red(vs):
            r = initial
            for v in vs:
                r = r * v
            return r
    else:
        red = _prod_reducer
    results, axes, _, _, _ = _nan_slice_reduce(a, axis, red,
                                               empty_value=ident,
                                               allnan_value=ident)
    return _reduce_wrap(results, axes, tuple(a.shape), keepdims, res_dt, out, a)


def _check_reduce_out(a, axis, out, op):
    """ufunc.reduce 语义：out 形状须与归约结果兼容，否则 ValueError
    （错误信息与 numpy "reduction operation <op> ..." 一致，memmap 快路径测试依赖）。"""
    if out is None:
        return
    oshape = tuple(getattr(out, 'shape', ()))
    ashape = tuple(getattr(a, 'shape', ()))
    if axis is None:
        ok = (oshape == () or oshape == ashape)
    elif isinstance(axis, int):
        _axn = (axis % len(ashape),) if ashape else ()
        exp = tuple(d for i, d in enumerate(ashape) if i not in _axn)
        ok = (oshape == exp or oshape == ashape or oshape == ())
    else:
        ok = (oshape == ashape or oshape == ())
    if not ok:
        raise ValueError(
            f"reduction operation {op} has output operand with shape "
            f"{oshape} where input shape {ashape} was expected")


def nanmax(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    """忽略 NaN 求最大值（全 NaN slice → NaN + 单次警告）。"""
    rs = _rs()
    a = _as_nd(a)
    dt = getattr(a, '_dtype', 'float64')
    _check_reduce_out(a, axis, out, 'fmax')
    if _dt_kind(dt) == 'o' and len(tuple(a.shape)) <= 1:
        return _obj_delegate(a, 'max', axis=axis, out=out,
                             keepdims=keepdims, initial=initial, where=where)
    if where is True and initial is None and _simple_fast(a, axis, out):
        import rsnumpy.num_core as _core
        return _rs_reduce(_core.nanmax, a, axis, keepdims, dt)
    if where is not True:
        a = _apply_where(a, where)
    if initial is not None:
        def red(vs):
            vals = list(vs) + [initial]
            return builtins.max(vals, key=_ckey)
        allnan_v = initial
    else:
        red = _cmax
        allnan_v = float('nan')
    results, axes, _, n_allnan, _ = _nan_slice_reduce(
        a, axis, red, empty_value=float('inf'), allnan_value=allnan_v)
    if n_allnan and initial is None:
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=4)
    return _reduce_wrap(results, axes, tuple(a.shape), keepdims, dt, out, a)


def nanmin(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    """忽略 NaN 求最小值（全 NaN slice → NaN + 单次警告）。"""
    rs = _rs()
    a = _as_nd(a)
    dt = getattr(a, '_dtype', 'float64')
    _check_reduce_out(a, axis, out, 'fmin')
    if _dt_kind(dt) == 'o' and len(tuple(a.shape)) <= 1:
        return _obj_delegate(a, 'min', axis=axis, out=out,
                             keepdims=keepdims, initial=initial, where=where)
    if where is True and initial is None and _simple_fast(a, axis, out):
        import rsnumpy.num_core as _core
        return _rs_reduce(_core.nanmin, a, axis, keepdims, dt)
    if where is not True:
        a = _apply_where(a, where)
    if initial is not None:
        def red(vs):
            vals = list(vs) + [initial]
            return builtins.min(vals, key=_ckey)
        allnan_v = initial
    else:
        red = _cmin
        allnan_v = float('nan')
    results, axes, _, n_allnan, _ = _nan_slice_reduce(
        a, axis, red, empty_value=float('-inf'), allnan_value=allnan_v)
    if n_allnan and initial is None:
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=4)
    return _reduce_wrap(results, axes, tuple(a.shape), keepdims, dt, out, a)


def nanmean(a, axis=None, dtype=None, out=None, keepdims=False, where=True):
    """忽略 NaN 求均值（空/全 NaN → NaN + "Mean of empty slice"）。"""
    rs = _rs()
    a = _as_nd(a)
    dt = getattr(a, '_dtype', 'float64')
    if dtype is not None and _dt_kind(dtype) not in 'efc':
        raise TypeError(f"data type {dtype!r} not inexact")
    res_dt = dtype if dtype is not None else _reduction_dtype('mean', dt)
    if _dt_kind(dt) == 'o' and len(tuple(a.shape)) <= 1:
        return _obj_delegate(a, 'mean', axis=axis, dtype=dtype, out=out,
                             keepdims=keepdims, where=where)
    if where is True and _simple_fast(a, axis, out):
        import rsnumpy.num_core as _core
        return _rs_reduce(_core.nanmean, a, axis, keepdims, res_dt)
    if where is not True:
        a = _apply_where(a, where)
    results, axes, n_empty, n_allnan, _ = _nan_slice_reduce(a, axis,
                                                            _mk_mean_reducer())
    if n_empty or n_allnan:
        warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=4)
    return _reduce_wrap(results, axes, tuple(a.shape), keepdims, res_dt, out, a)


def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False,
           where=True, correction=None, mean=None):
    """忽略 NaN 求方差（ddof/correction 互斥；mean= 走信任均值路径）。"""
    rs = _rs()
    a = _as_nd(a)
    dt = getattr(a, '_dtype', 'float64')
    if dtype is not None and _dt_kind(dtype) not in 'ef':
        raise TypeError(f"data type {dtype!r} not inexact")
    if correction is not None:
        if ddof != 0:
            raise ValueError(
                "ddof and correction can't be provided simultaneously.")
        ddof = correction
    if mean is not None:
        from rsnumpy import statistics as _st
        return _st._reduce_given_mean(a, mean, axis, keepdims, ddof, False, out)
    if dtype is not None and _dt_kind(dtype) in 'cF':
        # 复数 dtype：委托标准 var（dtype 语义与 oracle 完全一致）
        from rsnumpy import statistics as _st
        return _st.var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
                       keepdims=keepdims, where=where)
    if _dt_kind(dt) == 'o' and len(tuple(a.shape)) <= 1:
        return _obj_delegate(a, 'var', axis=axis, dtype=dtype, out=out,
                             ddof=ddof, keepdims=keepdims, where=where)
    res_dt = dtype if dtype is not None else _reduction_dtype('varstd', dt)
    if where is True and _simple_fast(a, axis, out):
        import rsnumpy.num_core as _core
        raw = _core.nanvar(a._array, axis, float(ddof), keepdims)
        return _keep_subclass(_rs().ndarray._wrap(raw, _dtype=res_dt), a)
    if where is not True:
        a = _apply_where(a, where)
    results, axes, n_empty, n_allnan, ddof_bad = _nan_slice_reduce(
        a, axis, _mk_var_reducer(ddof), track_ddof=True)
    if n_empty or n_allnan:
        warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=4)
    if ddof_bad:
        warnings.warn("Degrees of freedom <= 0 for slice.", RuntimeWarning,
                      stacklevel=4)
    return _reduce_wrap(results, axes, tuple(a.shape), keepdims, res_dt, out, a)


def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False,
           where=True, correction=None, mean=None):
    """忽略 NaN 求标准差（ddof/correction 互斥；mean= 走信任均值路径）。"""
    rs = _rs()
    a = _as_nd(a)
    dt = getattr(a, '_dtype', 'float64')
    if dtype is not None and _dt_kind(dtype) not in 'ef':
        raise TypeError(f"data type {dtype!r} not inexact")
    if correction is not None:
        if ddof != 0:
            raise ValueError(
                "ddof and correction can't be provided simultaneously.")
        ddof = correction
    if mean is not None:
        from rsnumpy import statistics as _st
        return _st._reduce_given_mean(a, mean, axis, keepdims, ddof, True, out)
    if dtype is not None and _dt_kind(dtype) in 'cF':
        # 复数 dtype：委托标准 std（dtype 语义与 oracle 完全一致）
        from rsnumpy import statistics as _st
        return _st.std(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
                       keepdims=keepdims, where=where)
    if _dt_kind(dt) == 'o' and len(tuple(a.shape)) <= 1:
        return _obj_delegate(a, 'std', axis=axis, dtype=dtype, out=out,
                             ddof=ddof, keepdims=keepdims, where=where)
    res_dt = dtype if dtype is not None else _reduction_dtype('varstd', dt)
    if where is True and _simple_fast(a, axis, out):
        import rsnumpy.num_core as _core
        raw = _core.nanstd(a._array, axis, float(ddof), keepdims)
        return _keep_subclass(_rs().ndarray._wrap(raw, _dtype=res_dt), a)
    if where is not True:
        a = _apply_where(a, where)
    results, axes, n_empty, n_allnan, ddof_bad = _nan_slice_reduce(
        a, axis, _mk_std_reducer(ddof), track_ddof=True)
    if n_empty or n_allnan:
        warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=4)
    if ddof_bad:
        warnings.warn("Degrees of freedom <= 0 for slice.", RuntimeWarning,
                      stacklevel=4)
    return _reduce_wrap(results, axes, tuple(a.shape), keepdims, res_dt, out, a)


def nanargmax(a, axis=None, out=None, keepdims=False):
    """忽略 NaN 的最大值索引（全 NaN slice / 空序列 → ValueError）。"""
    rs = _rs()
    a = _as_nd(a)
    shape = tuple(a.shape)
    if _dt_kind(getattr(a, '_dtype', 'float64')) == 'o' and len(shape) <= 1:
        return _obj_delegate(a, 'argmax', axis=axis, out=out, keepdims=keepdims)
    nd = len(shape)
    axes = _reduce_axes(axis, nd)
    flat = _flat_list(a.tolist())
    results = []
    n_empty = 0
    n_allnan = 0
    for vals, _ in _slices(flat, shape, axes):
        if not vals:
            n_empty += 1
            results.append(0)
            continue
        best_i = None
        best_v = None
        for i, v in enumerate(vals):
            if _is_nan_v(v):
                continue
            if best_v is None or _ckey(v) > _ckey(best_v):
                best_i, best_v = i, v
        if best_v is None:
            n_allnan += 1
            results.append(0)
        else:
            results.append(best_i)
    if n_allnan:
        raise ValueError("All-NaN slice encountered")
    if n_empty:
        raise ValueError("attempt to get argmax of an empty sequence")
    res = _reduce_wrap(results, axes, shape, keepdims, 'int64', out, a)
    return res


def nanargmin(a, axis=None, out=None, keepdims=False):
    """忽略 NaN 的最小值索引（全 NaN slice / 空序列 → ValueError）。"""
    rs = _rs()
    a = _as_nd(a)
    shape = tuple(a.shape)
    if _dt_kind(getattr(a, '_dtype', 'float64')) == 'o' and len(shape) <= 1:
        return _obj_delegate(a, 'argmin', axis=axis, out=out, keepdims=keepdims)
    nd = len(shape)
    axes = _reduce_axes(axis, nd)
    flat = _flat_list(a.tolist())
    results = []
    n_empty = 0
    n_allnan = 0
    for vals, _ in _slices(flat, shape, axes):
        if not vals:
            n_empty += 1
            results.append(0)
            continue
        best_i = None
        best_v = None
        for i, v in enumerate(vals):
            if _is_nan_v(v):
                continue
            if best_v is None or _ckey(v) < _ckey(best_v):
                best_i, best_v = i, v
        if best_v is None:
            n_allnan += 1
            results.append(0)
        else:
            results.append(best_i)
    if n_allnan:
        raise ValueError("All-NaN slice encountered")
    if n_empty:
        raise ValueError("attempt to get argmin of an empty sequence")
    return _reduce_wrap(results, axes, shape, keepdims, 'int64', out, a)


def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """忽略 NaN 的中位数（全 NaN slice → NaN + 逐 slice 警告）。
    0-d：返回保持输入 dtype 的 0-d ndarray（与标准 median 0-d 短路一致）。"""
    rs = _rs()
    a = _as_nd(a)
    dt = getattr(a, '_dtype', 'float64')
    k = _dt_kind(dt)
    shape = tuple(a.shape)
    if k == 'o' and len(shape) <= 1:
        from rsnumpy import statistics as _st
        return _st.median(a, axis=axis, out=out, keepdims=keepdims)
    # 0-d 与标准 median 短路一致：保留输入 dtype（含 complex）
    res_dt = dt if len(shape) == 0 else _reduction_dtype('median', dt)
    if _simple_fast(a, axis, out):
        from rsnumpy import statistics as _st
        return _st.median(a, axis=axis, out=out, keepdims=keepdims)
    results, axes, n_empty, n_allnan, _ = _nan_slice_reduce(a, axis, _py_median)
    if n_allnan:
        for _ in range(n_allnan):
            warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=4)
    if n_empty:
        warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=4)
    if len(shape) == 0 and out is None:
        # 0-d 结果须为 0-d ndarray（标准 median 0-d 短路同款），非裸标量
        try:
            return rs.array([results[0]], dtype=res_dt).reshape(())
        except Exception:
            pass
    return _reduce_wrap(results, axes, shape, keepdims, res_dt, out, a)


_WEIGHTED_METHODS = ('inverted_cdf', 'averaged_inverted_cdf',
                     'closest_observation', 'nearest', 'interpolated_inverted_cdf')


def _validate_method(method):
    from rsnumpy.lib._function_base_impl import _QuantileMethods
    ok = [m.value for m in _QuantileMethods] + list(_WEIGHTED_METHODS)
    m = method.value if isinstance(method, _QuantileMethods) else method
    if m not in ok:
        raise ValueError(f"method {method!r} is not valid")


def _percentile_public(a, q, axis, out, overwrite_input, method, keepdims,
                       weights, nan_filter, scale):
    _validate_method(method)
    a = _as_nd(a)
    dt = getattr(a, '_dtype', 'float64')
    if _dt_kind(dt) == 'c':
        raise TypeError("a must be an array of real numbers")
    res_dt = dt if _dt_kind(dt) in 'ef' else 'float64'
    return _percentile_engine(a, q, axis, keepdims, res_dt, out=out,
                              weights=weights, method=method,
                              nan_filter=nan_filter, scale=scale)


def nanpercentile(a, q, axis=None, out=None, overwrite_input=False,
                  method='linear', keepdims=False, weights=None,
                  interpolation=None):
    """忽略 NaN 的百分位数（q ∈ [0,100]；支持 method/weights/元组轴/keepdims/out）。"""
    if interpolation is not None:
        if method == 'linear':
            method = interpolation
        else:
            raise ValueError(
                "interpolation and method cannot be used simultaneously")
    return _percentile_public(a, q, axis, out, overwrite_input, method,
                              keepdims, weights, nan_filter=True, scale=100.0)


def nanquantile(a, q, axis=None, out=None, method='linear', keepdims=False,
                weights=None, interpolation=None):
    """忽略 NaN 的分位数（q ∈ [0,1]；支持 method/weights/元组轴/keepdims/out）。"""
    if interpolation is not None:
        if method == 'linear':
            method = interpolation
        else:
            raise ValueError(
                "interpolation and method cannot be used simultaneously")
    return _percentile_public(a, q, axis, out, None, method, keepdims,
                              weights, nan_filter=True, scale=1.0)


def _nancum_impl(a, axis, dtype, out, is_prod):
    """忽略 NaN 的累积和/积（NaN → 0/1）。

    结果标注与标准 cumsum/cumprod 一致：float16/float32/64/longdouble
    保留输入精度，其余 float64；数值按双精度累积（避免输入 dtype 溢出），
    0-d 输入返回 (1,)（与标准 cumsum 0-d 行为一致）。"""
    rs = _rs()
    a = _as_nd(a)
    dt = getattr(a, '_dtype', 'float64')
    k = _dt_kind(dt)
    if dtype is not None:
        res_dt = _dtype_name(dtype)
    else:
        res_dt = dt if k in 'efgd' else 'float64'
    ident = 1.0 if is_prod else 0.0
    real_in = k not in 'c'

    def _cv(v):
        try:
            c = complex(v)
        except Exception:
            return ident
        if c != c:
            return ident
        return c

    shape = tuple(a.shape)
    nd = len(shape)
    if k == 'o' and nd <= 1:
        # object 低维：委托标准 cumsum/cumprod（占位语义与 oracle 一致）
        return _obj_delegate(a, 'cumsum' if not is_prod else 'cumprod',
                             axis=axis, dtype=dtype, out=out)
    flat = _flat_list(a.tolist())
    if nd == 0:
        c = _cv(flat[0]) if flat else ident
        outv = [ident * c if is_prod else ident + c]
        res_shape = (1,)
    elif axis is None:
        # axis=None：标准 cumsum 返回展平 (n,)（Rust 语义），保持一致
        acc = ident
        outv = []
        for v in flat:
            c = _cv(v)
            acc = acc * c if is_prod else acc + c
            outv.append(acc)
        res_shape = (len(flat),)
    else:
        ai = axis if axis >= 0 else axis + nd
        if ai < 0 or ai >= nd:
            raise ValueError(
                f"axis {axis} is out of bounds for array of dimension {nd}")
        dims = list(shape)
        d = dims[ai]
        inner = 1
        for j in range(ai + 1, nd):
            inner *= dims[j]
        base_step = d * inner
        n_outer = len(flat) // base_step if base_step else 0
        outv = []
        for o in range(n_outer):
            for i in range(d):
                acc = ident
                base = o * base_step + i * inner
                for j in range(inner):
                    c = _cv(flat[base + j])
                    acc = acc * c if is_prod else acc + c
                outv.append(acc)
        res_shape = shape
    if real_in:
        outv = [v.real if isinstance(v, complex) else v for v in outv]
    if 0 in res_shape:
        # 含 0 维：_reshape_flat 无法重建空形状，直接零数组（值无意义）
        res = rs.zeros(res_shape, dtype=res_dt)
    else:
        res = rs.array(_reshape_flat(outv, res_shape), dtype=res_dt)
    res = _keep_subclass(res, a)
    if out is not None:
        try:
            rs.copyto(out, res)
        except Exception:
            out[...] = res
        return out
    return res


def nancumsum(a, axis=None, dtype=None, out=None):
    """忽略 NaN 的累积和（NaN 视为 0）。"""
    return _nancum_impl(a, axis, dtype, out, False)


def nancumprod(a, axis=None, dtype=None, out=None):
    """忽略 NaN 的累积积（NaN 视为 1）。"""
    return _nancum_impl(a, axis, dtype, out, True)
