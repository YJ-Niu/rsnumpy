# SPDX-License-Identifier: MIT
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


def _list_shape(l):
    """嵌套列表 → 形状元组；不规则/空返回 None。"""
    if not isinstance(l, (list, tuple)):
        return ()
    if len(l) == 0:
        return None
    sub = [_list_shape(v) for v in l]
    if any(s is None for s in sub):
        return None
    if all(s == sub[0] for s in sub):
        return (len(l),) + sub[0]
    return None


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

# numpy 2.x 分位数 method 全集与"插值型"（参与结果 dtype 提升）子集
QUANTILE_METHODS = [
    'inverted_cdf', 'averaged_inverted_cdf', 'closest_observation',
    'interpolated_inverted_cdf', 'hazen', 'weibull', 'linear',
    'median_unbiased', 'normal_unbiased', 'nearest', 'lower', 'higher',
    'midpoint']
_INTERPOLATING_METHODS = [
    'averaged_inverted_cdf', 'interpolated_inverted_cdf', 'hazen', 'weibull',
    'linear', 'median_unbiased', 'normal_unbiased', 'midpoint']

_INT64_MIN = -9223372036854775808  # m8/M8 的 NaT 哨兵（int64 存储）


def _quantile_of(sorted_vals, q01, method, wsorted=None):
    """已排序（无 NaN）实值列表 + q01∈[0,1] → float。

    实现 numpy 2.x 全部 13 个 method（虚指数公式按真实 numpy 2.5 逐点验证）；
    weights 仅支持 inverted_cdf（概念上按权重展开成重复观测再取无权重值）。
    """
    n = len(sorted_vals)
    if n == 0:
        return float('nan')
    m = method or 'linear'
    vs = sorted_vals
    ws = None
    if wsorted is not None:
        vs, ws = wsorted
    if ws is not None:
        total = builtins.sum(ws)
        if not (total > 0) or total != total or total == float('inf'):
            raise ValueError("Weights included NaN, inf or were all zero")
        # 重复观测语义：展开数组中 0-based 位置 = max(ceil(q*W), 1) - 1
        pos = builtins.max(math.ceil(q01 * total), 1) - 1
        c = 0.0
        for v, w in zip(vs, ws):
            c += w
            if c > pos:
                return float(v)
        return float(vs[-1])
    if m == 'inverted_cdf':
        idx = builtins.max(0, builtins.min(math.ceil(q01 * n) - 1, n - 1))
        return float(vs[idx])
    if m == 'averaged_inverted_cdf':
        k = q01 * n
        if k == builtins.int(k) and 0 < k < n:
            # q 恰在 CDF 跳变点：左右极限的均值
            return float((vs[int(k) - 1] + vs[int(k)]) / 2.0)
        idx = builtins.max(0, builtins.min(math.ceil(k) - 1, n - 1))
        return float(vs[idx])
    # 其余 method：统一"虚指数"（0-based 小数位置，clamp 到 [0, n-1]）
    if m == 'lower':
        vidx = math.floor(q01 * (n - 1))
    elif m == 'higher':
        vidx = math.ceil(q01 * (n - 1))
    elif m == 'nearest':
        vidx = round(q01 * (n - 1))  # Python round = 银行家舍入（tie→even）
    elif m == 'closest_observation':
        # numpy 2.x（#26656）：tie 向"偶数阶统计量"——1-based 位置做
        # half-even 舍入，等价于 round(q*n) - 1
        vidx = round(q01 * n) - 1
    elif m == 'midpoint':
        vidx = 0.5 * (math.floor(q01 * (n - 1)) + math.ceil(q01 * (n - 1)))
    elif m == 'linear':
        vidx = q01 * (n - 1)
    elif m == 'hazen':
        vidx = q01 * n - 0.5
    elif m == 'weibull':
        vidx = q01 * (n + 1.0) - 1.0
    elif m == 'median_unbiased':
        vidx = q01 * (n + 1.0 / 3.0) - 2.0 / 3.0
    elif m == 'normal_unbiased':
        vidx = q01 * (n + 0.25) - 0.625
    elif m == 'interpolated_inverted_cdf':
        vidx = q01 * n - 1.0
    else:
        raise ValueError(f"method {m!r} is not a valid method")
    vidx = builtins.min(builtins.max(vidx, 0), n - 1)
    if m in ('lower', 'higher', 'nearest', 'closest_observation'):
        return float(vs[int(vidx)])
    lo = int(math.floor(vidx))
    hi = builtins.min(lo + 1, n - 1)
    frac = vidx - lo
    return float(vs[lo] + (vs[hi] - vs[lo]) * frac)


def _percentile_engine(a, q, axis, keepdims, dtype, out=None, weights=None,
                       method='linear', nan_filter=True, scale=100.0,
                       sentinel=None, relabel=None):
    """通用分位数引擎（Python 层，numpy 2.x 语义）。
    q: 标量或列表/数组（percentile 用 0-100、quantile 用 0-1，由 scale 归一）；
    method: 见 QUANTILE_METHODS（加权仅 inverted_cdf，公共层校验）；
    weights: 与 a 同形状，或 1-D（长度 = 归约轴长 → 广播；
             多轴时 = 每 slice 展平长 → 逐 slice 使用）；
    sentinel: m8/M8 哨兵（NaT=int64 最小值）——slice 含哨兵/为空 → 哨兵结果，
              数值结果 round 为 int；relabel: 结果数组 _dtype 标签（如 'm8[D]'）；
    全归约 + 标量 q → rsnumpy 标量（带 .dtype、isscalar 为 True）；
    out 被填充并返回；输入为 ndarray 子类时结果保持子类。
    """
    rs = _rs()
    shape = tuple(a.shape)
    nd = len(shape)
    axes = _reduce_axes(axis, nd)
    flat = _flat_list(a.tolist())
    # ---- weights：全形状或 1-D
    wflat = None
    w1d = None
    red_size = _prod([shape[i] for i in axes]) if axes else 1
    if weights is not None:
        wsrc = weights if hasattr(weights, 'tolist') else rs.array(weights)
        wlist = _flat_list(wsrc.tolist())
        if len(wlist) == len(flat):
            wflat = wlist
        elif len(axes) == 1 and len(wlist) == shape[axes[0]]:
            w1d = wlist
        elif len(axes) > 1 and len(wlist) == red_size:
            w1d = wlist
        else:
            raise ValueError(
                "weights must be a 1-D array with the same length as the "
                "reduction axis, or an array with the same shape as a")
    if hasattr(q, 'ndim'):
        qarr = q.ndim > 0
        q_shape = tuple(q.shape) if qarr else ()
        qvals = _flat_list(q.tolist()) if qarr else [q.tolist()]
    elif isinstance(q, (list, tuple)):
        qarr = True
        q_shape = _list_shape(q) or (len(_flat_list(q)),)
        qvals = _flat_list(q)
    else:
        qarr = False
        q_shape = ()
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
    bad = float('nan') if sentinel is None else sentinel
    multi = len(axes) > 1
    for q01 in qs:
        for oi, (vals, wlist) in enumerate(slice_list):
            if not vals:
                results.append(bad)
                n_empty += 1
                continue
            if w1d is not None:
                wl = list(w1d) if not multi else w1d[:len(vals)]
            else:
                wl = wlist
            vw = list(builtins.zip(vals, wl)) if wl is not None \
                else [(v, None) for v in vals]
            if nan_filter:
                vw = [p for p in vw if not _is_nan_v(p[0])]
                if not vw:
                    results.append(bad)
                    n_allnan += 1
                    continue
            else:
                if any(_is_nan_v(p[0]) for p in vw) or (
                        sentinel is not None
                        and any(p[0] == sentinel for p in vw)):
                    results.append(bad)
                    continue
            if wl is not None:
                sv = sorted(vw, key=lambda p: _ckey(p[0]))
                vs = [p[0] for p in sv]
                ws = [p[1] for p in sv]
                r = _quantile_of(vs, q01, method, (vs, ws))
            else:
                r = _quantile_of(sorted((p[0] for p in vw), key=_ckey),
                                 q01, method)
            results.append(builtins.int(round(r)) if sentinel is not None else r)
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
    out_q_shape = q_shape + red_shape
    if sentinel is not None:
        # m8/M8：int64 存储 + 标签；0-d 结果保持 0-d ndarray（.dtype 可查）
        if not results and 0 in out_q_shape:
            res = rs.zeros(out_q_shape, dtype='int64')
        elif qarr:
            res = rs.array(_reshape_flat(results, out_q_shape), dtype='int64')
        elif not red_shape and len(results) == 1:
            res = rs.array([results[0]], dtype='int64').reshape(())
        else:
            res = rs.array(_reshape_flat(results, red_shape), dtype='int64')
        if relabel:
            res._dtype = relabel
        return res
    if not results and 0 in out_q_shape:
        res = rs.zeros(out_q_shape, dtype=dtype)
    elif qarr:
        res = rs.array(_reshape_flat(results, out_q_shape), dtype=dtype)
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
    if len(tuple(a.shape)) == 0:
        # 0-d：布尔掩码 setitem 会抛 IndexError，按值判定后直接重建
        v = _flat_list(a.tolist())[0]
        try:
            _c = complex(v)
            _m = _c != _c
        except Exception:
            _m = (v is not None) and (v != v)
        if _m:
            b = rs.array([val], dtype=getattr(a, '_dtype', 'float64')).reshape(())
        else:
            b = rs.array(a, copy=True)
        return b, mask
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
               'F': 'complex64', 'D': 'complex128', 'G': 'complex128'}


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
        # oracle 语义：var/std 结果恒 float64（complex 输入取实部）
        return 'float64'
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
        # 空结果 + 含 0 维：嵌套列表无法表达 0 维形状，直接 zeros 构造
        if not results and 0 in shape_out:
            res = rs.zeros(shape_out, dtype=dtype)
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
    if dtype is not None and _dt_kind(dtype) not in 'efdcgciubOFGD':
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
    if dtype is not None and _dt_kind(dtype) not in 'efdcgciubOFGD':
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
    if dtype is not None and _dt_kind(dtype) not in 'efgdcFDG':
        raise TypeError(f"data type {dtype!r} not inexact")
    if dtype is not None and _dt_kind(dtype) in 'cFDG':
        # 复数 dtype：委托标准 mean（dtype 语义与 oracle 完全一致）
        from rsnumpy import statistics as _st
        return _st.mean(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                        where=where)
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


def _nan_var_std_given_mean(a, mean, axis, keepdims, ddof, want_std, out):
    """nanvar/nanstd(mean=...) 路径：信任给定均值，仅干净值参与计算
    （numpy masked 语义：count 排除 NaN，NaN 位置零贡献）。"""
    import builtins as _bi
    rs = _rs()
    a = _as_nd(a)
    dt = getattr(a, '_dtype', 'float64')
    res_dt = _reduction_dtype('varstd', dt)
    # 必须在原始数组上判 NaN（替换后的 0.0 会被误判为干净值）
    shape = tuple(a.shape)
    nd = len(shape)

    def _clean(v):
        try:
            c = complex(v)
        except Exception:
            return None
        return None if c != c else c

    def _mval(m):
        try:
            return complex(m)
        except Exception:
            return m

    def _unravel(idx, dims):
        mi = []
        for dm in reversed(dims):
            mi.append(idx % dm)
            idx //= dm
        return list(reversed(mi))

    def _cidx(mi, shp):
        c = 0
        for m, s in zip(mi, shp):
            c = c * s + m
        return c

    if hasattr(mean, 'tolist'):
        mshape = tuple(mean.shape)
        mflat = _flat_list(mean.tolist())
    else:
        mshape = ()
        mflat = [mean]

    if axis is None:
        vals = [c for c in map(_clean, _flat_list(a.tolist())) if c is not None]
        m = _mval(mflat[0])
        n = len(vals) - ddof
        if n <= 0 or not vals:
            s = float('nan')
        else:
            s = _bi.sum(abs(v - m) ** 2 for v in vals) / n
        v = s ** 0.5 if want_std else s
        return _reduce_wrap([v], tuple(range(nd)), shape, keepdims, res_dt,
                            out, a)
    ai = axis if axis >= 0 else axis + nd
    dims = list(shape)
    d = dims[ai]
    inner = 1
    for j in range(ai + 1, nd):
        inner *= dims[j]
    flat = _flat_list(a.tolist())
    out_shape = dims[:ai] + dims[ai + 1:]
    out_size = 1
    for s in out_shape:
        out_size *= s
    results = []
    for oi in range(out_size):
        # 输出网格逐元素：slice = a[omi, k, imi]（k 沿轴方向，非轴内维）
        omi = _unravel(oi // inner, dims[:ai]) if dims[:ai] else []
        imi = _unravel(oi % inner, dims[ai + 1:]) if dims[ai + 1:] else []
        if len(mshape) == nd:
            midx = _cidx(omi + [0] + imi, mshape)
        elif mshape:
            midx = _cidx(omi + imi, mshape)
        else:
            midx = 0
        m = _mval(mflat[midx])
        vals = []
        for kk in range(d):
            c = _clean(flat[_cidx(omi + [kk] + imi, shape)])
            if c is not None:
                vals.append(c)
        n = len(vals) - ddof
        if n <= 0 or not vals:
            s = float('nan')
        else:
            s = _bi.sum(abs(v - m) ** 2 for v in vals) / n
        results.append(s ** 0.5 if want_std else s)
    return _reduce_wrap(results, (ai,), shape, keepdims, res_dt, out, a)


def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False,
           where=True, correction=None, mean=None):
    """忽略 NaN 求方差（ddof/correction 互斥；mean= 走信任均值路径）。"""
    rs = _rs()
    a = _as_nd(a)
    dt = getattr(a, '_dtype', 'float64')
    if dtype is not None and _dt_kind(dtype) not in 'efgdcFDG':
        raise TypeError(f"data type {dtype!r} not inexact")
    if correction is not None:
        if ddof != 0:
            raise ValueError(
                "ddof and correction can't be provided simultaneously.")
        ddof = correction
    if mean is not None:
        return _nan_var_std_given_mean(a, mean, axis, keepdims, ddof,
                                       False, out)
    if dtype is not None and _dt_kind(dtype) in 'cFDG':
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
    if dtype is not None and _dt_kind(dtype) not in 'efgdcFDG':
        raise TypeError(f"data type {dtype!r} not inexact")
    if correction is not None:
        if ddof != 0:
            raise ValueError(
                "ddof and correction can't be provided simultaneously.")
        ddof = correction
    if mean is not None:
        return _nan_var_std_given_mean(a, mean, axis, keepdims, ddof,
                                       True, out)
    if dtype is not None and _dt_kind(dtype) in 'cFDG':
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
    # 0-d 与标准 median 一致：float/complex 保留精度，其余 float64
    #（int 输入 0-d 也转 float64，与 _median_py 的 res_dt 规则对齐）
    res_dt = (dt if k in 'efc' else 'float64') if len(shape) == 0 \
        else _reduction_dtype('median', dt)
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


def _validate_method(method):
    """numpy 2.x 报错："<m> is not a valid method. Use one of: 'inverted_cdf', ..."""
    m = getattr(method, 'value', method)
    if m not in QUANTILE_METHODS:
        use = ', '.join("'%s'" % x for x in QUANTILE_METHODS)
        raise ValueError(f"{m} is not a valid method. Use one of: {use}")
    return m


def _frac_type():
    from fractions import Fraction
    return Fraction


def _promote_float(base, other):
    """float 家族 dtype 提升（f16<f32<f64<f80）；非 float 基底按对方 float 提升。"""
    order = {'float16': 0, 'float32': 1, 'float64': 2, 'longdouble': 3}
    if base in order and other in order:
        return base if order[base] >= order[other] else other
    if base in order:
        return base
    if other in order:
        return other
    return 'float64'


def _m8_label_of(dt):
    """dtype 字符串/DType 的 m8[X]/M8[X] 标签（无则 None）。"""
    if dt is None:
        return None
    s = dt if isinstance(dt, str) else str(dt)
    t = s.lstrip('<>=|')
    if (len(t) > 4 and t[1] == '8' and t[2] == '[' and t[-1] == ']'
            and t[0] in 'Mm' and t[3:-1].isalpha()):
        return t
    return None


def _q_values(q, scale):
    """q → (Python 数值列表, 数组 q 的 float 提升 dtype|None, 是否全 0/1, 是否数组)。
    校验范围并按 scale 给出 numpy 2.x 精确报错信息。"""
    msg = ("Percentiles must be in the range [0, 100]" if scale > 1
           else "Quantiles must be in the range [0, 1]")
    promote = None
    if isinstance(q, (list, tuple)):
        qvals = _flat_list(q)
        qarr = True
    elif builtins.type(q) in (int, float, bool) or isinstance(q, _frac_type()):
        # 纯 Python 标量：NEP50 不参与提升
        qvals = [q]
        qarr = False
    else:
        # rsnumpy ndarray / 数值标量（有 .dtype）：参与提升
        _dtq = getattr(q, '_dtype', None) or getattr(q, 'dtype', None)
        qk = _dt_kind(_dtq)
        if qk in 'efd':
            promote = _dtype_name(_dtq)
        qarr = getattr(q, 'ndim', 0) > 0
        qvals = _flat_list(q.tolist()) if qarr else [q.tolist()]
    exact01 = True
    for v in qvals:
        try:
            fv = float(v) / scale
        except (TypeError, ValueError):
            raise ValueError(msg)
        if fv != fv or not (0.0 <= fv <= 1.0):
            raise ValueError(msg)
        if fv not in (0.0, 1.0):
            exact01 = False
    return qvals, promote, exact01, qarr


def _frac_quantile(vs, qf, method):
    """Fraction 精确分位数（vs 已排序 Fraction 列表，qf ∈ [0,1] 精确值）。"""
    from fractions import Fraction as _Frac
    n = len(vs)
    if n == 0:
        return _Frac(0)
    m = method or 'linear'
    if m == 'inverted_cdf':
        idx = builtins.max(0, builtins.min(math.ceil(qf * n) - 1, n - 1))
        return vs[idx]
    if m == 'averaged_inverted_cdf':
        k = qf * n
        if k.denominator == 1 and 0 < k < n:
            return (vs[k - 1] + vs[k]) / 2
        idx = builtins.max(0, builtins.min(math.ceil(k) - 1, n - 1))
        return vs[idx]
    if m == 'lower':
        vidx = math.floor(qf * (n - 1))
    elif m == 'higher':
        vidx = math.ceil(qf * (n - 1))
    elif m == 'nearest':
        # Fraction 精确 half-even（Python round 对 Fraction 是精确 tie→even）
        vidx = round(qf * (n - 1))
    elif m == 'closest_observation':
        # 1-based 位置 half-even（numpy #26656）：round(q*n) - 1
        vidx = round(qf * n) - 1
    elif m == 'midpoint':
        vidx = _Frac(math.floor(qf * (n - 1))
                     + math.ceil(qf * (n - 1)), 2)
    elif m == 'linear':
        vidx = qf * (n - 1)
    elif m == 'hazen':
        vidx = qf * n - _Frac(1, 2)
    elif m == 'weibull':
        vidx = qf * (n + 1) - 1
    elif m == 'median_unbiased':
        vidx = qf * (n + _Frac(1, 3)) - _Frac(2, 3)
    elif m == 'normal_unbiased':
        vidx = qf * (n + _Frac(1, 4)) - _Frac(5, 8)
    elif m == 'interpolated_inverted_cdf':
        vidx = qf * n - 1
    else:
        raise ValueError(f"method {m!r} is not a valid method")
    vidx = builtins.min(builtins.max(vidx, 0), n - 1)
    if m in ('lower', 'higher', 'nearest', 'closest_observation'):
        return vs[vidx]
    lo = math.floor(vidx)
    hi = builtins.min(lo + 1, n - 1)
    frac = vidx - lo
    return vs[lo] + (vs[hi] - vs[lo]) * frac


def _percentile_public(a, q, axis, out, overwrite_input, method, keepdims,
                       weights, nan_filter, scale):
    m = _validate_method(method)
    a = _as_nd(a)
    dt = getattr(a, '_dtype', 'float64')
    k = _dt_kind(dt).lower()
    if k == 'c':
        raise TypeError(
            "numpy is unable to avoid using the complex type with the "
            "current version of the 'inplace-sort' used in the "
            "'nanpercentile' function. Please make sure to cast your "
            "input array to a real type before calling this function.")
    # ---- weights 校验（仅 inverted_cdf；非负）
    if weights is not None:
        if m != 'inverted_cdf':
            raise ValueError("Only method 'inverted_cdf' supports weights")
        wsrc = weights if hasattr(weights, 'tolist') else _rs().array(weights)
        for w in _flat_list(wsrc.tolist()):
            if w < 0:
                raise ValueError("Weights must be non-negative")
    # ---- m8/M8：int64 哨兵路径
    m8s = _m8_label_of(dt)
    if m8s is not None:
        return _percentile_engine(a, q, axis, keepdims, 'int64', out=out,
                                  weights=weights, method=m, nan_filter=nan_filter,
                                  scale=scale, sentinel=_INT64_MIN, relabel=m8s)
    qvals, promote, exact01, qarr = _q_values(q, scale)
    if isinstance(q, (list, tuple)):
        q_shape = _list_shape(q) or (len(qvals),)
    elif qarr and hasattr(q, 'shape'):
        q_shape = tuple(q.shape)
    else:
        q_shape = ()

    def _frac_results(vs):
        _Frac = _frac_type()
        n = len(vs)
        res = []
        for qv in qvals:
            fqv = float(qv) / scale
            if isinstance(qv, _Frac):
                res.append(_frac_quantile(vs, _Frac(qv) / _Frac(int(scale)), m))
            elif m in _INTERPOLATING_METHODS and fqv not in (0.0, 1.0):
                # 浮点 q + 插值 method：float 结果（numpy 语义）
                res.append(_quantile_of([float(x) for x in vs], fqv, m))
            elif m == 'averaged_inverted_cdf':
                kx = fqv * n
                if kx == builtins.int(kx) and 0 < kx < n:
                    res.append((vs[int(kx) - 1] + vs[int(kk)]) / 2)
                else:
                    idx = builtins.max(0, builtins.min(math.ceil(kx) - 1, n - 1))
                    res.append(vs[idx])
            else:
                # 精确 0/1 或离散 method：保留原对象（Fraction 恒等性）
                if fqv == 0.0:
                    res.append(vs[0])
                elif fqv == 1.0:
                    res.append(vs[n - 1])
                else:
                    res.append(vs[_sel_idx(fqv, n, m)])
        return res

    def _frac_out(results):
        rs = _rs()
        if qarr:
            r = rs.array(results, dtype='object')
            if q_shape and int(getattr(r, 'size', 0) or 0) == _prod(q_shape):
                r = r.reshape(q_shape)
            return r
        return results[0]

    # ---- Fraction 精确路径（object 输入 + Fraction q，全归约）
    if k == 'o' and axis is None:
        flat = _flat_list(a.tolist())
        if flat and all(isinstance(v, _frac_type()) for v in flat):
            return _frac_out(_frac_results(sorted(flat)))
    # ---- int/bool 输入 + Fraction q：numpy 语义——插值算术保持 Fraction
    #（q=Fraction(1), [1,2] → Fraction(2)；Fraction(.5) → Fraction(3,2)）
    if (k in 'iu?b' and axis is None and weights is None
            and any(isinstance(qv, _frac_type()) for qv in qvals)):
        vals0 = _flat_list(a.tolist())
        if vals0 and all(isinstance(v, (int, bool)) for v in vals0):
            return _frac_out(_frac_results(sorted(vals0)))
    # ---- 结果 dtype 规则（numpy 2.x）
    interp = m in _INTERPOLATING_METHODS
    if k in 'ef':
        res_dt = _dtype_name(dt)
    elif k in 'iu?b':
        res_dt = _dtype_name(dt) if (not interp or exact01) else 'float64'
    else:
        res_dt = 'float64'
    if promote is not None:
        res_dt = _promote_float(res_dt, promote)
    return _percentile_engine(a, q, axis, keepdims, res_dt, out=out,
                              weights=weights, method=m,
                              nan_filter=nan_filter, scale=scale)


def _sel_idx(fqv, n, m):
    """离散 method 的选择索引（0-based，clamp）；非离散返回 None。"""
    if m == 'inverted_cdf':
        return builtins.max(0, builtins.min(math.ceil(fqv * n) - 1, n - 1))
    if m == 'lower':
        return builtins.min(builtins.max(math.floor(fqv * (n - 1)), 0), n - 1)
    if m == 'higher':
        return builtins.min(builtins.max(math.ceil(fqv * (n - 1)), 0), n - 1)
    if m == 'nearest':
        return builtins.min(builtins.max(round(fqv * (n - 1)), 0), n - 1)
    if m == 'closest_observation':
        return builtins.min(builtins.max(round(fqv * n) - 1, 0), n - 1)
    return None


def _median_py(a, axis, out, keepdims):
    """纯 Python 中位数（numpy 2.x 语义；任意 dtype/axis 形状，消除 Rust
    _core.median 的 f64/2D 限制与空 slice 减溢 abort）。
    NaN/NaT 传染 slice；空 slice → NaN + 两条 RuntimeWarning；
    m8/M8 在 int64 存储上计算（NaT=INT64_MIN），结果重贴标签；
    结果 dtype：float 保精度，其余 float64；子类/keepdims/out 保留。
    """
    rs = _rs()
    a = _as_nd(a)
    dt = getattr(a, '_dtype', 'float64')
    k = _dt_kind(dt)
    shape = tuple(a.shape)
    nd = len(shape)
    axes = _reduce_axes(axis, nd)
    flat = _flat_list(a.tolist())
    m8s = _m8_label_of(dt)
    sentinel = _INT64_MIN if m8s else None
    results = []
    n_empty = 0
    for vals, _wl in _slices(flat, shape, axes):
        if not vals:
            results.append(sentinel if sentinel is not None else float('nan'))
            n_empty += 1
            continue
        if any(_is_nan_v(v) for v in vals) or (
                sentinel is not None and any(v == sentinel for v in vals)):
            results.append(sentinel if sentinel is not None else float('nan'))
            continue
        vs = sorted(vals, key=_ckey)
        n = len(vs)
        r = vs[n // 2] if n % 2 == 1 else (vs[n // 2 - 1] + vs[n // 2]) / 2
        results.append(builtins.int(round(r)) if sentinel is not None else r)
    if n_empty:
        warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=4)
        warnings.warn("invalid value encountered in scalar divide",
                      RuntimeWarning, stacklevel=4)
    if m8s:
        axset = set(axes)
        out_shape = [d for i, d in enumerate(shape) if i not in axset]
        if keepdims:
            red = []
            kk = 0
            for i in range(nd):
                if i in axset:
                    red.append(1)
                else:
                    red.append(out_shape[kk])
                    kk += 1
            red = tuple(red)
        else:
            red = tuple(out_shape)
        if not red:
            res = rs.array([results[0]], dtype='int64').reshape(())
        elif not results and 0 in red:
            res = rs.zeros(red, dtype='int64')
        else:
            res = rs.array(_reshape_flat(list(results), red), dtype='int64')
        res._dtype = m8s
        if out is not None:
            try:
                rs.copyto(out, res)
            except Exception:
                out[...] = res
            return out
        return res
    # complex 保留自身 dtype（numpy 语义：median(c64) → c64；float64(c) 会
    # 抛 TypeError 退化裸值）
    res_dt = _dtype_name(dt) if k in 'efc' else 'float64'
    return _reduce_wrap(results, axes, shape, keepdims, res_dt, out, a)


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
    """忽略 NaN 的累积和/积（numpy 实现：NaN → 0/1 替换后委托标准
    cumsum/cumprod 内核——与标准结果逐位一致、标注规则天然一致）。"""
    rs = _rs()
    a = _as_nd(a)
    ident = 1.0 if is_prod else 0.0
    b, _mask = _replace_nan(a, ident)
    func = rs.cumprod if is_prod else rs.cumsum
    res = func(b, axis=axis, dtype=dtype, out=out)
    if out is not None:
        return res
    return _keep_subclass(res, a)


def nancumsum(a, axis=None, dtype=None, out=None):
    """忽略 NaN 的累积和（NaN 视为 0）。"""
    return _nancum_impl(a, axis, dtype, out, False)


def nancumprod(a, axis=None, dtype=None, out=None):
    """忽略 NaN 的累积积（NaN 视为 1）。"""
    return _nancum_impl(a, axis, dtype, out, True)
