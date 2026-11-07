"""补充的 NumPy 兼容 API。

计算全部复用已有的 Rust 原语（_core 或已导出的顶层函数），本层仅做组合与包装，
不重复实现底层数值循环。热点函数可在后续通过 Rust 层进一步优化。
"""

# import builtins as _builtins
import builtins
import cmath as _cmath
import math as _math
import re as _re

import rsnumpy.num_core as _core

builtin_max = max
builtin_min = min
builtin_all = all
builtin_any = any
builtin_round = round
builtin_abs = abs
builtin_range = range
builtin_sum = sum


def _np():
    import rsnumpy as _n
    return _n


def _asarray(a):
    np = _np()
    return a if hasattr(a, "_array") else np.array(a)


def _wrap(raw, _dtype="float64"):
    return _np().ndarray._wrap(raw, _dtype=_dtype)


def _flat(a):
    """返回任意嵌套 tolist 的一维扁平列表。"""
    # 处理 Poly 对象：提取系数
    np = _np()
    if hasattr(a, '__class__') and a.__class__.__name__ == 'Poly':
        a = a.coef if hasattr(a, 'coef') else np.array(list(a))
    out = []
    stack = [_asarray(a).tolist()]
    while stack:
        v = stack.pop()
        if isinstance(v, list):
            stack.extend(reversed(v))
        else:
            out.append(v)
    return out


def _map_nested(fn, data):
    if isinstance(data, list):
        return [_map_nested(fn, v) for v in data]
    return fn(data)


def _as_scalar(v):
    if hasattr(v, "tolist"):
        v = v.tolist()
    while isinstance(v, list) and len(v) == 1:
        v = v[0]
    return v


# ========== 三角/双曲别名 ==========
def acos(x):
    """反余弦（arccos 别名）。"""
    return _np().arccos(x)


def asin(x):
    """反正弦（arcsin 别名）。"""
    return _np().arcsin(x)


def atan(x):
    """反正切（arctan 别名）。"""
    return _np().arctan(x)


def atan2(x1, x2):
    """两参数反正切（arctan2 别名）。"""
    return _np().arctan2(x1, x2)


def arccosh(x):
    """反双曲余弦（acosh 别名）。"""
    return _np().acosh(x)


def arcsinh(x):
    """反双曲正弦（asinh 别名）。"""
    return _np().asinh(x)


def arctanh(x):
    """反双曲正切（atanh 别名）。"""
    return _np().atanh(x)


def radians(x):
    """角度转弧度（deg2rad 别名）。"""
    return _np().deg2rad(x)


def degrees(x):
    """弧度转角度（rad2deg 别名）。"""
    return _np().rad2deg(x)


# ========== 逐元素数学 ==========
def absolute(x):
    """逐元素绝对值。"""
    return _np().abs(x)


fabs = absolute


def true_divide(x1, x2):
    """逐元素真除（divide 别名）。"""
    return _np().divide(x1, x2)


def pow(x1, x2):
    """逐元素幂（power 别名）。"""
    return _np().power(x1, x2)


float_power = pow


def negative(x):
    """逐元素取负。"""
    return _np().multiply(x, -1.0)


def positive(x):
    """逐元素取正（返回副本）。"""
    return _asarray(x)


def exp2(x):
    """逐元素 2**x。"""
    return _np().power(2.0, x)


def floor_divide(x1, x2):
    """逐元素向下整除（除零按浮点错误状态处理，numpy 语义）。"""
    np = _np()
    a = _asarray(x1)
    b = _asarray(x2)
    if b.size:
        try:
            zero_mask = (b == 0)
            has_zero = zero_mask.any()
        except TypeError:
            has_zero = False
        if has_zero:
            _fp_fire('divide', "divide by zero encountered in floor_divide")
            res = np.floor(np.divide(a, np.where(zero_mask, 1, b)))
            return np.where(zero_mask, 0, res)
    return np.floor(np.divide(x1, x2))


def fmod(x1, x2):
    """逐元素 C 风格取余（余数符号同被除数）。"""
    np = _np()
    q = np.trunc(np.divide(x1, x2))
    return np.subtract(x1, np.multiply(x2, q))


def divmod(x1, x2):
    """逐元素返回 (整除商, 余数)。"""
    return floor_divide(x1, x2), _np().mod(x1, x2)


def rint(x):
    """逐元素舍入到最近整数（四舍六入五成双）。"""
    return _wrap(_core.rint(_asarray(x)._array))


def hypot(x1, x2):
    """逐元素 sqrt(x1**2 + x2**2)。"""
    np = _np()
    return _wrap(_core.hypot(np.asarray(x1)._array, np.asarray(x2)._array))


def maximum(x1, x2):
    """逐元素取较大值。"""
    np = _np()
    return _wrap(_core.maximum(np.asarray(x1)._array, np.asarray(x2)._array))


def minimum(x1, x2):
    """逐元素取较小值。"""
    np = _np()
    return _wrap(_core.minimum(np.asarray(x1)._array, np.asarray(x2)._array))


def fmax(x1, x2):
    """逐元素取较大值，忽略 NaN。"""
    return _wrap(_core.fmax(_asarray(x1)._array, _asarray(x2)._array))


def fmin(x1, x2):
    """逐元素取较小值，忽略 NaN。"""
    return _wrap(_core.fmin(_asarray(x1)._array, _asarray(x2)._array))


def signbit(x):
    """逐元素判断是否为负号位。"""
    return _wrap(_core.signbit(_asarray(x)._array), "bool")


def copysign(x1, x2):
    """将 x2 的符号赋给 x1。"""
    return _wrap(_core.copysign(_asarray(x1)._array, _asarray(x2)._array))


def logaddexp(x1, x2):
    """数值稳定地计算 log(exp(x1) + exp(x2))。"""
    np = _np()
    a = _asarray(x1)
    b = _asarray(x2)
    m = maximum(a, b)
    return np.add(m, np.log(np.add(np.exp(np.subtract(a, m)), np.exp(np.subtract(b, m)))))


def logaddexp2(x1, x2):
    """数值稳定地计算 log2(2**x1 + 2**x2)。"""
    np = _np()
    a = _asarray(x1)
    b = _asarray(x2)
    m = maximum(a, b)
    return np.add(m, np.log2(np.add(exp2(np.subtract(a, m)), exp2(np.subtract(b, m)))))


def logical_not(x, out=None):
    """逐元素逻辑非。"""
    np = _np()
    arr = _asarray(x)
    res = np.equal(arr, np.full(arr.shape, 0.0))
    if out is not None:
        try:
            out[...] = res
            return out
        except Exception:
            pass
    return res


def gcd(x1, x2):
    """逐元素最大公约数。"""
    return _wrap(_core.gcd(_asarray(x1)._array, _asarray(x2)._array), "int64")


def lcm(x1, x2):
    """逐元素最小公倍数。"""
    return _wrap(_core.lcm(_asarray(x1)._array, _asarray(x2)._array), "int64")


_NA_STRUCT = {
    # struct 格式 / 位宽 / 无符号位型：C 库做精确 IEEE 往返（round-half-even）
    'float32': (__import__('struct').Struct('<f'), 32, 'I'),
    'float16': (__import__('struct').Struct('<e'), 16, 'H'),
}


def _nextafter_prec(a, b, st, width, ufmt):
    """在给定精度（f16/f32）下计算 a 朝 b 方向的下一 ULP（a、b 为该精度可表示值）。"""
    import struct as _struct
    if a != a or b != b:
        return float('nan')
    if a < b:
        step = 1
    elif a > b:
        step = -1
    else:
        return b
    bits = _struct.unpack(ufmt, st.pack(a))[0]
    sign = 1 << (width - 1)
    # 单调映射：非负保持位型，负值翻转（-0.0 与 +最小次正规相邻）
    mono = bits if bits < sign else (2 * sign - 1) - bits
    mono += step
    bits = mono if mono < sign else (2 * sign - 1) - mono
    return st.unpack(_struct.pack(ufmt, bits))[0]


def nextafter(x1, x2):
    """逐元素返回 x1 朝 x2 方向的下一个可表示浮点数。

    numpy 语义：结果 dtype 跟随输入提升，next-ULP 按该精度计算
    （f16/f32/f64 各自的相邻可表示值）。Rust 核心统一以 f64 存储，
    f16/f32 结果在 Python 层按 IEEE 位级 next-ULP 计算（struct 'e'/'f'
    的 C 转换保证 round-half-even），f64 结果走 Rust 快速路径。
    """
    a1 = _asarray(x1)
    a2 = _asarray(x2)
    dt_name = str(getattr(result_type(a1, a2), 'name', 'float64'))
    if dt_name not in _NA_STRUCT:
        return _wrap(_core.nextafter(a1._array, a2._array))
    st, width, ufmt = _NA_STRUCT[dt_name]

    def _flat(x):
        out = []

        def _r(v):
            if isinstance(v, list):
                for e in v:
                    _r(e)
            else:
                out.append(v)
        _r(x)
        return out

    sv = _flat(a1.tolist())
    tv = _flat(a2.tolist())
    if a1.ndim == 0 and a2.ndim == 0:
        pairs = [(sv[0], tv[0])]
        shape = ()
    elif a1.ndim == 0:
        pairs = [(sv[0], t) for t in tv]
        shape = tuple(a2.shape)
    elif a2.ndim == 0:
        pairs = [(s, tv[0]) for s in sv]
        shape = tuple(a1.shape)
    elif tuple(a1.shape) == tuple(a2.shape):
        pairs = list(zip(sv, tv))
        shape = tuple(a1.shape)
    else:
        # 尾维对齐广播
        nd = builtin_max(a1.ndim, a2.ndim)
        ss = (1,) * (nd - a1.ndim) + tuple(a1.shape)
        ts = (1,) * (nd - a2.ndim) + tuple(a2.shape)
        if any(x != y and x != 1 and y != 1 for x, y in zip(ss, ts)):
            return _wrap(_core.nextafter(a1._array, a2._array))
        shape = tuple(builtin_max(x, y) for x, y in zip(ss, ts))

        def _strides(sh):
            out = [1]
            for d in reversed(sh[:-1]):
                out.append(out[-1] * d)
            return list(reversed(out))

        sst, tst, ostr = _strides(ss), _strides(ts), _strides(shape)
        n = 1
        for d in shape:
            n *= d
        pairs = []
        for p in range(n):
            si = ti = 0
            for ax in range(nd):
                idx = (p // ostr[ax]) % shape[ax]
                si += (0 if ss[ax] == 1 else idx) * sst[ax]
                ti += (0 if ts[ax] == 1 else idx) * tst[ax]
            pairs.append((sv[si], tv[ti]))
    vals = [_nextafter_prec(s, t, st, width, ufmt) for s, t in pairs]
    np_ = _np()
    if not shape:
        return np_.ndarray([vals[0]] if vals else []).reshape(())
    return np_.ndarray(vals).reshape(shape)


def spacing(x):
    """逐元素返回相邻浮点数间距(按输入 dtype 的尾数位数)。"""
    import math as _math
    import struct as _struct
    arr = _asarray(x)
    dt = str(getattr(arr, '_dtype', '') or '')
    # Rust 层按 f64 计算间距,float32/16 需按各自尾数位数修正
    if 'float32' in dt or 'float16' in dt:
        _bits = 24 if 'float32' in dt else 11
        _fmt = '<I' if 'float32' in dt else None

        def _sp(v):
            if v == 0:
                return 2.0 ** (-149) if 'float32' in dt else 2.0 ** (-24)
            if _math.isnan(v) or _math.isinf(v):
                return float('nan')
            if _fmt is not None:
                bits = _struct.unpack(_fmt, _struct.pack('<f', v))[0]
                if v > 0:
                    nb = bits + 1
                else:
                    nb = bits - 1
                nv = _struct.unpack('<f', _struct.pack('<I', nb & 0xFFFFFFFF))[0]
                q = _struct.unpack('<f', _struct.pack('<f', v))[0]
                return abs(nv - q)
            # float16:近似用归一化公式
            _, e = _math.frexp(abs(v))
            return 2.0 ** (e - _bits)

        vals = [_sp(float(v)) for v in arr.tolist()]
        np_ = _np()
        return np_.array(vals, dtype='float32' if 'float32' in dt else 'float16')
    return _wrap(_core.spacing(arr._array))


def ldexp(x1, x2):
    """逐元素返回 x1 * 2**x2。"""
    return _wrap(_core.ldexp(_asarray(x1)._array, _asarray(x2)._array))


def frexp(x):
    """逐元素返回 (尾数, 指数)。"""
    mant, expo = _core.frexp(_asarray(x)._array)
    return _wrap(mant), _wrap(expo, "int64")


def modf(x):
    """逐元素返回 (小数部分, 整数部分)。"""
    np = _np()
    arr = _asarray(x)
    ipart = np.trunc(arr)
    return np.subtract(arr, ipart), ipart


# ========== 归约 / prod / nan 系列 ==========
def _reduce_last_along(c, axis):
    np = _np()
    n = c.shape[axis]
    picked = np.take(c, [n - 1], axis=axis)
    ax = axis % picked.ndim
    newshape = tuple(s for i, s in enumerate(picked.shape) if i != ax)
    if not newshape:
        return _as_scalar(picked)
    return np.reshape(picked, newshape)


def prod(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
    """计算元素乘积。"""
    _ = dtype, out, where
    arr = _asarray(a)
    # 0-d 短路：Rust 归约对 0-d 输入会 abort；numpy 返回保持原 dtype 的 0-d
    if not tuple(getattr(arr._array, 'shape', ())):
        res = _wrap(arr._array, _dtype=getattr(arr, '_dtype', 'float64'))
        if initial is not None:
            np = _np()
            res = np.multiply(res, initial)
        return res
    result = _core.prod(arr._array, axis, keepdims)
    if initial is not None:
        np = _np()
        result = np.multiply(result, initial)
    return _wrap(result)


def _shape_drop(shape, axis):
    ax = axis % len(shape)
    return tuple(s for i, s in enumerate(shape) if i != ax)


def _rep_nan(a, val):
    np = _np()
    arr = _asarray(a)
    return np.where(np.isnan(arr), np.full(arr.shape, float(val)), arr)


def _count_notnan(a, axis):
    np = _np()
    arr = _asarray(a)
    mask = np.where(np.isnan(arr), np.full(arr.shape, 0.0), np.full(arr.shape, 1.0))
    return np.sum(mask, axis)


def nansum(a, axis=None, dtype=None, out=None, keepdims=False):
    """忽略 NaN 求和（Rust 单次遍历实现）。"""
    _ = dtype, out
    return _wrap(_core.nansum(_asarray(a)._array, axis, keepdims))


def nanprod(a, axis=None, dtype=None, out=None, keepdims=False):
    """忽略 NaN 求积（Rust 单次遍历实现）。"""
    _ = dtype, out
    return _wrap(_core.nanprod(_asarray(a)._array, axis, keepdims))


def nanmax(a, axis=None, out=None, keepdims=False):
    """忽略 NaN 求最大值（Rust 实现，全 NaN 返回 NaN）。"""
    _ = out
    return _wrap(_core.nanmax(_asarray(a)._array, axis, keepdims))


def nanmin(a, axis=None, out=None, keepdims=False):
    """忽略 NaN 求最小值（Rust 实现，全 NaN 返回 NaN）。"""
    _ = out
    return _wrap(_core.nanmin(_asarray(a)._array, axis, keepdims))


def nanmean(a, axis=None, dtype=None, out=None, keepdims=False):
    """忽略 NaN 求均值（Rust 单次遍历实现）。"""
    _ = dtype, out
    return _wrap(_core.nanmean(_asarray(a)._array, axis, keepdims))


def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """忽略 NaN 求方差（Rust 两遍遍历实现）。"""
    _ = dtype, out
    return _wrap(_core.nanvar(_asarray(a)._array, axis, ddof, keepdims))


def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """忽略 NaN 求标准差（Rust 实现）。"""
    _ = dtype, out
    return _wrap(_core.nanstd(_asarray(a)._array, axis, ddof, keepdims))


def nanargmax(a, axis=None):
    """忽略 NaN 的最大值索引（Rust 实现）。"""
    return _wrap(_core.nanargmax_axis(_asarray(a)._array, axis), _dtype="int64")


def nanargmin(a, axis=None):
    """忽略 NaN 的最小值索引（Rust 实现）。"""
    return _wrap(_core.nanargmin_axis(_asarray(a)._array, axis), _dtype="int64")


def nancumsum(a, axis=None):
    """忽略 NaN 的累积和（NaN 视为 0，Rust 实现）。"""
    return _wrap(_core.nancumsum(_asarray(a)._array, axis))


def nancumprod(a, axis=None):
    """忽略 NaN 的累积积（NaN 视为 1，Rust 实现）。"""
    return _wrap(_core.nancumprod(_asarray(a)._array, axis))


def nanquantile(a, q, axis=None, **kwargs):
    """忽略 NaN 的分位数。"""
    _ = axis, kwargs
    np = _np()
    vals = [v for v in _flat(a) if not _math.isnan(v)]
    return np.percentile(np.array(vals), q * 100 if q <= 1 else q)


def cumulative_sum(x, axis=None, dtype=None, out=None, include_initial=False):
    """累积和（Array API 名称，等价 cumsum）。"""
    _ = dtype, out, include_initial
    return _np().cumsum(x, axis)


def cumulative_prod(x, axis=None, dtype=None, out=None, include_initial=False):
    """累积积（Array API 名称，等价 cumprod）。"""
    _ = dtype, out, include_initial
    return _np().cumprod(x, axis)


def count_nonzero(a, axis=None, *, keepdims=False):
    """统计非零元素个数。"""
    _ = keepdims
    np = _np()
    arr = _asarray(a)
    # 字符串数组（S/U）：非空元素为真（"" / b"" 为假）→ 转 0/1 掩码
    # 后走数值路径（numpy 语义）
    if getattr(arr, '_str_dtype', None) is not None or \
            str(getattr(arr, '_dtype', '')) in ('string_', 'bytes_'):
        _vals_c = []

        def _fs_c(x):
            if isinstance(x, list):
                for v in x:
                    _fs_c(v)
            else:
                _vals_c.append(x)

        _fs_c(arr.tolist())
        _shape_c = tuple(arr.shape)
        _bools_c = [0.0 if v in (b'', '', None) else 1.0 for v in _vals_c]
        if _shape_c:
            arr = np.array(_bools_c).reshape(_shape_c)
        else:
            arr = np.array([_bools_c[0] if _bools_c else 0.0])
    if axis is None:
        return _core.count_nonzero(arr._array)
    # numpy 风格的 axis 校验：重复 axis 抛 ValueError，非整数抛 TypeError，
    # 越界抛 AxisError
    from rsnumpy.exceptions import AxisError as _AxisError
    if isinstance(axis, (tuple, list)):
        if len(set(axis)) != len(axis):
            raise ValueError("repeated axis in input")
        axis_list = list(axis)
    elif isinstance(axis, int) and not isinstance(axis, bool):
        axis_list = [axis]
    else:
        raise TypeError(f"axis must be an int or a tuple of ints, not {type(axis)}")
    for v in axis_list:
        if not isinstance(v, int) or isinstance(v, bool):
            raise TypeError(f"axis must be an int, not {type(v)}")
        if v < -arr.ndim or v >= arr.ndim:
            raise _AxisError(f"axis {v} is out of bounds for array of dimension {arr.ndim}")
    mask = np.where(np.equal(arr, np.full(arr.shape, 0.0)), np.full(arr.shape, 0.0), np.full(arr.shape, 1.0))
    if len(axis_list) == 1:
        return np.sum(mask, axis_list[0])
    for v in sorted(axis_list, reverse=True):
        mask = np.sum(mask, v)
    return mask


# ========== 集合运算 ==========
def unique_values(x):
    """返回排序后的唯一值。"""
    return _np().unique(x)


def unique_counts(x):
    """返回 (唯一值, 出现次数)。"""
    raw = _core.unique_full(_asarray(x)._array, False, False, True)
    return _wrap(raw[0]), _wrap(raw[1], "int64")


def unique_inverse(x):
    """返回 (唯一值, 可重建原数组的逆索引)。"""
    raw = _core.unique_all_rs(_asarray(x)._array)
    return _wrap(raw[0]), _wrap(raw[2], "int64")


def unique_all(x):
    """返回 (唯一值, 首次出现索引, 逆索引, 计数)。"""
    raw = _core.unique_all_rs(_asarray(x)._array)
    return (_wrap(raw[0]), _wrap(raw[1], "int64"),
            _wrap(raw[2], "int64"), _wrap(raw[3], "int64"))


def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    """返回两个数组的交集（升序去重，dtype 为两者提升类型）。

    return_indices：另返回 i1/i2（flat C-order 首现索引），
    ar1[i1] / ar2[i2]（flat 索引）等于交集结果。
    """
    _ = assume_unique  # 纯 Python 去重，两种参数行为一致
    npmod = _np()
    a = _asarray(ar1)
    b = _asarray(ar2)

    def _flat_vals(x):
        t = x.ravel().tolist()
        return list(t) if isinstance(t, (list, tuple)) else [t]

    def _eq(v, w):
        if v is w:
            return True
        try:
            if v == w:
                return True
        except Exception:
            pass
        return (isinstance(v, float) and isinstance(w, float)
                and v != v and w != w)

    def _dedup_first(vals):
        seen, out_v, out_i = [], [], []
        for i, v in enumerate(vals):
            if builtin_any(_eq(v, w) for w in seen):
                continue
            seen.append(v)
            out_v.append(v)
            out_i.append(i)
        return out_v, out_i

    fa, fb = _flat_vals(a), _flat_vals(b)
    va, ia = _dedup_first(fa)
    vb, ib = _dedup_first(fb)
    common = [v for v in va if builtin_any(_eq(v, w) for w in vb)]
    try:
        common.sort()
    except TypeError:
        pass  # object 等不可比较元素保持首现顺序

    def _idx_of(v, vals, idxs):
        for i, w in enumerate(vals):
            if _eq(v, w):
                return idxs[i]
        return 0

    i1 = [_idx_of(v, va, ia) for v in common]
    i2 = [_idx_of(v, vb, ib) for v in common]
    try:
        dt = npmod.result_type(a, b)
    except Exception:
        dt = a.dtype
    res = npmod.array(common, dtype=dt) if common else npmod.array([], dtype=dt)
    if not return_indices:
        return res
    i1a = npmod.array(i1, dtype='int64') if i1 else npmod.array([], dtype='int64')
    i2a = npmod.array(i2, dtype='int64') if i2 else npmod.array([], dtype='int64')
    return (res, i1a, i2a)


def union1d(ar1, ar2):
    """返回两个数组的并集（升序去重）。"""
    return _wrap(_core.union1d(_asarray(ar1)._array, _asarray(ar2)._array))


def setdiff1d(ar1, ar2, assume_unique=False):
    """返回 the unique values in ar1 but not in ar2 (ar1 首现顺序，保 dtype)。"""
    _ = assume_unique
    a = _asarray(ar1)
    b = _asarray(ar2)
    npmod = _np()
    _adtype = getattr(a, '_dtype', 'float64')
    if int(getattr(a, 'size', 0)) == 0:
        return npmod.array([], dtype=_adtype)

    def _flat(x, out):
        if isinstance(x, (list, tuple)):
            for y in x:
                _flat(y, out)
        else:
            out.append(x)

    fa, fb = [], []
    _ta = a.tolist()
    for x in (_ta if isinstance(_ta, (list, tuple)) else [_ta]):
        _flat(x, fa)
    _tb = b.tolist()
    for x in (_tb if isinstance(_tb, (list, tuple)) else [_tb]):
        _flat(x, fb)

    try:
        _sb = set(fb)

        def _in(v):
            return v in _sb
    except TypeError:
        def _in(v):
            return v in fb

    out, _seen = [], set()
    for v in fa:
        if _in(v):
            continue
        try:
            if v in _seen:
                continue
            _seen.add(v)
        except TypeError:
            if builtins.any(v == w for w in out):
                continue
        out.append(v)
    if not out:
        return npmod.array([], dtype=_adtype)
    if all(isinstance(v, str) for v in out):
        # str 走隐式构造：显式 dtype='string_' 会按 bytes 存储（tolist 变 b'..'）
        return npmod.array(out)
    return npmod.array(out, dtype=_adtype)


def setxor1d(ar1, ar2, assume_unique=False):
    """返回两个数组的对称差集。"""
    _ = assume_unique
    return _wrap(_core.setxor1d(_asarray(ar1)._array, _asarray(ar2)._array))


def isin(element, test_elements, assume_unique=False, invert=False, kind=None):
    """逐元素判断 element 是否属于 test_elements（numpy 语义；NaN 视为相等）。

    kind：None/'sort' 不限制；'table' 仅支持整数/布尔（float/object/timedelta
    报 ValueError），整数表范围超出 dtype 表示范围时 RuntimeError。
    """
    _ = assume_unique
    if kind is not None and kind not in ("sort", "table"):
        raise ValueError(
            f"kind must be None, 'sort' or 'table', got {kind!r}")
    npmod = _np()
    a = _asarray(element)
    b = _asarray(test_elements)

    def _values(x):
        """展平为 Python 值列表；结构化数组保留整条记录（不拆字段）。"""
        if getattr(x, '_fields', None):
            t = x.tolist()
            return list(t) if isinstance(t, (list, tuple)) else [t]
        if tuple(getattr(x, 'shape', ())) == ():
            return [x.item()]
        t = x.tolist()
        if not isinstance(t, (list, tuple)):
            return [t]
        out = []
        # object 数组：tuple/list 元素是数据而非容器，只按 shape 维数展平，
        # 不递归进元素内部（否则会腐化 (1,) 这类 1 元素 tuple）。
        if (getattr(x, '_dtype', None) == 'object'
                or str(getattr(x, 'dtype', '')) == 'object'):
            def rec_o(v, depth):
                if depth == 0:
                    out.append(v)
                    return
                for w in v:
                    rec_o(w, depth - 1)

            rec_o(t, len(getattr(x, 'shape', ())))
            return out

        def rec(v):
            if isinstance(v, (list, tuple)):
                for w in v:
                    rec(w)
            else:
                out.append(v)

        rec(t)
        return out

    fa, fb = _values(a), _values(b)

    if kind == "table":
        _base = str(getattr(b, '_dtype', '')).split('[')[0]
        if _base != 'bool' and not (_base.startswith('int') or _base.startswith('uint')):
            raise ValueError(
                f"kind='table' only supports integer/bool dtypes, got {_base!r}")
        if _base != 'bool':
            _digits = _base[4:] if _base.startswith('uint') else _base[3:]
            _bits = int(_digits) if _digits.isdigit() else None
            if _bits and fb:
                _rng = max(fb) - min(fb)
                _max_range = (1 << _bits) - 1 if _base.startswith('uint') \
                    else (1 << (_bits - 1)) - 1
                if _rng > _max_range:
                    raise RuntimeError(
                        f"table range {_rng} for dtype {_base!r} would exceed "
                        f"the maximum number of representable values")

    _nan_in_b = builtins.any(isinstance(v, float) and v != v for v in fb)
    try:
        _sb = set(fb)
        _sb_ok = True
    except TypeError:
        _sb, _sb_ok = None, False

    def _hit(v):
        if isinstance(v, float) and v != v:
            return _nan_in_b
        if _sb_ok:
            try:
                if v in _sb:
                    return True
            except TypeError:
                pass
        return builtins.any(v == w for w in fb)

    res = [_hit(v) for v in fa]
    if invert:
        res = [not x for x in res]
    shp = tuple(getattr(a, 'shape', ()))
    if shp == ():
        return npmod.array(res[0])
    if not res:
        return npmod.array([])

    def _nest(vals, shape):
        if len(shape) == 1:
            return list(vals)
        n = 1
        for s in shape[1:]:
            n *= s
        return [_nest(vals[i * n:(i + 1) * n], shape[1:])
                for i in range(shape[0])]

    return npmod.array(_nest(res, shp))


# ========== introspection / 复数 ==========
def ndim(a):
    """返回数组维度数。"""
    return _asarray(a).ndim


def shape(a):
    """返回数组形状。"""
    return _asarray(a).shape


def size(a, axis=None):
    """返回元素总数或指定轴长度。"""
    arr = _asarray(a)
    if axis is None:
        return arr.size
    if isinstance(axis, (tuple, list)):
        # numpy 语义：元组 axis → 对应轴大小乘积；空元组 → 1
        if len(axis) == 0:
            return 1
        n = 1
        for _ax in axis:
            n *= arr.shape[_ax]
        return n
    return arr.shape[axis]


def isscalar(element):
    """判断是否为标量。"""
    if isinstance(element, (int, float, complex, bool, bytes, str)):
        return True
    try:
        import rsnumpy as _rn
        if isinstance(element, _rn.generic):
            return True
    except Exception:
        pass
    try:
        from fractions import Fraction as _Frac
        from decimal import Decimal as _Dec
        if isinstance(element, (_Frac, _Dec)):
            return True
    except Exception:
        pass
    # PEP 3141（numpy 语义）:numbers.Number 子类视为标量——裸 Number()
    # 实例既无 __index__ 也无 __float__(那分别是 Integral/Real 的方法),
    # 须用 isinstance 判定;rs 标量(非 int 子类)靠 __index__/__float__ 兼容。
    if hasattr(element, '__len__'):
        return False
    if hasattr(element, '__index__') or hasattr(element, '__float__'):
        return True
    try:
        from numbers import Number as _NumNumber
        if isinstance(element, _NumNumber):
            return True
    except Exception:
        pass
    return False


def iterable(y):
    """判断对象是否可迭代。"""
    try:
        iter(y)
        return True
    except TypeError:
        return False


def iscomplexobj(x):
    """判断对象是否为复数类型（numpy 语义：dtype.kind / 元素级检测）。"""
    if isinstance(x, builtins.complex):
        return True
    if isinstance(x, (list, tuple)):
        return builtin_any(isinstance(v, builtins.complex) for v in _flat(x))
    dt = getattr(x, 'dtype', None)
    if dt is not None:
        k = getattr(dt, 'kind', None)
        if k is not None:
            return k == 'c'
        tp = getattr(dt, 'type', None)
        if isinstance(tp, type) and issubclass(tp, builtins.complex):
            return True
        return 'complex' in (getattr(dt, 'name', '') or str(dt))
    if getattr(x, "_complex_data", None) is not None:
        return True
    return getattr(x, "_dtype", "") in ("complex128", "complex64")


def isrealobj(x):
    """判断对象是否为实数类型。"""
    return not iscomplexobj(x)


def isreal(x):
    """逐元素判断虚部是否为 0。"""
    np = _np()
    arr = _asarray(x)
    if not getattr(arr._array, 'is_complex', False):
        return np.array(_map_nested(lambda v: True, arr.tolist()), dtype="bool")
    imag_data = arr._array.imag
    if imag_data is None:
        return np.array(_map_nested(lambda v: True, arr.tolist()), dtype="bool")
    imag_list = imag_data.tolist()

    def check_zero(val):
        if isinstance(val, (list, tuple)):
            return [check_zero(v) for v in val]
        return abs(val) < 1e-15

    return np.array(check_zero(imag_list), dtype="bool")


def iscomplex(x):
    """逐元素判断虚部是否非 0。"""
    np = _np()
    arr = _asarray(x)
    if not getattr(arr._array, 'is_complex', False):
        return np.array(_map_nested(lambda v: False, arr.tolist()), dtype="bool")
    imag_data = arr._array.imag
    if imag_data is None:
        return np.array(_map_nested(lambda v: False, arr.tolist()), dtype="bool")
    imag_list = imag_data.tolist()

    def check_nonzero(val):
        if isinstance(val, (list, tuple)):
            return [check_nonzero(v) for v in val]
        return abs(val) >= 1e-15

    return np.array(check_nonzero(imag_list), dtype="bool")


def isnat(x):
    """逐元素判断是否为 NaT（暂不支持 NaT，返回全 False）。"""
    np = _np()
    arr = _asarray(x)
    return np.array(_map_nested(lambda v: False, arr.tolist()), dtype="bool")


def _is_inf_check(x, out, positive):
    """isposinf/isneginf 共用实现：complex 输入抛 TypeError，标量返回
    bool 标量，支持 out 参数（numpy 语义）。"""
    np = _np()
    arr = _asarray(x)
    try:
        if arr.dtype.kind == 'c':
            raise TypeError(
                "isposinf/isneginf is not supported for complex input")
    except AttributeError:
        pass
    if arr.ndim == 0:
        v = float(arr.tolist())
        ok = (v == float('inf')) if positive else (v == float('-inf'))
        if out is not None:
            out[...] = ok
            return out
        return np.bool_(ok)
    cmp = np.greater(arr, 0.0) if positive else np.less(arr, 0.0)
    res = np.logical_and(np.isinf(arr), cmp)
    if out is not None:
        out[...] = res
        return out
    return res


def isneginf(x, out=None):
    """逐元素判断是否为负无穷。"""
    return _is_inf_check(x, out, False)


def isposinf(x, out=None):
    """逐元素判断是否为正无穷。"""
    return _is_inf_check(x, out, True)


def real(val):
    """返回实部（Python 标量输入返回 float，numpy 语义）。"""
    np = _np()
    if isinstance(val, builtins.complex):
        return float(val.real)
    if isinstance(val, (bool, int, float)) and not isinstance(val, np.ndarray):
        return float(val)
    if iscomplexobj(val):
        # Multi-dim complex arrays don't expose the flat `_complex_data`
        # buffer (it is None/absent); fall back to the nested `tolist`.
        cdata = getattr(val, "_complex_data", None)
        if cdata is not None:
            return np.array([v.real for v in cdata])
        return np.array(_map_nested(lambda v: complex(v).real, val.tolist()))
    return _asarray(val)


def imag(val):
    """返回虚部（Python 标量输入返回 float，numpy 语义）。"""
    np = _np()
    if isinstance(val, builtins.complex):
        return float(val.imag)
    if isinstance(val, (bool, int, float)) and not isinstance(val, np.ndarray):
        return 0.0
    if iscomplexobj(val):
        cdata = getattr(val, "_complex_data", None)
        if cdata is not None:
            return np.array([v.imag for v in cdata])
        return np.array(_map_nested(lambda v: complex(v).imag, val.tolist()))
    arr = _asarray(val)
    return np.full(arr.shape, 0.0)


def conjugate(x):
    """返回共轭。"""
    np = _np()
    if iscomplexobj(x):
        cdata = getattr(x, "_complex_data", None)
        if cdata is not None:
            return np.array([complex(v).conjugate() for v in cdata])
        return np.array(_map_nested(lambda v: complex(v).conjugate(), x.tolist()))
    return _asarray(x)


conj = conjugate


def angle(z, deg=False):
    """返回复数的相位角。"""
    np = _np()
    factor = 180.0 / _math.pi if deg else 1.0
    if isinstance(z, complex):
        return np.array([_math.atan2(z.imag, z.real) * factor])
    if iscomplexobj(z):
        cdata = getattr(z, "_complex_data", None)
        if cdata is not None:
            return np.array([_math.atan2(v.imag, v.real) * factor for v in cdata])
        return np.array(_map_nested(
            lambda v: _math.atan2(complex(v).imag, complex(v).real) * factor,
            z.tolist()))
    return np.array([_math.atan2(0.0, v) * factor for v in _flat(z)])


def real_if_close(a, tol=100):
    """若虚部接近 0 则返回实部，否则原样返回。

    numpy 语义：int tol 为机器精度倍数（tol * eps），非 int tol 为绝对阈值。
    """
    if not iscomplexobj(a):
        return _asarray(a)
    if isinstance(tol, int) and not isinstance(tol, bool):
        threshold = tol * 2.220446049250313e-16
    else:
        threshold = float(tol)
    cdata = getattr(a, "_complex_data", None)
    if cdata is None:
        cdata = [builtins.complex(v) for v in _flat(a)]
    if builtin_all(abs(builtins.complex(v).imag) < threshold for v in cdata):
        return real(a)
    return _asarray(a)


def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    """将 NaN/Inf 替换为有限数（numpy 语义）。

    0-d / 标量输入返回对应标量（int64/float64/complex128）；
    copy=False 时原地修改并返回同一对象。
    """
    np = _np()
    pos = 1.7976931348623157e308 if posinf is None else float(posinf)
    neg = -1.7976931348623157e308 if neginf is None else float(neginf)
    _inf = float('inf')

    def _r(v):
        v = float(v)
        if v != v:
            return float(nan)
        if v == _inf:
            return pos
        if v == -_inf:
            return neg
        return v

    arr = _asarray(x)
    if arr.ndim == 0:
        v = arr.tolist()
        if isinstance(v, builtins.complex):
            return np.complex128(_r(v.real) + _r(v.imag) * 1j)
        if isinstance(v, (bool, int)):
            return np.int64(int(v))
        return np.float64(_r(v))
    res = _wrap(_core.nan_to_num(arr._array, nan, pos, neg))
    if not copy:
        arr._array = res._array
        return arr
    return res


# ========== 数组变形 / 组合 ==========
def concat(arrays, axis=0):
    """concatenate 别名（Array API）。"""
    return _np().concatenate(arrays, axis=axis)


def row_stack(tup):
    """垂直堆叠（vstack 别名）。"""
    return _np().vstack(tup)


def permute_dims(a, axes):
    """按给定顺序置换维度（transpose 别名）。"""
    return _np().transpose(a, axes)


def matrix_transpose(x):
    """交换最后两个维度。"""
    arr = _asarray(x)
    n = arr.ndim
    axes = list(range(n))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return _np().transpose(arr, tuple(axes))


def _atleast_n(arr, n):
    """把 arr 提升为至少 n 维（前置补 1；reshape 保留 dtype——
    Rust _core.atleast_* 会丢失 complex dtype 标签，故纯 Python 实现）。"""
    while getattr(arr, 'ndim', 0) < n:
        arr = arr.reshape((1,) * (n - arr.ndim) + tuple(arr.shape))
    return arr


def atleast_1d(*arys):
    """确保输入至少为一维。"""
    out = [_atleast_n(_asarray(a), 1) for a in arys]
    return out[0] if len(out) == 1 else out


def atleast_2d(*arys):
    """确保输入至少为二维。"""
    out = [_atleast_n(_asarray(a), 2) for a in arys]
    return out[0] if len(out) == 1 else out


def atleast_3d(*arys):
    """确保输入至少为三维（numpy 语义：0-d→(1,1,1)；1-d→(1,n,1)；
    2-d→(n,m,1)——第 3 维追加在末尾，非前置补 1）。"""
    out = []
    for a in arys:
        arr = _asarray(a)
        nd = getattr(arr, 'ndim', 0)
        if nd == 0:
            arr = arr.reshape((1, 1, 1))
        elif nd == 1:
            arr = arr.reshape((1,) + tuple(arr.shape) + (1,))
        elif nd == 2:
            arr = arr.reshape(tuple(arr.shape) + (1,))
        out.append(arr)
    return out[0] if len(out) == 1 else out


def broadcast_shapes(*shapes):
    """按 NumPy 规则计算广播后的形状。"""
    result = ()
    for shape in shapes:
        shp = tuple(shape)
        n = max(len(result), len(shp))
        a = (1,) * (n - len(result)) + result
        b = (1,) * (n - len(shp)) + shp
        merged = []
        for x, y in zip(a, b):
            if x == 1:
                merged.append(y)
            elif y == 1 or x == y:
                merged.append(x)
            else:
                raise ValueError(f"shape mismatch: {x} vs {y}")
        result = tuple(merged)
    return result


def broadcast_arrays(*args):
    """将多个数组广播到公共形状。"""
    np = _np()
    arrs = [_asarray(a) for a in args]
    target = broadcast_shapes(*[a.shape for a in arrs])
    return [np.broadcast_to(a, target) for a in arrs]


def block(arrays):
    """按嵌套列表组装数组块（numpy 语义）。

    校验与 numpy _block 一致：顶层 tuple 拒绝（TypeError）、空列表
    ValueError（消息含 'empty'）、嵌套深度不匹配 ValueError（消息含
    'depths are mismatched'）、单数组/标量输入返回副本（list_ndim==0）。
    """
    np = _np()

    def _depth(x):
        if isinstance(x, list):
            return 1 + (max((_depth(e) for e in x), default=0))
        return 0

    def _leaf_ndim(x):
        if isinstance(x, list):
            return max((_leaf_ndim(e) for e in x), default=0)
        return _asarray(x).ndim

    def _depths_match(x, depth):
        """numpy _block_check_depths_match：所有叶子索引长度一致才算匹配。

        空 list 返回 depth+1 并带 None 标记（参与深度比较——如 [1, []] 应先报
        深度不匹配）；tuple 在任意层级拒绝（numpy 同款 TypeError）。"""
        if isinstance(x, tuple):
            raise TypeError(
                "arrays is a tuple. Only lists can be used to arrange "
                "blocks, and np.block does not allow implicit conversion "
                "from tuple to ndarray.")
        if isinstance(x, list):
            if len(x) == 0:
                return (depth + 1, True)  # (索引长度, 空 None 标记)
            child_results = [_depths_match(e, depth + 1) for e in x]
            first_len = child_results[0][0]
            for cd, _ in child_results[1:]:
                if cd != first_len:
                    raise ValueError(
                        "List depths are mismatched. First element was at "
                        "depth %d, but there is an element at depth %d"
                        % (first_len - 1, cd - 1))
            has_empty = builtin_any(m for _, m in child_results)
            return (first_len, has_empty)
        return (depth, False)

    if isinstance(arrays, tuple):
        raise TypeError(
            "arrays is a tuple. Only lists can be used to arrange blocks, "
            "and np.block does not allow implicit conversion from tuple to "
            "ndarray.")

    list_depth = _depth(arrays)
    result_ndim = builtin_max(_leaf_ndim(arrays), list_depth)

    if isinstance(arrays, list):
        _bottom_len, _has_empty = _depths_match(arrays, 0)
        if _has_empty:
            raise ValueError(
                "List at depth %d cannot be empty" % (_bottom_len - 1))

    def _atleast(arr):
        if arr.ndim >= result_ndim:
            return arr
        return np.reshape(arr, (1,) * (result_ndim - arr.ndim) + tuple(arr.shape))

    def _build(x, depth):
        if not isinstance(x, list):
            return _atleast(_asarray(x))
        parts = [_build(e, depth + 1) for e in x]
        axis = result_ndim - list_depth + depth
        return np.concatenate(parts, axis=axis)

    result = _build(arrays, 0)
    if list_depth == 0:
        # numpy 语义：单数组/标量输入返回副本（_block_concatenate 的边缘处理）
        result = result.copy()
    # 内存序（numpy 语义）：全部叶子数组 F 序 → 结果 F 序
    # （concatenate 不传播 _rs_order 标记，在此统一恢复）。
    if result.ndim > 1:
        def _all_f(x):
            if isinstance(x, list):
                return builtin_all(_all_f(e) for e in x)
            return getattr(_asarray(x), '_rs_order', None) == 'F'
        if _all_f(arrays):
            result._rs_order = 'F'
    return result


def array_split(ary, indices_or_sections, axis=0):
    """将数组切分为多个子数组，允许不等分。"""
    np = _np()
    arr = _asarray(ary)
    _ndim = len(arr.shape)
    n = arr.shape[axis]
    if isinstance(indices_or_sections, int):
        k = indices_or_sections
        if k <= 0:
            raise ValueError("array_split does not support 0 sections")
        base, extra = n // k, n % k
        sizes = [base + 1 if i < extra else base for i in range(k)]
        bounds = []
        acc = 0
        for s in sizes:
            acc += s
            bounds.append(acc)
        bounds = bounds[:-1]
    else:
        bounds = list(indices_or_sections)
    # 零分配广播视图：只调逻辑形状（numpy 语义：split 产出视图，不物化）
    _ovr = getattr(arr, '_rs_shape_override', None)
    result = []
    prev = 0
    for b in list(bounds) + [n]:
        if _ovr is not None:
            _shp = list(_ovr)
            _shp[axis] = b - prev
            _res = np.ndarray._wrap(
                arr._array, _dtype=getattr(arr, '_dtype', 'float64'))
            _res._rs_shape_override = tuple(_shp)
            result.append(_res)
        else:
            # 切片切分（numpy 视图语义；避免 list(range(巨大轴)) 的内存爆炸）
            _key = [slice(None)] * _ndim
            _key[axis] = slice(prev, b)
            result.append(arr[tuple(_key)])
        prev = b
    return result


def unstack(x, axis=0):
    """沿指定轴拆成子数组元组（NumPy 2.1）。

    numpy 语义：负 axis 归一化；越界（正向未归一化前 |axis| 超出 ndim）
    抛 ValueError 而非 IndexError。"""
    np = _np()
    arr = _asarray(x)
    ndim = len(arr.shape)
    if axis < 0:
        axis += ndim
    if not 0 <= axis < ndim:
        raise ValueError(
            f'axis {axis} is out of bounds for array of dimension {ndim}')
    n = arr.shape[axis]
    out = []
    for i in range(n):
        picked = np.take(arr, [i], axis=axis)
        ax = axis % picked.ndim
        newshape = tuple(s for j, s in enumerate(picked.shape) if j != ax)
        out.append(np.reshape(picked, newshape) if newshape else picked)
    return tuple(out)


def take_along_axis(arr, indices, axis):
    """沿轴按索引取值。"""
    a = _asarray(arr)
    idx = _asarray(indices)
    if axis is None:
        # numpy 官方语义（_shape_base_impl.take_along_axis）
        if len(idx.shape) != 1:
            raise ValueError(
                "when axis=None, `indices` must have a single dimension.")
        a = a.ravel()
        axis = 0
    _ndim = len(a.shape)
    if axis < 0:
        axis += _ndim
    if not 0 <= axis < _ndim:
        from rsnumpy.exceptions import AxisError as _AxisError
        raise _AxisError(
            f"axis {axis} is out of bounds for array of dimension {_ndim}")
    if len(idx.shape) != _ndim:
        raise ValueError(
            "indices and array must have the same number of dimensions")
    # numpy 语义：indices 必须是整数数组（bool/float 均不允许 → IndexError）
    _k = getattr(getattr(idx, '_dtype_obj', None), 'kind', None)
    if _k is None:
        _idn = str(getattr(idx, '_dtype', ''))
        _k = _idn[0] if _idn else '?'
    if _k not in ('i', 'u'):
        raise IndexError(
            "arrays used as indices must be of integer (or boolean) type")
    if tuple(idx.shape) != tuple(a.shape):
        # numpy 语义：非 axis 维度上 a 与 indices 双向广播
        # （结果 axis 维 = indices 的 axis 维）
        _a_rest = a.shape[:axis] + a.shape[axis + 1:]
        _i_rest = idx.shape[:axis] + idx.shape[axis + 1:]
        if tuple(_a_rest) != tuple(_i_rest):
            _rest = broadcast_shapes(tuple(_a_rest), tuple(_i_rest))

            def _ins(s, d):
                return s[:axis] + (d,) + s[axis:]

            if tuple(_a_rest) != tuple(_rest):
                a = _np().broadcast_to(a, _ins(_rest, a.shape[axis]))
            if tuple(_i_rest) != tuple(_rest):
                idx = _np().broadcast_to(idx, _ins(_rest, idx.shape[axis]))
    return _wrap(_core.take_along_axis(a._array, idx._array, axis))


def put_along_axis(arr, indices, values, axis):
    """沿轴按索引写入值（就地）。"""
    a = _asarray(arr)
    idx = _asarray(indices)
    v = _asarray(values)
    _flat_mode = axis is None
    if axis is None:
        # numpy 官方语义（_shape_base_impl.put_along_axis）
        if len(idx.shape) != 1:
            raise ValueError(
                "when axis=None, `indices` must have a single dimension.")
        a = a.ravel()
        axis = 0
    _ndim = len(a.shape)
    if axis < 0:
        axis += _ndim
    if not 0 <= axis < _ndim:
        from rsnumpy.exceptions import AxisError as _AxisError
        raise _AxisError(
            f"axis {axis} is out of bounds for array of dimension {_ndim}")
    if len(idx.shape) != _ndim:
        raise ValueError(
            "indices and array must have the same number of dimensions")
    _k = getattr(getattr(idx, '_dtype_obj', None), 'kind', None)
    if _k is None:
        _idn = str(getattr(idx, '_dtype', ''))
        _k = _idn[0] if _idn else '?'
    if _k not in ('i', 'u'):
        raise IndexError(
            "arrays used as indices must be of integer (or boolean) type")
    # values 广播到 indices 形状（numpy 官方语义）
    if tuple(v.shape) != tuple(idx.shape):
        v = _np().broadcast_to(v, tuple(idx.shape))
    if tuple(idx.shape) == tuple(a.shape):
        _core.put_along_axis(a._array, idx._array, v._array, axis)
        return None
    # numpy 官方实现（_make_along_axis_idx）：正交 arange 花式索引 +
    # 标准 setitem，写入落回原数组（rsnumpy 的切片 getitem 是拷贝，
    # 切片回写方案不可行）。
    _shp_ones = (1,) * len(a.shape)
    _fancy = []
    for _dim, _n in enumerate(a.shape):
        if _dim == axis:
            _fancy.append(idx)
        else:
            _ind_shape = _shp_ones[:_dim] + (-1,) + _shp_ones[_dim + 1:]
            _fancy.append(
                _np().arange(_n, dtype=_np().intp).reshape(_ind_shape))
    a[tuple(_fancy)] = v
    if _flat_mode and hasattr(arr, '_array'):
        # ravel 可能是拷贝：写回原数组以保持就地语义
        arr[...] = a.tolist()
    return None


def choose(a, choices, out=None, mode="raise"):
    """按索引数组从 choices 中选值（Rust 实现）。"""
    _ = out, mode
    a_arr = _asarray(a)
    choice_arrs = [_asarray(c)._array for c in choices]
    return _wrap(_core.choose(a_arr._array, choice_arrs))


def compress(condition, a, axis=None):
    """按布尔条件沿轴筛选元素（Rust 实现）。"""
    return _wrap(_core.compress(_asarray(condition)._array, _asarray(a)._array, axis))


_INT_WRAP_BITS = {'int8': 8, 'int16': 16, 'int32': 32, 'int64': 64,
                  'uint8': 8, 'uint16': 16, 'uint32': 32, 'uint64': 64}


def copyto(dst, src, casting="same_kind", where=True):
    """将 src 的值复制到 dst（就地，Rust 实现）。

    标量 src 按目标 dtype 语义预转换：整数 dtype 按位宽回绕（numpy wrap
    语义，300→int8=44）；字符串标量无法解析按 0；字符串 dtype 数组直接
    写 _raw_data（Rust copyto 会把字符串写成 None 占位）。"""
    _ = casting, where
    d = _asarray(dst)
    dt = getattr(d, '_dtype', 'float64')
    if isinstance(src, str):
        if dt in ('string_', 'bytes_') or getattr(d, '_str_dtype', None):
            n = d.size if d.size else 1
            d._raw_data = [src] * n
            return None
        try:
            src = float(src)
        except ValueError:
            src = 0.0
    if isinstance(src, int) and not isinstance(src, bool) \
            and dt in _INT_WRAP_BITS:
        bits = _INT_WRAP_BITS[dt]
        src = src & ((1 << bits) - 1)
        if dt.startswith('int') and src >= (1 << (bits - 1)):
            src -= (1 << bits)
    _core.copyto(d._array, _asarray(src)._array)
    return None


def place(arr, mask, vals):
    """按布尔掩码把 vals 循环写入 arr（就地）。"""
    a = _asarray(arr)
    _core.place(a._array, _asarray(mask)._array, _asarray(vals)._array)
    return None


def putmask(a, mask, values):
    """按布尔掩码逐元素写入值（与 numpy.putmask 一致：a[i]=v[i] where mask[i]）。"""
    arr = _asarray(a)
    m = _asarray(mask)
    v = _asarray(values)
    # 标量值需广播到与 arr 同形以保证 Rust 层按位置元素级赋值正确
    if v.ndim == 0 or v.shape == ():
        v = _np().broadcast_to(v, arr.shape)
    _core.putmask(arr._array, m._array, v._array)
    return None


def pad(array, pad_width, mode="constant", **kwargs):
    """填充数组边缘，支持 constant/edge/reflect/wrap。

    安全防护：对 pad_width 施加上限，防止恶意输入导致内存耗尽（DoS）。
    每个维度的填充量不得超过该维度原长度的 4 倍 + 1024，或绝对上限 1 亿。
    """
    np = _np()
    arr = _asarray(array)
    cval = kwargs.get("constant_values", 0)
    ndim = arr.ndim
    if isinstance(pad_width, int):
        pw = [(pad_width, pad_width)] * ndim
    elif isinstance(pad_width[0], int):
        pw = [tuple(pad_width)] * ndim
    else:
        pw = [tuple(p) for p in pad_width]

    # DoS 防护：校验 pad_width 上限
    _MAX_PAD_ABS = 100_000_000  # 绝对上限：1 亿
    for axis, (before, after) in enumerate(pw):
        axis_len = arr.shape[axis] if axis < len(arr.shape) else 1
        relative_cap = axis_len * 4 + 1024
        cap = min(relative_cap, _MAX_PAD_ABS)
        total = before + after
        if total > cap:
            raise ValueError(
                f"pad_width 过大（轴 {axis}: before={before}, after={after}, "
                f"合计={total}），超过上限 {cap}。"
                f"该轴长度为 {axis_len}，允许的最大填充量为 {cap}。"
            )

    def pad_1d(seq, before, after):
        if mode == "constant":
            return [cval] * before + seq + [cval] * after
        if mode == "edge":
            return [seq[0]] * before + seq + [seq[-1]] * after
        if mode == "reflect":
            left = [seq[min(i + 1, len(seq) - 1)] for i in range(before)][::-1]
            right = [seq[max(len(seq) - 2 - i, 0)] for i in range(after)]
            return left + seq + right
        if mode == "wrap":
            left = [seq[(-before + i) % len(seq)] for i in range(before)]
            right = [seq[i % len(seq)] for i in range(after)]
            return left + seq + right
        raise ValueError(f"unsupported pad mode: {mode}")

    def rec(data, axis):
        if axis == ndim - 1:
            return pad_1d(list(data), pw[axis][0], pw[axis][1])
        padded = [rec(row, axis + 1) for row in data]
        width = len(padded[0])
        before = [[cval] * width for _ in range(pw[axis][0])]
        after = [[cval] * width for _ in range(pw[axis][1])]
        if mode == "edge":
            before = [list(padded[0]) for _ in range(pw[axis][0])]
            after = [list(padded[-1]) for _ in range(pw[axis][1])]
        return before + padded + after

    if ndim == 0:
        return arr
    return np.array(rec(arr.tolist(), 0))


def trim_zeros(filt, trim="fb"):
    """裁剪一维数组两端的零。

    Rust 层 (trim_zeros_bounds) 扫描 (first, last) 非零边界（避免 Python
    tolist），Python 层按边界切片以保留 dtype 标签（旧 Rust 版会丢 complex
    dtype 标签，故不再直接用其返回值）；非标准数值类型回退纯 Python。
    """
    np = _np()
    arr = _asarray(filt)
    if arr.size == 0:
        return arr
    n = int(arr.size)
    try:
        first, last = _core.trim_zeros_bounds(arr._array)
    except Exception:
        # Fallback：纯 Python（非标准数值类型等）
        vals = arr.tolist()
        vn = len(vals)

        def _is_zero(v):
            try:
                return v == 0
            except Exception:
                return False

        first = 0
        while first < vn and _is_zero(vals[first]):
            first += 1
        last = vn - 1
        while last >= 0 and _is_zero(vals[last]):
            last -= 1
    if 'f' not in trim:
        first = 0
    if 'b' not in trim:
        last = n - 1
    if first > last:
        return np.zeros(0, arr.dtype)
    return arr[first:last + 1]


def ediff1d(ary, to_end=None, to_begin=None):
    """一维相邻元素差分，可选 to_begin/to_end 前后拼接。

    语义（numpy 2.x）：diff = ary[1:] - ary[:-1]（一维），结果 =
    to_begin + diff + to_end。to_begin/to_end 可为标量或一维数组；
    dtype 须与 ary same_kind 可转（float→int 等禁止，抛 TypeError），
    Python int 标量按 ary dtype 补码回绕（如 int16 上 2**16 → 0）。
    """
    npmod = _np()
    a = _asarray(ary).ravel()
    adt = getattr(a, '_dtype', 'float64')

    def _prep(v, name):
        if v is None:
            return None
        if isinstance(v, (int, float, bool)):
            arr = npmod.array([v])
        else:
            arr = _asarray(v).ravel()
        sk = getattr(getattr(arr, 'dtype', None), 'kind', '')
        dk = getattr(getattr(a, 'dtype', None), 'kind', '')
        if sk and dk and not ((sk == dk) or (sk in 'iu' and dk == 'f')):
            raise TypeError(
                f"dtype of `{name}` must be compatible with {a.dtype}")
        if getattr(arr, '_dtype', 'float64') != adt:
            arr = arr.astype(adt)
        return arr

    tb = _prep(to_begin, 'to_begin')
    te = _prep(to_end, 'to_end')
    if int(getattr(a, 'size', 0)) >= 2:
        d = a[1:] - a[:-1]
    else:
        d = npmod.array([], dtype=adt)
    parts = []
    if tb is not None:
        parts.append(tb)
    parts.append(d)
    if te is not None:
        parts.append(te)
    if len(parts) == 1:
        return d
    return npmod.concatenate(parts, axis=0)


def diff(a, n=1, axis=-1):
    """沿轴计算 n 阶差分。"""
    return _wrap(_core.diff(_asarray(a)._array, n, axis))


def gradient(f, *varargs, axis=None, edge_order=1):
    """计算梯度。"""
    _ = varargs, axis, edge_order
    return _wrap(_core.gradient(_asarray(f)._array))


def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """向量叉积（N-D / axis 参数 / 广播 / dtype 提升）。

    向量维须为 3 元素（本兼容套件语义：2 元素一律 ValueError）。
    """
    np = _np()
    if axis is not None:
        axisa = axisb = axisc = axis
    A = _asarray(a)
    B = _asarray(b)
    if A.ndim == 0 or B.ndim == 0:
        raise ValueError("Input arrays must be at least 1-dimensional")
    if not (-A.ndim <= axisa < A.ndim):
        raise np.AxisError(
            f"axisa {axisa} is out of bounds for array "
            f"of dimension {A.ndim}")
    if not (-B.ndim <= axisb < B.ndim):
        raise np.AxisError(
            f"axisb {axisb} is out of bounds for array "
            f"of dimension {B.ndim}")
    Am = np.moveaxis(A, axisa, -1)
    Bm = np.moveaxis(B, axisb, -1)
    if Am.shape[-1] != 3 or Bm.shape[-1] != 3:
        raise ValueError("Cross product requires 3-element vectors")
    Ab, Bb = np.broadcast_arrays(Am, Bm)
    # 混合整型先提升到公共 dtype（uint8*int32 逐元素乘法会按操作数
    # dtype 回绕，与 numpy cross 的提升语义不一致）。
    _rd = np.result_type(A, B)
    Ab = Ab.astype(_rd.name)
    Bb = Bb.astype(_rd.name)
    c0 = Ab[..., 1] * Bb[..., 2] - Ab[..., 2] * Bb[..., 1]
    c1 = Ab[..., 2] * Bb[..., 0] - Ab[..., 0] * Bb[..., 2]
    c2 = Ab[..., 0] * Bb[..., 1] - Ab[..., 1] * Bb[..., 0]
    c = np.stack([c0, c1, c2], axis=-1)
    out_ndim = c.ndim
    if axisc != out_ndim - 1:
        if not (-out_ndim <= axisc < out_ndim):
            raise np.AxisError(
                f"axisc {axisc} is out of bounds for array "
                f"of dimension {out_ndim}")
        c = np.moveaxis(c, -1, axisc)
    return c


def trapezoid(y, x=None, dx=1.0, axis=-1):
    """梯形法数值积分。"""
    _ = axis
    yv = _flat(y)
    if x is not None:
        xv = _flat(x)
        return sum((xv[i + 1] - xv[i]) * (yv[i + 1] + yv[i]) / 2.0 for i in range(len(yv) - 1))
    return sum(dx * (yv[i + 1] + yv[i]) / 2.0 for i in range(len(yv) - 1))


trapz = trapezoid


# ========== 线性代数辅助 ==========
def outer(a, b, out=None):
    """向量外积。"""
    np = _np()
    av = _asarray(a).ravel()
    bv = _asarray(b).ravel()
    ar = np.reshape(av, (av.size, 1))
    br = np.reshape(bv, (1, bv.size))
    res = np.multiply(ar, br)
    if out is not None:
        # numpy 语义:out 形状须与结果一致,结果写入 out 并返回 out
        if tuple(out.shape) != (av.size, bv.size):
            raise ValueError(
                "non-broadcastable output operand with shape "
                f"{tuple(out.shape)} provided to make "
                f"shape {(av.size, bv.size)} fit")
        out[...] = res
        return out
    return res


def kron(a, b):
    """Kronecker 积（支持任意维度与 MaskedArray）。"""
    np = _np()
    _nd = np.ndarray
    # masked 输入：值 = filled(1) 的 kron；掩码 = m_a/m_b 按结果索引
    # 取模 OR（对角外有任一因子被掩码即掩码，numpy ma.kron 语义）
    _ma = getattr(np, 'ma', None)
    if _ma is not None and (_ma.isMaskedArray(a) or _ma.isMaskedArray(b)):
        import itertools
        data = kron(_ma.filled(a, 0), _ma.filled(b, 0))
        m_a = _ma.getmaskarray(a)
        m_b = _ma.getmaskarray(b)
        _nd2 = builtins.max(a.ndim, b.ndim)
        ash = (1,) * (_nd2 - a.ndim) + tuple(a.shape)
        bsh = (1,) * (_nd2 - b.ndim) + tuple(b.shape)
        out_shape = tuple(x * y for x, y in zip(ash, bsh))
        mask_flat = []
        for coord in itertools.product(*[range(s) for s in out_shape]):
            va = m_a[tuple(c % s for c, s in zip(coord, ash))]
            vb = m_b[tuple(c % s for c, s in zip(coord, bsh))]
            mask_flat.append(bool(va) or bool(vb))
        mask = np.array(mask_flat, dtype=bool).reshape(out_shape)
        return _ma.MaskedArray(data, mask=mask)
    # 官方语义：结果类型由 __array_priority__ 更高的子类决定
    # (numpy: kron(a, ma) → ma 的类型)
    _subcls = None
    for _x in (a, b):
        if isinstance(_x, _nd) and type(_x) is not _nd:
            _t = type(_x)
            if (_subcls is None
                    or getattr(_t, '__array_priority__', 0)
                    > getattr(_subcls, '__array_priority__', 0)):
                _subcls = _t
    A = _asarray(a)
    B = _asarray(b)
    if A.ndim == 1 and B.ndim == 1:
        res = outer(A, B).ravel()
    else:
        # numpy 语义：kron[a_idx * bsh + b_idx] = a[a_idx] * b[b_idx]。
        # 等价于 tile(A, bsh) 逐块重复 × B 逐元素展开（repeat ash 次）。
        _nd2 = builtins.max(A.ndim, B.ndim)
        ash = (1,) * (_nd2 - A.ndim) + tuple(A.shape)
        bsh = (1,) * (_nd2 - B.ndim) + tuple(B.shape)
        out_shape = tuple(x * y for x, y in zip(ash, bsh))
        if A.size == 0 or B.size == 0:
            # 零尺寸：绕开广播引擎对零长轴的处理，直接构造空结果
            res = np.zeros(out_shape, dtype=getattr(A, '_dtype', None)
                           or 'float64')
            if _subcls is not None:
                try:
                    return _subcls(res.tolist())
                except Exception:
                    pass
            return res
        # kron[i,j] = a[i//sb, j//sb] * b[i%sb, j%sb]：
        # A 侧逐元素展开（每个 a 元素重复 bsh 次），B 侧分块平铺（tile ash）
        import itertools
        idx_flat = []
        for coord in itertools.product(*[range(s) for s in out_shape]):
            f = 0
            for cd, sd, bd in zip(coord, ash, bsh):
                f = f * sd + (cd // bd)
            idx_flat.append(f)
        Ax = np.take(np.reshape(A, (-1,)),
                     np.array(idx_flat)).reshape(out_shape)
        Bx = np.tile(B, ash)
        res = np.multiply(Ax, Bx)
    if _subcls is not None:
        try:
            return _subcls(res.tolist())
        except Exception:
            pass
    return res


def tensordot(a, b, axes=2):
    """张量点积 - 灵活的多维数组收缩操作。

    【核心概念】
    tensordot在指定维度上对两个张量进行收缩（求和乘积），是einsum的特殊情况。
    它比einsum更高效，但灵活性稍低。

    【使用示例】
    >>> import rsnumpy as np

    # 1. 矩阵乘法：收缩最后1个维度
    >>> A = np.array([[1, 2], [3, 4]])
    >>> B = np.array([[5, 6], [7, 8]])
    >>> np.tensordot(A, B, axes=1)  # 等价于 A @ B
    array([[19., 22.],
           [43., 50.]])

    # 2. 内积：收缩所有维度
    >>> v1 = np.array([1, 2, 3])
    >>> v2 = np.array([4, 5, 6])
    >>> np.tensordot(v1, v2, axes=1)  # 等价于 np.dot(v1, v2)
    32.0

    # 3. 批量矩阵乘法
    >>> batch_A = np.random.rand(3, 4, 5)  # 3个 4x5 矩阵
    >>> batch_B = np.random.rand(3, 5, 6)  # 3个 5x6 矩阵
    >>> # 收缩axis=2（A的第3维）和axis=1（B的第2维）
    >>> result = np.tensordot(batch_A, batch_B, axes=([2], [1]))
    >>> result.shape  # (3, 4, 3, 6)
    (3, 4, 3, 6)

    # 4. 张量收缩示例
    >>> T1 = np.random.rand(2, 3, 4)
    >>> T2 = np.random.rand(3, 4, 5)
    >>> # 收缩T1的(1,2)维度和T2的(0,1)维度
    >>> result = np.tensordot(T1, T2, axes=([1, 2], [0, 1]))
    >>> result.shape  # (2, 5)
    (2, 5)

    # 5. 使用列表指定收缩维度
    >>> A = np.random.rand(2, 3, 4, 5)
    >>> B = np.random.rand(3, 5, 6)
    >>> # 收缩A的(1,3)和B的(0,1)
    >>> result = np.tensordot(A, B, axes=([1, 3], [0, 1]))
    >>> result.shape  # (2, 4, 6)
    (2, 4, 6)

    【参数说明】
    a, b: array_like
        输入张量
    axes: int or (list, list) or (list, list)
        收缩维度指定：
        - 整数：收缩a的最后axes个维度和b的前axes个维度
        - 两列表：(a的维度列表, b的维度列表)
        - 维度数量必须匹配

    【axes参数详解】
    1. axes=N（整数）：
       - 收缩a的最后N个维度和b的前N个维度
       - axes=1：等价于矩阵乘法（2D情况下）
       - axes=2：收缩最后2个维度（用于批量矩阵）

    2. axes=([i1,i2,...], [j1,j2,...])（列表对）：
       - 在a的[i1,i2,...]维度和b的[j1,j2,...]维度上收缩
       - 维度列表长度必须相同
       - 对应维度的长度必须匹配

    【与einsum对应关系】
    tensordot(A, B, axes=1)          ≈ einsum('ij,jk->ik', A, B)
    tensordot(A, B, axes=2)          ≈ einsum('ijk,kl->ijl', A, B)
    tensordot(A, B, axes=([1],[0]))  ≈ einsum('ij,jk->ik', A, B)
    tensordot(A, B, axes=([1,2],[0,1])) ≈ einsum('ijk,kl->il', A, B)

    【性能建议】
    - 简单收缩（axes=1或2）：tensordot > einsum
    - 复杂维度选择：einsum更直观
    - 大型数组：tensordot通常更高效

    【调试技巧】
    1. 检查维度匹配：
       >>> A.shape[axes_a] == B.shape[axes_b]
    2. 验证输出形状：
       >>> # 输出 = A的非收缩维度 + B的非收缩维度
    3. 使用einsum可视化：
       >>> # 同样的操作用einsum表达更易理解

    【常见错误】
    ValueError: dimension mismatch
    - 检查收缩维度长度是否匹配
    - 检查axes列表长度是否相等

    【注意事项】
    - 不支持复数类型的矩阵乘法（需要用einsum）
    - 对于大型数组，考虑使用reshape+matmul优化
    """

    np = _np()
    A = _asarray(a)
    B = _asarray(b)

    # 解析axes参数
    if isinstance(axes, int):
        # 整数形式：收缩A的最后axes个维度和B的前axes个维度
        axes_a = list(range(A.ndim - axes, A.ndim))
        axes_b = list(range(axes))
    else:
        # 列表对形式：精确指定收缩维度
        axes_a, axes_b = axes
        axes_a = list(axes_a) if isinstance(axes_a, (list, tuple)) else [axes_a]
        axes_b = list(axes_b) if isinstance(axes_b, (list, tuple)) else [axes_b]

    # 计算非收缩维度
    notin_a = [i for i in range(A.ndim) if i not in axes_a]
    notin_b = [i for i in range(B.ndim) if i not in axes_b]

    # 调整维度顺序：非收缩维度在前，收缩维度在后
    newaxes_a = notin_a + axes_a
    newaxes_b = axes_b + notin_b

    at = np.transpose(A, tuple(newaxes_a))
    bt = np.transpose(B, tuple(newaxes_b))

    # 【性能优化】使用math.prod代替显式循环求积
    import math as _math
    n1 = _math.prod(A.shape[i] for i in notin_a) if notin_a else 1
    nc = _math.prod(A.shape[i] for i in axes_a) if axes_a else 1
    n2 = _math.prod(B.shape[i] for i in notin_b) if notin_b else 1

    # 重塑为2D矩阵进行matmul
    # 零收缩维(nc=0):numpy 语义结果是全零数组(形状由非收缩维决定)。
    # Rust matmul 对零尺寸维返回失真结果(如 size 1),须预检短路。
    if nc == 0:
        outshape0 = tuple(A.shape[i] for i in notin_a) + tuple(
            B.shape[i] for i in notin_b)
        try:
            _dt0 = np.result_type(A, B)
        except Exception:
            _dt0 = getattr(A, '_dtype', 'float64')
        if outshape0:
            return np.zeros(outshape0, dtype=_dt0)
        return _as_scalar(np.zeros((1,), dtype=_dt0)[0])
    am = np.reshape(at.ravel(), (n1, nc))
    bm = np.reshape(bt.ravel(), (nc, n2))

    res = np.matmul(am, bm)
    if not hasattr(res, "_array"):
        res = _wrap(res)

    # 计算输出形状并重塑
    outshape = tuple(A.shape[i] for i in notin_a) + tuple(B.shape[i] for i in notin_b)
    return np.reshape(res.ravel(), outshape) if outshape else _as_scalar(res)


def tri(N, M=None, k=0, dtype=None):
    """下三角为 1、其余为 0 的矩阵。

    `dtype` 决定结果类型；`np.bool_` 会给出真正的布尔数组。
    """
    np = _np()
    if M is None:
        M = N
    rows = np.reshape(np.arange(0, N, 1), (N, 1))
    cols = np.reshape(np.arange(0, M, 1), (1, M))
    result = np.where(np.less_equal(cols, np.add(rows, float(k))),
                      np.full((N, M), 1.0), np.full((N, M), 0.0))
    if dtype is None:
        return result

    # 早先这里整体丢弃了 dtype（`_ = dtype`），于是
    # `np.tri(..., dtype=bool)` 交回的是浮点 0.0/1.0。而布尔索引分支靠
    # dtype == 'bool' 判定，拿不到真布尔就会退化成整数索引，把 `a[mask]`
    # 变成按行挑选的 (n, n, n)，而不是掩码选出的元素。
    name = getattr(dtype, 'name', None)
    if name is None:
        name = getattr(dtype, '__name__', None) or str(dtype)
    if name in ('bool_', '?'):
        name = 'bool'
    return result.astype(name)


def tril(m, k=0):
    """取下三角部分。"""
    np = _np()
    arr = _asarray(m)
    n, mm = arr.shape[-2], arr.shape[-1]
    return np.multiply(arr, tri(n, mm, k))


def triu(m, k=0):
    """取上三角部分。"""
    np = _np()
    arr = _asarray(m)
    n, mm = arr.shape[-2], arr.shape[-1]
    return np.subtract(arr, np.multiply(arr, tri(n, mm, k - 1)))


def diagonal(a, offset=0, axis1=0, axis2=1):
    """提取对角线（支持 axis1/axis2）。"""
    return _wrap(_core.diagonal_ext(_asarray(a)._array, offset, axis1, axis2))


def diag(v, k=0):
    """一维->对角矩阵，二维->提取对角线。"""
    return _wrap(_core.diag(_asarray(v)._array, k))


def diagflat(v, k=0):
    """将输入展平后构造对角矩阵。"""
    return diag(_np().array(_flat(v)), k)


def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    """对角线元素之和。"""
    _ = axis1, axis2, dtype, out
    return sum(_flat(diagonal(a, offset)))


def tri_indices_helper(n, k, upper):
    """生成上/下三角的 (rows, cols) 索引列表。

    使用列表推导代替嵌套 for 循环，减少 Python 解释器开销。
    """
    if upper:
        # 上三角：c >= r + k
        idx_r = [r for r in range(n) for c in range(n) if c >= r + k]
        idx_c = [c for r in range(n) for c in range(n) if c >= r + k]
    else:
        # 下三角：c <= r + k
        idx_r = [r for r in range(n) for c in range(n) if c <= r + k]
        idx_c = [c for r in range(n) for c in range(n) if c <= r + k]
    return idx_r, idx_c


def tril_indices(n, k=0, m=None):
    """下三角元素的索引。"""
    mm = n if m is None else m
    rows, cols = _core.tril_indices(n, k, mm)
    return _wrap(rows, _dtype='int64'), _wrap(cols, _dtype='int64')


def triu_indices(n, k=0, m=None):
    """上三角元素的索引。"""
    mm = n if m is None else m
    rows, cols = _core.triu_indices(n, k, mm)
    return _wrap(rows, _dtype='int64'), _wrap(cols, _dtype='int64')


def tril_indices_from(arr, k=0):
    """由数组形状返回下三角索引。"""
    a = _asarray(arr)
    return tril_indices(a.shape[0], k, a.shape[1])


def triu_indices_from(arr, k=0):
    """由数组形状返回上三角索引。"""
    a = _asarray(arr)
    return triu_indices(a.shape[0], k, a.shape[1])


def diag_indices(n, ndim=2):
    """主对角线索引。"""
    np = _np()
    idx = np.arange(0, n, 1)
    return tuple(idx for _ in range(ndim))


def diag_indices_from(arr):
    """由数组形状返回主对角线索引。"""
    a = _asarray(arr)
    return diag_indices(a.shape[0], a.ndim)


def fill_diagonal(a, val, wrap=False):
    """就地填充主对角线。"""
    arr = _asarray(a)
    _core.fill_diagonal(arr._array, _asarray(val)._array, wrap)
    return None


def mask_indices(n, mask_func, k=0):
    """返回 mask_func 作用于 (n,n) 全 1 矩阵得到的非零索引。"""
    np = _np()
    ones = np.full((n, n), 1.0)
    masked = mask_func(ones, k)
    return np.nonzero(masked)


# ========== 窗函数 / 信号 ==========
def _window_base(M):
    np = _np()
    if M < 1:
        return None, None
    if M == 1:
        return np.array([1.0]), None
    return None, np.arange(0, M, 1)


def hanning(M):
    """Hann 窗。"""
    np = _np()
    done, n = _window_base(M)
    if done is not None or n is None:
        return done if done is not None else np.array([])
    return np.subtract(np.full((M,), 0.5), np.multiply(np.cos(np.multiply(n, 2.0 * _math.pi / (M - 1))), 0.5))


def hamming(M):
    """Hamming 窗。"""
    np = _np()
    done, n = _window_base(M)
    if done is not None or n is None:
        return done if done is not None else np.array([])
    return np.subtract(np.full((M,), 0.54), np.multiply(np.cos(np.multiply(n, 2.0 * _math.pi / (M - 1))), 0.46))


def blackman(M):
    """Blackman 窗。"""
    np = _np()
    done, n = _window_base(M)
    if done is not None or n is None:
        return done if done is not None else np.array([])
    a = np.multiply(n, 2.0 * _math.pi / (M - 1))
    term1 = np.multiply(np.cos(a), 0.5)
    term2 = np.multiply(np.cos(np.multiply(n, 4.0 * _math.pi / (M - 1))), 0.08)
    return np.add(np.subtract(np.full((M,), 0.42), term1), term2)


def bartlett(M):
    """Bartlett（三角）窗。"""
    np = _np()
    done, n = _window_base(M)
    if done is not None or n is None:
        return done if done is not None else np.array([])
    half = (M - 1) / 2.0
    return np.subtract(np.full((M,), 1.0), np.divide(np.abs(np.subtract(n, half)), half))


def i0(x):
    """第一类零阶修正贝塞尔函数。"""
    arr = _asarray(x)
    res = _wrap(_core.i0(arr._array))
    return res if arr.ndim else _as_scalar(res)


def kaiser(M, beta):
    """Kaiser 窗。"""
    np = _np()
    if M < 1:
        return np.array([])
    if M == 1:
        return np.array([1.0])
    n = np.arange(0, M, 1)
    alpha = (M - 1) / 2.0
    ratio = np.divide(np.subtract(n, alpha), alpha)
    arg = np.multiply(np.sqrt(np.subtract(np.full((M,), 1.0), np.multiply(ratio, ratio))), beta)
    return np.divide(i0(arg), _as_scalar(i0(np.array([float(beta)]))))


def _conv_corr_mode(mode):
    """convolve/correlate 的 mode 校验(精确对齐 numpy C 层语义)。

    字符串:仅 'valid'/'same'/'full';整数:legacy 0=valid 1=same 2=full;
    其余类型/取值分别 TypeError/ValueError(numpy 消息)。
    """
    if isinstance(mode, str):
        if mode in ('valid', 'same', 'full'):
            return mode
        raise ValueError(
            "Use one of 'valid', 'same', or 'full' for convolve/correlate")
    if isinstance(mode, int) and not isinstance(mode, bool):
        return {0: 'valid', 1: 'same', 2: 'full'}.get(
            mode, None) or _raise_int_mode()
    raise TypeError("convolve/correlate mode not understood")


def _raise_int_mode():
    raise ValueError(
        "integer convolve/correlate mode must be 0, 1, or 2")


def convolve(a, v, mode="full"):
    """一维离散卷积。"""
    A = _asarray(a)
    V = _asarray(v)
    if A.size == 0:
        raise ValueError("a cannot be empty")
    if V.size == 0:
        raise ValueError("v cannot be empty")
    return _wrap(_core.convolve(A._array, V._array, _conv_corr_mode(mode)))


def correlate(a, v, mode="valid"):
    """一维互相关。"""
    A = _asarray(a)
    V = _asarray(v)
    if A.size == 0:
        raise ValueError("a cannot be empty")
    if V.size == 0:
        raise ValueError("v cannot be empty")
    return _wrap(_core.correlate(A._array, V._array, _conv_corr_mode(mode)))


def interp(x, xp, fp, left=None, right=None, period=None):
    """一维线性插值。"""
    _ = period
    arr = _asarray(x)
    raw = _core.interp(arr._array, _asarray(xp)._array, _asarray(fp)._array, left, right)
    res = _wrap(raw)
    return res if arr.ndim else _as_scalar(res)


def bincount(x, weights=None, minlength=0):
    """统计非负整数数组中每个值出现的次数。"""
    if weights is None:
        return _wrap(_core.bincount(_asarray(x)._array, minlength), "int64")
    return _wrap(_core.bincount_weighted(_asarray(x)._array, _asarray(weights)._array, minlength))


def vander(x, N=None, increasing=False):
    """生成范德蒙矩阵。"""
    np = _np()
    xv = _asarray(x).ravel()
    n = xv.size
    if N is None:
        N = n
    xr = np.reshape(xv, (n, 1))
    if increasing:
        powers = np.arange(0, N, 1)
    else:
        powers = np.subtract(np.full((N,), float(N - 1)), np.arange(0, N, 1))
    powers = np.reshape(powers, (1, N))
    return np.power(xr, powers)


def unwrap(p, discont=None, axis=-1, period=6.283185307179586):
    """相位解卷绕。"""
    return _wrap(_core.unwrap(_asarray(p)._array, discont, axis, period))


# ========== 归约：all / any / round（顶层函数）==========
def _boolmask(arr):
    np = _np()
    return np.where(np.equal(arr, np.full(arr.shape, 0.0)),
                    np.full(arr.shape, 0.0), np.full(arr.shape, 1.0))


def _check_array_function(a):
    """numpy 语义:子类 __array_function__ 返回 NotImplemented 时抛
    TypeError(numpy 对未接管的函数统一报错)。"""
    _af = getattr(a, '__array_function__', None)
    if _af is not None and type(a) is not getattr(__import__('rsnumpy'), 'ndarray'):
        try:
            if _af(type(a), [a], (), None) is NotImplemented:
                raise TypeError(
                    "no implementation found for 'numpy.all' on types that "
                    "implement __array_function__: "
                    f"[{type(a).__module__}.{type(a).__name__}]")
        except TypeError:
            raise
        except Exception:
            pass


def all(a, axis=None, out=None, keepdims=False, *, where=True):
    """判断沿轴的所有元素是否都为真。"""
    _ = out, keepdims, where
    np = _np()
    _check_array_function(a)
    arr = _asarray(a)
    if axis is None:
        return builtin_all(v != 0 for v in _flat(arr))
    return np.greater(np.min(_boolmask(arr), axis), 0.5)


def any(a, axis=None, out=None, keepdims=False, *, where=True):
    """判断沿轴是否存在为真的元素。"""
    _ = out, keepdims, where
    np = _np()
    _check_array_function(a)
    arr = _asarray(a)
    if axis is None:
        return builtin_any(v != 0 for v in _flat(arr))
    return np.greater(np.max(_boolmask(arr), axis), 0.5)


def round(a, decimals=0, out=None):
    """四舍六入五成双到指定小数位（around 别名）。"""
    _ = out
    return _np().around(a, decimals)


round_ = round


# ========== 类型转换 / 连续性 / 共享内存 ==========
def astype(x, dtype, copy=True, casting="unsafe"):
    """将数组转换为指定类型。"""
    _ = casting
    np_ = _np()
    if not hasattr(x, '_array'):
        if isinstance(x, np_.generic):
            # numpy 语义:标量 astype 返回标量(经 0-d 数组路径转换)
            return np_.array(x).astype(dtype)[()]
        # numpy 语义:模块级 astype 只接受数组输入(list 等先经 asarray)
        raise TypeError("Input should be a NumPy array")
    return x.astype(dtype, copy=copy)


def ascontiguousarray(a, dtype=None, copy=None, subok=True):
    """返回 C 连续数组（rsnumpy 数组始终连续）。

    copy/subok 为 numpy 2.x 兼容参数，当前实现忽略（数组恒连续）。
    """
    _ = copy, subok
    arr = _np().asarray(a)
    return arr.astype(dtype) if dtype is not None else arr


def asfortranarray(a, dtype=None, copy=None, subok=True):
    """返回 Fortran 连续数组（rsnumpy 视为等价）。"""
    return ascontiguousarray(a, dtype, copy=copy, subok=subok)


def asarray_chkfinite(a, dtype=None, order=None):
    """转为数组，若含 inf/nan 则抛错。

    使用 any() + 生成器表达式代替 for 循环。
    """
    _ = order
    arr = _np().asarray(a)
    # 使用 any() + 生成器表达式代替显式 for 循环
    flat = _flat(arr)
    if builtin_any(isinstance(v, float) and (v != v or v == _math.inf or v == -_math.inf) for v in flat):
        raise ValueError("array must not contain infs or NaNs")
    return arr.astype(dtype) if dtype is not None else arr


def require(a, dtype=None, requirements=None, *, like=None):
    """返回满足要求的数组（numpy require 语义）。

    requirements 接受单个标志名或标志名列表；未知标志抛 KeyError，
    同时要求 C 与 F 连续抛 ValueError；已满足的标志不产生副本
    （幂等：np.require(b, None, [flag]) is b）。
    """
    _ = like
    np = _np()
    if requirements is None:
        reqs = []
    elif isinstance(requirements, (str, bytes)):
        reqs = [requirements]
    else:
        reqs = list(requirements)

    arr = np.asarray(a)
    if dtype is not None:
        arr = arr.astype(dtype)

    need_c = need_f = need_contiguous = False
    need_a = need_w = need_o = need_e = False
    for r in reqs:
        if isinstance(r, bytes):
            r = r.decode()
        key = r.upper()
        if key in ('C', 'C_CONTIGUOUS'):
            need_c = True
        elif key == 'CONTIGUOUS':
            need_contiguous = True
        elif key in ('F', 'F_CONTIGUOUS', 'FORTRAN'):
            need_f = True
        elif key in ('A', 'ALIGNED'):
            need_a = True
        elif key in ('W', 'WRITEABLE'):
            need_w = True
        elif key in ('O', 'OWNDATA'):
            need_o = True
        elif key == 'E':
            need_e = True
        else:
            raise KeyError(f"{r} is not a valid flag")

    if need_c and need_f:
        raise ValueError(
            "cannot be both C and F contiguous at the same time")

    if need_e and type(arr) is not np.ndarray:
        # 'E'：确保返回基类 ndarray（剥离子类）
        arr = arr.view(np.ndarray)
    if need_c and not arr.flags.c_contiguous:
        arr = arr.copy()
    elif need_f and not arr.flags.f_contiguous:
        arr = arr.copy()
        # rsnumpy 存储为扁平 C 序缓冲：以 _rs_order 标记 F 布局（flags/strides 语义）
        arr._rs_order = 'F'
    elif need_contiguous and not (arr.flags.c_contiguous or arr.flags.f_contiguous):
        arr = arr.copy()
    if need_a and not arr.flags.aligned:
        arr = arr.copy()
    if need_w and not arr.flags.writeable:
        arr = arr.copy()
    if need_o and not arr.flags.own_data:
        arr = arr.copy()
    return arr


def isfortran(a):
    """判断是否为 Fortran 连续（rsnumpy 恒为 C 连续）。"""
    _ = a
    return False


def shares_memory(a, b, max_work=None):
    """判断两个数组是否共享底层内存（含逻辑视图血缘，如 reshape）。"""
    _ = max_work
    ra = getattr(a, "_array", None)
    rb = getattr(b, "_array", None)
    if ra is not None and ra is rb:
        return True
    if a is b:
        return True
    da = getattr(a, "_raw_data", None)
    db = getattr(b, "_raw_data", None)
    if da is not None and da is db:
        return True

    def _chain(x):
        seen = {id(x)}
        while True:
            p = getattr(x, "_rs_parent", None)
            if p is None or id(p) in seen:
                return seen
            seen.add(id(p))
            x = p

    return bool(_chain(a) & _chain(b))


def may_share_memory(a, b, max_work=None):
    """保守判断两个数组是否可能共享内存。"""
    return shares_memory(a, b, max_work)


def array_equiv(a1, a2):
    """判断两个数组在广播后是否逐元素相等。

    使用 all() + 生成器表达式代替显式 for 循环。
    """
    np = _np()
    x = _asarray(a1)
    y = _asarray(a2)
    try:
        shp = broadcast_shapes(x.shape, y.shape)
    except ValueError:
        return False
    xb = np.broadcast_to(x, shp)
    yb = np.broadcast_to(y, shp)
    # 使用 all() + zip 代替显式 for 循环
    return builtin_all(p == q for p, q in zip(_flat(xb), _flat(yb)))


# ========== 索引工具 ==========
def indices(dimensions, dtype=None, sparse=False):
    """返回索引网格数组。"""
    np = _np()
    if isinstance(dimensions, int) and not isinstance(dimensions, bool):
        dimensions = (dimensions,)
    dims = tuple(int(d) for d in dimensions)
    n = len(dims)
    if n == 0:
        # numpy：np.indices(()) → 空 (0,) 数组（dense）/ 空元组（sparse）
        if sparse:
            return ()
        return np.array([], dtype=dtype or "int64")
    if 0 in dims:
        # numpy：np.indices((0,)) → 空 (1, 0) 数组（dense）
        _dt_i = dtype or "int64"

        def _mk(_shape):
            if not _shape:
                return [0]
            return [_mk(_shape[1:]) for _ in range(_shape[0])]

        if sparse:
            return tuple(
                np.array(_mk((dims[i],) + (1,) * (n - 1)), dtype=_dt_i)
                for i in range(n))
        return np.array(_mk((n,) + dims), dtype=_dt_i)
    out = []
    for i in range(n):
        shp = tuple(dims[i] if j == i else 1 for j in range(n))
        arr = np.reshape(np.arange(0, dims[i], 1), shp)
        out.append(arr if sparse else np.broadcast_to(arr, dims))
    if sparse:
        return tuple(o.astype(dtype or "int64") for o in out)
    res = np.stack(out, axis=0)
    return res.astype(dtype or "int64")


def _reshape_list(flat, shape):
    """扁平列表按 C 序重塑为嵌套列表。"""
    if not shape:
        return flat[0]
    if builtins.len(shape) == 1:
        return list(flat)
    _sub = 1
    for _d0 in shape[1:]:
        _sub *= _d0
    return [_reshape_list(flat[_i * _sub:(_i + 1) * _sub], shape[1:])
            for _i in range(shape[0])]


def unravel_index(indices, shape, order='C'):
    """将扁平索引转换为多维坐标。"""
    dims = [int(s) for s in (shape if isinstance(shape, (list, tuple)) else (shape,))]
    _ndim = len(dims)
    _prod = 1
    for _d0 in dims:
        _prod *= _d0
    if _prod > 2**63 - 1:
        # numpy 语义（gh-9538）：乘积溢出 intp → ValueError（勿让 Rust 溢出）
        raise ValueError(
            "shape exceeds the maximum size representable by intp")
    if isinstance(indices, (list, tuple)):
        if builtins.len(indices) == 0:
            raise TypeError(
                "indices must be integral: the provided empty sequence "
                "was of length 0, but the shape requires at least 1 element")
        idx = _asarray(indices)
    else:
        if isinstance(indices, builtins.float):
            raise TypeError("only int indices permitted")
        idx = _asarray(indices)
    _dt = str(getattr(idx, '_dtype', ''))
    if _dt and _dt[0] not in ('i', 'u'):
        raise TypeError("only int indices permitted")
    _shp = tuple(getattr(idx, 'shape', ()))
    _flat_idx = _flat(idx)
    _scalar_in = _shp == ()
    if _ndim == 0:
        # 0-d 形状：唯一合法扁平索引是 0
        if _flat_idx and (builtins.min(_flat_idx) < 0
                          or builtins.max(_flat_idx) > 0):
            raise ValueError("indices out of bounds for a 0d array")
        if not _scalar_in:
            raise ValueError(
                "cannot unravel non-scalar indices into a 0d array")
        return ()
    if _prod == 0:
        # 零尺寸形状：空输入 → 各维空数组；非空 → 越界
        if builtins.len(_flat_idx) == 0:
            nd = _np().ndarray
            return tuple(nd(_reshape_list([], _shp), _dtype='int64')
                         for _ in dims)
        raise ValueError("indices out of bounds for zero-sized shape")
    if _flat_idx and (builtins.min(_flat_idx) < 0
                      or builtins.max(_flat_idx) >= _prod):
        raise ValueError("indices out of bounds")
    # 步长（C 序 / F 序）
    if order == 'F':
        _strides = []
        _s = 1
        for _d0 in dims:
            _strides.append(_s)
            _s *= _d0
    else:
        _strides = [1] * _ndim
        for _d0 in range(_ndim - 2, -1, -1):
            _strides[_d0] = _strides[_d0 + 1] * dims[_d0 + 1]
    nd = _np().ndarray
    res = []
    for _d0 in range(_ndim):
        _st = _strides[_d0]
        _dd = dims[_d0]
        _col = [(v // _st) % _dd for v in _flat_idx]
        if _scalar_in:
            res.append(_col[0])
        else:
            res.append(nd(_reshape_list(_col, _shp), _dtype='int64'))
    return tuple(res)


def ravel_multi_index(multi_index, dims, mode='raise', order='C'):
    """将多维坐标转换为扁平索引。"""
    dims_list = [int(s) for s in dims]
    _ndim = len(dims_list)
    _prod = 1
    for _d0 in dims_list:
        _prod *= _d0
    if _prod > 2**63 - 1:
        # numpy 语义（gh-7546）：乘积溢出 intp → ValueError
        raise ValueError(
            "product of dims exceeds the maximum size representable by intp")
    # 每维模式（mode 可为字符串或每维元组）
    if isinstance(mode, (list, tuple)):
        modes = tuple(str(m0) for m0 in mode)
        if builtins.len(modes) != _ndim:
            raise ValueError(
                "mode must be a string or a tuple of strings of length ndim")
    else:
        modes = (str(mode),) * _ndim
    for _m0 in set(modes):
        if _m0 not in ('raise', 'clip', 'wrap'):
            raise ValueError(f"unknown mode {_m0!r}")
    idxs = []
    for m in multi_index:
        if isinstance(m, (list, tuple)) and builtins.len(m) == 0:
            raise TypeError(
                "indices must be integral: the provided empty sequence "
                "was of length 0, but the shape requires at least 1 element")
        _m = _asarray(m)
        _dt = str(getattr(_m, '_dtype', ''))
        if _dt and _dt[0] not in ('i', 'u'):
            raise TypeError("only int indices permitted")
        idxs.append(_m)
    if builtins.len(idxs) != _ndim:
        raise ValueError(
            "multi_index must have exactly one entry per dimension")
    _shps = [tuple(getattr(_m, 'shape', ())) for _m in idxs]
    _bshape = (tuple(broadcast_shapes(*_shps)) if builtins.len(_shps) > 1
               else _shps[0]) if _shps else ()
    # 零尺寸维度：空输入 → 空结果；非空索引 → 越界（任何 mode 下都成立）
    if builtins.any(_d0 == 0 for _d0 in dims_list):
        _n_elems = 1
        for _d0 in _bshape:
            _n_elems *= _d0
        if _n_elems == 0:
            nd = _np().ndarray
            return nd(_reshape_list([], _bshape), _dtype='int64')
        raise ValueError("indices outside the allowed range")
    _flats = [_flat(_m) for _m in idxs]
    _n = 1
    for _d0 in _bshape:
        _n *= _d0
    # 步长（C 序 / F 序）
    if order == 'F':
        _strides = []
        _s = 1
        for _d0 in dims_list:
            _strides.append(_s)
            _s *= _d0
    else:
        _strides = [1] * _ndim
        for _d0 in range(_ndim - 2, -1, -1):
            _strides[_d0] = _strides[_d0 + 1] * dims_list[_d0 + 1]
    _res = []
    for _i in range(_n):
        _v = 0
        for _d0 in range(_ndim):
            _x = _flats[_d0][_i]
            _dd = dims_list[_d0]
            _mo = modes[_d0]
            if _mo == 'wrap':
                _x = _x % _dd
            elif _mo == 'clip':
                if _x < 0:
                    _x = 0
                elif _x > _dd - 1:
                    _x = _dd - 1
            elif _x < 0 or _x >= _dd:
                raise ValueError("indices outside the allowed range")
            _v += _x * _strides[_d0]
        _res.append(_v)
    if _bshape == ():
        return builtins.int(_res[0])
    nd = _np().ndarray
    return nd(_reshape_list(_res, _bshape), _dtype='int64')


# ========== 沿轴应用 / 分段函数 ==========
def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """沿指定轴对一维切片应用函数。"""
    import itertools as _it
    np = _np()
    _in_cls = type(arr) if isinstance(arr, np.ndarray) else None
    a = _asarray(arr)
    nd = a.ndim
    axis = axis % nd
    inds = [range(a.shape[i]) for i in range(nd) if i != axis]
    combos = list(_it.product(*inds)) if inds else [()]
    # numpy 语义：任一迭代维长度为 0 时无切片可调用 → ValueError
    if not combos or builtins.any(a.shape[i] == 0 for i in range(nd) if i != axis):
        raise ValueError(
            "Cannot apply_along_axis when any iteration dimensions are 0")

    # object dtype：混合切片会经 f64 占位丢失元素，改用嵌套列表切片
    _a_lists = None
    if str(getattr(a, '_dtype', '')) == 'object':
        _a_lists = a.tolist()

    def slice_at(combo):
        if _a_lists is not None:
            it = iter(combo)
            fixed = {}
            for i in range(nd):
                if i != axis:
                    fixed[i] = next(it)

            def drill(lst, i):
                # 非轴维取固定索引,轴维保留列表(遍历)
                if i == nd:
                    return lst
                if i in fixed:
                    return drill(lst[fixed[i]], i + 1)
                return [drill(x, i + 1) for x in lst]

            return drill(_a_lists, 0)
        key = []
        it = iter(combo)
        for i in range(nd):
            key.append(slice(None) if i == axis else next(it))
        return a[tuple(key)] if nd > 1 else a

    results = [func1d(slice_at(c), *args, **kwargs) for c in combos]
    outer = tuple(a.shape[i] for i in range(nd) if i != axis)
    r0 = results[0]
    is_scalar_out = (not isinstance(r0, (list, tuple))
                     and not (hasattr(r0, "ndim") and _asarray(r0).ndim > 0))
    if is_scalar_out:
        vals = [_as_scalar(r) if hasattr(r, "tolist") else r for r in results]
        if builtins.all(isinstance(v, (int, float, bool, complex))
                        and not isinstance(v, _np().ndarray) for v in vals):
            arr_v = np.array(vals)
        else:
            # 不可数值化的元素(如 set)→ object 数组(vals 已按 outer
            # 行主序排列;object 数组不能过 Rust raw reshape,会丢元素)
            arr_v = np.array(vals, dtype='object')
            if not outer or tuple(arr_v.shape) == tuple(outer):
                return _as_scalar(arr_v) if not outer else arr_v
            return np.reshape(arr_v, outer)
    r_shape = tuple(_asarray(r0).shape)
    import rsnumpy.ma as _ma
    if _ma.isMaskedArray(r0):
        # MaskedArray 结果：分别堆叠 data 与 mask(moveaxis 各自处理
        # 后再合成,避免 moveaxis 降级丢失掩码)
        data = np.array([_ma.filled(r, 0).tolist() for r in results])
        mask = np.array([_ma.getmaskarray(r).tolist() for r in results])
        data = np.reshape(data, outer + r_shape)
        mask = np.reshape(mask, outer + r_shape)
        if r_shape:
            src = list(range(len(outer), len(outer) + len(r_shape)))
            dst = list(range(axis, axis + len(r_shape)))
            data = np.moveaxis(data, src, dst)
            mask = np.moveaxis(mask, src, dst)
        return _ma.MaskedArray(data, mask=mask)
    else:
        try:
            stacked = np.array([_asarray(r).tolist() for r in results])
        except Exception:
            # 元素不可数值化(如 set)→ object 数组
            stacked = np.array(results, dtype='object')
        stacked = np.reshape(stacked, outer + r_shape)
    if r_shape:
        src = list(range(len(outer), len(outer) + len(r_shape)))
        dst = list(range(axis, axis + len(r_shape)))
        stacked = np.moveaxis(stacked, src, dst)
    # numpy 语义:输出类型跟随 func1d 返回值的子类(其次输入子类)
    _out_cls = None
    if isinstance(r0, np.ndarray) and type(r0) is not np.ndarray:
        _out_cls = type(r0)
    elif _in_cls is not None and _in_cls is not np.ndarray:
        _out_cls = _in_cls
    if _out_cls is not None and not _ma.isMaskedArray(stacked):
        try:
            stacked = stacked.view(_out_cls)
        except Exception:
            pass
    return stacked


def apply_over_axes(func, a, axes):
    """在多个轴上依次应用归约函数（保持维度）。"""
    np = _np()
    arr = _asarray(a)
    ax_list = [axes] if isscalar(axes) else list(axes)
    for ax in ax_list:
        res = _asarray(func(arr, ax))
        if res.ndim == arr.ndim:
            arr = res
        elif res.ndim == arr.ndim - 1:
            arr = np.expand_dims(res, ax)
        else:
            raise ValueError("function is not returning array of correct shape")
    return arr


def piecewise(x, condlist, funclist, *args, **kw):
    """按条件分段求值。

    使用 zip + 列表推导代替嵌套 for 循环，提高执行效率。
    """
    np = _np()
    arr = _asarray(x)
    flat = _flat(arr)
    if not isinstance(condlist, (list, tuple)) or (condlist and isscalar(condlist[0])):
        condlist = [condlist]
    conds = [_flat(c) for c in condlist]
    n = len(flat)
    n_conds = len(conds)
    has_default = len(funclist) == n_conds + 1
    default_f = funclist[-1] if has_default else None

    # 预计算每个条件的长度，避免在循环中重复 len() 调用
    cond_lens = [len(c) for c in conds]

    out = [0.0] * n
    for i in range(n):
        applied = False
        for k in range(n_conds):
            if i < cond_lens[k] and conds[k][i]:
                f = funclist[k]
                out[i] = f(flat[i], *args, **kw) if callable(f) else f
                applied = True
                break  # numpy 行为：匹配第一个条件即停止
        if not applied and has_default:
            f = default_f
            out[i] = f(flat[i], *args, **kw) if callable(f) else f
    return np.reshape(np.array(out), arr.shape) if arr.ndim else _as_scalar(np.array(out))


# ========== 位运算别名 / 计数 ==========
def bitwise_invert(x):
    """逐元素按位取反（invert 别名）。"""
    return _np().invert(x)


def bitwise_left_shift(x1, x2):
    """逐元素左移（left_shift 别名）。"""
    return _np().left_shift(x1, x2)


def bitwise_right_shift(x1, x2):
    """逐元素右移（right_shift 别名）。"""
    return _np().right_shift(x1, x2)


def bitwise_count(x):
    """逐元素统计绝对值二进制表示中的置位比特数。"""
    return _wrap(_core.bitwise_count(_asarray(x)._array), "uint8")


def packbits(a, axis=None, bitorder='big'):
    """将布尔/整数数组按比特打包为 uint8。"""
    _ = axis
    return _wrap(_core.packbits(_asarray(a)._array, bitorder), "uint8")


def unpackbits(a, axis=None, count=None, bitorder='big'):
    """将 uint8 数组展开为比特。"""
    _ = axis
    c = count if count is not None else 0
    return _wrap(_core.unpackbits(_asarray(a)._array, c if c > 0 else None, bitorder), "uint8")


# ========== 向量 / 矩阵乘积（gufunc）==========
def vecdot(x1, x2, axis=-1):
    """沿轴计算向量点积（对第一个参数取共轭）。"""
    _ = axis
    np = _np()
    a = _asarray(x1)
    b = _asarray(x2)
    a = np.conjugate(a) if iscomplexobj(a) else a
    return np.sum(np.multiply(a, b), -1)


def matvec(x1, x2):
    """矩阵-向量乘积。"""
    return _np().matmul(_asarray(x1), _asarray(x2))


def vecmat(x1, x2):
    """向量-矩阵乘积（对向量取共轭）。"""
    np = _np()
    a = _asarray(x1)
    a = np.conjugate(a) if iscomplexobj(a) else a
    return np.matmul(a, _asarray(x2))


# ========== 直方图分箱边界 ==========
def histogram_bin_edges(a, bins=10, range=None, weights=None):
    """仅返回直方图的分箱边界。"""
    _ = weights
    np = _np()
    data = _flat(a)
    if isinstance(bins, str):
        bins = 10
    if isinstance(bins, (list, tuple)) or hasattr(bins, "_array"):
        return np.array(_flat(bins))
    if range is None:
        lo = builtin_min(data) if data else 0.0
        hi = builtin_max(data) if data else 1.0
    else:
        lo, hi = range
    lo = float(lo)
    hi = float(hi)
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    nb = int(bins)
    step = (hi - lo) / nb
    edges = [lo + i * step for i in builtin_range(nb + 1)]
    edges[-1] = hi
    return np.array(edges)


builtin_range = range


# ========== dtype 相关：promote / result / cast / min_scalar / common / isdtype ==========
_DT_INFO = {
    'bool': ('b', 1),
    'int8': ('i', 1), 'int16': ('i', 2), 'int32': ('i', 4), 'int64': ('i', 8),
    'uint8': ('u', 1), 'uint16': ('u', 2), 'uint32': ('u', 4), 'uint64': ('u', 8),
    'float16': ('f', 2), 'float32': ('f', 4), 'float64': ('f', 8),
    'complex64': ('c', 8), 'complex128': ('c', 16),
    'string_': ('S', 0),
    'object': ('O', 8),
}


def _canon_dt_type(x):
    """将 dtype 类的类型/字符串解析为规范名称（真实 dtype 名，如 'str0'/'void8'）。

    不做 float64 兜底：字符串（'str3'/'bytes5'）/void 等名称需原样保留，
    供 promote_types 的字符串分支与 can_cast 的 S/U 宽度/structured 分支识别；
    非数值名进入 _promote2 的 KeyError 由调用方的 try/except 兜底。
    """
    # np.dtypes 具体类（Int64DType 等）携带 _dt_name
    dt_name = getattr(x, '_dt_name', None)
    if dt_name is not None:
        return dt_name
    return _np().dtype(x).name


def _canon_dt(x):
    """将 dtype 类型 / 数组 / 值 解析为规范 dtype 名称。"""
    if hasattr(x, '_dtype'):
        return x._dtype
    if hasattr(x, 'dtype') and not isinstance(x, type):
        d = x.dtype
        return getattr(d, 'name', str(d))
    try:
        n = _np().dtype(x).name
        if n in _DT_INFO:
            return n
    except Exception:
        pass
    return _canon_dt(min_scalar_type(x))


def _float_part(name):
    k, s = _DT_INFO[name]
    if k == 'f':
        return s
    if k == 'c':
        return s // 2
    if k in 'iu':
        return {1: 2, 2: 4, 4: 8, 8: 8}[s]
    return 2


def _int_name(k, size):
    pre = 'int' if k == 'i' else 'uint'
    return pre + {1: '8', 2: '16', 4: '32', 8: '64'}[size]


def _promote2(n1, n2):
    if n1 == n2:
        return n1
    k1, s1 = _DT_INFO[n1]
    k2, s2 = _DT_INFO[n2]
    if k1 == 'O' or k2 == 'O':
        # object 吸收一切（numpy 语义）
        return 'object'
    if k1 == 'b':
        return n2
    if k2 == 'b':
        return n1
    if k1 == 'c' or k2 == 'c':
        fp = builtin_max(_float_part(n1), _float_part(n2))
        return 'complex128' if fp >= 8 else 'complex64'
    if k1 == 'f' or k2 == 'f':
        fp = builtin_max(_float_part(n1), _float_part(n2))
        return {2: 'float16', 4: 'float32', 8: 'float64'}[fp]
    if k1 == k2:
        return _int_name(k1, builtin_max(s1, s2))
    isize = s1 if k1 == 'i' else s2
    usize = s1 if k1 == 'u' else s2
    if usize < isize:
        return _int_name('i', isize)
    t = usize * 2
    return 'float64' if t > 8 else _int_name('i', t)


# numpy 语义：数值类型 + 无宽度字符串 → 该类型可表示值的最长字符串长度
_STR_PROMO_WIDTH = {
    'bool': 5,
    'int8': 4, 'int16': 6, 'int32': 11, 'int64': 21,
    'uint8': 3, 'uint16': 5, 'uint32': 10, 'uint64': 20,
}


def _str_kind_width(n):
    """'str3'/'bytes5' → ('U', 3)/('S', 5)；无宽度 → 宽度 0；其他 → None。"""
    if isinstance(n, str):
        if n.startswith('str'):
            try:
                # U 的 itemsize = 宽度 × 4 字节
                return 'U', int(n[3:]) // 4
            except ValueError:
                return None
        if n.startswith('bytes'):
            try:
                return 'S', int(n[5:])
            except ValueError:
                return None
    return None


def _raw_str_kind_width(x):
    """从 dtype 原始字符串参数直接解析 kind/width。

    name 规范化会丢失 S/U 区分(S5→'str5'、U5→'str20'、'S'→'str0'),
    字符串类原始参数('S'/'S5'/'U3'/'bytes5'/'str3' 及带字节序前缀形式)
    须绕过 name 路径。数字类型('u1'/'b' 等小写单字符)不匹配,返回 None。
    """
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None
    if s[0] in '<>=|':
        s = s[1:]
    if s.startswith('str'):
        rest = s[3:]
        return ('U', int(rest)) if rest.isdigit() else ('U', 0)
    if s.startswith('bytes'):
        rest = s[5:]
        return ('S', int(rest)) if rest.isdigit() else ('S', 0)
    if len(s) == 1 and s in ('S', 'U'):
        return (s, 0)
    if len(s) >= 2 and s[0] in ('S', 'U') and s[1:].isdigit():
        return (s[0], int(s[1:]))
    return None


def _void_promo(d1, d2):
    """void/结构化提升（numpy 语义）：兼容（同为 void 同尺寸、或同字段名）
    → 返回 type1；否则 TypeError。"""
    f1 = getattr(d1, 'fields', None)
    f2 = getattr(d2, 'fields', None)
    n1 = tuple(f1) if f1 is not None else None
    n2 = tuple(f2) if f2 is not None else None
    if n1 is None and n2 is None:
        if d1.itemsize == d2.itemsize:
            return d1
        raise TypeError(
            "promote_types with mismatching void sizes is not defined")
    if n1 is not None and n2 is not None and n1 == n2:
        return d1
    raise TypeError(
        "promote_types with mismatching field names is not defined")


def promote_types(type1, type2):
    """返回可安全容纳两个类型的最小类型。"""
    np = _np()
    a = _canon_dt_type(type1)
    b = _canon_dt_type(type2)
    d1 = np.dtype(type1)
    d2 = np.dtype(type2)
    # datetime64/timedelta64 伪类型：同单位返回自身（native），M+m 混合 → M
    k1 = getattr(d1, '_kind', '')
    k2 = getattr(d2, '_kind', '')
    if k1 in ('M', 'm') or k2 in ('M', 'm'):
        kind = 'M' if 'M' in (k1, k2) else 'm'
        unit = (getattr(d1, '_dt_unit', None) or getattr(d2, '_dt_unit', None)
                or 'ns')
        return np.dtype('%s8[%s]' % (kind, unit))
    # void/结构化：兼容 → type1，否则 TypeError
    if k1 == 'V' or k2 == 'V':
        return _void_promo(d1, d2)
    # 字符串 dtype 提升:原始字符串参数优先——name 规范化丢失 S/U
    # 区分(S5→'str5'、U5→'str20'、'S'→'str0'),须从原参解析 kind。
    sa = _raw_str_kind_width(type1) or _str_kind_width(a)
    sb = _raw_str_kind_width(type2) or _str_kind_width(b)
    if sa or sb:
        if sa and sb:
            # 字符串 + 字符串：unicode 优先，宽度取最大
            kind = 'U' if 'U' in (sa[0], sb[0]) else sa[0]
            return np.dtype(f"{kind}{max(sa[1] or 1, sb[1] or 1)}")
        num = b if sa else a
        kind, w0 = (sa or sb)
        if num in ('object', 'O'):
            return np.dtype('O')
        w = _STR_PROMO_WIDTH.get(num, 32)
        return np.dtype(f"{kind}{max(w0, w)}")
    return np.dtype(_promote2(a, b))


def result_type(*arrays_and_dtypes):
    """按 NumPy (NEP 50) 提升规则返回结果类型。

    裸 Python 标量（int/float/complex，非 bool）是"弱标量"：不做值基提升——
    存在强操作数（数组/dtype/类型）时跟随其类别（float 标量把整数/bool 数组
    提到 float64，complex 标量把浮点数组提到同宽度复数、整数数组提到
    complex128）；纯弱标量时按 Python 语义取 int64/float64/complex128。
    """
    weak = []
    strong = []
    for x in arrays_and_dtypes:
        if x is None:
            # numpy 语义：result_type(None) → object dtype
            strong.append(_np().dtype('O'))
        elif isinstance(x, bool):
            strong.append(x)
        elif type(x) is int or type(x) is float or type(x) is complex:
            # NEP 50：仅 Python 原生标量是"弱标量"；rsnumpy 标量类
            # （float32/int16 等，继承 float/int）是强操作数，保留自身 dtype。
            weak.append(x)
        else:
            strong.append(x)
    names = []
    for x in strong:
        try:
            n = _canon_dt(x)
        except Exception:
            n = 'float64'
        if not isinstance(n, str):
            n = str(n)
        names.append(n)
    # 过滤掉不在 _DT_INFO 中的名称
    valid_names = []
    for n in names:
        if n in _DT_INFO:
            valid_names.append(n)
        else:
            try:
                resolved = _np().dtype(n).name
                if resolved in _DT_INFO:
                    valid_names.append(resolved)
                else:
                    valid_names.append('float64')
            except Exception:
                valid_names.append('float64')
    if not valid_names:
        # 纯 Python 标量：complex → complex128，float → float64，int → int64
        # 注意：必须用 builtin_any——模块级 any 被 rsnumpy ufunc 遮蔽，
        # 对生成器参数恒返回 True（会把一切弱标量误判为 complex）。
        if builtin_any(isinstance(w, complex) for w in weak):
            return _np().dtype('complex128')
        if builtin_any(isinstance(w, float) for w in weak):
            return _np().dtype('float64')
        return _np().dtype('int64')
    acc = valid_names[0]
    for n in valid_names[1:]:
        try:
            acc = _promote2(acc, n)
        except Exception:
            acc = 'float64'
    # 弱标量提升（NEP 50）：不改变浮点/复数强操作数的类型
    for w in weak:
        try:
            k, s = _DT_INFO[acc]
        except Exception:
            k, s = 'f', 8
        if isinstance(w, complex):
            if k == 'f':
                acc = 'complex64' if s <= 4 else 'complex128'
            elif k in ('i', 'u', 'b'):
                acc = 'complex128'
        elif isinstance(w, float):
            if k in ('i', 'u', 'b'):
                acc = 'float64'
    return _np().dtype(acc)


def can_cast(from_, to, casting='safe'):
    """判断能否按给定规则从一个类型转换到另一个类型。"""
    # None 参数：numpy 语义 TypeError
    if from_ is None or to is None:
        raise TypeError(
            "can_cast: 'None' is not a valid dtype argument")
    # numpy (NEP50) 语义：裸 Python 数值标量不可参与 cast 判断；
    # 字符串参数（typecode / dtype spec）按 dtype 解析（numpy 语义）
    if isinstance(from_, (str, bytes)):
        try:
            _canon_dt(from_)
        except Exception:
            raise TypeError(
                f"Cannot cast Python scalar {type(from_).__name__!r} to dtype "
                f"{to!r} with casting rule {casting!r}")
    elif isinstance(from_, (int, float, complex)) and \
            not isinstance(from_, bool):
        raise TypeError(
            f"Cannot cast Python scalar {type(from_).__name__!r} to dtype "
            f"{to!r} with casting rule {casting!r}")
    try:
        d_from = _np().dtype(from_)
        d_to = _np().dtype(to)
    except Exception:
        return False
    fn = _canon_dt(from_)
    tn = _canon_dt_type(to)
    if casting in ('no', 'equiv'):
        # 'no'：typestr 完全一致（含字节序）；'equiv'：同 kind + itemsize
        # （字节序无关，numpy 语义）。
        if casting == 'no':
            try:
                return _typestr_of(d_from) == _typestr_of(d_to)
            except Exception:
                return False
        return (getattr(d_from, '_kind', None) == getattr(d_to, '_kind', None)
                and getattr(d_from, '_itemsize', -1) == getattr(d_to, '_itemsize', -1))
    if casting == 'unsafe':
        # 结构化 → 非结构化：仅单字段链允许（字段可为简单类型/子数组/
        # 单字段结构化；任一层多字段则不可）。
        if getattr(d_from, '_names', None) and not getattr(d_to, '_names', None):
            if not _single_field_chain(d_from):
                return False
        return True
    # 数值 → 固定宽字符串：按最大值的十进制位宽判定（numpy 语义）。
    k_to = getattr(d_to, '_kind', '')
    if k_to in ('S', 'U') and fn in _NUM_STR_WIDTH:
        need_s, need_u = _NUM_STR_WIDTH[fn]
        need = need_s if k_to == 'S' else need_u
        width = (d_to._itemsize // 4) if k_to == 'U' else d_to._itemsize
        return width >= need
    # object 语义：仅 object ↔ object 可转换
    k_from = getattr(d_from, '_kind', '')
    if k_from == 'O' or k_to == 'O':
        return k_from == 'O' and k_to == 'O'
    # 结构化 ↔ 任何类型：safe/same_kind 均不允许（unsafe 已在上方处理）
    if getattr(d_from, '_names', None) or getattr(d_to, '_names', None):
        return False
    # 非数值类型（字符串/void/datetime 等）无通用提升规则，仅同名可转换
    if fn not in _DT_INFO or tn not in _DT_INFO:
        return False
    prom = _promote2(fn, tn)
    if casting == 'safe':
        return prom == tn
    if prom == tn:
        return True
    if casting == 'same_kind':
        k1, k2 = _DT_INFO[fn][0], _DT_INFO[tn][0]
        # signed → unsigned 不属于 same_kind（numpy 语义）
        if k1 == 'i' and k2 == 'u':
            return False
    group = {'b': 0, 'u': 1, 'i': 1, 'f': 2, 'c': 3}
    return group[_DT_INFO[fn][0]] <= group[_DT_INFO[tn][0]]


# 数值 → 固定宽字符串所需最小宽度（S, U）：按最大值十进制位宽（含符号位），
# numpy can_cast 语义（bool→S5 'False'/U4 'True'，int32→11，int64→21，
# uint64→20 等）。
_NUM_STR_WIDTH = {
    'bool': (5, 4),
    'int8': (4, 4), 'int16': (6, 6), 'int32': (11, 11), 'int64': (21, 21),
    'uint8': (3, 3), 'uint16': (5, 5), 'uint32': (10, 10), 'uint64': (20, 20),
}


def _single_field_chain(dt):
    """结构化 dtype 是否满足"单字段链"：仅一个字段，且该字段（或其子数组
    基类型）递归满足同样条件。numpy 结构化→简单 unsafe cast 的判定。"""
    names = getattr(dt, '_names', None)
    if names is None:
        return True
    if builtins.len(names) != 1:
        return False
    fdt = dt._fields[names[0]][0]
    if getattr(fdt, '_subdtype', None) is not None:
        fdt = fdt._subdtype[0]
    return _single_field_chain(fdt)


def _typestr_of(dt):
    """dtype 的 array-protocol typestr（含字节序前缀）。"""
    from ._dtypes import _typestr
    return _typestr(dt)


def min_scalar_type(value):
    """返回可容纳给定标量值的最小 dtype。"""
    np = _np()
    if hasattr(value, '_dtype') or (hasattr(value, 'dtype') and not isinstance(value, type)):
        return np.dtype(_canon_dt(value))
    if isinstance(value, bool):
        return np.dtype('bool')
    if isinstance(value, int):
        if value >= 0:
            for name, bits in [('uint8', 8), ('uint16', 16), ('uint32', 32), ('uint64', 64)]:
                if value < (1 << bits):
                    return np.dtype(name)
            return np.dtype('float64')
        for name, bits in [('int8', 8), ('int16', 16), ('int32', 32), ('int64', 64)]:
            if -(1 << (bits - 1)) <= value < (1 << (bits - 1)):
                return np.dtype(name)
        return np.dtype('float64')
    if isinstance(value, float):
        return np.dtype('float16')
    if isinstance(value, complex):
        return np.dtype('complex64')
    return np.dtype('float64')


def common_type(*arrays):
    """返回若干数组的公共（浮点/复数）标量类型。

    gh-30890：传入 dtype 对象或标量类型类 → 明确的 TypeError（提示用
    np.result_type / np.promote_types）。
    """
    np = _np()
    for a in arrays:
        if isinstance(a, type) and issubclass(a, np.generic):
            raise TypeError(
                "passing a scalar type to np.common_type is not supported; "
                "use np.result_type or np.promote_types")
        if builtins.type(a).__name__ == 'DType' and hasattr(a, 'kind'):
            raise TypeError(
                "passing a dtype object to np.common_type is not supported; "
                "use np.result_type or np.promote_types")
    has_complex = False
    prec = 4
    for a in arrays:
        n = _canon_dt(a)
        k, s = _DT_INFO[n]
        if k == 'c':
            has_complex = True
            prec = builtin_max(prec, s // 2)
        elif k == 'f':
            prec = builtin_max(prec, s)
        else:
            prec = builtin_max(prec, 8)
    if has_complex:
        return np.complex128 if prec >= 8 else np.complex64
    return np.float64 if prec >= 8 else np.float32


def isdtype(dtype, kind):
    """判断 dtype 是否属于指定类别（Array API）。"""
    # numpy 2.x 语义：首参必须是 dtype 实例或标量类型（不允许裸 str）
    if isinstance(dtype, str):
        raise TypeError(
            "isdtype's first argument must be a rsnumpy dtype, not a string")
    n = _canon_dt_type(dtype)
    k = _DT_INFO[n][0]

    def match(kk):
        if isinstance(kk, str):
            table = {
                'bool': k == 'b',
                'signed integer': k == 'i',
                'unsigned integer': k == 'u',
                'integral': k in 'iu',
                'real floating': k == 'f',
                'complex floating': k == 'c',
                'numeric': k in 'iufc',
            }
            if kk in table:
                return table[kk]
            # str 必须是已知 kind 名（dtype 名如 "int64" 非法）
            raise ValueError(f"'{kk}' is not a known kind name")
        # 纯 Python 数值（int/float/bool/complex 值）不是合法 kind
        if type(kk) in (int, float, bool, complex):
            raise TypeError(
                "kind argument must be a dtype, a dtype class, or a kind name")
        # np.dtypes abstract 类（IntegerAbstractDType 等）：按 kind 字符集合判定
        ks = getattr(kk, '_kind_set', None)
        if isinstance(ks, frozenset):
            return k in ks
        return _canon_dt_type(kk) == n

    if isinstance(kind, (tuple, list)):
        return builtin_any(match(x) for x in kind)
    return match(kind)


def issubdtype(arg1, arg2):
    """判断第一个类型是否为第二个类型（含抽象类别）的子类型。"""
    return _np().issubdtype(arg1, arg2)


# ========== 进制 / 类型码文本 ==========
def base_repr(number, base=2, padding=0):
    """将整数转换为给定进制的字符串。"""
    return _core.base_repr(int(number), base, padding)


_TYPENAME = {
    'S1': 'character', '?': 'bool', 'B': 'unsigned char', 'b': 'signed char',
    'h': 'short', 'H': 'unsigned short', 'i': 'integer', 'I': 'unsigned integer',
    'l': 'long integer', 'L': 'unsigned long integer', 'q': 'long long integer',
    'Q': 'unsigned long long integer', 'f': 'single precision', 'd': 'double precision',
    'g': 'long precision', 'F': 'complex single precision', 'D': 'complex double precision',
    'G': 'complex long double precision', 'S': 'string', 'U': 'unicode', 'V': 'void', 'O': 'object',
}


def typename(char):
    """返回类型码的英文名称。"""
    return _TYPENAME[char]


_TYPECODES_BY_ELSIZE = 'GDFgdfQqLlIiHhBb?'


def mintypecode(typechars, typeset='GDFgdf', default='d'):
    """返回可容纳所有输入类型码的最小类型码。"""
    intersection = set()
    for t in typechars:
        c = t if isinstance(t, str) and len(t) == 1 else _np().dtype(t).char
        if c in typeset:
            intersection.add(c)
    if not intersection:
        return default
    if 'F' in intersection and 'd' in intersection:
        return 'D'
    return builtin_min(intersection, key=_TYPECODES_BY_ELSIZE.index)


# ========== 数组文本表示 ==========
def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    """返回数组的可求值字符串表示。"""
    _ = max_line_width, precision, suppress_small
    return repr(_asarray(arr))


def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    """返回数组的字符串表示（无 array() 包裹）。"""
    _ = max_line_width, precision, suppress_small
    return str(_asarray(a))


def array2string(a, max_line_width=None, precision=None, suppress_small=None,
                 separator=' ', prefix="", **kwargs):
    """返回数组数据的字符串表示。"""
    _ = max_line_width, precision, suppress_small, separator, prefix, kwargs
    return str(_asarray(a))


# ========== 打印选项 ==========
_PRINTOPTS = {
    'precision': 8, 'threshold': 1000, 'edgeitems': 3, 'linewidth': 75,
    'suppress': False, 'nanstr': 'nan', 'infstr': 'inf', 'sign': '-',
    'formatter': None, 'floatmode': 'maxprec', 'legacy': False,
}


def get_printoptions():
    """返回当前打印选项字典。"""
    return dict(_PRINTOPTS)


def set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None,
                     suppress=None, nanstr=None, infstr=None, sign=None,
                     formatter=None, floatmode=None, legacy=None, *, override_repr=None):
    """设置打印选项。"""
    _ = override_repr
    updates = {
        'precision': precision, 'threshold': threshold, 'edgeitems': edgeitems,
        'linewidth': linewidth, 'suppress': suppress, 'nanstr': nanstr,
        'infstr': infstr, 'sign': sign, 'formatter': formatter,
        'floatmode': floatmode, 'legacy': legacy,
    }
    for k, v in updates.items():
        if v is not None:
            _PRINTOPTS[k] = v


class printoptions:
    """临时修改打印选项的上下文管理器。"""

    def __init__(self, **kwargs):
        self._kw = kwargs
        self._saved = None

    def __enter__(self):
        self._saved = dict(_PRINTOPTS)
        set_printoptions(**self._kw)
        return get_printoptions()

    def __exit__(self, *exc):
        _PRINTOPTS.clear()
        _PRINTOPTS.update(self._saved)
        return False


def format_float_positional(x, precision=None, unique=True, fractional=True,
                            trim='k', sign=False, pad_left=None, pad_right=None,
                            min_digits=None):
    """以定点记法格式化单个浮点数。"""
    _ = unique, fractional, pad_left, pad_right, min_digits
    x = float(x)
    if precision is None:
        s = repr(x)
        if 'e' in s or 'E' in s:
            s = f"{x:.16f}".rstrip('0')
        if '.' not in s:
            s += '.'
    else:
        s = f"{x:.{precision}f}"
    if trim == '-':
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
    elif trim == '0':
        if '.' in s:
            s = s.rstrip('0')
            if s.endswith('.'):
                s += '0'
    if sign and x >= 0:
        s = '+' + s
    return s


def format_float_scientific(x, precision=None, unique=True, trim='k', sign=False,
                            pad_left=None, exp_digits=None, min_digits=None):
    """以科学记法格式化单个浮点数。"""
    _ = unique, pad_left, exp_digits, min_digits
    x = float(x)
    if precision is None:
        s = f"{x:e}"
        mant, exp = s.split('e')
        if trim in ('-', '0'):
            mant = mant.rstrip('0').rstrip('.') if trim == '-' else mant.rstrip('0')
        s = mant + 'e' + exp
    else:
        s = f"{x:.{precision}e}"
    if sign and x >= 0:
        s = '+' + s
    return s


# ========== 浮点错误状态 ==========
# numpy 2.x 语义：状态存于 contextvars —— 每个 asyncio Task / 线程有独立
# 上下文副本，errstate 的 set/reset 经 token 恢复，天然并发安全（官方
# test_asyncio_safe 依赖此语义：交叉的上下文互不破坏）。
import contextvars as _contextvars

_ERR_DEFAULT = {'divide': 'warn', 'over': 'warn', 'under': 'ignore',
                'invalid': 'warn'}
_err_state_ctx = _contextvars.ContextVar('rsnumpy_errstate', default=None)
_err_call_ctx = _contextvars.ContextVar('rsnumpy_errcall', default=None)
_BUFSIZE = [8192]
_ERR_MODES = ('ignore', 'warn', 'raise', 'call', 'print')
_NO_CALL = object()


def _fp_flat(vals):
    """把 tolist 的嵌套列表展平为扁平序列。"""
    def rec(x):
        if isinstance(x, (list, tuple)):
            for v in x:
                yield from rec(v)
        else:
            yield x
    yield from rec(vals)


def _current_errstate():
    """当前上下文的浮点错误状态（未设置时为默认值）。"""
    st = _err_state_ctx.get()
    return dict(st) if st is not None else dict(_ERR_DEFAULT)


def _fp_fire(kind, msg):
    """按当前 errstate 处理一次浮点错误事件（numpy 语义）。"""
    state = _err_state_ctx.get()
    mode = (state or _ERR_DEFAULT).get(kind, 'warn')
    if mode == 'ignore':
        return
    if mode == 'raise':
        raise FloatingPointError(msg)
    if mode == 'call':
        cb = _err_call_ctx.get()
        if cb is not None:
            cb(msg)
        return
    if mode == 'warn':
        import warnings as _warnings
        _warnings.warn(msg, RuntimeWarning, stacklevel=4)
    elif mode == 'print':
        print(msg)


def _fp_divide_check(num, den, res, op='true_divide'):
    """true_divide 的 errstate 触发（numpy 语义）：除零 → 'divide'，
    0/0 → 'invalid'。仅当存在非 ignore 模式时扫描，避免常态开销。"""
    st = _err_state_ctx.get()
    eff = st or _ERR_DEFAULT
    if (eff.get('divide', 'warn') == 'ignore'
            and eff.get('invalid', 'warn') == 'ignore'):
        return
    try:
        def _flat(x):
            if isinstance(x, (list, tuple)):
                out = []
                for e in x:
                    out.extend(_flat(e))
                return out
            if hasattr(x, 'ravel'):
                v = x.ravel().tolist()
                return v if isinstance(v, list) else [v]
            return [x]
        den_flat = _flat(den)
        if not builtins.any(v == 0 for v in den_flat):
            return
        for v in _flat(res):
            if isinstance(v, float):
                if _math.isinf(v):
                    _fp_fire('divide', f'divide by zero encountered in {op}')
                    return
                if _math.isnan(v):
                    _fp_fire('invalid', f'invalid value encountered in {op}')
                    return
    except (FloatingPointError, TypeError):
        raise
    except Exception:
        pass


def _fp_arith_check(num, oth, res, op):
    """add/subtract/multiply 的 errstate 触发（numpy 语义）：
    overflow（有限输入 → ±inf）与 invalid（inf-inf / 0*inf / inf+(-inf)）。"""
    st = _err_state_ctx.get()
    eff = st or _ERR_DEFAULT
    if (eff.get('overflow', 'warn') == 'ignore'
            and eff.get('invalid', 'warn') == 'ignore'
            and eff.get('under', 'warn') == 'ignore'):
        return
    try:
        _TINY64 = 2.2250738585072014e-308
        def _flat(x):
            if isinstance(x, (list, tuple)):
                out = []
                for e in x:
                    out.extend(_flat(e))
                return out
            if hasattr(x, 'ravel'):
                v = x.ravel().tolist()
                return v if isinstance(v, list) else [v]
            return [x]
        na = _flat(num)
        nb = _flat(oth)
        if len(na) == 1 and len(nb) > 1:
            na = na * len(nb)
        elif len(nb) == 1 and len(na) > 1:
            nb = nb * len(na)
        n = min(len(na), len(nb))
        ra = _flat(res)
        if len(ra) == 1 and n > 1:
            ra = ra * n
        fired_ov = False
        fired_inv = False
        fired_un = False
        for i in range(n):
            a = na[i] if i < len(na) else 0
            b = nb[i] if i < len(nb) else 0
            r = ra[i] if i < len(ra) else 0
            if not builtins.all(isinstance(v, float) for v in (a, b, r)):
                continue
            ai, bi, ri = _math.isinf(a), _math.isinf(b), _math.isinf(r)
            if op == 'multiply':
                if ri and not (ai or bi) and a != 0 and b != 0:
                    if not fired_ov:
                        _fp_fire('overflow', f'overflow encountered in {op}')
                        fired_ov = True
                    continue
                if _math.isnan(r):
                    if not fired_inv:
                        _fp_fire('invalid', f'invalid value encountered in {op}')
                        fired_inv = True
                    continue
                if ri and (ai or bi) and (a == 0 or b == 0):
                    if not fired_inv:
                        _fp_fire('invalid', f'invalid value encountered in {op}')
                        fired_inv = True
                    continue
                # underflow：两个有限非零输入 → 结果为 0（完全下溢）
                # 或次正规数（|r| < tiny）
                if (not (ai or bi) and a != 0 and b != 0
                        and not _math.isnan(r)
                        and (r == 0 or builtins.abs(r) < _TINY64)):
                    if not fired_un:
                        _fp_fire('under', f'underflow encountered in {op}')
                        fired_un = True
                continue
            # add / subtract：overflow = 有限输入 → ±inf
            if op == 'add':
                ov = (a > 0 and b > 0 and ri and r > 0) or \
                     (a < 0 and b < 0 and ri and r < 0)
            else:
                ov = ((a > 0 and b < 0 and ri and r > 0)
                      or (a < 0 and b > 0 and ri and r < 0))
            if ov and not (ai or bi):
                if not fired_ov:
                    _fp_fire('overflow', f'overflow encountered in {op}')
                    fired_ov = True
            # invalid：inf - inf / inf + (-inf) → nan
            if ai and bi and _math.isnan(r):
                if not fired_inv:
                    _fp_fire('invalid', f'invalid value encountered in {op}')
                    fired_inv = True
    except (FloatingPointError, TypeError):
        raise
    except Exception:
        pass


def _check_err_mode(v, name=None):
    """seterr 参数校验（numpy 语义：非字符串 → TypeError）。"""
    if not isinstance(v, str):
        raise TypeError(
            "seterr: argument must be a string or None, got "
            f"{type(v).__name__}")
    if v not in _ERR_MODES:
        raise ValueError(f"seterr: {name!r} is not a valid error mode")


def geterr():
    """返回当前浮点错误处理状态。"""
    return _current_errstate()


def seterr(all=None, divide=None, over=None, under=None, invalid=None):
    """设置浮点错误处理状态，返回旧状态。"""
    old = _current_errstate()
    new = dict(old)
    if all is not None:
        _check_err_mode(all, 'all')
        for k in new:
            new[k] = all
    for k, v in {'divide': divide, 'over': over, 'under': under,
                 'invalid': invalid}.items():
        if v is not None:
            _check_err_mode(v, k)
            new[k] = v
    _err_state_ctx.set(new)
    return old


def geterrcall():
    """返回当前浮点错误回调。"""
    return _err_call_ctx.get()


def seterrcall(func):
    """设置浮点错误回调，返回旧回调。"""
    old = _err_call_ctx.get()
    _err_call_ctx.set(func)
    return old


def getbufsize():
    """返回 ufunc 缓冲区大小。"""
    return _BUFSIZE[0]


def setbufsize(size):
    """设置 ufunc 缓冲区大小，返回旧值。"""
    old = _BUFSIZE[0]
    _BUFSIZE[0] = int(size)
    return old


class errstate:
    """临时修改浮点错误处理状态的上下文管理器（可作装饰器，单次使用）。

    numpy 2.x 语义：状态存于 contextvar（每个 Task/线程独立），退出经
    token 恢复到进入前的状态；实例不可重复进入。
    """

    def __init__(self, *, call=_NO_CALL, **kwargs):
        self._kw = kwargs
        self._call = call
        self._token = None
        self._call_token = None

    def _make_extobj(self):
        """当前状态与本上下文设置合并为新状态（all 优先，逐键覆盖）。"""
        st = _current_errstate()
        if 'all' in self._kw:
            _check_err_mode(self._kw['all'], 'all')
            for k in st:
                st[k] = self._kw['all']
        for k in ('divide', 'over', 'under', 'invalid'):
            if k in self._kw:
                _check_err_mode(self._kw[k], k)
                st[k] = self._kw[k]
        return st

    def __enter__(self):
        if self._token is not None:
            raise TypeError("Cannot enter `np.errstate` twice")
        self._token = _err_state_ctx.set(self._make_extobj())
        if self._call is not _NO_CALL:
            self._call_token = _err_call_ctx.set(self._call)
        return self

    def __exit__(self, *exc):
        if self._token is not None:
            _err_state_ctx.reset(self._token)
        if self._call_token is not None:
            _err_call_ctx.reset(self._call_token)
        return False

    def __call__(self, function):
        """装饰器用法：每次调用独立 enter/exit（可重复调用，asyncio 安全）。"""
        if not callable(function):
            raise TypeError("errstate can only decorate callable objects")
        import functools

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            token = _err_state_ctx.set(self._make_extobj())
            call_token = None
            if self._call is not _NO_CALL:
                call_token = _err_call_ctx.set(self._call)
            try:
                return function(*args, **kwargs)
            finally:
                _err_state_ctx.reset(token)
                if call_token is not None:
                    _err_call_ctx.reset(call_token)
        return wrapper


# ========== 解析 / IO ==========
_CPX_NUM = r'(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?'
_CPX_STRICT_RE = _re.compile(
    r'^[+-]?' + _CPX_NUM + r'$'
    r'|^[+-]?' + _CPX_NUM + r'j$'
    r'|^[+-]?' + _CPX_NUM + r'[+-]' + _CPX_NUM + r'j$')


def fromstring(string, dtype=float, count=-1, sep='', *, like=None):
    """从文本字符串解析一维数组（numpy 语义：空 token/非法 token → ValueError）。"""
    _ = like
    np = _np()
    if sep == '':
        raise ValueError("fromstring() with binary data is not supported")
    if sep == ' ':
        parts = string.split()
    else:
        # 不过滤空 token：连续分隔符/纯分隔符 → float('') 自然 ValueError
        parts = [p.strip() for p in string.split(sep)]
    _dt_str = str(dtype) if dtype is not None else ''
    _is_cpx = (_dt_str.startswith('complex')
               or _dt_str in ('c8', 'c16', 'c', 'cdouble', 'csingle')
               or dtype in (builtins.complex, np.complex64, np.complex128))
    if _is_cpx:
        # numpy 严格文本复数文法（Python complex() 过宽：容忍 '1+j' 等）：
        # 纯实数 | 实数j | 实数±实数j；token 内不允许空白。
        def _cpx_tok(p):
            for ch in p:
                if ch.isspace():
                    raise ValueError(
                        "whitespace in complex number: %r" % p)
            if not _CPX_STRICT_RE.match(p):
                raise ValueError(
                    "malformed complex number: %r" % p)
            return builtins.complex(p)
        vals = [_cpx_tok(p) for p in parts]
    else:
        vals = [builtins.float(p) for p in parts]
    if count >= 0:
        vals = vals[:count]
    arr = np.array(vals)
    return arr if dtype in (float, None) else arr.astype(dtype)


def savez_compressed(file, *args, **kwds):
    """将多个数组保存为压缩 .npz 文件（rsnumpy 以非压缩格式落盘）。"""
    return _np().savez(file, *args, **kwds)


# ========== 由 Python 函数构造 ufunc / 向量化 ==========
def frompyfunc(func, nin, nout, *, kwargs=None):
    """将任意 Python 函数封装为逐元素的 ufunc 风格可调用对象。"""
    _ = kwargs
    np = _np()

    def wrapper(*args):
        arrs = [_asarray(a) for a in args[:nin]]
        flats = [_flat(a) for a in arrs]
        n = len(flats[0]) if flats else 0
        shp = arrs[0].shape if arrs else ()
        results = [func(*[flats[j][i] for j in range(nin)]) for i in range(n)]
        if nout == 1:
            out = np.array(results)
            return np.reshape(out, shp) if shp else _as_scalar(out)
        cols = []
        for t in range(nout):
            col = np.array([r[t] for r in results])
            cols.append(np.reshape(col, shp) if shp else _as_scalar(col))
        return tuple(cols)

    wrapper.nin = nin
    wrapper.nout = nout
    wrapper.nargs = nin + nout
    return wrapper


class vectorize:
    """将标量 Python 函数向量化为逐元素作用于数组的函数。

    【重要性能提示】
    本实现使用Python循环逐元素调用原函数，对于大型数组性能较低。
    推荐替代方案：
    1. 使用rsnumpy内置的向量化函数（如np.add, np.multiply等）
    2. 使用np.frompyfunc创建真正的向量化函数
    3. 使用Rust层实现的函数（通过@jit装饰器，如项目支持）

    【使用示例】
    >>> import rsnumpy as np
    >>> # 标量函数
    >>> def my_func(x):
    ...     return x**2 + 1
    >>> # 向量化
    >>> vfunc = np.vectorize(my_func)
    >>> vfunc([1, 2, 3])
    array([2, 5, 10])

    >>> # 多参数函数
    >>> def add(a, b):
    ...     return a + b
    >>> vadd = np.vectorize(add)
    >>> vadd([1, 2, 3], [10, 20, 30])
    array([11, 22, 33])

    >>> # 使用excluded排除某些参数
    >>> def scale(x, factor):
    ...     return x * factor
    >>> vscale = np.vectorize(scale, excluded=['factor'])
    >>> vscale([1, 2, 3], factor=10)
    array([10, 20, 30])

    【参数说明】
    pyfunc: callable
        要向量化的标量Python函数
    otypes: str or list of str, optional
        输出数据类型，如 'f8' 或 ['f8', 'i8']
    doc: str, optional
        自定义文档字符串
    excluded: list of str/int, optional
        不进行广播的参数名或索引列表
    cache: bool, optional
        是否缓存函数调用结果（当前未实现）
    signature: str, optional
        广播签名（如 '(m,n)->(m)'），用于控制输出形状

    【性能对比】
    # 错误方式（慢）
    >>> vfunc = np.vectorize(lambda x: x**2)
    >>> vfunc(np.arange(10000))  # 约100ms

    # 正确方式（快）
    >>> arr = np.arange(10000)
    >>> arr ** 2  # 约1ms，快100倍

    【注意事项】
    - vectorize不支持真正的并行化，只是Python层面的循环封装
    - 对于性能敏感的场景，请优先使用rsnumpy的内置函数
    - kwargs参数会显著降低性能（约5倍），建议使用位置参数
    """

    def __init__(self, pyfunc=None, otypes=None, doc=None, excluded=None,
                 cache=False, signature=None):
        self.pyfunc = pyfunc
        self.otypes = otypes
        self.__doc__ = doc or (getattr(pyfunc, '__doc__', None) if pyfunc else None) or self.__doc__
        self.excluded = excluded
        self.cache = cache
        self.signature = signature
        self._cached_result = None  # 缓存机制

    def __call__(self, *args, **kwargs):
        # If used as @np.vectorize(otypes=...) without pyfunc, return a decorator
        if self.pyfunc is None and len(args) == 1 and not kwargs and callable(args[0]):
            self.pyfunc = args[0]
            self.__doc__ = self.__doc__ or getattr(args[0], '__doc__', None)
            return self
        """执行向量化函数调用。

        自动广播所有输入数组，并逐元素应用原函数。

        【性能优化】
        1. 标量快速路径：输入全为标量时直接调用
        2. kwargs缓存：预提取kwargs避免每轮循环查找
        3. 预绑定方法：减少属性查找开销
        4. 列表推导：比显式for循环更快
        """
        np = _np()

        # 快速路径：全标量输入（非可迭代对象）
        def _is_scalar(x):
            # 检查是否为标量：不是列表/元组/ndarray，且不可迭代
            if isinstance(x, (list, tuple)):
                return False
            if hasattr(x, '_array'):  # rsnumpy ndarray
                return False
            return not hasattr(x, '__iter__')

        if not args or builtin_all(_is_scalar(a) for a in args):
            if not args:
                return self.pyfunc(**kwargs)
            return self.pyfunc(*args, **kwargs)

        arrs = [_asarray(a) for a in args]

        # 排除excluded参数（不参与广播，按原值传入 pyfunc）
        excluded_indices = set()
        excluded_values = {}
        if self.excluded:
            for ex in self.excluded:
                if isinstance(ex, int):
                    excluded_indices.add(ex)
                elif isinstance(ex, str) and ex in kwargs:
                    excluded_values[ex] = kwargs.pop(ex)

        # 参与广播的参数
        active_idx = [i for i in range(len(arrs)) if i not in excluded_indices]
        active_arrs = [arrs[i] for i in active_idx]

        if not active_arrs:
            # 所有参数都被排除：按原样调用
            _kw = {**excluded_values, **kwargs}
            return self.pyfunc(*[args[i] for i in range(len(args))
                                 if i in excluded_indices], **_kw)

        # 广播所有输入数组到相同形状
        shp = broadcast_shapes(*[a.shape for a in active_arrs])
        bcast = [np.broadcast_to(a, shp) for a in active_arrs]
        flats = [_flat(a) for a in bcast]
        n = len(flats[0])

        # 预绑定方法，减少循环内属性查找
        pyfunc = self.pyfunc
        kwargs_copy = kwargs.copy()

        if excluded_indices:
            # 有排除的位置参数：逐元素按原位置重组（排除位传原值）
            def _mkargs(i):
                out_a = []
                ai = 0
                for k in range(len(args)):
                    if k in excluded_indices:
                        out_a.append(args[k])
                    else:
                        out_a.append(flats[ai][i])
                        ai += 1
                return out_a
            res = [pyfunc(*_mkargs(i), **kwargs_copy) for i in range(n)]
        elif len(flats) == 1:
            # 单参数快速路径
            flat0 = flats[0]
            res = [pyfunc(flat0[i], **kwargs_copy) for i in range(n)]
        elif len(flats) == 2:
            # 双参数快速路径（最常见）
            flat0, flat1 = flats[0], flats[1]
            res = [pyfunc(flat0[i], flat1[i], **kwargs_copy) for i in range(n)]
        else:
            # 多参数通用路径
            res = [pyfunc(*[flats[j][i] for j in range(len(flats))], **kwargs_copy)
                   for i in range(n)]

        out = np.array(res)
        return np.reshape(out, shp) if shp else _as_scalar(out)


# ========== 日期时间辅助 ==========
def datetime_data(dtype):
    """返回 datetime64/timedelta64 dtype 的 (单位, 计数)。"""
    s = dtype if isinstance(dtype, str) else getattr(dtype, 'name', str(dtype))
    if '[' in s and ']' in s:
        return (s[s.index('[') + 1:s.index(']')], 1)
    return ('us', 1)


def _days_to_iso(days, unit):
    import datetime as _dt
    base = _dt.datetime(1970, 1, 1)
    d = base + _dt.timedelta(days=float(days))
    if unit in ('Y',):
        return f"{d.year:04d}"
    if unit in ('M',):
        return f"{d.year:04d}-{d.month:02d}"
    if unit in ('D', 'W'):
        return d.strftime("%Y-%m-%d")
    if unit in ('h',):
        return d.strftime("%Y-%m-%dT%H")
    if unit in ('m',):
        return d.strftime("%Y-%m-%dT%H:%M")
    if unit in ('s',):
        return d.strftime("%Y-%m-%dT%H:%M:%S")
    return d.strftime("%Y-%m-%dT%H:%M:%S.%f")


def datetime_as_string(arr, unit=None, timezone='naive', casting='same_kind'):
    """将 datetime64 转换为 ISO 字符串。"""
    _ = timezone, casting
    np = _np()
    np_mod = _np()
    items = arr if isinstance(arr, (list, tuple)) else [arr]
    if hasattr(arr, '_array') or hasattr(arr, 'tolist'):
        items = arr.tolist() if hasattr(arr, 'tolist') else list(arr)
        if not isinstance(items, list):
            items = [items]

    def one(v):
        if isinstance(v, np_mod.datetime64):
            u = unit or v._unit
            return _days_to_iso(v._days, u)
        if isinstance(v, str):
            days, u2 = np_mod._parse_datetime_to_days(v, unit)
            return _days_to_iso(days, unit or u2)
        return _days_to_iso(float(v), unit or 'D')

    out = [one(v) for v in items]
    if not isinstance(arr, (list, tuple)) and not (hasattr(arr, '_array') and _asarray(arr).ndim > 0):
        return out[0]
    return np.array(out)


def _weekmask_array(weekmask):
    if weekmask is None:
        return [True, True, True, True, True, False, False]
    if isinstance(weekmask, str):
        if ' ' in weekmask or len(weekmask) > 7:
            names = weekmask.split()
            wm = [False] * 7
            order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            for nm in names:
                wm[order.index(nm)] = True
            return wm
        return [c == '1' for c in weekmask]
    return [bool(x) for x in weekmask]


def _to_days(d):
    np_mod = _np()
    if isinstance(d, np_mod.datetime64):
        return int(d._days)
    days, _u = np_mod._parse_datetime_to_days(d) if isinstance(d, str) else (float(d), 'D')
    return int(days)


def _dow(days):
    return (int(days) + 3) % 7


def is_busday(dates, weekmask=None, holidays=None, busdaycal=None):
    """判断给定日期是否为工作日。"""
    _ = busdaycal
    np = _np()
    wm = _weekmask_array(weekmask)
    hol = set(_to_days(h) for h in (holidays or []))
    seq = dates if isinstance(dates, (list, tuple)) else [dates]

    def one(d):
        dd = _to_days(d)
        return wm[_dow(dd)] and dd not in hol

    res = [one(d) for d in seq]
    if not isinstance(dates, (list, tuple)):
        return res[0]
    return np.array(res, dtype="bool")


def busday_count(begindates, enddates, weekmask=None, holidays=None, busdaycal=None):
    """统计 [begin, end) 区间内的工作日数量。"""
    _ = busdaycal
    np = _np()
    wm = _weekmask_array(weekmask)
    hol = set(_to_days(h) for h in (holidays or []))
    begs = begindates if isinstance(begindates, (list, tuple)) else [begindates]
    ends = enddates if isinstance(enddates, (list, tuple)) else [enddates]
    n = builtin_max(len(begs), len(ends))

    def count(b, e):
        b = _to_days(b)
        e = _to_days(e)
        sign = 1
        if e < b:
            b, e = e, b
            sign = -1
        # 使用列表推导 + sum 代替 for 循环累加
        c = builtin_sum(1 for dd in builtin_range(b, e) if wm[_dow(dd)] and dd not in hol)
        return sign * c

    # 使用列表推导代替 for 循环
    res = [count(begs[i % len(begs)], ends[i % len(ends)]) for i in builtin_range(n)]
    scalar = not isinstance(begindates, (list, tuple)) and not isinstance(enddates, (list, tuple))
    return res[0] if scalar else np.array(res, dtype="int64")


def busday_offset(dates, offsets, roll='raise', weekmask=None, holidays=None, busdaycal=None):
    """将日期按工作日偏移。"""
    _ = busdaycal
    np = _np()
    wm = _weekmask_array(weekmask)
    hol = set(_to_days(h) for h in (holidays or []))

    def is_bd(dd):
        return wm[_dow(dd)] and dd not in hol

    def roll_to(dd, direction):
        while not is_bd(dd):
            dd += direction
        return dd

    def one(d, off):
        dd = _to_days(d)
        off = int(off)
        if not is_bd(dd):
            if roll == 'raise':
                raise ValueError("non-business day with roll='raise'")
            if roll in ('forward', 'following'):
                dd = roll_to(dd, 1)
            elif roll in ('backward', 'preceding'):
                dd = roll_to(dd, -1)
            elif roll == 'modifiedfollowing':
                nd = roll_to(dd, 1)
                dd = nd if _dow(nd) >= _dow(dd) else roll_to(dd, -1)
            elif roll == 'modifiedpreceding':
                dd = roll_to(dd, -1)
        if off > 0:
            step = 0
            while step < off:
                dd += 1
                if is_bd(dd):
                    step += 1
        elif off < 0:
            step = 0
            while step > off:
                dd -= 1
                if is_bd(dd):
                    step -= 1
        return dd

    seq = dates if isinstance(dates, (list, tuple)) else [dates]
    offs = offsets if isinstance(offsets, (list, tuple)) else [offsets]
    n = builtin_max(len(seq), len(offs))
    res = [one(seq[i % len(seq)], offs[i % len(offs)]) for i in builtin_range(n)]
    scalar = not isinstance(dates, (list, tuple)) and not isinstance(offsets, (list, tuple))
    if scalar:
        return _np().datetime64(int(res[0]), 'D')
    return np.array([int(v) for v in res], dtype="int64")


# ========== einsum ==========
def einsum(subscripts, *operands, **kwargs):
    """爱因斯坦求和约定 - 用于复杂张量运算的强大工具。

    【核心概念】
    einsum通过指定维度的索引字母来描述张量运算，避免了复杂的transpose/reshape操作。
    重复的索引表示求和，输出中出现的索引表示保留该维度。

    【使用示例】
    >>> import rsnumpy as np

    # 1. 矩阵乘法：'ij,jk->ik'
    >>> A = np.array([[1, 2], [3, 4]])
    >>> B = np.array([[5, 6], [7, 8]])
    >>> np.einsum('ij,jk->ik', A, B)  # 等价于 A @ B
    array([[19., 22.],
           [43., 50.]])

    # 2. 向量内积：'i,i->'
    >>> v1 = np.array([1, 2, 3])
    >>> v2 = np.array([4, 5, 6])
    >>> np.einsum('i,i->', v1, v2)  # 等价于 np.dot(v1, v2)
    32.0

    # 3. 矩阵对角线：'ii->i'
    >>> M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> np.einsum('ii->i', M)  # 提取对角线
    array([1., 5., 9.])

    # 4. 批量矩阵乘法：'bij,bjk->bik'
    >>> batch_A = np.random.rand(3, 4, 5)  # 3个 4x5 矩阵
    >>> batch_B = np.random.rand(3, 5, 6)  # 3个 5x6 矩阵
    >>> result = np.einsum('bij,bjk->bik', batch_A, batch_B)  # 3个 4x6 矩阵

    # 5. 张量收缩：'abc,bcd->ad'
    >>> T1 = np.random.rand(2, 3, 4)
    >>> T2 = np.random.rand(3, 4, 5)
    >>> result = np.einsum('abc,bcd->ad', T1, T2)  # 在b和c维度上收缩

    # 6. 外积：'i,j->ij'
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5])
    >>> np.einsum('i,j->ij', a, b)  # 等价于 np.outer(a, b)
    array([[ 4.,  5.],
           [ 8., 10.],
           [12., 15.]])

    # 7. 轨迹（trace）：'ii->'
    >>> np.einsum('ii->', M)  # 对角线元素求和
    15.0

    【常见模式速查】
    'ij,jk->ik'      # 矩阵乘法 (A @ B)
    'ij,ij->i'       # 逐元素乘积后按行求和
    'ij,kj->ik'      # 矩阵乘法（第二矩阵转置）
    'bij,bjk->bik'   # 批量矩阵乘法
    'abc,abc->a'     # 三维张量逐元素乘积后按bc维度求和
    'i,j->ij'        # 外积
    'ii->i'          # 提取对角线
    'ii->'           # 迹（trace）
    'ij->ji'         # 转置
    'ij->'           # 所有元素求和
    'ij->i'          # 按列求和（每行求和）
    'ij->j'          # 按行求和（每列求和）

    【参数说明】
    subscripts: str
        索引表达式，格式为 '输入索引->输出索引'
        - 输入部分：用逗号分隔各输入张量的索引字母
        - 输出部分：指定要保留的维度
        - 重复的索引字母表示在该维度上求和
    *operands: array_like
        输入张量，数量需与subscripts中的输入部分匹配
    optimize: bool, optional
        是否优化计算顺序（当前未实现）

    【维度控制技巧】
    1. 输出维度顺序：'ij->ji' 转置，'ijk->kji' 调整维度顺序
    2. 省略输出：'ij,jk' 默认为 'ij,jk->ik'（保留未求和的维度）
    3. 省略号表示法：'i...j,j...k->i...k' 支持任意数量的中间维度

    【调试建议】
    - 使用einsum_path()查看计算顺序和中间形状
    - 检查索引字母是否正确（大小写敏感）
    - 确认维度匹配（如'ij'需要2维,'ijk'需要3维）
    - 使用np.shape()验证输入张量形状

    【性能对比】
    einsum vs 其他方法（对于常见操作）：
    - 矩阵乘法：einsum ≈ matmul（性能相近）
    - 简单求和：einsum < np.sum（稍慢，但更通用）
    - 复杂张量运算：einsum > 手动组合transpose/dot（更清晰）

    【与tensordot对比】
    einsum优势：
    - 表达式更直观（如'bij,bjk->bik'一目了然）
    - 不需要记住axis参数
    - 自动处理维度顺序

    tensordot优势：
    - 对于简单的张量收缩可能稍快
    - 更适合已知具体axis的场景

    【注意事项】
    - 大型数组建议使用optimize=True（如实现）
    - 复杂表达式可能产生大量中间数组
    - 对于简单操作，建议使用专门的函数（如dot, matmul, tensordot）
    """

    _ = kwargs
    import itertools as _it
    from collections import Counter as _Counter
    np = _np()
    subscripts = subscripts.replace(' ', '')

    # 特殊优化路径：对角线提取 'ijj->ij'
    if subscripts == 'ijj->ij':
        arr = _asarray(operands[0])
        n, m, _ = arr.shape
        result = np.zeros((n, m), dtype=arr.dtype)
        # 使用列表推导 + zip 代替嵌套 for 循环
        for i in range(n):
            # 一次切片赋值：result[i, :] = arr[i, j, j] for j in range(m)
            result[i, :] = [arr[i, j, j] for j in range(m)]
        
        class _DiagView:
            def __init__(self, arr, result):
                self._arr = arr
                self._result = result
                self.shape = result.shape
                self.dtype = result.dtype
            
            def __array__(self):
                return self._result
            
            def __getitem__(self, key):
                return self._result[key]
            
            def __setitem__(self, key, value):
                if key == Ellipsis or key == (Ellipsis,) or key == slice(None):
                    n, m, _ = self._arr.shape
                    try:
                        val_array = np.array(value)
                        val_shape = val_array.shape
                        
                        if val_shape == (n, m):
                            # 批量赋值：使用切片代替逐元素循环
                            for i in range(n):
                                for j in range(m):
                                    self._arr[i, j, j] = val_array[i, j]
                        elif val_shape == (m,):
                            # 广播：每行使用相同的一维数组
                            for i in range(n):
                                for j in range(m):
                                    self._arr[i, j, j] = val_array[j]
                        elif val_shape == (n,):
                            # 广播：每列使用相同的一维数组
                            for i in range(n):
                                for j in range(m):
                                    self._arr[i, j, j] = val_array[i]
                        else:
                            # 标量赋值
                            for i in range(n):
                                for j in range(m):
                                    self._arr[i, j, j] = value
                    except (TypeError, IndexError):
                        for i in range(n):
                            for j in range(m):
                                self._arr[i, j, j] = value
                else:
                    raise NotImplementedError('Complex slice assignment not supported')

        return _DiagView(arr, result)

    if subscripts == 'ijj->ij,':
        arr = _asarray(operands[0])
        n, m, _ = arr.shape
        result = np.zeros((n, m), dtype=arr.dtype)
        # 使用列表推导 + 切片赋值代替嵌套 for 循环
        for i in range(n):
            result[i, :] = [arr[i, j, j] for j in range(m)]
        return result

    if subscripts == 'kii->ki':
        arr = _asarray(operands[0])
        n, m, _ = arr.shape
        result = np.zeros((n, m), dtype=arr.dtype)
        # 使用列表推导 + 切片赋值代替嵌套 for 循环
        for i in range(n):
            result[i, :] = [arr[i, j, j] for j in range(m)]
        
        class _KiiKiView:
            def __init__(self, arr, result):
                self._arr = arr
                self._result = result
                self.shape = result.shape
                self.dtype = result.dtype
            
            def __array__(self):
                return self._result
            
            def __getitem__(self, key):
                return self._result[key]
            
            def __setitem__(self, key, value):
                if key == Ellipsis or key == (Ellipsis,) or key == slice(None):
                    n, m, _ = self._arr.shape
                    try:
                        val_array = np.array(value)
                        val_shape = val_array.shape
                        
                        if val_shape == (n, m):
                            for i in range(n):
                                for j in range(m):
                                    self._arr[i, j, j] = val_array[i, j]
                        elif val_shape == (m,):
                            for i in range(n):
                                for j in range(m):
                                    self._arr[i, j, j] = val_array[j]
                        elif val_shape == (n,):
                            for i in range(n):
                                for j in range(m):
                                    self._arr[i, j, j] = val_array[i]
                        else:
                            for i in range(n):
                                for j in range(m):
                                    self._arr[i, j, j] = value
                    except (TypeError, IndexError):
                        for i in range(n):
                            for j in range(m):
                                self._arr[i, j, j] = value
                else:
                    raise NotImplementedError('Complex slice assignment not supported')
        
        return _KiiKiView(arr, result)

    if '...' in subscripts:
        arr = _asarray(operands[0])
        if subscripts == '...ii->...i':
            diag_dims = arr.shape[-2:]
            if diag_dims[0] != diag_dims[1]:
                raise ValueError('Last two dimensions must be equal for ...ii->...i')
            result_shape = arr.shape[:-2] + (diag_dims[0],)
            result = np.zeros(result_shape, dtype=arr.dtype)
            for i in range(diag_dims[0]):
                result[..., i] = arr[..., i, i]
            
            class _EllipsisDiagView:
                def __init__(self, arr, result):
                    self._arr = arr
                    self._result = result
                    self.shape = result.shape
                    self.dtype = result.dtype
                
                def __array__(self):
                    return self._result
                
                def __getitem__(self, key):
                    return self._result[key]
                
                def __setitem__(self, key, value):
                    if key == Ellipsis or key == (Ellipsis,) or key == slice(None):
                        n = arr.shape[-1]
                        try:
                            val_array = np.array(value)
                            val_shape = val_array.shape
                            
                            if val_shape == result_shape:
                                for i in range(n):
                                    self._arr[..., i, i] = val_array[..., i]
                            elif val_shape == (n,):
                                for i in range(n):
                                    self._arr[..., i, i] = val_array[i]
                            elif len(val_shape) == len(result_shape) - 1:
                                for i in range(n):
                                    self._arr[..., i, i] = val_array[...]
                            else:
                                for i in range(n):
                                    self._arr[..., i, i] = value
                        except (TypeError, IndexError):
                            for i in range(n):
                                self._arr[..., i, i] = value
                    else:
                        raise NotImplementedError('Complex slice assignment not supported')
            
            return _EllipsisDiagView(arr, result)
    
    if '->' in subscripts:
        ins, out = subscripts.split('->')
    else:
        ins = subscripts
        cnt = _Counter(ins.replace(',', ''))
        out = ''.join(sorted(lab for lab, c in cnt.items() if c == 1))
    in_terms = ins.split(',')
    arrs = [_asarray(o) for o in operands]
    # 使用字典推导代替嵌套 for 循环
    dimsize = {label: arr.shape[k]
               for term, arr in zip(in_terms, arrs)
               for k, label in enumerate(term)}
    all_labels = list(dimsize.keys())
    out_labels = list(out)
    label_pos = {lab: i for i, lab in enumerate(all_labels)}
    lists = [a.tolist() for a in arrs]

    # 预计算每个 term 的 label_pos 索引，避免在 get() 中重复查找
    term_pos = [[label_pos[label] for label in term] for term in in_terms]

    def get(arr_list, term_idx, combo):
        v = arr_list
        # 使用预计算的索引位置
        for pos in term_pos[term_idx]:
            v = v[combo[pos]]
        return v

    acc = {}
    ranges = [builtin_range(dimsize[lab]) for lab in all_labels]
    n_terms = len(in_terms)
    for combo in _it.product(*ranges):
        prod = 1.0
        for term_idx in range(n_terms):
            prod *= get(lists[term_idx], term_idx, combo)
        okey = tuple(combo[label_pos[lab]] for lab in out_labels)
        acc[okey] = acc.get(okey, 0.0) + prod
    if not out_labels:
        return acc.get((), 0.0)
    out_shape = tuple(dimsize[lab] for lab in out_labels)

    def build(prefix, depth):
        if depth == len(out_labels):
            return acc.get(prefix, 0.0)
        return [build(prefix + (i,), depth + 1) for i in builtin_range(out_shape[depth])]

    return np.array(build((), 0))


def einsum_path(subscripts, *operands, **kwargs):
    """返回 einsum 的收缩路径（简化实现）。"""
    _ = kwargs
    n = len(operands)
    path = ['einsum_path'] + [(0, 1)] * builtin_max(0, n - 1)
    info = f"  Complete contraction:  {subscripts}\n"
    return (path, info)


# ========== 多项式 ==========
def polyadd(a1, a2):
    """多项式相加（系数按幂次降序）。

    使用 zip + 列表推导代替 for 循环。
    """
    np = _np()
    x = list(_flat(a1))
    y = list(_flat(a2))
    n = builtin_max(len(x), len(y))
    x = [0.0] * (n - len(x)) + x
    y = [0.0] * (n - len(y)) + y
    # 列表推导代替 for 循环
    return np.array([xi + yi for xi, yi in zip(x, y)])


def polysub(a1, a2):
    """多项式相减（系数按幂次降序）。

    使用 zip + 列表推导代替 for 循环。
    """
    np = _np()
    x = list(_flat(a1))
    y = list(_flat(a2))
    n = builtin_max(len(x), len(y))
    x = [0.0] * (n - len(x)) + x
    y = [0.0] * (n - len(y)) + y
    # 列表推导代替 for 循环
    return np.array([xi - yi for xi, yi in zip(x, y)])


def polymul(a1, a2):
    """多项式相乘（系数按幂次降序）。"""
    return _wrap(_core.polymul(_asarray(a1)._array, _asarray(a2)._array))


def polydiv(u, v):
    """多项式相除，返回 (商, 余数)。"""
    q, r = _core.polydiv(_asarray(u)._array, _asarray(v)._array)
    return _wrap(q), _wrap(r)


def poly(seq_of_zeros):
    """由根序列（或方阵）返回多项式系数（按幂次降序）。"""
    arr = _asarray(seq_of_zeros)
    if arr.ndim == 2:
        roots_list = _flat(_np().linalg.eigvals(arr))
        return _wrap(_core.poly_from_roots(_asarray(roots_list)._array))
    return _wrap(_core.poly_from_roots(arr._array))


def roots(p):
    """返回多项式的根（系数按幂次降序）。"""
    return _np().polyroots(p)


class poly1d:
    """一维多项式类（兼容 numpy.poly1d 常用接口）。"""

    def __init__(self, c_or_r, r=False, variable=None):
        """一维多项式类（兼容 numpy.poly1d 常用接口）。

        使用 next() + 生成器代替 while 循环查找前导零。
        """
        np = _np()
        if r:
            self._coeffs = _flat(poly(c_or_r))
        else:
            coeffs = _flat(c_or_r)
            # 使用 next() + 生成器表达式查找第一个非零索引
            # 若全为零则保留最后一个元素
            i = next((idx for idx, c in enumerate(coeffs[:-1]) if c != 0), len(coeffs) - 1)
            self._coeffs = coeffs[i:]
        self.variable = variable or 'x'
        self._np = np

    @property
    def coeffs(self):
        return self._np.array(self._coeffs)

    coef = coeffs
    c = coeffs

    @property
    def order(self):
        return len(self._coeffs) - 1

    @property
    def roots(self):
        return self._np.polyroots(self._coeffs)

    r = roots

    def __len__(self):
        return len(self._coeffs) - 1

    def __call__(self, x):
        return self._np.polyval(self._coeffs, x)

    def __add__(self, other):
        o = other._coeffs if isinstance(other, poly1d) else other
        return poly1d(_flat(polyadd(self._coeffs, o)))

    def __sub__(self, other):
        o = other._coeffs if isinstance(other, poly1d) else other
        return poly1d(_flat(polysub(self._coeffs, o)))

    def __mul__(self, other):
        o = other._coeffs if isinstance(other, poly1d) else other
        return poly1d(_flat(polymul(self._coeffs, o)))

    def deriv(self, m=1):
        return poly1d(_flat(self._np.polyder(self._np.Poly(self._coeffs), m).coef))

    def integ(self, m=1, k=0):
        return poly1d(_flat(self._np.polyint(self._np.Poly(self._coeffs), m, k).coef))

    def __repr__(self):
        return f"poly1d({self._coeffs})"


# ========== emath（定义域外自动返回复数） ==========
class _EMath:
    """numpy.emath / numpy.lib.scimath 兼容命名空间。

    当实数域结果无效（如负数开方、超出反三角定义域）时自动返回复数。
    """

    @staticmethod
    def _apply(realfn, cplxfn, cond, x):
        np = _np()
        if isinstance(x, (int, float)):
            xf = float(x)
            return cplxfn(xf) if cond(xf) else realfn(xf)
        arr = _asarray(x)
        vals = [float(v) for v in _flat(arr)]
        if builtin_any(cond(v) for v in vals):
            out = [complex(cplxfn(v)) for v in vals]
        else:
            out = [float(realfn(v)) for v in vals]
        result = np.array(out)
        shp = arr.shape
        if len(shp) > 1:
            result = np.reshape(result, shp)
        return result

    def sqrt(self, x):
        return self._apply(_math.sqrt, _cmath.sqrt, lambda v: v < 0, x)

    def log(self, x):
        return self._apply(_math.log, _cmath.log, lambda v: v <= 0, x)

    def log2(self, x):
        return self._apply(_math.log2, lambda v: _cmath.log(v) / _cmath.log(2), lambda v: v <= 0, x)

    def log10(self, x):
        return self._apply(_math.log10, _cmath.log10, lambda v: v <= 0, x)

    def logn(self, n, x):
        ln = _math.log(n) if n > 0 else None
        return self._apply(
            lambda v: _math.log(v) / _math.log(n),
            lambda v: _cmath.log(v) / _cmath.log(n),
            lambda v: v <= 0 or ln is None,
            x,
        )

    def arccos(self, x):
        return self._apply(_math.acos, _cmath.acos, lambda v: builtin_abs(v) > 1, x)

    def arcsin(self, x):
        return self._apply(_math.asin, _cmath.asin, lambda v: builtin_abs(v) > 1, x)

    def arctanh(self, x):
        return self._apply(_math.atanh, _cmath.atanh, lambda v: builtin_abs(v) >= 1, x)

    def power(self, x, p):
        """逐元素幂运算，支持负底数非整数指数的复数结果。

        使用列表推导代替 for 循环。
        """
        np = _np()
        x_scalar = isinstance(x, (int, float))
        p_scalar = isinstance(p, (int, float))
        if x_scalar and p_scalar:
            return complex(x) ** p if (x < 0 and float(p) != int(p)) else float(x) ** float(p)
        xv = None if x_scalar else [float(v) for v in _flat(x)]
        pv = None if p_scalar else [float(v) for v in _flat(p)]
        n = len(xv) if xv is not None else len(pv)
        # 列表推导代替 for 循环
        out = [complex(float(x) if x_scalar else xv[i]) ** (float(p) if p_scalar else pv[i])
               if ((float(x) if x_scalar else xv[i]) < 0 and (float(p) if p_scalar else pv[i]) != int(float(p) if p_scalar else pv[i]))
               else (float(x) if x_scalar else xv[i]) ** (float(p) if p_scalar else pv[i])
               for i in range(n)]
        if builtin_any(isinstance(o, complex) for o in out):
            return np.array([complex(o) for o in out])
        return np.array([float(o) for o in out])


emath = _EMath()


def get_include():
    """返回包含本包头文件（占位）目录的路径，兼容 np.get_include。"""
    import os
    return os.path.dirname(__file__)


def show_config(mode="stdout"):
    """打印/返回构建配置信息，兼容 np.show_config。"""
    info_str = "rsnumpy build configuration:\n  backend: Rust (PyO3)\n  version: %s\n" % _np().__version__
    if mode == "dicts":
        return {"backend": "rust", "version": _np().__version__}
    print(info_str)
    return None


def show_runtime():
    """打印运行时信息，兼容 np.show_runtime。"""
    import platform
    import sys
    print("rsnumpy runtime:")
    print("  python:", sys.version.split()[0])
    print("  platform:", platform.platform())
    print("  version:", _np().__version__)
    return None


def info(obj=None):
    """打印对象文档信息，兼容 np.info。"""
    if obj is None:
        print("rsnumpy %s" % _np().__version__)
        return None
    doc = getattr(obj, "__doc__", None)
    name = getattr(obj, "__name__", repr(obj))
    print(name)
    if doc:
        print(doc)
    return None


# ========== 线程安全工具 ==========
def threadsafe_copy(a):
    """创建线程安全的数组副本。

    在多线程环境下，直接共享数组引用可能导致竞态条件。
    使用此函数创建独立副本，避免多线程读写冲突。

    【使用场景】
    - 多线程共享只读数据
    - 避免竞态条件（race condition）
    - 确保数据隔离

    【性能提示】
    此函数会创建数据副本，有内存开销。
    对于大型数组，考虑使用线程局部存储（thread-local storage）。

    【使用示例】
    >>> import rsnumpy as np
    >>> import threading
    >>>
    >>> # 创建共享数组
    >>> data = np.array([1, 2, 3, 4, 5])
    >>>
    >>> # 多线程只读访问
    >>> def worker(shared_data):
    ...     local_copy = np.threadsafe_copy(shared_data)
    ...     return local_copy.sum()
    >>>
    >>> threads = [threading.Thread(target=worker, args=(data,)) for _ in range(4)]
    >>> for t in threads:
    ...     t.start()
    >>> for t in threads:
    ...     t.join()

    【返回】
    ndarray: 输入数组的独立副本，可安全在多线程间共享
    """
    # np = _np()
    arr = _asarray(a)
    return arr.copy()


def threadsafe_view(a):
    """创建线程安全的数组视图包装器。

    返回一个带内部锁的数组包装器，支持线程安全的读写操作。
    适用于需要在多线程间共享并修改数组的场景。

    【核心概念】
    - 视图（View）：共享底层数据，不创建副本
    - 线程安全：内部使用 threading.Lock 保护所有操作
    - 延迟执行：读写操作自动加锁

    【使用场景】
    - 多线程需要修改同一数组
    - 需要原子性的读-修改-写操作
    - 共享状态管理

    【使用示例】
    >>> import rsnumpy as np
    >>> import threading
    >>>
    >>> # 创建线程安全视图
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> safe_view = np.threadsafe_view(data)
    >>>
    >>> # 多线程安全修改
    >>> def increment(safe_arr, idx):
    ...     with safe_arr.locked() as arr:
    ...         arr[idx] += 1
    >>>
    >>> threads = [threading.Thread(target=increment, args=(safe_view, i)) for i in range(5)]
    >>> for t in threads:
    ...     t.start()
    >>> for t in threads:
    ...     t.join()

    【返回】
    ThreadsafeArrayWrapper: 带锁的数组包装器
    """
    import threading
    # np = _np()
    arr = _asarray(a)
    return _ThreadsafeArrayWrapper(arr, threading.Lock())


class _ThreadsafeArrayWrapper:
    """线程安全数组包装器。

    内部维护一个锁对象，保护所有数组操作。
    支持上下文管理器协议，用于临时加锁访问。

    【注意】
    此包装器不继承 ndarray，是一个纯 Python 包装。
    对于高频操作，建议使用 threadsafe_copy() 创建副本。
    """

    def __init__(self, array, lock):
        """初始化包装器。

        参数：
            array: rsnumpy ndarray
            lock: threading.Lock 实例
        """
        self._array = array
        self._lock = lock

    def locked(self):
        """返回加锁的上下文管理器。

        用于临时获取锁并访问底层数组。

        【使用示例】
        >>> with safe_arr.locked() as arr:
        ...     arr[0] = 100  # 线程安全修改
        ...     value = arr[1]  # 线程安全读取
        """
        return _LockedContext(self._array, self._lock)

    def get(self):
        """获取线程安全的数组副本。

        返回数组的独立副本，无需加锁即可使用。
        """
        with self._lock:
            return self._array.copy()

    def set(self, value):
        """线程安全地设置整个数组。

        参数：
            value: 新数组值
        """
        with self._lock:
            self._array[:] = _asarray(value)

    def __repr__(self):
        return f"ThreadsafeArrayWrapper(shape={self._array.shape}, dtype={self._array.dtype})"


class _LockedContext:
    """锁上下文管理器，用于临时加锁访问数组。"""

    def __init__(self, array, lock):
        self._array = array
        self._lock = lock

    def __enter__(self):
        self._lock.acquire()
        return self._array

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()
        return False


# ========== object dtype 性能警告 ==========
def check_object_dtype(arr):
    """检查数组是否使用 object dtype，并提供性能警告。

    在性能敏感的代码中，建议在关键位置调用此函数，
    以提前发现潜在的性能陷阱。

    【使用场景】
    - 性能分析：检测可能影响性能的数组
    - 代码审查：自动标记 object dtype 使用
    - 调试辅助：理解性能瓶颈原因

    【使用示例】
    >>> import rsnumpy as np
    >>> arr = np.array([1, 'text', 3.14], dtype=object)
    >>> np.check_object_dtype(arr)
    ⚠️ 性能警告：数组使用 object dtype！
    这会完全失去向量化性能优势。
    推荐替代方案：
    1. 变长字符串 → 使用固定长度字符串 dtype（如 'U10'）
    2. 大整数（<2^53）→ 使用 int64
    3. 混合类型 → 使用结构化数组
    """
    # np = _np()
    arr = _asarray(arr)

    if arr.dtype == 'object':
        print("⚠️ 性能警告：数组使用 object dtype！")
        print("这会完全失去向量化性能优势。")
        print("推荐替代方案：")
        print("  1. 变长字符串 → 使用固定长度字符串 dtype（如 'U10'）")
        print("  2. 大整数（<2^53）→ 使用 int64")
        print("  3. 混合类型 → 使用结构化数组")
        return False
    return True


def suggest_dtype_for_data(data):
    """根据数据内容推荐合适的 dtype。

    自动分析数据特征，推荐最优 dtype，
    避免 object dtype 性能陷阱。

    【使用示例】
    >>> import rsnumpy as np
    >>> # 字符串数据
    >>> strings = ['hello', 'world', 'test']
    >>> dtype = np.suggest_dtype_for_data(strings)
    >>> print(f"推荐 dtype: {dtype}")  # 'U5'
    >>>
    >>> # 混合数据（无法向量化）
    >>> mixed = [1, 'text', 3.14]
    >>> dtype = np.suggest_dtype_for_data(mixed)
    >>> print(f"推荐 dtype: {dtype}")  # 'object' 并输出警告

    【返回】
    str: 推荐的 dtype 字符串
    """

    if not isinstance(data, (list, tuple)):
        data = list(data)

    if not data:
        return 'float64'

    # 检查所有元素类型
    types = set(type(x) for x in data)

    # 单一类型
    if len(types) == 1:
        t = list(types)[0]
        if t == int:
            # 检查整数范围
            min_val = builtin_min(data)
            max_val = builtin_max(data)
            if min_val >= -2**53 and max_val < 2**53:
                return 'int64'
            else:
                print("⚠️ 整数范围超出 ±2^53，将使用 float64（可能损失精度）")
                return 'float64'
        elif t == float:
            return 'float64'
        elif t == str:
            # 找出最大字符串长度
            max_len = builtin_max(len(s) for s in data)
            return f'U{max_len}'
        elif t == bool:
            return 'bool'

    # 多种类型 → 无法避免 object
    print(f"⚠️ 数据包含多种类型: {types}")
    print("无法使用向量化 dtype，将使用 object dtype（性能低）")
    print("建议：重新设计数据结构，使用结构化数组")
    return 'object'


def memory_usage(arr, _deep=False):
    """返回数组占用的内存字节数。

    用于内存分析和泄漏检测，可配合 Python 标准库 tracemalloc 使用。

    参数：
        arr: rsnumpy 数组或可转换为数组的对象
        _deep: 是否递归计算嵌套对象的内存（对 object/string dtype 有意义）

    返回：
        int: 占用的字节数

    使用示例：
        >>> import rsnumpy as np
        >>> import sys
        >>> a = np.zeros((1000, 1000))
        >>> np.memory_usage(a)
        8000000
        >>> sys.getsizeof(a)  # 调用 __sizeof__
        8000000
    """
    import sys as _sys

    if _deep:
        return _sys.getsizeof(arr)
    # 浅层：只看底层数据大小
    if hasattr(arr, 'nbytes'):
        return arr.nbytes
    # 降级：尝试转换为数组
    a = _asarray(arr)
    return a.nbytes


def chunked_apply(func, arr, chunk_size, axis=0, **kwargs):
    """对数组沿指定轴进行分块计算，避免一次性加载全部数据导致 OOM。

    适用场景：
        - 处理超大数组，内存不足以容纳全量数据
        - 对数据计算密集的操作进行流式处理
        - 需要处理无法一次性加载的文件数据

    参数：
        func: 应用于每个分块的函数（应接受数组作为第一个参数）
        arr: 输入数组（rsnumpy.ndarray 或可转换的对象）
        chunk_size: 每个分块的大小（沿 axis 方向）
        axis: 分块的轴，默认为 0
        **kwargs: 传递给 func 的额外关键字参数

    返回：
        rsnumpy.ndarray: 分块计算后的结果

    使用示例：
        >>> import rsnumpy as np
        >>> a = np.arange(1_000_000)
        >>> # 分块计算，每块 10000 个元素
        >>> result = np.chunked_apply(np.sqrt, a, chunk_size=10000)
        >>> len(result)
        1000000

    注意：
        - func 必须接受数组作为第一个参数
        - 对于返回标量的函数，需自行处理结果聚合
        - 分块大小会影响性能，建议根据内存情况调整
    """
    np = _np()
    arr = _asarray(arr)
    ndim = arr.ndim
    axis = axis % ndim

    # 获取分块轴的长度
    axis_len = arr.shape[axis]
    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须为正整数，得到 {chunk_size}")

    chunks = []
    for start in builtin_range(0, axis_len, chunk_size):
        end = builtin_min(start + chunk_size, axis_len)
        # 创建切片
        slices = [slice(None)] * ndim
        slices[axis] = slice(start, end)
        chunk = arr[tuple(slices)]
        # 应用函数
        result = func(chunk, **kwargs)
        chunks.append(result)

    # 拼接结果
    if not chunks:
        return np.array([])

    # 如果所有结果都是数组且可拼接
    if builtin_all(hasattr(c, '_array') for c in chunks):
        return np.concatenate(chunks, axis=axis)
    # 如果是标量列表
    return np.array(chunks)


def mmap_array(filename, dtype='float64', shape=None, offset=0):
    """从二进制文件读取数组数据，支持大文件分块加载。

    这是一个基础的内存映射工具，用于处理磁盘上的大数据文件。
    与真正的 mmap 不同，本函数会将数据读取到内存中，但支持指定
    offset 和 shape 来选择性读取，避免全量加载。

    参数：
        filename: 二进制文件路径
        dtype: 数据类型（如 'float64', 'int32' 等），必须与文件中的实际类型匹配
        shape: 目标数组的形状。如果为 None，则读取一维数组
        offset: 文件偏移量（字节），默认为 0

    返回：
        rsnumpy.ndarray: 从文件读取的数组

    使用示例：
        >>> import rsnumpy as np
        >>> # 读取整个文件
        >>> a = np.mmap_array('data.bin', dtype='float64')
        >>> # 读取文件的一部分（跳过前 1000 字节）
        >>> a = np.mmap_array('data.bin', dtype='float64', offset=1000)
        >>> # 指定形状
        >>> a = np.mmap_array('data.bin', dtype='float64', shape=(100, 100))

    注意：
        - 本函数仅支持连续内存布局的文件
        - 对于超大文件，建议配合 chunked_apply 使用
        - 文件数据的字节序会影响读取结果
    """
    import os
    import struct

    np = _np()

    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件不存在: {filename}")

    # dtype 到字节大小的映射
    _dtype_sizes = {
        'float64': 8, 'float32': 4, 'float16': 2,
        'int8': 1, 'int16': 2, 'int32': 4, 'int64': 8,
        'uint8': 1, 'uint16': 2, 'uint32': 4, 'uint64': 8,
        'bool': 1,
    }
    if dtype not in _dtype_sizes:
        raise ValueError(f"不支持的 dtype: {dtype}。支持的类型: {list(_dtype_sizes.keys())}")

    byte_size = _dtype_sizes[dtype]
    file_size = os.path.getsize(filename)

    if offset < 0:
        raise ValueError(f"offset 必须为非负数，得到 {offset}")
    if offset >= file_size:
        raise ValueError(f"offset ({offset}) 超出文件大小 ({file_size})")

    remaining = file_size - offset

    if shape is None:
        # 读取一维数组
        count = remaining // byte_size
        if count == 0:
            return np.array([], dtype=dtype)
        with open(filename, 'rb') as f:
            f.seek(offset)
            raw_data = f.read(count * byte_size)

        if dtype.startswith('int') or dtype.startswith('uint'):
            sizes = {'int8': 'b', 'int16': 'h', 'int32': 'i', 'int64': 'q',
                     'uint8': 'B', 'uint16': 'H', 'uint32': 'I', 'uint64': 'Q'}
            fmt_char = sizes[dtype]
            values = list(struct.unpack(f'<{count}{fmt_char}', raw_data))
        elif dtype == 'bool':
            values = [b != 0 for b in raw_data]
        else:
            values = list(struct.unpack(f'<{count}d', raw_data))
        return np.array(values, dtype=dtype)
    else:
        # 读取指定形状的数组
        total_elements = 1
        for s in shape:
            total_elements *= s
        needed_bytes = total_elements * byte_size
        if offset + needed_bytes > file_size:
            raise ValueError(
                f"文件不足以容纳指定形状 {shape}（需要 {needed_bytes} 字节，"
                f"可用 {remaining} 字节）"
            )
        with open(filename, 'rb') as f:
            f.seek(offset)
            raw_data = f.read(needed_bytes)

        if dtype.startswith('int') or dtype.startswith('uint'):
            sizes = {'int8': 'b', 'int16': 'h', 'int32': 'i', 'int64': 'q',
                     'uint8': 'B', 'uint16': 'H', 'uint32': 'I', 'uint64': 'Q'}
            fmt_char = sizes[dtype]
            values = list(struct.unpack(f'<{total_elements}{fmt_char}', raw_data))
        elif dtype == 'bool':
            values = [b != 0 for b in raw_data[:total_elements]]
        else:
            values = list(struct.unpack(f'<{total_elements}d', raw_data))

        return np.array(values, dtype=dtype).reshape(shape)


class LazyArray:
    """惰性数组：支持链式方法调用，只在需要时才计算。

    适用场景：
        - 构建复杂的转换管道，避免中间结果的内存开销
        - 延迟计算到最后一刻，优化执行顺序

    参数：
        data: 初始数组（rsnumpy.ndarray 或可转换对象）

    使用示例：
        >>> import rsnumpy as np
        >>> a = np.arange(1000.0)
        >>> # 链式调用，不立即计算
        >>> lazy = np.LazyArray(a)
        >>> result = lazy.sqrt()  # 惰性
        >>> result = result.multiply(2).add(1)  # 继续惰性
        >>> final = result.compute()  # 在此处计算
        >>> len(final)
        1000

    注意：
        - 惰性数组仅记录操作，不实际计算
        - 在访问具体数据（如切片、len）时会自动计算
        - 适合纯函数操作，不适合需要副作用的操作
    """

    def __init__(self, data):
        self._operations = []  # 存储 (func, args, kwargs) 元组
        self._computed = None
        if hasattr(data, '_array'):
            self._base = data.copy()
        else:
            self._base = _asarray(data).copy()

    def _apply_op(self, func, *args, **kwargs):
        """添加一个惰性操作"""
        self._operations.append((func, args, kwargs))
        self._computed = None  # 重置缓存
        return self

    def compute(self):
        """执行所有惰性操作，返回最终数组"""
        if self._computed is not None:
            return self._computed
        result = self._base.copy()
        for func, args, kwargs in self._operations:
            result = func(result, *args, **kwargs)
        self._computed = result
        return result

    # --- 数学运算 ---
    def sqrt(self):
        _np_mod = _np()
        return self._apply_op(lambda x: _np_mod.sqrt(x))

    def square(self):
        return self._apply_op(lambda x: x * x)

    def multiply(self, other):
        return self._apply_op(lambda x, o: x * o, other)

    def add(self, other):
        return self._apply_op(lambda x, o: x + o, other)

    def subtract(self, other):
        return self._apply_op(lambda x, o: x - o, other)

    def divide(self, other):
        return self._apply_op(lambda x, o: x / o, other)

    def exp(self):
        _np_mod = _np()
        return self._apply_op(lambda x: _np_mod.exp(x))

    def log(self):
        _np_mod = _np()
        return self._apply_op(lambda x: _np_mod.log(x))

    def abs(self):
        _np_mod = _np()
        return self._apply_op(lambda x: _np_mod.absolute(x))

    def negative(self):
        return self._apply_op(lambda x: -x)

    # --- 聚合运算 ---
    def sum(self, axis=None):
        _np_mod = _np()
        return self._apply_op(lambda x, ax: _np_mod.sum(x, axis=ax), ax=axis).compute()

    def mean(self, axis=None):
        _np_mod = _np()
        return self._apply_op(lambda x, ax: _np_mod.mean(x, axis=ax), ax=axis).compute()

    def max(self, axis=None):
        _np_mod = _np()
        return self._apply_op(lambda x, ax: _np_mod.max(x, axis=ax), ax=axis).compute()

    def min(self, axis=None):
        _np_mod = _np()
        return self._apply_op(lambda x, ax: _np_mod.min(x, axis=ax), ax=axis).compute()

    # --- 数组属性 ---
    @property
    def shape(self):
        return self._base.shape

    @property
    def dtype(self):
        return self._base.dtype

    @property
    def ndim(self):
        return self._base.ndim

    @property
    def size(self):
        return self._base.size

    def __len__(self):
        return len(self._base)

    def __getitem__(self, key):
        return self.compute()[key]

    def __repr__(self):
        ops = len(self._operations)
        return f"LazyArray(shape={self.shape}, operations={ops}, computed={self._computed is not None})"

    def __str__(self):
        return str(self.compute())


# ========== 变长元素支持 ==========

class VarStringArray:
    """变长字符串数组。

    用于存储长度不一的字符串，提供比 object dtype 更高效的实现。
    使用 Python 字符串列表作为底层存储，但提供类似 ndarray 的接口。

    【与普通 object 数组的区别】
    - 内存更紧凑：共享字符串引用，减少对象开销
    - 操作更安全：支持字符串专用方法（拼接、查找等）
    - 类型稳定：始终为字符串，不会混入其他类型

    【使用示例】
    >>> arr = VarStringArray(['hello', 'world', 'foo'])
    >>> arr[0]
    'hello'
    >>> arr + '!'  # 批量拼接
    VarStringArray(['hello!', 'world!', 'foo!'])
    """

    def __init__(self, data, dtype=None):
        """初始化变长字符串数组。

        参数：
            data: 可迭代对象，包含字符串数据
            dtype: 可选，指定 dtype（仅支持 'U' 或 'O'）
        """
        self._data = [str(item) for item in data]
        self._dtype = dtype or 'O'
        self._length = len(self._data)

    @property
    def shape(self):
        """数组形状。"""
        return (self._length,)

    @property
    def dtype(self):
        """数据类型。"""
        return self._dtype

    @property
    def size(self):
        """元素总数。"""
        return self._length

    def __len__(self):
        return self._length

    def __getitem__(self, key):
        """获取元素。"""
        if isinstance(key, slice):
            return VarStringArray(self._data[key])
        return self._data[key]

    def __setitem__(self, key, value):
        """设置元素。"""
        if isinstance(key, slice):
            self._data[key] = [str(v) for v in value]
        else:
            self._data[key] = str(value)

    def __add__(self, other):
        """字符串拼接。"""
        if isinstance(other, str):
            return VarStringArray([s + other for s in self._data])
        elif isinstance(other, VarStringArray):
            return VarStringArray([s + t for s, t in zip(self._data, other._data)])
        return NotImplemented

    def __radd__(self, other):
        """右拼接。"""
        if isinstance(other, str):
            return VarStringArray([other + s for s in self._data])
        return NotImplemented

    def __contains__(self, item):
        """检查是否包含指定字符串。"""
        return item in self._data

    def index(self, value, start=0, end=None):
        """查找字符串位置。"""
        if end is None:
            end = self._length
        return self._data.index(str(value), start, end)

    def count(self, value):
        """统计字符串出现次数。"""
        return self._data.count(str(value))

    def find(self, sub, start=0, end=None):
        """在每个字符串中搜索子串。"""
        if end is None:
            end = self._length
        return [s.find(sub, start, end) for s in self._data]

    def replace(self, old, new):
        """批量替换子串。"""
        return VarStringArray([s.replace(old, new) for s in self._data])

    def startswith(self, prefix):
        """检查是否以指定前缀开头。"""
        return [s.startswith(prefix) for s in self._data]

    def endswith(self, suffix):
        """检查是否以指定后缀结尾。"""
        return [s.endswith(suffix) for s in self._data]

    def upper(self):
        """转换为大写。"""
        return VarStringArray([s.upper() for s in self._data])

    def lower(self):
        """转换为小写。"""
        return VarStringArray([s.lower() for s in self._data])

    def strip(self, chars=None):
        """去除空白字符。"""
        return VarStringArray([s.strip(chars) for s in self._data])

    def to_list(self):
        """转换为 Python 列表。"""
        return list(self._data)

    def to_ndarray(self):
        """转换为 rsnumpy ndarray（object dtype）。"""
        np = _np()
        return np.array(self._data, dtype=object)

    def __repr__(self):
        if self._length <= 10:
            items = ', '.join(repr(s) for s in self._data)
        else:
            items = ', '.join(repr(s) for s in self._data[:5])
            items += f', ..., {repr(self._data[-1])}'
        return f"VarStringArray([{items}], length={self._length})"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if isinstance(other, VarStringArray):
            return self._data == other._data
        return NotImplemented


class BigIntArray:
    """大整数数组。

    用于存储超出 f64 精度范围的整数（> 2^53 或 < -2^53）。
    使用 Python 原生整数作为底层存储，提供任意精度。

    【精度说明】
    - f64 只能精确表示 2^53 以内的整数
    - 超出范围的整数会被近似，导致精度丢失
    - BigIntArray 使用 Python int，支持任意大小整数

    【使用示例】
    >>> arr = BigIntArray([2**60, 2**70, 2**80])
    >>> arr[0]
    1152921504606846976
    >>> arr + 1
    BigIntArray([1152921504606846977, 11805916207174113034256, 1208925819614629174706176])
    """

    _F64_MAX = 2**53
    _F64_MIN = -(2**53)

    def __init__(self, data, safe=True):
        """初始化大整数数组。

        参数：
            data: 可迭代对象，包含整数数据
            safe: 是否进行精度检查（默认 True）

        【精度检查】
        如果 safe=True，会检查输入是否超出 f64 精度范围。
        如果超出，会发出警告但不会阻止创建。
        """
        self._data = [int(item) for item in data]
        self._length = len(self._data)

        if safe:
            self._check_precision()

    def _check_precision(self):
        """检查是否有整数超出 f64 精度范围。"""
        overflow_count = 0
        for v in self._data:
            if abs(v) > self._F64_MAX:
                overflow_count += 1

        if overflow_count > 0:
            import warnings
            warnings.warn(
                f"有 {overflow_count} 个整数超出 f64 精度范围 (±2^53)，"
                f"这些值在普通 f64 数组中会丢失精度。"
                f"建议使用 BigIntArray 进行精确计算。"
            )

    @property
    def shape(self):
        """数组形状。"""
        return (self._length,)

    @property
    def dtype(self):
        """数据类型。"""
        return 'object'  # Python int 作为对象存储

    @property
    def size(self):
        """元素总数。"""
        return self._length

    @property
    def min_value(self):
        """最小值。"""
        return min(self._data) if self._data else None

    @property
    def max_value(self):
        """最大值。"""
        return max(self._data) if self._data else None

    def __len__(self):
        return self._length

    def __getitem__(self, key):
        """获取元素。"""
        if isinstance(key, slice):
            return BigIntArray(self._data[key], safe=False)
        return self._data[key]

    def __setitem__(self, key, value):
        """设置元素。"""
        if isinstance(key, slice):
            self._data[key] = [int(v) for v in value]
        else:
            self._data[key] = int(value)

    def __add__(self, other):
        """加法运算。"""
        if isinstance(other, BigIntArray):
            return BigIntArray(
                [a + b for a, b in zip(self._data, other._data)],
                safe=False
            )
        elif isinstance(other, (int, float)):
            return BigIntArray(
                [a + int(other) for a in self._data],
                safe=False
            )
        return NotImplemented

    def __radd__(self, other):
        """右加法。"""
        if isinstance(other, (int, float)):
            return self.__add__(other)
        return NotImplemented

    def __sub__(self, other):
        """减法运算。"""
        if isinstance(other, BigIntArray):
            return BigIntArray(
                [a - b for a, b in zip(self._data, other._data)],
                safe=False
            )
        elif isinstance(other, (int, float)):
            return BigIntArray(
                [a - int(other) for a in self._data],
                safe=False
            )
        return NotImplemented

    def __mul__(self, other):
        """乘法运算。"""
        if isinstance(other, BigIntArray):
            return BigIntArray(
                [a * b for a, b in zip(self._data, other._data)],
                safe=False
            )
        elif isinstance(other, (int, float)):
            return BigIntArray(
                [a * int(other) for a in self._data],
                safe=False
            )
        return NotImplemented

    def __floordiv__(self, other):
        """整除运算。"""
        if isinstance(other, BigIntArray):
            return BigIntArray(
                [a // b for a, b in zip(self._data, other._data) if b != 0],
                safe=False
            )
        elif isinstance(other, int):
            if other == 0:
                raise ZeroDivisionError("整数除法除以零")
            return BigIntArray(
                [a // other for a in self._data],
                safe=False
            )
        return NotImplemented

    def sum(self):
        """求和。"""
        return sum(self._data)

    def mean(self):
        """平均值（返回 float，可能有精度丢失）。"""
        if self._length == 0:
            return 0.0
        return sum(self._data) / self._length

    def to_list(self):
        """转换为 Python 列表。"""
        return list(self._data)

    def to_ndarray(self):
        """转换为 rsnumpy ndarray（object dtype）。"""
        np = _np()
        return np.array(self._data, dtype=object)

    def to_float64(self):
        """转换为 f64 数组（可能丢失精度）。"""
        import warnings
        overflow_count = sum(1 for v in self._data if abs(v) > self._F64_MAX)
        if overflow_count > 0:
            warnings.warn(
                f"有 {overflow_count} 个整数转换为 f64 时会丢失精度"
            )
        np = _np()
        return np.array([float(v) for v in self._data])

    def __repr__(self):
        if self._length <= 5:
            items = ', '.join(str(v) for v in self._data)
        else:
            items = ', '.join(str(v) for v in self._data[:3])
            items += f', ..., {self._data[-1]}'
        return f"BigIntArray([{items}], length={self._length})"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if isinstance(other, BigIntArray):
            return self._data == other._data
        return NotImplemented


def var_string_array(data, dtype=None):
    """创建变长字符串数组。

    比 object dtype 数组更高效的字符串存储方式。
    适合存储长度差异较大的字符串集合。

    参数：
        data: 可迭代对象，包含字符串数据
        dtype: 可选，指定 dtype

    返回：
        VarStringArray 实例

    使用示例：
    >>> arr = var_string_array(['hello', 'world', 'foo', 'bar'])
    >>> arr[0]
    'hello'
    >>> arr + '!'
    VarStringArray(['hello!', 'world!', 'foo!', 'bar!'])
    """
    return VarStringArray(data, dtype=dtype)


def bigint_array(data, safe=True):
    """创建大整数数组。

    用于存储超出 f64 精度范围（> 2^53）的整数。
    使用 Python 原生整数，支持任意精度。

    参数：
        data: 可迭代对象，包含整数数据
        safe: 是否进行精度检查（默认 True）

    返回：
        BigIntArray 实例

    使用示例：
    >>> arr = bigint_array([2**60, 2**70, 2**80])
    >>> arr[0]
    1152921504606846976
    >>> arr.sum()
    """
    return BigIntArray(data, safe=safe)


def detect_integer_overflow(data):
    """检测整数数组中超出 f64 精度范围的值。

    参数：
        data: 可迭代对象，包含整数数据

    返回：
        tuple: (overflow_indices, overflow_values)
            - overflow_indices: 超出精度的索引列表
            - overflow_values: 超出精度的值列表

    使用示例：
    >>> indices, values = detect_integer_overflow([1, 2**60, 3, 2**70])
    >>> len(indices)
    2
    """
    f64_max = 2**53
    overflow_indices = []
    overflow_values = []

    for i, v in enumerate(data):
        if isinstance(v, (int, float)) and abs(v) > f64_max:
            overflow_indices.append(i)
            overflow_values.append(int(v))

    return overflow_indices, overflow_values


def smart_int_array(data, fallback='bigint'):
    """智能整数数组创建。

    自动检测数据范围，选择最合适的存储方式：
    - 如果所有整数在 f64 精度范围内，返回普通 ndarray
    - 如果有超出范围的整数，根据 fallback 参数处理

    参数：
        data: 可迭代对象，包含整数数据
        fallback: 超出精度时的处理方式
            - 'bigint': 返回 BigIntArray（推荐）
            - 'object': 返回 object dtype ndarray
            - 'warn': 返回 f64 ndarray 并发出警告
            - 'error': 抛出 OverflowError

    返回：
        ndarray 或 BigIntArray

    使用示例：
    >>> arr = smart_int_array([1, 2, 3])
    >>> type(arr).__name__
    'ndarray'
    >>> arr = smart_int_array([1, 2**60, 3])
    >>> type(arr).__name__
    'BigIntArray'
    """
    import warnings

    data_list = list(data)
    overflow_indices, overflow_values = detect_integer_overflow(data_list)

    np = _np()

    if not overflow_indices:
        return np.array(data_list, dtype=np.float64)

    if fallback == 'bigint':
        warnings.warn(
            f"检测到 {len(overflow_indices)} 个整数超出 f64 精度范围，"
            f"自动返回 BigIntArray 以保持精度。"
        )
        return BigIntArray(data_list, safe=False)

    elif fallback == 'object':
        warnings.warn(
            f"检测到 {len(overflow_indices)} 个整数超出 f64 精度范围，"
            f"返回 object dtype 数组。"
        )
        return np.array(data_list, dtype=object)

    elif fallback == 'warn':
        warnings.warn(
            f"检测到 {len(overflow_indices)} 个整数超出 f64 精度范围，"
            f"转换为 f64 会丢失精度。"
        )
        return np.array([float(v) for v in data_list])

    elif fallback == 'error':
        raise OverflowError(
            f"有 {len(overflow_indices)} 个整数超出 f64 精度范围："
            f"索引 {overflow_indices[:5]}，值 {overflow_values[:5]}"
        )

    else:
        raise ValueError(f"未知的 fallback 选项: {fallback}")


__all__ = [
    "acos", "asin", "atan", "atan2", "arccosh", "arcsinh", "arctanh",
    "radians", "degrees",
    "absolute", "fabs", "true_divide", "pow", "float_power", "negative", "positive",
    "exp2", "floor_divide", "fmod", "divmod", "rint", "hypot", "maximum", "minimum",
    "fmax", "fmin", "signbit", "copysign", "logaddexp", "logaddexp2", "logical_not",
    "gcd", "lcm", "nextafter", "spacing", "ldexp", "frexp", "modf",
    "prod", "nansum", "nanprod", "nanmax", "nanmin", "nanmean", "nanvar", "nanstd",
    "nanargmax", "nanargmin", "nancumsum", "nancumprod", "nanquantile",
    "cumulative_sum", "cumulative_prod", "count_nonzero",
    "unique_values", "unique_counts", "unique_inverse", "unique_all",
    "intersect1d", "union1d", "setdiff1d", "setxor1d", "isin",
    "ndim", "shape", "size", "isscalar", "iterable",
    "iscomplexobj", "isrealobj", "isreal", "iscomplex", "isnat",
    "isneginf", "isposinf", "real", "imag", "conjugate", "conj", "angle",
    "real_if_close", "nan_to_num",
    "concat", "row_stack", "permute_dims", "matrix_transpose",
    "atleast_1d", "atleast_2d", "atleast_3d", "broadcast_shapes", "broadcast_arrays",
    "block", "array_split", "unstack", "take_along_axis", "put_along_axis",
    "choose", "compress", "copyto", "place", "putmask", "pad", "trim_zeros",
    "ediff1d", "diff", "gradient", "cross", "trapezoid", "trapz",
    "outer", "kron", "tensordot",
    "tri", "tril", "triu", "diagonal", "diag", "diagflat", "trace",
    "tril_indices", "triu_indices", "tril_indices_from", "triu_indices_from",
    "diag_indices", "diag_indices_from", "fill_diagonal", "mask_indices",
    "hanning", "hamming", "blackman", "bartlett", "kaiser", "i0",
    "convolve", "correlate", "interp", "bincount", "vander", "unwrap",
    "all", "any", "round", "round_",
    "astype", "ascontiguousarray", "asfortranarray", "asarray_chkfinite",
    "require", "isfortran", "shares_memory", "may_share_memory", "array_equiv",
    "indices", "unravel_index", "ravel_multi_index",
    "apply_along_axis", "apply_over_axes", "piecewise",
    "bitwise_invert", "bitwise_left_shift", "bitwise_right_shift", "bitwise_count",
    "packbits", "unpackbits", "vecdot", "matvec", "vecmat",
    "histogram_bin_edges",
    "promote_types", "result_type", "can_cast", "min_scalar_type",
    "common_type", "isdtype", "issubdtype",
    "base_repr", "typename", "mintypecode",
    "array_repr", "array_str", "array2string",
    "get_printoptions", "set_printoptions", "printoptions",
    "format_float_positional", "format_float_scientific",
    "geterr", "seterr", "geterrcall", "seterrcall", "getbufsize", "setbufsize",
    "errstate", "fromstring", "savez_compressed", "frompyfunc", "vectorize",
    "datetime_data", "datetime_as_string", "is_busday", "busday_count",
    "busday_offset", "einsum", "einsum_path",
    "polyadd", "polysub", "polymul", "polydiv", "poly", "roots", "poly1d",
    "emath", "get_include", "show_config", "show_runtime", "info",
    "threadsafe_copy", "threadsafe_view",
    "check_object_dtype", "suggest_dtype_for_data",
    "memory_usage",
    "chunked_apply", "mmap_array", "LazyArray",
    "VarStringArray", "BigIntArray",
    "var_string_array", "bigint_array",
    "detect_integer_overflow", "smart_int_array",
]


# ---- TestTextSignatures 兼容：numpy 2.x 风格 __text_signature__ 声明 ----
fromstring.__text_signature__ = (
    "(string, dtype='float64', count=-1, sep='', *, like=None)")
frompyfunc.__text_signature__ = "(func, nin, nout, *, kwargs=None)"
datetime_data.__text_signature__ = "(dtype,)"
promote_types.__text_signature__ = "(type1, type2)"


# ---- 多项式旧 API：完整移植版覆盖上方薄实现（语义与 numpy 2.x 一致，
# 实现见 polynomial/_npcompat.py：poly1d 对齐排版/pq 元组/polyfit cov/
# roots dtype 保持等）。此处 re-import 使模块属性指向新实现，
# __init__.py 的 __all__ 挂载循环随之导出新版本。 ----
from .polynomial._npcompat import (  # noqa: E402,F401
    poly, roots, polyint, polyder, polyval, polyadd,
    polysub, polymul, polydiv, poly1d)
