"""补充的 NumPy 兼容 API。

计算全部复用已有的 Rust 原语（_core 或已导出的顶层函数），本层仅做组合与包装，
不重复实现底层数值循环。热点函数可在后续通过 Rust 层进一步优化。
"""

import math as _math

import rsnumpy._core as _core

builtin_max = max


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
    """逐元素向下整除。"""
    np = _np()
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
    np = _np()
    a = _asarray(x1)
    b = _asarray(x2)
    m = maximum(a, b)
    m = np.where(np.isnan(a), b, m)
    return np.where(np.isnan(b), a, m)


def fmin(x1, x2):
    """逐元素取较小值，忽略 NaN。"""
    np = _np()
    a = _asarray(x1)
    b = _asarray(x2)
    m = minimum(a, b)
    m = np.where(np.isnan(a), b, m)
    return np.where(np.isnan(b), a, m)


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


def logical_not(x):
    """逐元素逻辑非。"""
    np = _np()
    arr = _asarray(x)
    return np.equal(arr, np.full(arr.shape, 0.0))


def _elementwise(fn, *arrays):
    """对齐形状的逐元素 Python 回退（用于 gcd/lcm/frexp 等无向量原语的场景）。"""
    np = _np()
    lists = [_asarray(a).tolist() for a in arrays]

    def rec(*vals):
        if isinstance(vals[0], list):
            return [rec(*t) for t in zip(*vals)]
        return fn(*vals)

    return np.array(rec(*lists))


def gcd(x1, x2):
    """逐元素最大公约数。"""
    return _wrap(_core.gcd(_asarray(x1)._array, _asarray(x2)._array), "int64")


def lcm(x1, x2):
    """逐元素最小公倍数。"""
    return _wrap(_core.lcm(_asarray(x1)._array, _asarray(x2)._array), "int64")


def nextafter(x1, x2):
    """逐元素返回 x1 朝 x2 方向的下一个可表示浮点数。"""
    return _wrap(_core.nextafter(_asarray(x1)._array, _asarray(x2)._array))


def spacing(x):
    """逐元素返回相邻浮点数间距。"""
    return _wrap(_core.spacing(_asarray(x)._array))


def ldexp(x1, x2):
    """逐元素返回 x1 * 2**x2。"""
    return _wrap(_core.ldexp(_asarray(x1)._array, _asarray(x2)._array))


def frexp(x):
    """逐元素返回 (尾数, 指数)。"""
    np = _np()
    fr = _elementwise(lambda a: _math.frexp(float(a))[0], x)
    ex = np.array([_math.frexp(float(v))[1] for v in _flat(x)], dtype="int64")
    return fr, ex


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
    _ = dtype, out, keepdims, initial, where
    np = _np()
    arr = _asarray(a)
    if arr.size == 0:
        return 1.0 if axis is None else np.full(_shape_drop(arr.shape, axis), 1.0)
    if axis is None:
        return _as_scalar(np.take(arr.ravel().cumprod(), [arr.size - 1]))
    return _reduce_last_along(arr.cumprod(axis), axis)


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
    """忽略 NaN 求和。"""
    _ = dtype, out, keepdims
    return _np().sum(_rep_nan(a, 0.0), axis)


def nanprod(a, axis=None, dtype=None, out=None, keepdims=False):
    """忽略 NaN 求积。"""
    _ = dtype, out, keepdims
    return prod(_rep_nan(a, 1.0), axis)


def nanmax(a, axis=None, out=None, keepdims=False):
    """忽略 NaN 求最大值。"""
    _ = out, keepdims
    return _np().max(_rep_nan(a, -_math.inf), axis)


def nanmin(a, axis=None, out=None, keepdims=False):
    """忽略 NaN 求最小值。"""
    _ = out, keepdims
    return _np().min(_rep_nan(a, _math.inf), axis)


def nanmean(a, axis=None, dtype=None, out=None, keepdims=False):
    """忽略 NaN 求均值。"""
    _ = dtype, out, keepdims
    np = _np()
    s = nansum(a, axis)
    cnt = _count_notnan(a, axis)
    if axis is None:
        return float(_as_scalar(s)) / float(_as_scalar(cnt))
    return np.divide(s, cnt)


def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """忽略 NaN 求方差。"""
    _ = dtype, out, keepdims
    np = _np()
    arr = _asarray(a)
    if axis is None:
        cnt = float(_as_scalar(_count_notnan(arr, None)))
        n = cnt - ddof
        if n <= 0:
            return _math.nan
        s = float(_as_scalar(np.sum(_rep_nan(arr, 0.0), None)))
        mu = s / cnt
        diff = np.subtract(arr, np.full(arr.shape, mu))
        sq = np.where(np.isnan(arr), np.full(arr.shape, 0.0), np.multiply(diff, diff))
        return float(_as_scalar(np.sum(sq, None))) / n
    m = nanmean(a, axis)
    # 广播均值后按 axis 归约
    diff = np.subtract(arr, np.expand_dims(m, axis))
    sq = np.multiply(diff, diff)
    sq = np.where(np.isnan(arr), np.full(arr.shape, 0.0), sq)
    cnt = _count_notnan(a, axis)
    return np.divide(np.sum(sq, axis), np.subtract(cnt, float(ddof)))


def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """忽略 NaN 求标准差。"""
    _ = dtype, out, keepdims
    np = _np()
    v = nanvar(a, axis, ddof=ddof)
    if axis is None:
        return _math.sqrt(v)
    return np.sqrt(v)


def nanargmax(a, axis=None):
    """忽略 NaN 的最大值索引。"""
    return _np().argmax(_rep_nan(a, -_math.inf), axis)


def nanargmin(a, axis=None):
    """忽略 NaN 的最小值索引。"""
    return _np().argmin(_rep_nan(a, _math.inf), axis)


def nancumsum(a, axis=None):
    """忽略 NaN 的累积和（NaN 视为 0）。"""
    return _np().cumsum(_rep_nan(a, 0.0), axis)


def nancumprod(a, axis=None):
    """忽略 NaN 的累积积（NaN 视为 1）。"""
    return _np().cumprod(_rep_nan(a, 1.0), axis)


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
    if axis is None:
        return _core.count_nonzero(_asarray(a)._array)
    arr = _asarray(a)
    mask = np.where(np.equal(arr, np.full(arr.shape, 0.0)), np.full(arr.shape, 0.0), np.full(arr.shape, 1.0))
    return np.sum(mask, axis)


# ========== 集合运算 ==========
def unique_values(x):
    """返回排序后的唯一值。"""
    return _np().unique(x)


def _sorted_unique_counts(x):
    vals = sorted(_flat(x))
    uniq = []
    counts = []
    for v in vals:
        if uniq and uniq[-1] == v:
            counts[-1] += 1
        else:
            uniq.append(v)
            counts.append(1)
    return uniq, counts


def unique_counts(x):
    """返回 (唯一值, 出现次数)。"""
    np = _np()
    uniq, counts = _sorted_unique_counts(x)
    return np.array(uniq), np.array(counts, dtype="int64")


def unique_inverse(x):
    """返回 (唯一值, 可重建原数组的逆索引)。"""
    np = _np()
    flat = _flat(x)
    uniq = sorted(set(flat))
    pos = {v: i for i, v in enumerate(uniq)}
    inv = [pos[v] for v in flat]
    return np.array(uniq), np.array(inv, dtype="int64")


def unique_all(x):
    """返回 (唯一值, 首次出现索引, 逆索引, 计数)。"""
    np = _np()
    flat = _flat(x)
    uniq = sorted(set(flat))
    pos = {v: i for i, v in enumerate(uniq)}
    first = {}
    counts = [0] * len(uniq)
    inv = []
    for idx, v in enumerate(flat):
        i = pos[v]
        inv.append(i)
        counts[i] += 1
        if v not in first:
            first[v] = idx
    indices = [first[v] for v in uniq]
    return (np.array(uniq), np.array(indices, dtype="int64"),
            np.array(inv, dtype="int64"), np.array(counts, dtype="int64"))


def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    """返回两个数组的交集（升序去重）。"""
    _ = assume_unique, return_indices
    s2 = set(_flat(ar2))
    return _np().array(sorted({v for v in _flat(ar1) if v in s2}))


def union1d(ar1, ar2):
    """返回两个数组的并集（升序去重）。"""
    return _np().array(sorted(set(_flat(ar1)) | set(_flat(ar2))))


def setdiff1d(ar1, ar2, assume_unique=False):
    """返回在 ar1 但不在 ar2 中的唯一值。"""
    _ = assume_unique
    s2 = set(_flat(ar2))
    return _np().array(sorted({v for v in _flat(ar1) if v not in s2}))


def setxor1d(ar1, ar2, assume_unique=False):
    """返回两个数组的对称差集。"""
    _ = assume_unique
    return _np().array(sorted(set(_flat(ar1)) ^ set(_flat(ar2))))


def isin(element, test_elements, assume_unique=False, invert=False):
    """逐元素判断是否属于 test_elements。"""
    _ = assume_unique
    raw = _core.isin(_asarray(element)._array, _asarray(test_elements)._array, invert)
    return _wrap(raw, "bool")


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
    return arr.shape[axis]


def isscalar(element):
    """判断是否为标量。"""
    return isinstance(element, (int, float, complex, bool, bytes, str))


def iterable(y):
    """判断对象是否可迭代。"""
    try:
        iter(y)
        return True
    except TypeError:
        return False


def iscomplexobj(x):
    """判断对象是否为复数类型。"""
    if getattr(x, "_complex_data", None) is not None:
        return True
    return getattr(x, "_dtype", "") in ("complex128", "complex64")


def isrealobj(x):
    """判断对象是否为实数类型。"""
    return not iscomplexobj(x)


def isreal(x):
    """逐元素判断虚部是否为 0。"""
    np = _np()
    if iscomplexobj(x):
        return np.array([v.imag == 0 for v in x._complex_data], dtype="bool")
    arr = _asarray(x)
    return np.array(_map_nested(lambda v: True, arr.tolist()), dtype="bool")


def iscomplex(x):
    """逐元素判断虚部是否非 0。"""
    np = _np()
    if iscomplexobj(x):
        return np.array([v.imag != 0 for v in x._complex_data], dtype="bool")
    arr = _asarray(x)
    return np.array(_map_nested(lambda v: False, arr.tolist()), dtype="bool")


def isnat(x):
    """逐元素判断是否为 NaT（暂不支持 NaT，返回全 False）。"""
    np = _np()
    arr = _asarray(x)
    return np.array(_map_nested(lambda v: False, arr.tolist()), dtype="bool")


def isneginf(x, out=None):
    """逐元素判断是否为负无穷。"""
    _ = out
    np = _np()
    arr = _asarray(x)
    return np.logical_and(np.isinf(arr), np.less(arr, 0.0))


def isposinf(x, out=None):
    """逐元素判断是否为正无穷。"""
    _ = out
    np = _np()
    arr = _asarray(x)
    return np.logical_and(np.isinf(arr), np.greater(arr, 0.0))


def real(val):
    """返回实部。"""
    np = _np()
    if iscomplexobj(val):
        return np.array([v.real for v in val._complex_data])
    return _asarray(val)


def imag(val):
    """返回虚部。"""
    np = _np()
    if iscomplexobj(val):
        return np.array([v.imag for v in val._complex_data])
    arr = _asarray(val)
    return np.full(arr.shape, 0.0)


def conjugate(x):
    """返回共轭。"""
    np = _np()
    if iscomplexobj(x):
        return np.array([complex(v).conjugate() for v in x._complex_data])
    return _asarray(x)


conj = conjugate


def angle(z, deg=False):
    """返回复数的相位角。"""
    np = _np()
    if iscomplexobj(z):
        vals = [_math.atan2(v.imag, v.real) for v in z._complex_data]
    else:
        vals = [_math.atan2(0.0, v) for v in _flat(z)]
    if deg:
        vals = [v * 180.0 / _math.pi for v in vals]
    return np.array(vals)


def real_if_close(a, tol=100):
    """若虚部接近 0 则返回实部，否则原样返回。"""
    _ = tol
    if iscomplexobj(a):
        if all(abs(complex(v).imag) < 1e-13 for v in a._complex_data):
            return real(a)
    return _asarray(a)


def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    """将 NaN/Inf 替换为有限数。"""
    _ = copy
    np = _np()
    arr = _asarray(x)
    big = _math.inf
    pos = 1.7976931348623157e308 if posinf is None else float(posinf)
    neg = -1.7976931348623157e308 if neginf is None else float(neginf)
    res = np.where(np.isnan(arr), np.full(arr.shape, float(nan)), arr)
    res = np.where(np.logical_and(np.isinf(res), np.greater(res, 0.0)), np.full(arr.shape, pos), res)
    res = np.where(np.logical_and(np.isinf(res), np.less(res, 0.0)), np.full(arr.shape, neg), res)
    _ = big
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


def atleast_1d(*arys):
    """确保输入至少为一维。"""
    np = _np()
    out = []
    for a in arys:
        arr = _asarray(a)
        if arr.ndim == 0:
            arr = np.reshape(arr, (1,))
        out.append(arr)
    return out[0] if len(out) == 1 else out


def atleast_2d(*arys):
    """确保输入至少为二维。"""
    np = _np()
    out = []
    for a in arys:
        arr = _asarray(a)
        if arr.ndim == 0:
            arr = np.reshape(arr, (1, 1))
        elif arr.ndim == 1:
            arr = np.reshape(arr, (1, arr.shape[0]))
        out.append(arr)
    return out[0] if len(out) == 1 else out


def atleast_3d(*arys):
    """确保输入至少为三维。"""
    np = _np()
    out = []
    for a in arys:
        arr = _asarray(a)
        if arr.ndim == 0:
            arr = np.reshape(arr, (1, 1, 1))
        elif arr.ndim == 1:
            arr = np.reshape(arr, (1, arr.shape[0], 1))
        elif arr.ndim == 2:
            arr = np.reshape(arr, (arr.shape[0], arr.shape[1], 1))
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
    """按嵌套列表组装数组块。"""
    np = _np()

    def _depth(x):
        if isinstance(x, list):
            return 1 + (max((_depth(e) for e in x), default=0))
        return 0

    def _leaf_ndim(x):
        if isinstance(x, list):
            return max((_leaf_ndim(e) for e in x), default=0)
        return _asarray(x).ndim

    list_depth = _depth(arrays)
    result_ndim = builtin_max(_leaf_ndim(arrays), list_depth)

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

    return _build(arrays, 0)


def array_split(ary, indices_or_sections, axis=0):
    """将数组切分为多个子数组，允许不等分。"""
    np = _np()
    arr = _asarray(ary)
    n = arr.shape[axis]
    if isinstance(indices_or_sections, int):
        k = indices_or_sections
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
    result = []
    prev = 0
    for b in list(bounds) + [n]:
        idx = list(range(prev, b))
        result.append(np.take(arr, idx, axis=axis))
        prev = b
    return result


def unstack(x, axis=0):
    """沿指定轴拆成子数组元组（NumPy 2.1）。"""
    np = _np()
    arr = _asarray(x)
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
    np = _np()
    a = _asarray(arr)
    idx = _asarray(indices)
    if axis is None:
        a = a.ravel()
        return np.array([a.tolist()[int(i)] for i in _flat(idx)])
    al = a.tolist()
    il = idx.tolist()

    def rec(ad, idd, ax):
        if ax == 0:
            return [ad[int(i)] for i in idd]
        return [rec(ad[k], idd[k], ax - 1) for k in range(len(idd))]

    return np.array(rec(al, il, axis))


def put_along_axis(arr, indices, values, axis):
    """沿轴按索引写入值（就地）。"""
    a = _asarray(arr)
    idx = _flat(indices)
    vals = _flat(values)
    if axis is None or a.ndim == 1:
        data = a.tolist()
        for k, i in enumerate(idx):
            data[int(i)] = vals[k % len(vals)]
        a._array = _np().array(data)._array
    return None


def choose(a, choices, out=None, mode="raise"):
    """按索引数组从 choices 中选值。"""
    _ = out, mode
    np = _np()
    idx = _flat(a)
    ch = [_flat(c) for c in choices]
    res = [ch[int(idx[k])][k] for k in range(len(idx))]
    shp = _asarray(a).shape
    return np.reshape(np.array(res), shp)


def compress(condition, a, axis=None):
    """按布尔条件沿轴筛选元素。"""
    np = _np()
    cond = _flat(condition)
    arr = _asarray(a)
    if axis is None:
        flat = _flat(arr)
        return np.array([flat[i] for i in range(len(flat)) if i < len(cond) and cond[i]])
    keep = [i for i, c in enumerate(cond) if c]
    return np.take(arr, keep, axis=axis)


def copyto(dst, src, casting="same_kind", where=True):
    """将 src 的值复制到 dst（就地）。"""
    _ = casting, where
    np = _np()
    s = _asarray(src)
    b = np.broadcast_to(s, dst.shape) if s.shape != dst.shape else s
    dst._array = np.array(b.tolist())._array
    return None


def place(arr, mask, vals):
    """按布尔掩码把 vals 循环写入 arr（就地）。"""
    a = _asarray(arr)
    m = _flat(mask)
    v = _flat(vals)
    data = _flat(a)
    j = 0
    for i in range(len(data)):
        if i < len(m) and m[i]:
            data[i] = v[j % len(v)]
            j += 1
    a._array = _np().reshape(_np().array(data), a.shape)._array
    return None


def putmask(a, mask, values):
    """按布尔掩码写入值（place 的别名语义）。"""
    return place(a, mask, values)


def pad(array, pad_width, mode="constant", **kwargs):
    """填充数组边缘，支持 constant/edge/reflect/wrap。"""
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
    """裁剪一维数组两端的零。"""
    np = _np()
    data = _flat(filt)
    start = 0
    end = len(data)
    if "f" in trim:
        while start < end and data[start] == 0:
            start += 1
    if "b" in trim:
        while end > start and data[end - 1] == 0:
            end -= 1
    return np.array(data[start:end])


def ediff1d(ary, to_end=None, to_begin=None):
    """一维相邻元素差分，可选拼接首尾。"""
    np = _np()
    data = _flat(ary)
    diffs = [data[i + 1] - data[i] for i in range(len(data) - 1)]
    if to_begin is not None:
        diffs = _flat(to_begin) + diffs
    if to_end is not None:
        diffs = diffs + _flat(to_end)
    return np.array(diffs)


def diff(a, n=1, axis=-1):
    """沿轴计算 n 阶差分。"""
    return _wrap(_core.diff(_asarray(a)._array, n, axis))


def gradient(f, *varargs, axis=None, edge_order=1):
    """计算梯度。"""
    _ = varargs, axis, edge_order
    return _wrap(_core.gradient(_asarray(f)._array))


def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """向量叉积。"""
    _ = axisa, axisb, axisc, axis
    return _wrap(_core.cross(_asarray(a)._array, _asarray(b)._array))


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
    _ = out
    np = _np()
    av = _asarray(a).ravel()
    bv = _asarray(b).ravel()
    ar = np.reshape(av, (av.size, 1))
    br = np.reshape(bv, (1, bv.size))
    return np.multiply(ar, br)


def kron(a, b):
    """Kronecker 积（支持 1D/2D）。"""
    np = _np()
    A = _asarray(a)
    B = _asarray(b)
    if A.ndim == 1 and B.ndim == 1:
        return outer(A, B).ravel()
    A2 = A if A.ndim == 2 else np.reshape(A, (1, A.size))
    B2 = B if B.ndim == 2 else np.reshape(B, (1, B.size))
    ia, ja = A2.shape
    ib, jb = B2.shape
    ar = np.reshape(A2, (ia, 1, ja, 1))
    br = np.reshape(B2, (1, ib, 1, jb))
    prod4 = np.multiply(ar, br)
    return np.reshape(prod4, (ia * ib, ja * jb))


def tensordot(a, b, axes=2):
    """张量点积（支持整数或两序列形式的 axes）。"""
    np = _np()
    A = _asarray(a)
    B = _asarray(b)
    if isinstance(axes, int):
        axes_a = list(range(A.ndim - axes, A.ndim))
        axes_b = list(range(axes))
    else:
        axes_a, axes_b = axes
        axes_a = list(axes_a) if isinstance(axes_a, (list, tuple)) else [axes_a]
        axes_b = list(axes_b) if isinstance(axes_b, (list, tuple)) else [axes_b]
    notin_a = [i for i in range(A.ndim) if i not in axes_a]
    notin_b = [i for i in range(B.ndim) if i not in axes_b]
    newaxes_a = notin_a + axes_a
    newaxes_b = axes_b + notin_b
    at = np.transpose(A, tuple(newaxes_a))
    bt = np.transpose(B, tuple(newaxes_b))
    n1 = 1
    for i in notin_a:
        n1 *= A.shape[i]
    nc = 1
    for i in axes_a:
        nc *= A.shape[i]
    n2 = 1
    for i in notin_b:
        n2 *= B.shape[i]
    am = np.reshape(at.ravel(), (n1, nc))
    bm = np.reshape(bt.ravel(), (nc, n2))
    res = np.matmul(am, bm)
    if not hasattr(res, "_array"):
        res = _wrap(res)
    outshape = tuple(A.shape[i] for i in notin_a) + tuple(B.shape[i] for i in notin_b)
    return np.reshape(res.ravel(), outshape) if outshape else _as_scalar(res)


def tri(N, M=None, k=0, dtype=None):
    """下三角为 1、其余为 0 的矩阵。"""
    _ = dtype
    np = _np()
    if M is None:
        M = N
    rows = np.reshape(np.arange(0, N, 1), (N, 1))
    cols = np.reshape(np.arange(0, M, 1), (1, M))
    return np.where(np.less_equal(cols, np.add(rows, float(k))),
                    np.full((N, M), 1.0), np.full((N, M), 0.0))


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
    """提取对角线。"""
    _ = axis1, axis2
    np = _np()
    arr = _asarray(a).tolist()
    n = len(arr)
    m = len(arr[0]) if n else 0
    out = []
    i = 0
    while True:
        r = i if offset >= 0 else i - offset
        c = i + offset if offset >= 0 else i
        if r >= n or c >= m:
            break
        out.append(arr[r][c])
        i += 1
    return np.array(out)


def diag(v, k=0):
    """一维->对角矩阵，二维->提取对角线。"""
    np = _np()
    arr = _asarray(v)
    if arr.ndim == 2:
        return diagonal(arr, k)
    data = _flat(arr)
    n = len(data) + abs(k)
    out = [[0.0] * n for _ in range(n)]
    for i, val in enumerate(data):
        if k >= 0:
            out[i][i + k] = val
        else:
            out[i - k][i] = val
    return np.array(out)


def diagflat(v, k=0):
    """将输入展平后构造对角矩阵。"""
    return diag(_np().array(_flat(v)), k)


def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    """对角线元素之和。"""
    _ = axis1, axis2, dtype, out
    return sum(_flat(diagonal(a, offset)))


def tri_indices_helper(n, k, upper):
    idx_r = []
    idx_c = []
    for r in range(n):
        for c in range(n):
            if (upper and c >= r + k) or (not upper and c <= r + k):
                idx_r.append(r)
                idx_c.append(c)
    return idx_r, idx_c


def tril_indices(n, k=0, m=None):
    """下三角元素的索引。"""
    np = _np()
    mm = n if m is None else m
    r = []
    c = []
    for i in range(n):
        for j in range(mm):
            if j <= i + k:
                r.append(i)
                c.append(j)
    return np.array(r, dtype="int64"), np.array(c, dtype="int64")


def triu_indices(n, k=0, m=None):
    """上三角元素的索引。"""
    np = _np()
    mm = n if m is None else m
    r = []
    c = []
    for i in range(n):
        for j in range(mm):
            if j >= i + k:
                r.append(i)
                c.append(j)
    return np.array(r, dtype="int64"), np.array(c, dtype="int64")


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
    _ = wrap
    np = _np()
    arr = _asarray(a)
    data = arr.tolist()
    n = min(len(data), len(data[0]))
    vals = _flat(val)
    for i in range(n):
        data[i][i] = vals[i % len(vals)]
    a._array = np.array(data)._array
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
    np = _np()
    arr = _asarray(x)
    xf = _flat(arr)

    def bessel(v):
        total = 1.0
        term = 1.0
        half = v / 2.0
        for k in range(1, 40):
            term *= (half / k) ** 2
            total += term
            if term < 1e-18 * total:
                break
        return total

    res = [bessel(float(v)) for v in xf]
    return np.reshape(np.array(res), arr.shape) if arr.ndim else _as_scalar(np.array(res))


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


def convolve(a, v, mode="full"):
    """一维离散卷积。"""
    np = _np()
    x = _flat(a)
    h = _flat(v)
    n, m = len(x), len(h)
    full = [0.0] * (n + m - 1)
    for i in range(n):
        xi = x[i]
        for j in range(m):
            full[i + j] += xi * h[j]
    if mode == "full":
        return np.array(full)
    if mode == "same":
        start = (m - 1) // 2
        return np.array(full[start:start + n])
    if mode == "valid":
        length = max(n, m) - min(n, m) + 1
        start = min(n, m) - 1
        return np.array(full[start:start + length])
    raise ValueError(f"unsupported mode: {mode}")


def correlate(a, v, mode="valid"):
    """一维互相关。"""
    return convolve(a, list(reversed(_flat(v))), mode)


def interp(x, xp, fp, left=None, right=None, period=None):
    """一维线性插值。"""
    _ = period
    np = _np()
    xpv = _flat(xp)
    fpv = _flat(fp)
    lo = fpv[0] if left is None else left
    hi = fpv[-1] if right is None else right

    def one(xi):
        if xi <= xpv[0]:
            return lo
        if xi >= xpv[-1]:
            return hi
        j = 0
        while j < len(xpv) - 1 and not (xpv[j] <= xi <= xpv[j + 1]):
            j += 1
        x0, x1 = xpv[j], xpv[j + 1]
        y0, y1 = fpv[j], fpv[j + 1]
        if x1 == x0:
            return y0
        return y0 + (y1 - y0) * (xi - x0) / (x1 - x0)

    arr = _asarray(x)
    res = _map_nested(one, arr.tolist()) if arr.ndim else one(_as_scalar(arr))
    return np.array(res) if arr.ndim else res


def bincount(x, weights=None, minlength=0):
    """统计非负整数数组中每个值出现的次数。"""
    np = _np()
    if weights is None:
        return _wrap(_core.bincount(_asarray(x)._array, minlength), "int64")
    xv = [int(v) for v in _flat(x)]
    n = builtin_max(minlength, (builtin_max(xv) + 1) if xv else 0)
    wv = _flat(weights)
    out = [0.0] * n
    for i, v in enumerate(xv):
        out[v] += wv[i]
    return np.array(out)


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
    _ = axis
    np = _np()
    data = _flat(p)
    if discont is None:
        discont = period / 2.0
    out = list(data)
    for i in range(1, len(out)):
        delta = out[i] - out[i - 1]
        steps = round(delta / period)
        if abs(delta - steps * period) > discont or abs(delta) > discont:
            out[i] -= steps * period
    return np.array(out)


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
]
